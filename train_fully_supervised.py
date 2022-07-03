import argparse
import os

import numpy as np
import torch
from apex import amp
import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from model import DocREModel
from utils import set_seed, collate_fn
from prepro import read_docred
from evaluation import official_evaluate, to_official


def train(args, model, train_features, dev_features):
    def finetune(features, optimizer, num_epoch, num_steps):
        best_score = -1
        best_dev_output = None
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        for epoch in tqdm(train_iterator):
            model.zero_grad()
            for step, batch in enumerate(tqdm(train_dataloader)):
                model.train()

                inputs = {'input_ids': batch[0].to(args.device),
                        'attention_mask': batch[1].to(args.device),
                        'labels': batch[2],
                        'entity_pos': batch[3],
                        'hts': batch[4],
                        }

                outputs = model(**inputs)
                loss = outputs[0] / args.gradient_accumulation_steps
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1

                if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                    dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
                    print(dev_output)

                    print("loss:", loss.item(), "   step:", num_steps)

                    if dev_score > best_score:
                        best_score = dev_score
                        best_dev_output = dev_output
                        torch.save(model.state_dict(), args.save_path)

        print('best dev f1:', best_dev_output)
        return num_steps

    new_layer = ["extractor", "bilinear"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 1e-4},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps)


def evaluate(args, model, features, tag="test"):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            logits = model(**inputs)
            logits = logits.cpu().numpy()

            if args.isrank:
                pred = np.zeros((logits.shape[0], logits.shape[1]))
                for i in range(1, logits.shape[1]):
                    pred[(logits[:, i] > logits[:, 0]), i] = 1
                pred[:, 0] = (pred.sum(1) == 0)
            else:
                pred = np.zeros((logits.shape[0], logits.shape[1] + 1))
                for i in range(logits.shape[1]):
                    pred[(logits[:, i] > 0.), i + 1] = 1
                pred[:, 0] = (pred.sum(1) == 0)

            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, features)
    if len(ans) > 0:
        best_f1, _, best_f1_ign, re_f1_ignore_train, re_p, re_r = official_evaluate(ans, args.data_dir, tag, args)
        output = {
            tag + "_F1": best_f1 * 100,
            tag + "_F1_ign": best_f1_ign * 100,
            "re_p": re_p * 100,
            "re_r": re_r * 100,
        }
    else:
        best_f1, best_f1_ign = -1, -1
        output = {
            tag + "_F1": best_f1 * 100,
            tag + "_F1_ign": best_f1_ign * 100,
        }
    return best_f1, output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)

    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_path", default="out", type=str)
    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")
    parser.add_argument("--isrank", type=int, default='1 means use ranking loss, 0 means not use')
    parser.add_argument("--m_tag", type=str, default='PN/PU/S-PU')
    parser.add_argument('--beta', type=float, default=0.0, help='beta of pu learning (default 0.0)')
    parser.add_argument('--gamma', type=float, default=1.0, help='gamma of pu learning (default 1.0)')
    parser.add_argument('--m', type=float, default=1.0, help='margin')
    parser.add_argument('--e', type=float, default=3.0, help='estimated a priors multiple')

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    file_name = "{}_{}_{}_{}_isrank_{}_m_{}_e_{}_seed_{}".format(
        args.train_file.split('.')[0],
        args.transformer_type,
        args.data_dir.split('/')[-1],
        args.m_tag,
        str(args.isrank),
        args.m,
        args.e,
        str(args.seed))
    args.save_path = os.path.join(args.save_path, file_name)
    print(args.save_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    read = read_docred

    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)
    train_features, priors = read(args, train_file, tokenizer, max_seq_length=args.max_seq_length)
    dev_features, _ = read(args, dev_file, tokenizer, max_seq_length=args.max_seq_length)
    test_features, _ = read(args, test_file, tokenizer, max_seq_length=args.max_seq_length)

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    model = DocREModel(args, config, priors, priors * args.e, model)
    model.to(0)

    print(args.m_tag, args.isrank)

    if args.load_path == "":  # Training
        train(args, model, train_features, dev_features)

        print("TEST")
        model = amp.initialize(model, opt_level="O1", verbosity=0)
        model.load_state_dict(torch.load(args.save_path))
        test_score, test_output = evaluate(args, model, test_features, tag="test")
        print(test_output)

    else:  # Testing
        args.load_path = os.path.join(args.load_path, file_name)
        print(args.load_path)
        
        print("TEST")
        model = amp.initialize(model, opt_level="O1", verbosity=0)
        model.load_state_dict(torch.load(args.load_path))
        test_score, test_output = evaluate(args, model, test_features, tag="test")
        print(test_output)


if __name__ == "__main__":
    main()
