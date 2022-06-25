from tqdm import tqdm
import ujson as json
import numpy as np
import unidecode
import random
import os

docred_rel2id = json.load(open('meta/rel2id.json', 'r'))
ctd_rel2id = json.load(open('meta/relation_map.json', 'r'))
ENTITY_PAIR_TYPE_SET = set(
    [("Chemical", "Disease"), ("Chemical", "Gene"), ("Gene", "Disease")])

def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res

def map_index(chars, tokens):
    # position index mapping from character level offset to token level offset
    ind_map = {}
    i, k = 0, 0  # (character i to token k)
    len_char = len(chars)
    num_token = len(tokens)
    while k < num_token:
        if i < len_char and chars[i].strip() == "":
            ind_map[i] = k
            i += 1
            continue
        token = tokens[k]
        if token[:2] == "##":
            token = token[2:]
        if token[:1] == "Ġ":
            token = token[1:]

        # assume that unk is always one character in the input text.
        if token != chars[i:(i+len(token))]:
            ind_map[i] = k
            i += 1
            k += 1
        else:
            for _ in range(len(token)):
                ind_map[i] = k
                i += 1
            k += 1

    return ind_map

def read_chemdisgene(args, file_in, tokenizer, max_seq_length=1024, lower=True):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    pos, neg, pos_labels, neg_labels = {}, {}, {}, {}
    for pair in list(ENTITY_PAIR_TYPE_SET):
        pos[pair] = 0
        neg[pair] = 0
        pos_labels[pair] = 0
        neg_labels[pair] = 0
    ent_nums = 0
    rel_nums = 0
    max_len = 0
    features = []
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    padid = tokenizer.pad_token_id
    cls_token_length = len(cls_token)
    print(cls_token, sep_token)
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)

    re_fre = np.zeros(len(ctd_rel2id))
    for idx, sample in tqdm(enumerate(data), desc="Example"):
        if "title" in sample and "abstract" in sample:
            text = sample["title"] + sample["abstract"]
            if lower == True:
                text = text.lower()
        else:
            text = sample["text"]
            if lower == True:
                text = text.lower()

        text = unidecode.unidecode(text)
        tokens = tokenizer.tokenize(text)
        tokens = [cls_token] + tokens + [sep_token]
        text = cls_token + " " + text + " " + sep_token

        ind_map = map_index(text, tokens)

        entities = sample['entity']
        entity_start, entity_end = [], []

        train_triple = {}
        if "relation" in sample:
            for label in sample['relation']:
                if label['type'] not in ctd_rel2id:
                    continue
                if 'evidence' not in label:
                    evidence = []
                else:
                    evidence = label['evidence']
                r = int(ctd_rel2id[label['type']])

                if (label['subj'], label['obj']) not in train_triple:
                    train_triple[(label['subj'], label['obj'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['subj'], label['obj'])].append(
                        {'relation': r, 'evidence': evidence})

        entity_pos = []
        entity_dict = {}
        entity2id = {}
        entity_type = {}
        eids = 0
        offset = 0

        for e in entities:

            entity_type[e["id"]] = e["type"]
            if e["start"] + cls_token_length in ind_map:
                startid = ind_map[e["start"] + cls_token_length] + offset
                tokens = tokens[:startid] + ['*'] + tokens[startid:]
                offset += 1
            else:
                continue
                startid = 0


            if e["end"] + cls_token_length in ind_map:
                endid = ind_map[e["end"] + cls_token_length] + offset
                if ind_map[e["start"] + cls_token_length] >= ind_map[e["end"] + cls_token_length]:
                    endid += 1
                tokens = tokens[:endid] + ['*'] + tokens[endid:]
                endid += 1
                offset += 1
            else:
                continue
                endid = 0

            if startid >= endid:
                endid = startid + 1

            if e["id"] not in entity_dict:
                entity_dict[e["id"]] = [(startid, endid,)]
                entity2id[e["id"]] = eids
                eids += 1
                if e["id"] != "-":
                    ent_nums += 1
            else:
                entity_dict[e["id"]].append((startid, endid,))

        relations, hts = [], []
        for h, t in train_triple.keys():
            if h not in entity2id or t not in entity2id or ((entity_type[h], entity_type[t]) not in ENTITY_PAIR_TYPE_SET):
                continue
            relation = [0] * (len(ctd_rel2id) + 1)
            for mention in train_triple[h, t]:
                if relation[mention["relation"] + 1] == 0:
                    re_fre[mention["relation"]] += 1
                relation[mention["relation"] + 1] = 1
                evidence = mention["evidence"]
                
            relations.append(relation)
            hts.append([entity2id[h], entity2id[t]])

            rel_num = sum(relation)
            rel_nums += rel_num
            pos_labels[(entity_type[h], entity_type[t])] += rel_num
            pos[(entity_type[h], entity_type[t])] += 1
            pos_samples += 1

        for h in entity_dict.keys():
            for t in entity_dict.keys():
                if (h != t) and ([entity2id[h], entity2id[t]] not in hts) and ((entity_type[h], entity_type[t]) in ENTITY_PAIR_TYPE_SET) and (h != "-") and (t != "-"):
                    if (entity_type[h], entity_type[t]) not in neg:
                        neg[(entity_type[h], entity_type[t])] = 1
                    else:
                        neg[(entity_type[h], entity_type[t])] += 1
                    
                    relation = [1] + [0] * (len(ctd_rel2id))
                    relations.append(relation)
                    hts.append([entity2id[h], entity2id[t]])
                    neg_samples += 1

        if len(tokens) > max_len:
            max_len = len(tokens)

        tokens = tokens[1:-1][:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        i_line += 1

        feature = {'input_ids': input_ids,
                'entity_pos': list(entity_dict.values()),
                'labels': relations,
                'hts': hts,
                'title': sample['docid'],
                }
        features.append(feature)

    print("# of documents {}.".format(i_line))
    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples))
    re_fre = 1. * re_fre / (pos_samples + neg_samples)
    print(re_fre)
    print(max_len)
    print(pos)
    print(pos_labels)
    print(neg)
    print("# ents per doc", 1. * ent_nums / i_line)
    print("# rels per doc", 1. * rel_nums / i_line)
    return features, re_fre

def read_docred(args, file_in, tokenizer, max_seq_length=1024):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    rel_nums = 0
    features = []
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)

    re_fre = np.zeros(len(docred_rel2id) - 1)
    for idx, sample in tqdm(enumerate(data), desc="Example"):
        sents = []
        sent_map = []

        entities = sample['vertexSet']
        entity_start, entity_end = [], []
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0],))
                entity_end.append((sent_id, pos[1] - 1,))
        for i_s, sent in enumerate(sample['sents']):
            new_map = {}
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)

                if (i_s, i_t) in entity_start:
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)

        train_triple = {}
        if "labels" in sample:
            for label in sample['labels']:
                if 'evidence' not in label:
                    evidence = []
                else:
                    evidence = label['evidence']
                r = int(docred_rel2id[label['r']])
                re_fre[r - 1] += 1
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence})

        entity_pos = []
        for e in entities:
            entity_pos.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end,))


        relations, hts = [], []
        for h, t in train_triple.keys():
            relation = [0] * len(docred_rel2id)
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1
                evidence = mention["evidence"]
                rel_nums += 1
            relations.append(relation)
            hts.append([h, t])
            pos_samples += 1

        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t and [h, t] not in hts:
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    relations.append(relation)
                    hts.append([h, t])
                    neg_samples += 1

        assert len(relations) == len(entities) * (len(entities) - 1)

        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        i_line += 1
        feature = {'input_ids': input_ids,
                'entity_pos': entity_pos,
                'labels': relations,
                'hts': hts,
                'title': sample['title'],
                }
        features.append(feature)

    print("# of documents {}.".format(i_line))
    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples))
    re_fre = 1. * re_fre / (pos_samples + neg_samples)
    print(re_fre)
    print("# rels per doc", 1. * rel_nums / i_line)
    return features, re_fre
