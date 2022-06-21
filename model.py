from cProfile import label
import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract
from long_seq import process_long_input
import numpy as np


class DocREModel(nn.Module):
    def __init__(self, args, config, priors_l, priors_o, model, emb_size=768, block_size=64):
        super().__init__()
        self.args = args
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.priors_l = priors_l
        self.priors_o = priors_o
        self.weight = ((1 - self.priors_o)/self.priors_o) ** 0.5
        self.margin = args.m
        if args.isrank:
            self.rels = args.num_class-1
        else:
            self.rels = args.num_class

        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)
            entity_atts = torch.stack(entity_atts, dim=0)

            if len(hts[i]) == 0:
                hss.append(torch.FloatTensor([]).to(sequence_output.device))
                tss.append(torch.FloatTensor([]).to(sequence_output.device))
                rss.append(torch.FloatTensor([]).to(sequence_output.device))
                continue
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)

        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss

    def square_loss(self, yPred, yTrue, margin=1.):
        if len(yPred) == 0:
            return torch.FloatTensor([0]).cuda()
        loss = 0.25 * (yPred * yTrue - margin) ** 2
        return torch.mean(loss.sum() / yPred.shape[0])

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                ):

        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)

        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))    # zs
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))    # zo
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)

        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            
            risk_sum = torch.FloatTensor([0]).cuda()
            for i in range(self.rels):
                
                if self.args.isrank:
                    neg = (logits[(labels[:, i + 1] != 1), i + 1] - logits[(labels[:, i + 1] != 1), 0])
                    pos = (logits[(labels[:, i + 1] == 1), i + 1] - logits[(labels[:, i + 1] == 1), 0])
                else:
                    neg = logits[(labels[:, i + 1] != 1), i]
                    pos = logits[(labels[:, i + 1] == 1), i]

                if self.args.m_tag == 'PN':
                    # risk1 = N(-)_risk
                    risk1 = (1. - self.priors_o[i]) * self.square_loss(neg, -1., self.margin)
                    # risk2 = P(+)_risk
                    risk2 = self.priors_o[i] * self.square_loss(pos, 1., self.margin) * self.weight[i]
                    risk = risk1 + risk2
                    
                elif self.args.m_tag == 'S-PU':
                    priors_u = (self.priors_o[i] - self.priors_l[i]) / (1. - self.priors_l[i])
                    risk1 = (((1. - self.priors_o[i]) / (1. - priors_u)) * self.square_loss(neg, -1., self.margin) - 
                                ((priors_u - priors_u * self.priors_o[i]) / (1. - priors_u)) * self.square_loss(pos, -1., self.margin))

                    risk2 = self.priors_o[i] * self.square_loss(pos, 1., self.margin) * self.weight[i]
                    risk = risk1 + risk2

                    if risk1 < self.args.beta:
                        risk = - self.args.gamma * risk1

                elif self.args.m_tag == 'PU':
                    # risk1 = U(-)_risk - P(-)_risk
                    risk1 = (self.square_loss(neg, -1., self.margin) -
                                self.priors_o[i] * self.square_loss(pos, -1., self.margin))
                    # risk2 = P(+)_risk
                    risk2 = self.priors_o[i] * self.square_loss(pos, 1., self.margin) * self.weight[i]
                    risk = risk1 + risk2

                    if risk1 < self.args.beta:
                        risk = - self.args.gamma * risk1

                risk_sum += risk
            return risk_sum, logits

        return logits

