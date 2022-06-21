# SSR-PU
Code for paper [A Unified Positive and Unlabeled Learning Framework for Document-Level
Relation Extraction with Different Levels of Labeling].

## Requirements
* Python (tested on 3.6.7)
* CUDA (tested on 11.0)
* [PyTorch](http://pytorch.org/) (tested on 1.7.1)
* [Transformers](https://github.com/huggingface/transformers) (tested on 4.18.0)
* numpy (tested on 1.19.5)
* [apex](https://github.com/NVIDIA/apex) (tested on 0.1)
* [opt-einsum](https://github.com/dgasmith/opt_einsum) (tested on 3.3.0)
* ujson
* tqdm

## Dataset
The [DocRED](https://www.aclweb.org/anthology/P19-1074/) dataset can be downloaded following the instructions at [link](https://github.com/thunlp/DocRED/tree/master/data).

The [Re-DocRED](https://arxiv.org/abs/2205.12696) dataset can be downloaded following the instructions at [link](https://github.com/tonytan48/Re-DocRED).

The [ChemDisGene](https://arxiv.org/abs/2204.06584) dataset can be downloaded following the instructions at [link](https://github.com/chanzuckerberg/ChemDisGene).
```
SSR-PU
 |-- dataset
 |    |-- docred
 |    |    |-- train_annotated.json
 |    |    |-- train_distant.json
 |    |    |-- train_ext.json
 |    |    |-- train_revised.json
 |    |    |-- dev_revised.json
 |    |    |-- test_revised.json
 |    |-- chemdisgene
 |    |    |-- train.json
 |    |    |-- test.anno_all.json
 |-- meta
 |    |-- rel2id.json
 |    |-- relation_map.json
```

## Training and Evaluation
### DocRED
Train DocRED model with the following command:

```bash
>> sh scripts/run_bert.sh  # S-PU BERT
>> sh scripts/run_bert_rank.sh  # SSR-PU BERT
>> sh scripts/run_roberta.sh  # S-PU RoBERTa
>> sh scripts/run_roberta_rank.sh  # SSR-PU RoBERTa
>> sh scripts/run_bert_rank_full.sh  # SSR-PU BERT Fully supervised
>> sh scripts/run_roberta_rank_full.sh  # SSR-PU RoBERTa Fully supervised
>> sh scripts/run_bert_rank_ext.sh  # SSR-PU BERT Extremely unlabeled
>> sh scripts/run_roberta_rank_ext.sh  # SSR-PU RoBERTa Extremely unlabeled
```

### ChemDisGene
Train ChemDisGene model with the following command:
```bash
>> sh scripts/run_bio.sh  # S-PU PubmedBERT
>> sh scripts/run_bio_rank.sh  # SSR-PU PubmedBERT
```
