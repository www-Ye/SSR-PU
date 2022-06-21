#!/bin/sh

python train_bio.py --data_dir ./dataset/chemdisgene \
    --transformer_type bert \
    --model_name_or_path ../../pretrain/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --train_file train.json \
    --test_file test.anno_all.json \
    --train_batch_size 8 \
    --test_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.06 \
    --num_train_epochs 30.0 \
    --seed 66 \
    --num_class 15 \
    --isrank 1 \
    --m_tag S-PU \
    --m 0.25 \
    --e 3.0
