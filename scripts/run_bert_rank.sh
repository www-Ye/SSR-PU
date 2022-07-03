#!/bin/sh

python train.py --data_dir ./dataset/docred \
    --transformer_type bert \
    --model_name_or_path ../../pretrain/bert-base-cased \
    --train_file train_annotated.json \
    --dev_file dev_revised.json \
    --test_file test_revised.json \
    --train_batch_size 4 \
    --test_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.06 \
    --num_train_epochs 30.0 \
    --seed 66 \
    --num_class 97 \
    --isrank 1 \
    --m_tag S-PU \
    --m 0.25 \
    --e 3.0
