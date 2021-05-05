#!/bin/bash
set -e -x

CODE=src
DATA=data/musicnet/preprocessed
EXP=umtcpc-wavenet-pretrained
export MASTER_PORT=29500

python3.7 ${CODE}/train.py \
    --data ${DATA}/Solo_Cello  \
           ${DATA}/Solo_Violin \
           ${DATA}/Beethoven_Solo_Piano \
    --model-name $1 \
    --checkpoint "checkpoints/${EXP}/lastmodel" \
    --epochs 10000 \
    --batch-size 8 \
    --lr-decay 0.995 \
    --epoch-len 1000 \
    --num-workers 0 \
    --lr 5e-4 \
    --seq-len 10000 \
    --d-lambda 1e-2 \
    --expName ${EXP} \
    --latent-d 64 \
    --layers 14 \
    --blocks 4 \
    --encoder-pool 1 \
    --data-aug \
    --grad-clip 1
