#!/bin/bash
set -e -x

CODE=src
DATA=/home/ChesterHuynh/Wavenet-CPC-Music-Translation/data/musicnet/preprocessed
EXP=musicnet
export MASTER_PORT=29500

python3.7 ${CODE}/train.py \
    --data ${DATA}/Solo_Cello  \
           ${DATA}/Solo_Violin \
           ${DATA}/Beethoven_Solo_Piano \
    --model-name $1 \
    --epochs 1 \
    --batch-size 1 \
    --lr-decay 0.995 \
    --epoch-len 1 \
    --num-workers 0 \
    --lr 1e-3 \
    --seq-len 10000 \
    --d-lambda 1e-2 \
    --expName ${EXP} \
    --latent-d 64 \
    --layers 14 \
    --blocks 4 \
    --data-aug \
    --grad-clip 1
