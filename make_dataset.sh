#!/bin/bash
set -e -x

CODE=src
DATA=data/musicnet

python3.7 ${CODE}/data/make_dataset.py \
    --input ${DATA}