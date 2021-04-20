#!/bin/bash
set -e -x

CODE=src
DATA=data/musicnet

python ${CODE}/data/make_dataset.py \
    --input ${DATA}