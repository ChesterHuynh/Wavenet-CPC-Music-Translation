# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

DATE=`date +%d_%m_%Y`
CODE=src
OUTPUT=results/${DATE}/$1

python3.7 ${CODE}/inference.py --model-name $3 --data-from-args checkpoints/$1/args.pth --output-sampled ${OUTPUT}-py  -n 2 --seq 80000 \
    --files ${OUTPUT}-py --batch-size 2 --checkpoint checkpoints/$1/lastmodel --output-next-to-orig --decoders $2 --py

