# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# Generate with nvidia wavenet kernels
# 
# usage: ./generate.sh experiment-name path-to-files decoders model-name
# example usage: ./generate.sh umtcpc-pretrained-wavenet results/05_05_21/umtcpc_pretrained_wavenet "0 1 2" umtcpc

#!/usr/bin/env bash

DATE=`date +%d_%m_%Y`
CODE=src

python3.7 ${CODE}/data/run_on_files.py  --batch-size 2 --checkpoint checkpoints/$1/lastmodel --output-next-to-orig --files $2 --decoders $3 --model-name $4 --sample-len 80000 --skip-filter
