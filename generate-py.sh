# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Python-based generation
#
# usage: ./generate.sh experiment-name path-to-files output-path decoders model-name
# example usage: ./generate.sh umtcpc-pretrained-wavenet results/05_05_21/umtcpc_pretrained_wavenet-py "0 1 2" umtcpc

#!/usr/bin/env bash

DATE=`date +%d_%m_%Y`
CODE=src

python3.7 ${CODE}/data/run_on_files.py  --batch-size 2 --checkpoint checkpoints/$1/lastmodel --files $2 --output-next-to-orig --decoders $4 --model-name $5 --py
