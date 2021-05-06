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

python3.7 ${CODE}/data/data_samples.py \
    --data data/musicnet/preprocessed/Solo_Cello \
           data/musicnet/preprocessed/Solo_Violin \
           data/musicnet/preprocessed/Beethoven_Solo_Piano \
    --output-sampled ${OUTPUT}  -n 2 

echo "Finished writing samples to ${OUTPUT}"
