# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference: https://github.com/facebookresearch/music-translation/blob/master/src/preprocess.py

import argparse

import sys
include = Path(__file__).parents[2]
if include not in sys.path:
    sys.path.append(str(include))

import src.data.data as data
from pathlib import Path


def preprocess(input_path, output_path, norm_db=False):
    dataset = data.EncodedFilesDataset(input_path)
    dataset.dump_to_folder(output_path, norm_db=norm_db)
    print('Preprocessing complete')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=Path, required=True,
                        help='Input directory')
    parser.add_argument('-o', '--output', type=Path, required=True,
                        help='Output directory')
    parser.add_argument('--norm-db', required=False, action='store_true')

    args = parser.parse_args()
    dataset = data.EncodedFilesDataset(args.input)
    dataset.dump_to_folder(args.output, norm_db=args.norm_db)