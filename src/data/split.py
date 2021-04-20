# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference: https://github.com/facebookresearch/music-translation/blob/master/src/split_dir.py

import argparse
import random
import src.data.data as data
import src.data.utils as utils
from pathlib import Path
from shutil import copy, move

logger = utils.setup_logger('__name__', 'train.log')

def copy_files(files, from_path, to_path: Path):
    for f in files:
        out_file_path = to_path / f.relative_to(from_path)
        out_file_path.parent.mkdir(parents=True, exist_ok=True)
        copy(f, out_file_path)


def move_files(files, from_path, to_path: Path):
    for f in files:
        out_file_path = to_path / f.relative_to(from_path)
        out_file_path.parent.mkdir(parents=True, exist_ok=True)
        move(f, out_file_path)


def split(input_path, output_path, train_ratio, val_ratio, filetype, copy=False):
    if filetype:
        filetypes = [filetype]
    else:
        filetypes = data.EncodedFilesDataset.FILE_TYPES
    
    input_files = data.EncodedFilesDataset.filter_paths(input_path.glob('**/*'), filetypes)
    random.shuffle(input_files)

    logger.info(f"Found {len(input_files)} files")

    n_train = int(len(input_files) * train_ratio)
    n_val = int(len(input_files) * val_ratio)
    if n_val == 0:
        n_val = 1
    n_test = len(input_files) - n_train - n_val

    logger.info(f'Split as follows: Train - {n_train}, Validation - {n_val}, Test - {n_test}')
    assert n_test > 0

    if copy:
        copy_files(input_files[:n_train], input_path, output_path / 'train')
        copy_files(input_files[n_train:n_train + n_val], input_path, output_path / 'val')
        copy_files(input_files[n_train + n_val:], input_path, output_path / 'test')
    else:
        move_files(input_files[:n_train], input_path, output_path / 'train')
        move_files(input_files[n_train:n_train + n_val], input_path, output_path / 'val')
        move_files(input_files[n_train + n_val:], input_path, output_path / 'test')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=Path, required=True,
                        help='Input files directory')
    parser.add_argument('-o', '--output', type=Path, required=True,
                        help='Output files directory')
    parser.add_argument('--train', type=float, default=0.8,
                        help='Training set ratio')
    parser.add_argument('--val', type=float, default=0.1,
                        help='Validation set ratio')
    parser.add_argument('--seed', type=int, default=18,
                        help='Random seed')
    parser.add_argument('--filetype',
                        help='Filename suffixes to copy (default from data.py)')
    parser.add_argument('--copy', required=False, action='store_true',
                        help='Whether to create copies of files')

    args = parser.parse_args()

    random.seed(args.seed)
    split(args.input, args.output, args.train, args.val, args.filetype, args.copy)


if __name__ == '__main__':
    main()