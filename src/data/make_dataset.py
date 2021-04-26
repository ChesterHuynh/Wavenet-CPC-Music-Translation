import argparse
import sys
import os
import random
from pathlib import Path

from src.data.utils import download_data
from src.data.parse import parse_data
from src.data.split import split
from src.data.preprocess import preprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=Path, required=True,
                        help='Root directory for data')
    parser.add_argument('--seed', type=int, default=18,
                        help='Random seed')

    args = parser.parse_args()
    root = args.input
    if not os.path.exists(root):
        os.makedirs(root)

    # Download raw data at root
    download_data(root)

    # Parse data into groups
    domains = [
        'Solo Cello',
        'Solo Violin',
        'Solo Piano'
    ]
    parsed_dir = root / 'parsed'
    parse_data(root, parsed_dir, domains)

    # Split data into train-val-test
    random.seed(args.seed)
    split_dir = root / 'split'
    for input_path in parsed_dir.glob("*/"):
        basename = os.path.basename(input_path)
        output_path = Path(split_dir / basename)
        split(input_path, output_path, train_ratio=0.8, val_ratio=0.1, 
              filetype='wav', copy=True)

    # Preprocess data
    preproc_dir = root / 'preprocessed'
    preprocess(split_dir, preproc_dir)


if __name__ == '__main__':
    main()