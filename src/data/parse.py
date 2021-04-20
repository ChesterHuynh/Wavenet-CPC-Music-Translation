
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference: https://github.com/facebookresearch/music-translation/blob/master/src/parse_musicnet.py

import argparse
import csv
import os
from intervaltree import IntervalTree
from pathlib import Path
from tqdm import tqdm

import pandas as pd
from shutil import copy


def process_labels(root, path):
    """Parse label CSVs for MusicNet and store in a dictionary
    containing IntervalTrees 
    
    Parameters
    ----------
    root : str, Path
        Absolute path to root of data directory
    
    path : str, Path
        Subdirectory in root to parse labels from

    Returns
    -------
    trees : dict
        Dictionary of IntervalTrees for each CSV found in the specified
        subdirectory path.
    """
    trees = dict()
    for item in os.listdir(os.path.join(root,path)):
        if not item.endswith('.csv'): continue
        uid = int(item[:-4])
        tree = IntervalTree()
        with open(os.path.join(root, path, item), 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            for label in reader:
                start_time = int(label['start_time'])
                end_time = int(label['end_time'])
                instrument = int(label['instrument'])
                note = int(label['note'])
                start_beat = float(label['start_beat'])
                end_beat = float(label['end_beat'])
                note_value = label['note_value']
                tree[start_time:end_time] = (instrument,note,start_beat,end_beat,note_value)
        trees[uid] = tree
    return trees


def curate_data(root, destination, metadata, groupby='composer', disable_progress_bar=True):
    """Organize original dataset structure into 
    
    """
    if not hasattr(metadata, "columns"):
        raise AttributeError('metadata must have a columns attribute')

    if groupby not in metadata.columns:
        raise ValueError(f'{groupby} column is not in metadata')

    root = Path(root)
    destination = Path(destination)

    if not os.path.isabs(root):
        root = Path(os.path.abspath(root))

    if not os.path.isabs(destination):
        destination = Path(os.path.abspath(destination))

    if not os.path.exists(destination):
        os.mkdir(destination)

    # Loop and move files from MusicNet into a train folder grouped by "groupby"
    train_dir = root / "train_data"
    test_dir = root / "test_data"
    for group_name, group_df in tqdm(metadata.groupby(groupby), disable=disable_progress_bar):
        group_ids = group_df.id.tolist()

        out_dir = destination / f"{group_name.replace(' ', '_')}"
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        
        for fid in group_ids:
            
            fname = train_dir / f"{fid}.wav"
            if not fname.exists():
                fname = test_dir / f"{fid}.wav"
            
            copy(str(fname), str(out_dir))

    print(f"Curated data at {destination}")


def parse_data(src, dst, domains):
    """
    Extract the desired domains from the raw MusicNet files

    Parameters
    ----------
    src: str
        Path to input data (e.g. /content/musicnet)
        
    """

    dst.mkdir(exist_ok=True, parents=True)
    
    db = pd.read_csv( src / 'musicnet_metadata.csv')
    traindir = src / 'train_data'
    testdir = src /'test_data'

    for (ensemble, composer) in domains:
        fid_list = db[(db["composer"] == composer) & (db["ensemble"] == ensemble)].id.tolist()
        total_time = sum(db[(db["composer"] == composer) & (db["ensemble"] == ensemble)].seconds.tolist())
        print(f"Total time for {composer} with {ensemble} is: {total_time} seconds")


        domaindir = dst / f"{composer}_{ensemble.replace(' ', '_')}"
        if not os.path.exists(domaindir):
            os.mkdir(domaindir)

        for fid in fid_list:
            fname = traindir / f'{fid}.wav'
            if not fname.exists():
                fname = testdir / f'{fid}.wav'

            copy(str(fname), str(domaindir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=Path, required=True,
                        help='MusicNet directory')
    parser.add_argument('-o', '--output', type=Path, required=True,
                        help='Output directory')

    args = parser.parse_args()
    print(args)

    src = args.input
    dst = args.output
    dst.mkdir(exist_ok=True, parents=True)

    domains = [
        ['Accompanied Violin', 'Beethoven'],
        ['Solo Cello', 'Bach'],
        ['Solo Piano', 'Bach'],
        ['Solo Piano', 'Beethoven'],
        ['String Quartet', 'Beethoven'],
        ['Wind Quintet', 'Cambini'],
    ]

    db = pd.read_csv(src / 'musicnet_metadata.csv')
    traindir = src / 'train_data'
    testdir = src / 'test_data'

    for (ensemble, composer) in domains:
        fid_list = db[(db["composer"] == composer) & (db["ensemble"] == ensemble)].id.tolist()
        total_time = sum(db[(db["composer"] == composer) & (db["ensemble"] == ensemble)].seconds.tolist())
        print(f"Total time for {composer} with {ensemble} is: {total_time} seconds")

        domaindir = dst / f"{composer}_{ensemble.replace(' ', '_')}"
        if not os.path.exists(domaindir):
            os.mkdir(domaindir)

        for fid in fid_list:
            fname = traindir / f'{fid}.wav'
            if not fname.exists():
                fname = testdir / f'{fid}.wav'

            copy(str(fname), str(domaindir))


if __name__ == '__main__':
    main()