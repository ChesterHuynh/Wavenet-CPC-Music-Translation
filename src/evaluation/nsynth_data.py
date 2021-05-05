"""
Download and process NSynth data
"""

import os
import glob
import numpy as np
import pandas as pd

import argparse

from subprocess import call
from shutil import copy
import errno

import librosa

from pathlib import Path

def download_nsynth(root):
    """Download NSynth data at root.
    Adapted from https://github.com/jthickstun/pytorch_musicnet

    Parameters
    ----------
    root : str, Path
        Directory to download NSynth data
    """
    from six.moves import urllib

    try:
        os.makedirs(os.path.join(root, "raw"))
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # Download nsynth-test.jsonwav.tar.gz
    url = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz"
    filename = url.rpartition('/')[2]
    file_path = os.path.join(root, "raw", filename)
    if not os.path.exists(file_path):
        print(f"Downloading {url}")
        data = urllib.request.urlopen(url)
        with open(file_path, 'wb') as f:
            # stream the download to disk (it might not fit in memory!)
            while True:
                chunk = data.read(16 * 1024)
                if not chunk:
                    break
                f.write(chunk)

    # Unpack nsynth-test.tar.gz
    extracted_folders = ["audio", "examples.json"]
    if not all(map(lambda f: os.path.exists(os.path.join(root, f)), extracted_folders)):
        print('Extracting ' + filename)
        if call(["tar", "-xf", file_path, '-C', root, '--strip', '1']) != 0:
            raise OSError("Failed tarball extraction")

def parse_nsynth(root, json_name):
    df_json = pd.read_json(os.path.join(root, json_name))

    instr_str_strip = df_json.loc["instrument_str"].apply(lambda x: "_".join(x.split("_")[:2])).unique()

    parsed_dir = os.path.join(root, 'parsed')
    os.makedirs(parsed_dir, exist_ok=True)

    for instr_str in instr_str_strip:
            os.makedirs(os.path.join(parsed_dir, instr_str), exist_ok=True)

    filenames = glob.glob(os.path.join(root, "audio/*.wav"))
    for fname in filenames:
        fname_base = os.path.basename(fname).split(".")[0]
        family = df_json[fname_base].loc["instrument_family_str"]
        src = df_json[fname_base].loc["instrument_source_str"]

        dst_dir = os.path.join(parsed_dir, "_".join([family, src]))

        copy(fname, dst_dir)

def cqt_nsynth(root):
    parsed_dir = os.path.join(root, 'parsed')

    instr_str_dirs = glob.glob(parsed_dir +'/*')
    for dir in instr_str_dirs:
        instr_files = glob.glob(dir + '/*.wav')
        for fname in instr_files:
            wav_data,sr = librosa.load(fname)
            cqt_data = librosa.cqt(wav_data, sr=sr)
            np.save(fname.replace('.wav', '_cqt'), cqt_data)

def split_nsynth(root, train_ratio=0.8, val_ratio=0.1):

    parsed_dir = os.path.join(root, 'parsed')
    split_dir = os.path.join(root, 'split')
    os.makedirs(split_dir, exist_ok=True)

    instr_str_dirs = glob.glob(parsed_dir +'/*')
    for dir in instr_str_dirs:
        instr_files = glob.glob(dir + '/*')
        num_files = len(instr_files)

        num_train = int(train_ratio*num_files)
        num_val = int(val_ratio*num_files)
        train_files = instr_files[:num_train]
        val_files = instr_files[num_train:num_train + num_val]
        test_files = instr_files[num_train + num_val:]

        train_dir = os.path.join(dir.replace('parsed', 'split'), 'train')
        os.makedirs(train_dir, exist_ok=True)
        val_dir = os.path.join(dir.replace('parsed', 'split'), 'val')
        os.makedirs(val_dir, exist_ok=True)
        test_dir = os.path.join(dir.replace('parsed', 'split'), 'test')
        os.makedirs(test_dir, exist_ok=True)

        for train_f in train_files:
            copy(train_f, train_dir)
        for val_f in val_files:
            copy(val_f, val_dir)
        for test_f in test_files:
            copy(test_f, test_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NSynth Data Curation')

    parser.add_argument('-i', '--input', type=Path, required=True,
                        help='Root directory for data')

    args = parser.parse_args()

    root = args.input
    if not os.path.exists(root):
        os.makedirs(root)

    # Download raw data at root
    download_nsynth(root)

    # Parse data into domains
    print("Parsing into domains")
    parse_nsynth(root, 'examples.json')

    # Compute and store Constant-Q transforms
    print("Computing Constant-Q")
    cqt_nsynth(root)

    # Split each domain into train, test, and val folders
    print("Splitting into training and testing")
    split_nsynth(root)