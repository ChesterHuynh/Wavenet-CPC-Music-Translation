# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference: https://github.com/facebookresearch/music-translation/blob/master/src/utils.py

import errno
import logging
import os
import sys
import time
from datetime import timedelta
from subprocess import call

import numpy as np
import torch
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('agg')
from scipy.io import wavfile


def _check_exists(root):
    return os.path.exists(os.path.join(root, "train_data")) and \
        os.path.exists(os.path.join(root, "test_data")) and \
        os.path.exists(os.path.join(root, "train_labels")) and \
        os.path.exists(os.path.join(root, "test_labels"))


def download_data(root):
    """Download MusicNet data at root.
    Adapted from https://github.com/jthickstun/pytorch_musicnet

    Parameters
    ----------
    root : str, Path
        Directory to download MusicNet data. Will create train_data, train_labels,
        test_data, test_labels, and raw subdirectories.
    """
    from six.moves import urllib

    if _check_exists(root):
        return

    try:
        os.makedirs(os.path.join(root, "raw"))
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    
    # Download musicnet.tar.gz
    url = "https://homes.cs.washington.edu/~thickstn/media/musicnet.tar.gz"
    filename = url.rpartition('/')[2]
    file_path = os.path.join(root, "raw", filename)
    if not os.path.exists(file_path):
        print(f"Downloading {url}")
        data = urllib.request.urlopen(url)
        with open(file_path, 'wb') as f:
            # stream the download to disk (it might not fit in memory!)
            while True:
                chunk = data.read(16*1024)
                if not chunk:
                    break
                f.write(chunk)

    # Unpack musicnet.tar.gz
    extracted_folders = ["train_data", "train_labels", "test_data", "test_labels"]
    if not all(map(lambda f: os.path.exists(os.path.join(root, f)), extracted_folders)):
        print('Extracting ' + filename)
        if call(["tar", "-xf", file_path, '-C', root, '--strip', '1']) != 0:
            raise OSError("Failed tarball extraction")

    # Download musicnet_metadata.csv
    url = "https://homes.cs.washington.edu/~thickstn/media/musicnet_metadata.csv"
    metadata = urllib.request.urlopen(url)
    with open(os.path.join(root, 'musicnet_metadata.csv'), 'wb') as f:
        while True:
            chunk = metadata.read(16*1024)
            if not chunk:
                break
            f.write(chunk)

    print('Download Complete')


class timeit:
    def __init__(self, name, logger=None):
        self.name = name
        self.logger = logger

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger is None:
            print(f'{self.name} took {(time.time() - self.start) * 1000} ms')
        else:
            self.logger.debug('%s took %s ms', self.name, (time.time() - self.start) * 1000)


def mu_law(x, mu=255):
    x = np.clip(x, -1, 1)
    x_mu = np.sign(x) * np.log(1 + mu*np.abs(x))/np.log(1 + mu)
    return ((x_mu + 1)/2 * mu).astype('int16')


def inv_mu_law(x, mu=255.0):
    x = np.array(x).astype(np.float32)
    y = 2. * (x - (mu+1.)/2.) / (mu+1.)
    return np.sign(y) * (1./mu) * ((1. + mu)**np.abs(y) - 1.)


class LossMeter(object):
    def __init__(self, name):
        self.name = name
        self.losses = []

    def reset(self):
        self.losses = []

    def add(self, val):
        self.losses.append(val)

    def summarize_epoch(self):
        if self.losses:
            return np.mean(self.losses)
        else:
            return 0

    def sum(self):
        return sum(self.losses)


class LogFormatter:
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)


def create_output_dir(opt, path: Path):
    if hasattr(opt, 'rank'):
        filepath = path / f'main_{opt.rank}.log'
    else:
        filepath = path / 'main.log'

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    if hasattr(opt, 'rank') and opt.rank != 0:
        sys.stdout = open(path / f'stdout_{opt.rank}.log', 'w')
        sys.stderr = open(path / f'stderr_{opt.rank}.log', 'w')

    # Safety check
    if filepath.exists() and not opt.checkpoint:
        logging.warning("Experiment already exists!")

    # Create log formatter
    log_formatter = LogFormatter()

    # Create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create file handler and set level to debug
    file_handler = logging.FileHandler(filepath, "a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    # create console handler and set level to info
    if hasattr(opt, 'rank') and opt.rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(log_formatter)
        logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    logger.info(opt)
    return logger


def setup_logger(logger_name, filename):
    logger = logging.getLogger(logger_name)
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    stderr_handler = logging.StreamHandler(sys.stderr)
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        stderr_handler.setLevel(logging.WARNING)
    else:
        stderr_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stderr_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)
    logger.addHandler(file_handler)
    return logger


def wrap(data, **kwargs):
    if torch.is_tensor(data):
        var = data.cuda(non_blocking=True)
        return var
    else:
        return tuple([wrap(x, **kwargs) for x in data])


def save_audio(x, path, rate):
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, rate, x)


def save_wav_image(wav, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(15, 5))
    plt.plot(wav)
    plt.savefig(path)