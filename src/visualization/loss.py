import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import cycler

from src.logparser import parse_epoch_loss, parse_batch_loss
plt.rcParams.update({
    "axes.spines.right" : False,
    "axes.spines.top" : False,
    "axes.labelsize" : "x-large",
    "axes.titlesize" : "xx-large",
    "xtick.labelsize" : "large",
    "ytick.labelsize" : "large",
    "font.size" : 10,
    "axes.prop_cycle": cycler(color=[
        "#0000FF",  # Blue
        "#DC143C",  # Crimson
        "#008000",  # Green
        "#FFD700",  # Gold
        "#FFA500",  # Orange
        "#029386",  # Teal
        "#9A0EEA",  # Violet
    ])
})

def plot_batch_loss(fpath: Path, dst: Path):
    train_losses, test_losses = parse_batch_loss(fpath)
    n_epochs = train_losses.shape[0]
    train_losses = pd.DataFrame(train_losses.T)
    test_losses = pd.DataFrame(test_losses.T)

    # Convert DataFrames from wide to long form
    train_losses = train_losses.reset_index()
    train_losses_long = pd.melt(train_losses, id_vars='index', 
                                value_vars=list(np.arange(n_epochs)), 
                                var_name='Epoch', value_name='Loss')
    test_losses = test_losses.reset_index()
    test_losses_long = pd.melt(test_losses, id_vars='index', 
                                value_vars=list(np.arange(n_epochs)), 
                                var_name='Epoch', value_name='Loss')

    fig, ax = plt.subplots(dpi=200, figsize=(6.4, 4.8))
    sns.lineplot(data=train_losses_long, x='Epoch', y='Loss', label='Train loss', ax=ax)
    sns.lineplot(data=test_losses_long, x='Epoch', y='Loss', label='Test loss', ax=ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    
    if not os.path.exists(dst):
        os.makedirs(dst)
    plt.savefig(dst / "batch_loss.png")


def plot_epoch_loss(fpath: Path, dst: Path):
    train_losses, test_losses = parse_epoch_loss(fpath)

    fig, ax = plt.subplots(dpi=200, figsize=(6.4, 4.8))
    ax.plot(train_losses.sum(axis=1), label = "Train Loss")
    ax.plot(test_losses.sum(axis=1), label="Test Loss")
    ax.set(xlabel="Epoch", ylabel="Loss")
    ax.legend(frameon=False)
    fig.tight_layout()

    if not os.path.exists(dst):
        os.makedirs(dst)
    plt.savefig(dst / "epoch_loss.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize train and test loss from log file')
    parser.add_argument('--file', type=Path, required=True, 
                        help="Path to log file")
    parser.add_argument('--dst', type=Path, required=True, 
                        help="Path to save plots")
    parser.add_argument('--how', type=str, choices=["batch", "epoch"], default="batch", 
                        help="Loss values to use")

    args = parser.parse_args()
    fpath = args.file
    dst = args.dst
    how = args.how

    if how == "batch":
        plot_batch_loss(fpath, dst)
    elif how == "epoch":
        plot_epoch_loss(fpath, dst)
