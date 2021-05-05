import re
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import cycler

plt.rcParams.update({
    "axes.spines.right" : False,
    "axes.spines.top" : False,
    "axes.labelsize" : "medium",
    "axes.titlesize" : "x-large",
    "font.size" : 10,
    "axes.prop_cycle": cycler(color=[
        "#348ABD",
        "#A60628",
        "#7A68A6",
        "#467821",
        "#CF4457",
        "#188487",
        "#E24A33"
    ])
})


def parse_epoch_loss(fpath):
    fpath = Path("/Users/ChesterHuynh/OneDrive - Johns Hopkins/classes/dl482/Wavenet-CPC-Music-Translation/checkpoints/umtcpc-pretrained/main_0.log")
    train_losses = []
    test_losses = []
    with open(fpath,'r') as f:
        for line in f:
            if not (line.startswith("INFO") and "Epoch" in line and "loss" in line):
                continue
            # _, lead_end = re.match("^[^Epoch]*", line).span()
            # line_ = line[lead_end:]
            # _, train_loss, test_loss = re.split(' loss', line_)

            s = line.strip()

            train_loss = s[s.find("Train loss: ") + len("Train loss: (") : s.find(")")]
            test_loss = s[s.find("Test loss ") + len("Test loss (") : -1]

            train_loss = [float(loss) for loss in train_loss.split(", ")]
            test_loss = [float(loss) for loss in test_loss.split(", ")]

            train_losses.append(train_loss)
            test_losses.append(test_loss)
    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)
    return train_losses, test_losses


# TODO: Batch losses are not present in current data log
def parse_batch_loss(fpath):
    return


if __name__ == "__main__":
    repo_path = Path(__file__).parents[1]
    fpath = repo_path / "checkpoints/umtcpc-pretrained/main_0.log"

    train_losses, test_losses = parse_epoch_loss(fpath)

    fig, ax = plt.subplots(dpi=100)
    ax.plot(train_losses.sum(axis=1), label = "Train Loss")
    ax.plot(test_losses.sum(axis=1), label="Test Loss")
    ax.set(xlabel="Epoch", ylabel="Loss")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
