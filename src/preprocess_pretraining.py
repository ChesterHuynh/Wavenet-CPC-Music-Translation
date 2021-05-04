import torch
from pathlib import Path
from shutil import copyfile
import os


if __name__ == "__main__":
    src = Path("../checkpoints/pretrained_musicnet/")
    dst = Path("/home/magjywang/checkpoints/umtcpc-pretrained")
    data_dir = Path("/home/magjywang/Wavenet-CPC-Music-Translation/data/musicnet/preprocessed")
    
    if not os.path.exists(dst):
        os.makedirs(dst)

    checkpoint = src / "args.pth"
    chkpt_args = torch.load(checkpoint)
    chkpt_args[0].batch_size = 8
    chkpt_args[0].checkpoint = dst / "lastmodel"

    chkpt_args[0].data = [data_dir / "Solo_Cello",
                        data_dir / "Solo_Violin",
                        data_dir / "Beethoven_Solo_Piano"]
    chkpt_args[0].distributed = False
    chkpt_args[0].expName = "umtcpc-pretrained"
    chkpt_args[0].lr = 5e-4
    chkpt_args[0].n_datasets = 3
    chkpt_args[0].num_workers = 0
    chkpt_args[0].seq_len = 10000
    chkpt_args[0].world_size = 1

    torch.save(chkpt_args, dst / "args.pth")

    copyfile(src / "lastmodel_0.pth", dst / "lastmodel_0.pth")
    copyfile(src / "bestmodel_0.pth", dst / "bestmodel_0.pth")

    copyfile(src / "lastmodel_4.pth", dst / "lastmodel_1.pth")
    copyfile(src / "bestmodel_4.pth", dst / "bestmodel_1.pth")

    copyfile(src / "lastmodel_1.pth", dst / "lastmodel_2.pth")
    copyfile(src / "bestmodel_1.pth", dst / "bestmodel_2.pth")

    print("Done with preprocessing")