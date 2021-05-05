import torch
from pathlib import Path
from shutil import copyfile
import os


if __name__ == "__main__":
    repo_path = Path(__file__).parents[1]
    src = repo_path / "checkpoints/pretrained_musicnet"
    dst = repo_path / "checkpoints/umtcpc-wavenet-pretrained"
    data_dir = repo_path / "data/musicnet/preprocessed"
    
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
    chkpt_args[0].encoder_pool = 1
    chkpt_args[0].lr = 5e-4
    chkpt_args[0].n_datasets = 3
    chkpt_args[0].num_workers = 0
    chkpt_args[0].seq_len = 10000
    chkpt_args[0].world_size = 1

    torch.save(chkpt_args, dst / "args.pth")

    # Bach Solo Cello --> Solo Cello
    copyfile(src / "lastmodel_0.pth", dst / "lastmodel_0.pth")
    copyfile(src / "bestmodel_0.pth", dst / "bestmodel_0.pth")

    # Beethoven Accompanied Violin --> Solo Violin
    copyfile(src / "lastmodel_4.pth", dst / "lastmodel_1.pth")
    copyfile(src / "bestmodel_4.pth", dst / "bestmodel_1.pth")

    # Beethoven Solo Piano --> Beethoven Solo Piano
    copyfile(src / "lastmodel_1.pth", dst / "lastmodel_2.pth")
    copyfile(src / "bestmodel_1.pth", dst / "bestmodel_2.pth")

    print("Done with preprocessing")