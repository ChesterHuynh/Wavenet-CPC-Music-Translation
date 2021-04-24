from src.data.train import UMTCPCTrainer

import argparse

parser = argparse.ArgumentParser(description='PyTorch Code for A Constrastive Predictive Coding Music Translation Network')
# Env options:
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 92)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--expName', type=str, required=True,
                    help='Experiment name')
parser.add_argument('--data',
                    metavar='D', type=Path, help='Data path', nargs='+')
parser.add_argument('--checkpoint', default='',
                    metavar='C', type=str, help='Checkpoint path')
parser.add_argument('--load-optimizer', action='store_true')
parser.add_argument('--per-epoch', action='store_true',
                    help='Save model per epoch')

# Distributed
parser.add_argument('--dist-url', default='env://',
                    help='Distributed training parameters URL')
parser.add_argument('--dist-backend', default='nccl')
parser.add_argument('--local_rank', type=int,
                    help='Ignored during training.')

# Data options
parser.add_argument('--seq-len', type=int, default=16000,
                    help='Sequence length')
parser.add_argument('--epoch-len', type=int, default=10000,
                    help='Samples per epoch')
parser.add_argument('--batch-size', type=int, default=32,
                    help='Batch size')
parser.add_argument('--num-workers', type=int, default=10,
                    help='DataLoader workers')
parser.add_argument('--data-aug', action='store_true',
                    help='Turns data aug on')
parser.add_argument('--magnitude', type=float, default=0.5,
                    help='Data augmentation magnitude.')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--lr-decay', type=float, default=0.98,
                    help='new LR = old LR * decay')
parser.add_argument('--short', action='store_true',
                    help='Run only a few batches per epoch for testing')
parser.add_argument('--h5-dataset-name', type=str, default='wav',
                    help='Dataset name in .h5 file')

# Encoder options
parser.add_argument('--latent-d', type=int, default=128,
                    help='Latent size')
parser.add_argument('--repeat-num', type=int, default=6,
                    help='No. of hidden layers')
parser.add_argument('--encoder-channels', type=int, default=128,
                    help='Hidden layer size')
parser.add_argument('--encoder-blocks', type=int, default=3,
                    help='No. of encoder blocks.')
parser.add_argument('--encoder-pool', type=int, default=800,
                    help='Number of encoder outputs to pool over.')
parser.add_argument('--encoder-final-kernel-size', type=int, default=1,
                    help='final conv kernel size')
parser.add_argument('--encoder-layers', type=int, default=10,
                    help='No. of layers in each encoder block.')
parser.add_argument('--encoder-func', type=str, default='relu',
                    help='Encoder activation func.')

# Decoder options
parser.add_argument('--blocks', type=int, default=4,
                    help='No. of wavenet blocks.')
parser.add_argument('--layers', type=int, default=10,
                    help='No. of layers in each block.')
parser.add_argument('--kernel-size', type=int, default=2,
                    help='Size of kernel.')
parser.add_argument('--residual-channels', type=int, default=128,
                    help='Residual channels to use.')
parser.add_argument('--skip-channels', type=int, default=128,
                    help='Skip channels to use.')

# Z discriminator options
parser.add_argument('--d-layers', type=int, default=3,
                    help='Number of 1d 1-kernel convolutions on the input Z vectors')
parser.add_argument('--d-channels', type=int, default=100,
                    help='1d convolutions channels')
parser.add_argument('--d-cond', type=int, default=1024,
                    help='WaveNet conditioning dimension')
parser.add_argument('--d-lambda', type=float, default=1e-2,
                    help='Adversarial loss weight.')
parser.add_argument('--p-dropout-discriminator', type=float, default=0.0,
                    help='Discriminator input dropout - if unspecified, no dropout applied')
parser.add_argument('--grad-clip', type=float,
                    help='If specified, clip gradients with specified magnitude')

model = UMTCPCTrainer(args)

model.train()