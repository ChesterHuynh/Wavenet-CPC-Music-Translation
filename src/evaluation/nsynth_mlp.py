"""
Perform domain classification on NSynth data with simple MLP

"""
import numpy as np
import glob
import time

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import argparse
from pathlib import Path


class NSynthDatasetCoQT(Dataset):
    """
    Pytorch dataset of Constant-Q transformed NSynth data

    Parameters
    ----------
    data: paths to directory domain directories split into train, test, val
    domains: names of domains to use in classification
    op: "train", "test", or "val"
    """
    def __init__(self, data, op):
        self.op = op
        self.data = data
        self.filenames = []
        for d in data:
            self.filenames += glob.glob(f'{d}/{op}/*.npy')

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        label = np.zeros(len(self.data))
        for i, c in enumerate(self.data):
            if str(c) in fname:
                label[i] = 1
        label = np.argmax(label)
        cqt_data = np.load(fname, allow_pickle=True)
        cqt_data = torch.tensor(np.abs(cqt_data), dtype=torch.float32)
        label = torch.tensor(label, dtype=int)

        return cqt_data, label

class NSynthClassifier(nn.Module):

  def __init__(self, input_size, num_classes):

    super(NSynthClassifier, self).__init__()

    self.ff1 = nn.Linear(input_size, input_size // 100)
    self.ff2 = nn.Linear(input_size // 100, input_size // 1000)
    self.ff3 = nn.Linear(input_size // 1000, num_classes)

    self.relu = nn.ReLU()

  def forward(self, x):
    out = self.relu(self.ff1(x))
    out = self.relu(self.ff2(out))
    out = self.ff3(out)
    return out


class Trainer():
    def __init__(self, args):
        self.epochs = args.epochs
        trainset = NSynthDatasetCoQT(args.data, 'train')
        self.trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        valset = NSynthDatasetCoQT(args.data, 'val')
        self.valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=True)
        testset = NSynthDatasetCoQT(args.data, 'test')
        self.testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)

        self.dataloaders = {'train': self.trainloader, 'val': self.valloader, 'test': self.testloader}

        self.gpu_boole = torch.cuda.is_available()

        self.net = NSynthClassifier(args.input_dim, len(args.data))
        self.loss_metric = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)

    def eval(self, op, verbose=1):
        correct = 0
        total = 0
        loss_sum = 0
        for x, labels in self.dataloaders[op]:
            if self.gpu_boole:
                x, labels = x.cuda(), labels.cuda()
            x = x.view(x.shape[0], -1)
            outputs = self.net.forward(x)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.float() == labels.float()).sum()

            loss_sum += self.loss_metric(outputs, labels).item()

        if verbose:
            print(f'{op} accuracy: {100.0 * correct / total}%')
            print(f'{op} loss: {loss_sum / total}')

        return 100.0 * correct / total, loss_sum / total

    def train(self):
        def weights_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')

        if self.gpu_boole:
            self.net = self.net.cuda()

        self.net.apply(weights_init)

        loss_batch_store = []
        print("Starting Training")
        # training loop:
        for epoch in range(self.epochs):
            time1 = time.time()  # timekeeping

            for i, (x, y) in enumerate(self.dataloaders['train']):

                if self.gpu_boole:
                    x = x.cuda()
                    y = y.cuda()

                # loss calculation and gradient update:
                if i > 0 or epoch > 0:
                    self.optimizer.zero_grad()
                x = x.view(x.shape[0], -1)
                outputs = self.net.forward(x)
                loss = self.loss_metric(outputs, y)
                loss.backward()

                if i > 0 or epoch > 0:
                    loss_batch_store.append(loss.cpu().data.numpy().item())

                # perform update:
                self.optimizer.step()

            print("Epoch", epoch + 1, ':')
            train_perc, train_loss = self.eval('train')
            val_perc, val_loss = self.eval('val')

            time2 = time.time()  # timekeeping
            print('Elapsed time for epoch:', time2 - time1, 's')
            print('ETA of completion:', (time2 - time1) * (self.epochs - epoch - 1) / 60, 'minutes')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MLP classifier for CoQT NSynth data')

    parser.add_argument('--data',
                        metavar='D', type=Path, help='Data path', nargs='+')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of samples per batch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--input-dim', type=int, default=14532, help='Input dimension')

    args = parser.parse_args()

    t = Trainer(args)
    t.train()
    t.eval('test')
