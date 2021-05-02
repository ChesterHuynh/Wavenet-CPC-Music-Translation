from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

# PyTorch implementation of CPC 
# Paper reference: 'Representation Learning with Contrastive Predictive Coding'
# Source code reference: https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch

class CPCEncoder(nn.Module):
    def __init__(self, args):

        timestep = args.timestep  
        seq_len = args.seq_len
        batch_size = args.batch_size

        super().__init__()
        self.args = args
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep
        self.encoder = nn.Sequential( # downsampling factor = 160
            nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=3, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.gru = nn.GRU(512, 64, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk  = nn.ModuleList([nn.Linear(64, 512) for i in range(timestep)])
        self.softmax  = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def forward(self, x, hidden):
        batch = x.size()[0]

        # Encoder downsamples x by 160
        #t_samples is a random index into the encoded sequence z
        t_samples = torch.randint(self.seq_len // 160 - self.timestep, size=(1,)).long() # randomly pick a time stamp

        # input sequence is N*C*L, e.g. 8*1*20480
        x = x / 255 - .5
        if x.dim() < 3:
            x = x.unsqueeze(1)
        z = self.encoder(x)
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512
        z = z.transpose(1,2)
        encode_samples = torch.empty((self.timestep, batch, 512)).float() # e.g. size 12*8*512
        for k in np.arange(1, self.timestep+1):
            encode_samples[k-1] = z[:,t_samples+k,:].view(batch, 512) # z_t+k e.g. size 8*512
        forward_seq = z[:,:t_samples+1,:] # e.g. size 8*100*512

        output, hidden = self.gru(forward_seq, hidden) # output size e.g. 8*100*256
        c_t = output[:,t_samples,:].view(batch, self.args.latent_d) # c_t e.g. size 8*256
        pred = torch.empty((self.timestep, batch, 512)).float() # e.g. size 12*8*512
        for i in np.arange(self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t) # Wk*c_t e.g. size 8*512

        output = output.transpose(1,2)

        return output, hidden, encode_samples, pred 

def InfoNCELoss(encode_samples, pred, timestep):
    batch = encode_samples.shape[1]

    nce = 0 # average over timestep and batch
    correct = 0
    for i in np.arange(timestep):
        total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1)) # e.g. size 8*8
        correct += torch.sum(torch.eq(torch.argmax(F.softmax(total, dim=0), dim=0), torch.arange(0, batch))) # correct is a tensor
        nce += torch.sum(torch.diag(F.log_softmax(total, dim=0))) # nce is a tensor
    nce /= -1.*batch*timestep
    accuracy = 1.*correct.item()/(batch * timestep)

    return nce