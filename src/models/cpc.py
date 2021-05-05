from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

from src.models.wavenet_models import Encoder

# PyTorch implementation of CPC 

class CPC(nn.Module):
    """
    Creates a contrastive predictive coding model with a strided convolutional 
    encoder and a WaveNet-like autoregressor as described by [1], [2] and implemented in [3], [4].

    References
    ----------
    [1] van der Oord et al., "Representation Learning with Contrastive 
        Predictive Coding", arXiv, 2019.
        https://arxiv.org/abs/1807.03748
    [2] Engel et al., "Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders",
        https://arxiv.org/pdf/1704.01279.pdf, 2017.
    [3] Lai, "Contrastive-Predictive-Coding-PyTorch", GitHub.
        https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch
    [4] Polyak, "music-translation", GitHub.
        https://github.com/facebookresearch/music-translation
    """
    def __init__(self, args):
        super().__init__()
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
            nn.Conv1d(512, 1, kernel_size=1, stride=1,bias=False),
            nn.ReLU(inplace=True)
        )
        self.ar = Encoder(args) 

    def forward(self, x):
        """
        Parameters
        ----------
            x : B x 1 x L torch.Tensor
                Input batch of audio sequence with B samples and length L.

        Returns
        -------
            z : B x (L // 160) x 1 torch.Tensor
                Encoded representation of audio sequence with 512 channels.
            c : B x (L // 160) x 1 torch.Tensor
                Context-encoded representation of audio sequence with 256 channels.
        """
        x = x / 255 - .5
        if x.dim() < 3:
            x = x.unsqueeze(1)

        # Use encoder to get sequence of latent representations z_t
        z = self.encoder(x)

        # Use autoregressive model to compute context latent representation c_t
        c = self.ar(z)
        
        z = z.transpose(1, 2)

        return z, c


class InfoNCELoss(nn.Module):
    """
    Creates a criterion that computes the InfoNCELoss as described in [1].

    Parameters
    ----------
        prediction_step : int
            Number of steps to predict into the future using context vector c

    References
    ----------
    [1] van der Oord et al., "Representation Learning with Contrastive
        Predictive Coding", arXiv, 2019.
        https://arxiv.org/abs/1807.03748
    """
    def __init__(self, args):
        super().__init__()
        self.prediction_step = args.prediction_step
        self.Wk = nn.ModuleList(
            nn.Linear(args.latent_d, 1) for _ in range(self.prediction_step)
        )

    def get_neg_z(self, z, k, t, n_replicates):
        """
        Parameters
        ----------
            z : B x L x 1 torch.Tensor
                Encoded representation of audio sequence.
            k : int
                Number of time steps in the future for prediction
            t : B torch.Tensor
                Current time step for each sample in the batch
            n_replicates : int
                Number of repetitions of the negative sampling procedure

        Returns
        -------
            neg_samples : B x L-1 x N_rep x 1 torch.Tensor
                Batch-wise average InfoNCE loss
        """
        cur_device = z.get_device() if z.get_device() != -1 else "cpu"

        neg_idx = torch.vstack([torch.cat([
            torch.arange(0, t_i + k),             # indices before t+k
            torch.arange(t_i + k + 1, z.size(1))  # indices after t+k
        ]) for t_i in t])

        neg_samples = torch.vstack([z[i, neg_idx[i]].unsqueeze(0) for i in range(len(t))])
        neg_samples = torch.stack(
            [
                torch.index_select(neg_samples, 1, torch.randint(neg_samples.size(1),
                                                                 (neg_samples.size(1), )).to(cur_device))
                for i in range(n_replicates)
            ],
            2,
        )
        return neg_samples

    def forward(self, z, c, n_replicates):
        """
        Parameters
        ----------
            z : B x L x 1 torch.Tensor
                Encoded representation of audio sequence.
            c : B x L x 1 torch.Tensor
                Context-encoded representation of audio sequence.
            n_replicates : int
                Number of times to make a set of negative samples.

        Returns
        -------
            loss : float Tensor
                Batch-wise average InfoNCE loss
        """
        loss = 0

        n_batches = z.size(0)

        # Sample random t for each batch
        cur_device = z.get_device() if z.get_device() != -1 else "cpu"
        t = torch.randint(z.size(1) - self.prediction_step - 1, (n_batches,)).to(cur_device)

        # Get context vector c_t
        c = c.transpose(1, 2)
        c_t = c[torch.arange(n_batches), t] # B x 1 

        self.Wk.to(cur_device)
        for k in range(1, self.prediction_step + 1):
            # Perform negative sampling
            neg_samples = self.get_neg_z(z, k, t, n_replicates)  # B x L-1 x N_rep x C

            # Compute W_k * c_t
            linear = self.Wk[k - 1]  # 1 x C
            pred = linear(c_t) # B x C

            # Get positive z_t+k sample
            pos_sample = z[torch.arange(n_batches), t+k]

            # Positive sample: compute f_k(x_t+k, c_t)
            # Only take diagonal elements to get product between matched batches
            fk_pos = torch.diag(torch.matmul(pos_sample, pred.T)) # B (1-D tensor)
            fk_pos_rep = fk_pos.repeat(n_replicates).view(1, 1, n_replicates, fk_pos.size(0)) # 1 x 1 x N_rep x B

            # Negative samples: compute f_k(x_j, c_t)
            # Only take diagonal elements to get products between matched batches
            fk_neg = torch.matmul(neg_samples, pred.T) # B x L-1 x N_rep x B
            fk_neg = torch.diagonal(fk_neg, dim1=0, dim2=-1).unsqueeze(0) # 1 x L-1 x N_rep x B

            # Concatenate fk for positive and negative samples
            fk = torch.hstack([fk_pos_rep, fk_neg]) # 1 x L x N_rep x B

            # Compute log softmax over all fk
            log_sm_fk = torch.nn.LogSoftmax(dim=1)(fk)  # 1 x L x N_rep x B

            # Compute expected value of log softmaxes over replicates
            exp_log_sm_fk = torch.mean(log_sm_fk, dim=2)  # 1 x L x B

            # Update loss with log softmax element corresponding to positive sample
            loss -= exp_log_sm_fk[:, 0] # 1 x B

        # Divide by number of predicted steps
        loss /= self.prediction_step

        # Average over batches
        loss = loss.sum() / n_batches

        return loss
