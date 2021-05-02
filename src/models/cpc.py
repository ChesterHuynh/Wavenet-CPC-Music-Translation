from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn

# PyTorch implementation of CPC 
# Paper reference: 'Representation Learning with Contrastive Predictive Coding'

class CPC(nn.Module):
    def __init__(self):
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
        )
        self.ar = nn.GRU(512, 256, num_layers=1, bidirectional=False, batch_first=True)

    def forward(self, x):
        """
        Input
        x : torch.Tensor [n_samples, n_steps]

        Output
        z : torch.Tensor [n_samples, ...?]

        c : torch.Tensor [n_samples, ...?]
        """
        # Use encoder to get sequence of latent representations z_t
        z = self.encoder(x)
        z = z.transpose(1,2)

        # Use autoregressive model to compute context latent representation c_t
        c, _ = self.ar(z)

        return z, c

class InfoNCELoss(nn.Module):
    def __init__(self, prediction_step):
        super().__init__()
        self.prediction_step = prediction_step
        self.Wk = nn.ModuleList(
            nn.Linear(256, 512) for _ in range(prediction_step)
        )

    def get_neg_z(self, z, k, t, n_replicates):
        cur_device = z.get_device() if z.get_device() != -1 else "cpu:0"

        neg_idx = torch.vstack([torch.cat([
            torch.arange(0, t_i + k), 
            torch.arange(t_i + k + 1, z.size(1))
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
        
    
    def forward(self, z, c, n_replicates=2):
        """
        z : (batch, encoding_len, 512)
        c : (batch, encoding_len, 256)
        """
        loss = 0

        n_batches = z.size(0)

        # Sample random t for each batch
        cur_device = z.get_device() if z.get_device() != -1 else "cpu:0"
        t = torch.randint(z.size(1) - self.prediction_step - 1, (n_batches,)).to(cur_device)

        # Get context vector c_t
        c_t = torch.vstack([c[i, t[i]].unsqueeze(0) for i in range(n_batches)]) # B x 256

        for k in range(1, self.prediction_step + 1):
            # Perform negative sampling
            neg_samples = self.get_neg_z(z, k, t, n_replicates)  # B x L-1 x N_rep x C

            # Compute W_k * c_t
            linear = self.Wk[k - 1]  # 256 x C
            pred = linear(c_t) # B x C

            # Get positive z_t+k sample
            pos_sample = torch.vstack([z[i, t[i] + k].unsqueeze(0) for i in range(n_batches)]) # B x C

            # Positive sample: compute f_k(x_t+k, c_t)
            # Only take diagonal elements to get product between matched batches
            fk_pos = torch.diag(torch.matmul(pos_sample, pred.T)) # B (1-D tensor)
            fk_pos_rep = torch.cat(
                    [fk_pos]*n_replicates
                ).view(1, 1, n_replicates, fk_pos.size(0)) # 1 x 1 x N_rep x B

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
        loss = loss.sum()/z.size(0)

        return loss
