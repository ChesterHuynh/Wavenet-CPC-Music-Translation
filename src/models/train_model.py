from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from pathlib import Path
import os
import tqdm

from src.data.utils import LossMeter
from src.data.utils import LogFormatter
from src.data.utils import create_output_dir
from src.data.data import DatasetSet

class UMTCPCTrainer:
    def __init__(self, args):
        self.args = args
        self.args.n_datasets = len(self.args.data)
        self.expPath = Path('/content/checkpoints') / args.expName

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        self.logger = create_output_dir(args, self.expPath)
        self.data = [DatasetSet(d, args.seq_len, args) for d in args.data]
        assert not args.distributed or len(self.data) == int(
            os.environ['WORLD_SIZE']), "Number of datasets must match number of nodes"

        self.losses_recon = [LossMeter(f'recon {i}') for i in range(self.args.n_datasets)]
        self.losses_nce = [LossMeter(f'nce {i}') for i in range(self.args.n_datasets)]
        self.loss_d_right = LossMeter('d')
        self.loss_total = LossMeter('total')

        self.evals_recon = [LossMeter(f'recon {i}') for i in range(self.args.n_datasets)]
        self.evals_nce = [LossMeter(f'nce {i}') for i in range(self.args.n_datasets)]
        self.eval_d_right = LossMeter('eval d')
        self.eval_total = LossMeter('eval total')

        self.cpc_encoder = CPC(args)
        self.encoder = Encoder(args)
        self.decoder = WaveNet(args)
        self.discriminator = ZDiscriminator(args)

        if args.checkpoint:
            checkpoint_args_path = os.path.dirname(args.checkpoint) + '/args.pth'
            checkpoint_args = torch.load(checkpoint_args_path)

            self.start_epoch = checkpoint_args[-1] + 1
            states = torch.load(args.checkpoint)

            self.cpc_encoder.load_state_dict(states['cpc_encoder_state'])
            self.decoder.load_state_dict(states['decoder_state'])
            self.discriminator.load_state_dict(states['discriminator_state'])

            self.logger.info('Loaded checkpoint parameters')
        else:
            self.start_epoch = 0

        if args.distributed:
            self.cpc_encoder.cuda()
            self.cpc_encoder = torch.nn.parallel.DistributedDataParallel(self.cpc_encoder)
            self.encoder.cuda()
            self.encoder = torch.nn.parallel.DistributedDataParallel(self.encoder)
            self.discriminator.cuda()
            self.discriminator = torch.nn.parallel.DistributedDataParallel(self.discriminator)
            self.logger.info('Created DistributedDataParallel')
        else:
            self.cpc_encoder = torch.nn.DataParallel(self.cpc_encoder).cuda()
            self.encoder = torch.nn.DataParallel(self.encoder).cuda()
            self.discriminator = torch.nn.DataParallel(self.discriminator).cuda()
        self.decoder = torch.nn.DataParallel(self.decoder).cuda()

        self.model_optimizer = optim.Adam(chain(self.cpc_encoder.parameters(),
                                                self.decoder.parameters()),
                                          lr=args.lr)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(),
                                      lr=args.lr)

        if args.checkpoint and args.load_optimizer:
            self.model_optimizer.load_state_dict(states['model_optimizer_state'])
            self.d_optimizer.load_state_dict(states['d_optimizer_state'])

        self.lr_manager = torch.optim.lr_scheduler.ExponentialLR(self.model_optimizer, args.lr_decay)
        self.lr_manager.last_epoch = self.start_epoch

    def init_hidden(self, use_gpu=True):
        if use_gpu: return torch.zeros(1, self.args.batch_size, self.args.latent_d).cuda()
        else: return torch.zeros(1, self.args.batch_size, self.args.latent_d)

    def eval_batch(self, x, x_aug, dset_num):
        x, x_aug = x.float(), x_aug.float()

        hidden = self.init_hidden()

        c, hidden, encoder_samples, pred = self.cpc_encoder(x, hidden)
        y = self.decoder(x, c)
        c_logits = self.discriminator(c)

        c_classification = torch.max(c_logits, dim=1)[1]

        c_accuracy = (c_classification == dset_num).float().mean()

        self.eval_d_right.add(c_accuracy.data.item())

        # discriminator_right = F.cross_entropy(c_logits, dset_num).mean()
        discriminator_right = F.cross_entropy(c_logits, torch.tensor([dset_num] * x.size(0)).long().cuda()).mean()
        recon_loss = cross_entropy_loss(y, x)
        self.evals_recon[dset_num].add(recon_loss.data.cpu().numpy().mean())

        nce_loss = InfoNCELoss(encoder_samples, pred, self.args.timestep)
        self.evals_nce[dset_num].add(nce_loss.data.cpu().numpy())

        total_loss = discriminator_right.data.item() * self.args.d_lambda + \
                     recon_loss.mean().data.item() + nce_loss.data.item()

        self.eval_total.add(total_loss)

        return total_loss

    def train_batch(self, x, x_aug, dset_num):
        x, x_aug = x.float(), x_aug.float()

        hidden = self.init_hidden()

        # Optimize D - discriminator right
        c, _, encoder_samples, pred = self.cpc_encoder(x, hidden)
        c_logits = self.discriminator(c)
        discriminator_right = F.cross_entropy(c_logits, torch.tensor([dset_num] * x.size(0)).long().cuda()).mean()
        self.loss_d_right.add(discriminator_right.data.cpu())

        # Get c_t for computing InfoNCE Loss
        nce_loss = InfoNCELoss(encoder_samples, pred, self.args.timestep)
        loss = discriminator_right * self.args.d_lambda + nce_loss
        self.d_optimizer.zero_grad()
        loss.backward()
        if self.args.grad_clip is not None:
            clip_grad_value_(self.discriminator.parameters(), self.args.grad_clip)

        self.d_optimizer.step()

        # optimize G - reconstructs well, discriminator wrong
        c, _, encoder_samples, pred = self.cpc_encoder(x_aug, hidden)
        y = self.decoder(x, c)
        c_logits = self.discriminator(c)
        discriminator_wrong = - F.cross_entropy(c_logits, torch.tensor([dset_num] * x.size(0)).long().cuda()).mean()

        if not (-100 < discriminator_right.data.item() < 100):
            self.logger.debug(f'c_logits: {c_logits.detach().cpu().numpy()}')
            self.logger.debug(f'dset_num: {dset_num}')

        nce_loss = InfoNCELoss(encoder_samples, pred, self.args.timestep)
        self.losses_nce[dset_num].add(nce_loss.data.cpu().numpy())

        recon_loss = cross_entropy_loss(y, x)
        self.losses_recon[dset_num].add(recon_loss.data.cpu().numpy().mean())

        loss = (recon_loss.mean() + self.args.d_lambda * discriminator_wrong) + nce_loss

        self.model_optimizer.zero_grad()
        loss.backward()
        if self.args.grad_clip is not None:
            clip_grad_value_(self.encoder.parameters(), self.args.grad_clip)
            clip_grad_value_(self.decoder.parameters(), self.args.grad_clip)
        self.model_optimizer.step()

        self.loss_total.add(loss.data.item())

        return loss.data.item()

    def train_epoch(self, epoch):
        for meter in self.losses_recon:
            meter.reset()
        for meter in self.losses_nce:
            meter.reset()
        self.loss_d_right.reset()
        self.loss_total.reset()

        self.cpc_encoder.train()
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()

        n_batches = self.args.epoch_len

        with tqdm.tqdm(total=n_batches, desc='Train epoch %d' % epoch) as train_enum:
            for batch_num in range(n_batches):
                if self.args.short and batch_num == 3:
                    break

                if self.args.distributed:
                    assert self.args.rank < self.args.n_datasets, "No. of workers must be equal to #dataset"
                    # dset_num = (batch_num + self.args.rank) % self.args.n_datasets
                    dset_num = self.args.rank
                else:
                    dset_num = batch_num % self.args.n_datasets

                x, x_aug = next(self.data[dset_num].train_iter)

                x = wrap(x)
                x_aug = wrap(x_aug)
                batch_loss = self.train_batch(x, x_aug, dset_num)

                train_enum.set_description(f'Train (loss: {batch_loss:.2f}) epoch {epoch}')
                train_enum.update()

    def evaluate_epoch(self, epoch):
        for meter in self.evals_recon:
            meter.reset()
        for meter in self.evals_nce:
            meter.reset()
        self.eval_d_right.reset()
        self.eval_total.reset()

        self.cpc_encoder.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()

        n_batches = int(np.ceil(self.args.epoch_len / 10))

        with tqdm.tqdm(total=n_batches) as valid_enum, \
                torch.no_grad():
            for batch_num in range(n_batches):
                if self.args.short and batch_num == 10:
                    break

                if self.args.distributed:
                    assert self.args.rank < self.args.n_datasets, "No. of workers must be equal to #dataset"
                    dset_num = self.args.rank
                else:
                    dset_num = batch_num % self.args.n_datasets

                x, x_aug = next(self.data[dset_num].valid_iter)

                x = wrap(x)
                x_aug = wrap(x_aug)
                batch_loss = self.eval_batch(x, x_aug, dset_num)

                valid_enum.set_description(f'Test (loss: {batch_loss:.2f}) epoch {epoch}')
                valid_enum.update()

    @staticmethod
    def format_losses(meters):
        losses = [meter.summarize_epoch() for meter in meters]
        return ', '.join('{:.4f}'.format(x) for x in losses)

    def train_losses(self):
        meters = [*self.losses_recon, *self.losses_nce, self.loss_d_right]
        return self.format_losses(meters)

    def eval_losses(self):
        meters = [*self.evals_recon, *self.evals_nce, self.eval_d_right]
        return self.format_losses(meters)


    def train(self):
        best_eval = float('inf')

        # Begin!
        for epoch in range(self.start_epoch, self.start_epoch + self.args.epochs):
            self.logger.info(f'Starting epoch, Rank {self.args.rank}, Dataset: {self.args.data[self.args.rank]}')
            self.train_epoch(epoch)
            self.evaluate_epoch(epoch)

            self.logger.info(f'Epoch %s Rank {self.args.rank} - Train loss: (%s), Test loss (%s)',
                             epoch, self.train_losses(), self.eval_losses())
            self.lr_manager.step()
            val_loss = self.eval_total.summarize_epoch()

            if val_loss < best_eval:
                self.save_model(f'bestmodel_{self.args.rank}.pth')
                best_eval = val_loss

            if not self.args.per_epoch:
                self.save_model(f'lastmodel_{self.args.rank}.pth')
            else:
                self.save_model(f'lastmodel_{epoch}_rank_{self.args.rank}.pth')

            if self.args.is_master:
                torch.save([self.args,
                            epoch],
                           '%s/args.pth' % self.expPath)

            self.logger.debug('Ended epoch')

    def save_model(self, filename):
        save_path = self.expPath / filename

        torch.save({'cpc_encoder_state': self.cpc_encoder.module.state_dict(),
                    'decoder_state': self.decoder.module.state_dict(),
                    'discriminator_state': self.discriminator.module.state_dict(),
                    'model_optimizer_state': self.model_optimizer.state_dict(),
                    'dataset': self.args.rank,
                    'd_optimizer_state': self.d_optimizer.state_dict()
                    },
                   save_path)

        self.logger.debug(f'Saved model to {save_path}')