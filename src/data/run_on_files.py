# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference: https://raw.githubusercontent.com/facebookresearch/music-translation/master/src/run_on_files.py

from pathlib import Path
import librosa
import torch
from argparse import ArgumentParser
import matplotlib
import h5py
import tqdm

import src.data.utils as utils
import src.models.wavenet_models as wavenet_models
from src.data.utils import save_audio
from src.models.wavenet import WaveNet
from src.models.wavenet_generator import WavenetGenerator
from src.models.nv_wavenet_generator import NVWavenetGenerator

from src.models.cpc import CPC 


def extract_id(path):
    decoder_id = str(path)[:-4].split('_')[-1]
    return int(decoder_id)

def generate(args):
    print('Starting')
    matplotlib.use('agg')

    checkpoints = args.checkpoint.parent.glob(args.checkpoint.name + '_*.pth')
    checkpoints = [c for c in checkpoints if extract_id(c) in args.decoders]
    assert len(checkpoints) >= 1, "No checkpoints found."

    model_args = torch.load(args.checkpoint.parent / 'args.pth')[0]
    if args.model_name == 'umt':
        encoder = wavenet_models.Encoder(model_args)
    else:
        encoder = CPC(model_args)
    encoder.load_state_dict(torch.load(checkpoints[0])['encoder_state'])
    encoder.eval()
    encoder = encoder.cuda()

    def init_hidden(size, use_gpu=True):
        if use_gpu: return torch.zeros(1, size, model_args.latent_d).cuda()
        else: return torch.zeros(1, size, model_args.latent_d)
    
    decoders = []
    decoder_ids = []
    for checkpoint in checkpoints:
        decoder = WaveNet(model_args)
        decoder.load_state_dict(torch.load(checkpoint)['decoder_state'])
        decoder.eval()
        decoder = decoder.cuda()
        if args.py:
            decoder = WavenetGenerator(decoder, args.batch_size, wav_freq=args.rate)
        else:
            decoder = NVWavenetGenerator(decoder, args.rate * (args.split_size // 20), args.batch_size, 3)

        decoders += [decoder]
        decoder_ids += [extract_id(checkpoint)]

    xs = []
    assert args.output_next_to_orig ^ (args.output_generated is not None)

    if len(args.files) == 1 and args.files[0].is_dir():
        top = args.files[0]
        file_paths = list(top.glob('**/*.wav')) + list(top.glob('**/*.h5'))
    else:
        file_paths = args.files

    if not args.skip_filter:
        file_paths = [f for f in file_paths if not '_' in str(f.name)]

    for file_path in file_paths:
        if file_path.suffix == '.wav':
            data, rate = librosa.load(file_path, sr=16000)
            assert rate == 16000
            data = utils.mu_law(data)
        elif file_path.suffix == '.h5':
            data = utils.mu_law(h5py.File(file_path, 'r')['wav'][:] / (2 ** 15))
            if data.shape[-1] % args.rate != 0:
                data = data[:-(data.shape[-1] % args.rate)]
            assert data.shape[-1] % args.rate == 0
            print(data.shape)
        else:
            raise Exception(f'Unsupported filetype {file_path}')

        if args.sample_len:
            data = data[:args.sample_len]
        else:
            args.sample_len = len(data)
        xs.append(torch.tensor(data).unsqueeze(0).float().cuda())

    xs = torch.stack(xs).contiguous()
    print(f'xs size: {xs.size()}')

    def save(x, decoder_ix, filepath):
        wav = utils.inv_mu_law(x.cpu().numpy())
        print(f'X size: {x.shape}')
        print(f'X min: {x.min()}, max: {x.max()}')

        if args.output_next_to_orig:
            save_audio(wav.squeeze(), filepath.parent / f'{filepath.stem}_{decoder_ix}.wav', rate=args.rate)
        else:
            save_audio(wav.squeeze(), args.output / str(decoder_ix) / filepath.with_suffix('.wav').name, rate=args.rate)

    yy = {}
    with torch.no_grad():
        zz = []
        for xs_batch in torch.split(xs, args.batch_size):
            if args.model_name == 'umt':
                output = encoder(xs_batch)
            else:
                _, output = encoder(xs_batch)
            zz += [output]
        zz = torch.cat(zz, dim=0)

        with utils.timeit("Generation timer"):
            for i, decoder_id in enumerate(decoder_ids):
                yy[decoder_id] = []
                decoder = decoders[i]
                for zz_batch in torch.split(zz, args.batch_size):
                    print(zz_batch.shape)
                    splits = torch.split(zz_batch, args.split_size, -1)
                    audio_data = []
                    decoder.reset()
                    for cond in tqdm.tqdm(splits):
                        audio_data += [decoder.generate(cond).cpu()]
                    audio_data = torch.cat(audio_data, -1)
                    yy[decoder_id] += [audio_data]
                yy[decoder_id] = torch.cat(yy[decoder_id], dim=0)
                del decoder

    for decoder_ix, decoder_result in yy.items():
        for sample_result, filepath in zip(decoder_result, file_paths):
            save(sample_result, decoder_ix, filepath)


def main():
    parser = ArgumentParser()
    
    parser.add_argument('--model-name', type=str, required=True, choices=['umt', 'umtcpc'],
                        help='Type of model architecture')
    parser.add_argument('--files', type=Path, nargs='+', required=False,
                        help='Top level directories of input music files')
    parser.add_argument('-o', '--output', type=Path,
                        help='Output directory for output files')
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Checkpoint path')
    parser.add_argument('--decoders', type=int, nargs='*', default=[],
                        help='Only output for the following decoder ID')
    parser.add_argument('--rate', type=int, default=16000,
                        help='Wav sample rate in samples/second')
    parser.add_argument('--batch-size', type=int, default=6,
                        help='Batch size during inference')
    parser.add_argument('--sample-len', type=int,
                        help='If specified, cuts sample lengths')
    parser.add_argument('--split-size', type=int, default=20,
                        help='Size of splits')
    parser.add_argument('--output-next-to-orig', action='store_true')
    parser.add_argument('--skip-filter', action='store_true')
    parser.add_argument('--py', action='store_true', help='Use python generator')

    args = parser.parse_args()

    generate(args)

if __name__ == '__main__':
    with torch.no_grad():
        main()
