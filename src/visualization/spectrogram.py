import os

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.fonttype'] = 'none'
import numpy as np
from scipy.io.wavfile import read as readwav

from pathlib import Path
import argparse


def rainbow(fname, dst, peak=70.0, use_cqt=True):
    # Constants
    n_fft = 512
    hop_length = 256
    over_sample = 4
    res_factor = 0.8
    octaves = 6
    notes_per_octave = 10

    # Plotting functions
    cdict = {'red': ((0.0, 0.0, 0.0),
                     (1.0, 0.0, 0.0)),

             'green': ((0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0)),

             'blue': ((0.0, 0.0, 0.0),
                      (1.0, 0.0, 0.0)),

             'alpha': ((0.0, 1.0, 1.0),
                       (1.0, 0.0, 0.0))
             }

    my_mask = matplotlib.colors.LinearSegmentedColormap('MyMask', cdict)
    plt.register_cmap(cmap=my_mask)

    fig, ax = plt.subplots(figsize=(6, 6))
    sr, audio = readwav(fname)
    audio = audio.astype(np.float32)
    if use_cqt:
        C = librosa.cqt(audio, sr=sr, hop_length=hop_length,
                        bins_per_octave=int(notes_per_octave*over_sample),
                        n_bins=int(octaves * notes_per_octave * over_sample),
                        filter_scale=res_factor,
                        fmin=librosa.note_to_hz('C2'))
    else:
        C = librosa.stft(audio, n_fft=n_fft, win_length=n_fft, hop_length=hop_length, center=True)
    mag, phase = librosa.core.magphase(C)
    phase_angle = np.angle(phase)
    phase_unwrapped = np.unwrap(phase_angle)
    dphase = phase_unwrapped[:, 1:] - phase_unwrapped[:, :-1]
    dphase = np.concatenate([phase_unwrapped[:, 0:1], dphase], axis=1) / np.pi
    mag = (librosa.power_to_db(mag ** 2, amin=1e-13, top_db=peak, ref=np.max) / peak) + 1
    ax.imshow(np.flipud(dphase[::-1, :]), cmap=plt.cm.rainbow, origin='lower')
    ax.imshow(np.flipud(mag[::-1, :]), cmap=my_mask, origin='lower')
    ax.set_title(f"Rainbow Spectrogram for {os.path.basename(fname)}")

    out_path = dst / os.path.basename(fname).replace('.wav', '_rainbow.jpg')
    plt.savefig(out_path, dpi=100)

def cqt(fname, dst):
    sr, audio = readwav(fname)
    CQT = librosa.amplitude_to_db(np.abs(librosa.cqt(audio.astype(np.float32), sr=sr)), ref=np.max)

    fig, ax = plt.subplots(figsize=(6, 6))
    d = librosa.display.specshow(CQT, y_axis='cqt_note', ax=ax)
    fig.colorbar(d, ax=ax, format='%+2.0f dB')
    ax.set_title('Constant-Q power spectrogram (note)')

    out_path = dst / os.path.basename(fname).replace('.wav', '_cqt.jpg')
    plt.savefig(out_path, dpi=100)

def chroma(fname, dst):
    sr, audio = readwav(fname)
    C = librosa.feature.chroma_cqt(y=audio.astype(np.float32), sr=sr)

    fig, ax = plt.subplots(figsize=(6, 6))
    d = librosa.display.specshow(C, y_axis='chroma')
    fig.colorbar(d, ax=ax)
    ax.set_title('Chromagram')

    if not os.path.exists(dst):
        os.makedirs(dst)

    out_path = dst / os.path.basename(fname).replace('.wav', '_chroma.jpg')
    plt.savefig(out_path, dpi=100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize sample with rainbowgram or mel spectrogram')
    parser.add_argument('--file', type=Path, required=True, help="Path to wav file")
    parser.add_argument('--dst', type=Path, required=True, help="Path to save plots")
    parser.add_argument('--rainbow', action='store_true', help="Plot rainbowgram")
    parser.add_argument('--cqt', action='store_true', help="Plot Constant-Q power spectrogram")
    parser.add_argument('--chroma', action='store_true', help="Plot chromagram with pitch classes")

    args = parser.parse_args()

    if args.rainbow:
        rainbow(args.file, args.dst)
    if args.cqt:
        cqt(args.file, args.dst)
    if args.chroma:
        chroma(args.file, args.dst)