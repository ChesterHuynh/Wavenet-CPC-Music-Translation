import os
import json
import glob
import numpy as np
from scipy.io import wavfile
import os
import sys
import random
import librosa
import librosa.display

def extract_max(pitches,magnitudes, shape):
    new_pitches = []
    new_magnitudes = []
    for i in range(0, shape[1]):
        new_pitches.append(np.max(pitches[:,i]))
        new_magnitudes.append(np.max(magnitudes[:,i]))
    return (new_pitches,new_magnitudes)

def smooth(x,window_len=11,window='hanning'):
    if window_len<3:
            return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': #moving average
            w=np.ones(window_len,'d')
    else:
            w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='same')
    return y[window_len:-window_len+1]

def corrCalc(pitches, pitches_2):
    a = (pitches - np.mean(pitches)) / (np.std(pitches))
    b = (pitches_2 - np.mean(pitches_2)) / (np.std(pitches_2))
    c = np.correlate(a, b, 'same') / max(len(a), len(b))

    return c

def compare(original, translated):
    # Get the original and translated samples
    wave_data_orig, samplerate = librosa.load(original, mono=True)
    wave_data_translated, _ = librosa.load(translated, mono=True)

    # Calculate their pitches and magnitudes using piptrack
    pitches_orig, magnitudes_orig = librosa.piptrack(wave_data_orig, sr=samplerate)
    piches_orig, magnitudes_orig = extract_max(pitches_orig,magnitudes, np.shape(pitches))
    piches_orig = smooth(np.asarray(piches_orig), window_len=20)
    pitches_trans = np.asarray(pitches_orig)

    pitches_trans, magnitudes_trans = librosa.piptrack(wave_data_translated, sr=samplerate)
    pitches_trans, magnitudes_trans = extract_max(pitches_trans,magnitudes, np.shape(pitches_2))
    pitches_trans = smooth(np.asarray(pitches_trans), window_len=20)
    pitches_trans = np.asarray(pitches_trans)

    # DTW on pitches, pitches_2
    if pitches_trans.shape[0] != pitches_orig.shape[0]:
        D, wp_pitches = librosa.sequence.dtw(pitches_orig, pitches_trans, subseq=True)

        # Warp the paths using dtw
        x_path, y_path = zip(*wp_pitches)
        x_path = np.asarray(x_path)
        y_path = np.asarray(y_path)
        pitches = pitches[:,x_path]
        pitches_2 = pitches_2[:,y_path]

    # Calculate the correlation for the best shift
    corr = corrCalc(pitches_orig, pitches_trans)
    print(f'Correlation after DTW: {corr:.4f}')
    
    return corr

def main():
    path = '/home/magjywang/music-translation/results/29_04_2021/musicnet-py/0/'
    correlations = []
    for filename in glob.glob(os.path.join(path, '*.wav')):
        if len(filename[0:-4]) > 1:
            original = filename[0] + '.wav'
            translated = filename

            corr = compare(original, translated)
            correlations.append([filename, str(corr)])
    
    with open('output.txt', 'w') as f:
    for i in correlations:
        f.write('%s\n' % i)

if __name__ == "__main__":
    main()