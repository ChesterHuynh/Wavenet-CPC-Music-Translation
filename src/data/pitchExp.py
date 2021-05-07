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
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
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

    return np.max(c)

def compare(original, translated):
    # Get the original and translated samples
    wave_data_orig, samplerate = librosa.load(original, mono=True)
    wave_data_translated, _ = librosa.load(translated, mono=True)

    # Calculate their pitches and magnitudes using piptrack
    pitches_orig, magnitudes_orig = librosa.piptrack(wave_data_orig, sr=samplerate)
    pitches_orig, magnitudes_orig = extract_max(pitches_orig,magnitudes_orig, np.shape(pitches_orig))
    pitches_orig = smooth(np.asarray(pitches_orig), window_len=20)
    pitches_orig = np.asarray(pitches_orig)

    pitches_trans, magnitudes_trans = librosa.piptrack(wave_data_translated, sr=samplerate)
    pitches_trans, magnitudes_trans = extract_max(pitches_trans,magnitudes_trans, np.shape(pitches_trans))
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
    print(corr)

    return corr

def main():
    directory = 'C:/Users/shizh/Documents/GitHub/Wavenet-CPC-Music-Translation/paired-5_new/'
    ls = [x[0] for x in os.walk(directory)][1:]
    correlations = []

    for d in ls:
        pitch = str(d[-3:])
        path = d
        original_kb = ''
        translated_kb_umtcpc = ''
        original_str = ''
        translated_str = ''
        for filename in glob.glob(os.path.join(path, '*.wav')):
            if len(filename[len(path):-4]) > 1:
                if 'keyboard_acoustic' in filename and not 'umt' in filename:
                    original_kb = filename
                    translated_kb = original_kb
                elif 'keyboard_acoustic' in filename and 'umt' in filename:
                    translated_kb = filename
                    print(original_str[len(path)+1:], translated_kb[len(path)+1:])
                    corr = compare(original_str, translated_kb)
                    correlations.append([original_str[len(path)+1:], translated_kb[len(path)+1:], str(corr)])
                elif 'string_acoustic' in filename and not 'umt' in filename:
                    original_str = filename
                    translated_str = original_str
                elif 'string_acoustic' in filename and 'umt' in filename:
                    translated_str = filename
                    print(original_kb[len(path)+1:], translated_str[len(path)+1:])
                    corr = compare(original_kb, translated_str)
                    correlations.append([original_kb[len(path)+1:], translated_str[len(path)+1:], str(corr)])

    with open('pitch_5_new/outputAll.txt', 'w') as f:
        for i in correlations:
            f.write('%s, ' % i[0])
            f.write('%s, ' % i[1])
            f.write('%s\n' % i[2])

if __name__ == "__main__":
    main()