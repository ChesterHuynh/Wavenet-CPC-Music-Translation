import os
import shutil
import argparse
from pathlib import Path
import glob

def pair_samples(root, num_pairs, num_pitches):
    path = root /'parsed'
    out = root / 'paired'

    folder_string = path / 'string_acoustic'
    folder_kb = path / 'keyboard_acoustic'
    t_str = os.listdir(folder_string)
    t_kb = os.listdir(folder_kb)
    
    done=False
    for j in t_str:
        for i in t_kb:
            if j[20:23] == i[22:25]:
                out_pitch = out / j[20:23]
                if not os.path.exists(out_pitch):
                    os.makedirs(out_pitch)
               
                if len(list(out_pitch.glob('**/*string*'))) < num_pairs:
                    shutil.copyfile(folder_string / j, out_pitch / j)
                if len(list(out_pitch.glob('**/*keyboard*'))) < num_pairs:
                    shutil.copyfile(folder_kb / i, out_pitch / i)
                
                if len(list(out.glob('**/*.wav'))) > num_pairs * num_pitches * 2:
                    done=True
                    break
        if done:
            break
               
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NSynth Pitch Pairing')

    parser.add_argument('-i', '--input', type=Path, required=True,
                        help='Root directory for data (NSynth folder)')
    parser.add_argument('--num-pairs', type=int, help='Number of pairs to put in each directory', default=1)

    parser.add_argument('--num-pitches', type=int, help='Number of pitches', default=5)

    args = parser.parse_args()

    # Pair samples from root
    pair_samples(args.input, args.num_pairs, args.num_pitches)
