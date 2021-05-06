import os
import shutil
import argpause

def pair_samples(root):
    root = str(root)
    path = root + '/nsynth/parsed/'
    out = root + '/paired/'

    folder_string = path + 'string_acoustic/'
    folder_kb = path + 'keyboard_acoustic/'
    t_str = os.listdir(folder_string)
    t_kb = os.listdir(folder_kb)

    for j in t_str:
        for i in t_kb:
            if j[20:23] == i[22:25]:
                out_pitch = out + j[20:23]
            if not os.path.exists(out_pitch):
                os.makedirs(out_pitch)
            shutil.copyfile(folder_string + j, out_pitch + '/' + j)
            shutil.copyfile(folder_kb + i, out_pitch + '/' + i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NSynth Pitch Pairing')

    parser.add_argument('-i', '--input', type=Path, required=True,
                        help='Root directory for data (musicnet folder)')

    args = parser.parse_args()

    root = args.input

    # Pair samples from root
    pair_samples(root)