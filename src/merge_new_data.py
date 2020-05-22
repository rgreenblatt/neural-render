import argparse
import pickle
import os
import shutil

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('other_dir')
    args = parser.parse_args()

    pickle_name = 'scenes.p'

    # TODO: fix path hard coding (constants file etc)
    with open(os.path.join(args.other_dir, pickle_name), 'rb') as f:
        other_data = pickle.load(f)

    orig_dir = 'data/'

    with open(os.path.join(orig_dir, pickle_name), 'rb') as f:
        orig_data = pickle.load(f)

    start = len(orig_data)

    print("appending starting at", start)

    other_img_path = os.path.join(args.other_dir, "imgs")
    orig_img_path = os.path.join(orig_dir, "imgs")

    for i in range(len(other_data)):
        shutil.move(
            os.path.join(other_img_path, "img_{}.exr".format(i)),
            os.path.join(orig_img_path, "img_{}.exr".format(i + start)))

    orig_data.extend(other_data)

    with open(os.path.join(orig_dir, pickle_name), 'wb') as f:
        pickle.dump(orig_data, f)

if __name__ == "__main__":
    main()
