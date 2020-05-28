import argparse
import pickle
import os
import shutil

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('other_dir')
    parser.add_argument('--merge-at', type=int, default=None)
    args = parser.parse_args()

    pickle_name = 'scenes.p'

    # TODO: fix path hard coding (constants file etc)
    with open(os.path.join(args.other_dir, pickle_name), 'rb') as f:
        other_data = pickle.load(f)

    orig_dir = 'data/'

    with open(os.path.join(orig_dir, pickle_name), 'rb') as f:
        orig_data = pickle.load(f)

    start = args.merge_at

    if start is None:
        start = len(orig_data)

    assert start <= len(orig_data) and start >= 0

    print("merging starting at", start)

    other_img_path = os.path.join(args.other_dir, "imgs")
    orig_img_path = os.path.join(orig_dir, "imgs")

    for i in range(len(orig_data[start:])):
        print("orig: {} to {}".format(i + start, i + start + len(other_data)))
        shutil.move(
            os.path.join(orig_img_path, "img_{}.exr".format(i + start)),
            os.path.join(orig_img_path,
                         "img_{}.exr".format(i + start + len(other_data))))

    for i in range(len(other_data)):
        print("other: {} to {}".format(i, i + start))
        shutil.move(
            os.path.join(other_img_path, "img_{}.exr".format(i)),
            os.path.join(orig_img_path, "img_{}.exr".format(i + start)))

    orig_data[start:start] = other_data

    with open(os.path.join(orig_dir, pickle_name), 'wb') as f:
        pickle.dump(orig_data, f)


if __name__ == "__main__":
    main()
