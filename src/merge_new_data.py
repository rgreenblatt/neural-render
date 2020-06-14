import argparse
import pickle
import os
import shutil

import numpy as np

from constants import data_path, pickle_name, pickle_path, get_img_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('other_dir')
    parser.add_argument('--merge-at', type=int, default=None)
    args = parser.parse_args()

    with open(os.path.join(args.other_dir, pickle_name), 'rb') as f:
        other_data = pickle.load(f)

    with open(pickle_path, 'rb') as f:
        orig_data = pickle.load(f)

    start = args.merge_at

    if start is None:
        start = len(orig_data)

    assert start <= len(orig_data) and start >= 0

    print("merging starting at", start)

    for i in range(len(orig_data[start:])):
        orig_index = i + start
        new_index = i + start + len(other_data)
        print("orig: {} to {}".format(orig_index, new_index))
        shutil.move(get_img_path(orig_index), get_img_path(new_index))

    for i in range(len(other_data)):
        orig_index = i
        new_index = i + start
        print("other: {} to {}".format(orig_index, new_index))
        shutil.move(get_img_path(orig_index, args.other_dir),
                    get_img_path(new_index))

    orig_data[start:start] = other_data

    with open(pickle_path, 'wb') as f:
        pickle.dump(orig_data, f)


if __name__ == "__main__":
    main()
