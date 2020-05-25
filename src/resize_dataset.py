import argparse
import os
import pickle
import shutil

from tqdm import tqdm

from utils import resize, mkdirs
from load_data import load_exr, write_exr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir')
    parser.add_argument('new_size', type=int)
    args = parser.parse_args()

    # TODO: fix path hard coding (constants file etc)
    pickle_name = 'scenes.p'
    orig_dir = 'data/'

    pickle_path = os.path.join(orig_dir, pickle_name)

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    mkdirs(os.path.join(args.output_dir, "imgs/"))

    for i in tqdm(range(len(data))):
        img = load_exr(os.path.join(orig_dir, "imgs", "img_{}.exr".format(i)))
        resized = resize(img, args.new_size)
        write_exr(
            os.path.join(args.output_dir, "imgs", "img_{}.exr".format(i)),
            resized)
    shutil.copy(pickle_path, os.path.join(args.output_dir, pickle_name))


if __name__ == "__main__":
    main()
