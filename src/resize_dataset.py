import argparse
import os
import pickle
import shutil

from tqdm import tqdm

from utils import resize, mkdirs
from data_utils import load_exr, write_exr
from constants import pickle_name, data_path, imgs_dir_name, get_img_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir')
    parser.add_argument('new_size', type=int)
    args = parser.parse_args()

    # TODO: fix path hard coding (constants file etc)

    pickle_path = os.path.join(data_path, pickle_name)

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    mkdirs(os.path.join(args.output_dir, imgs_dir_name))

    for i in tqdm(range(len(data))):
        img = load_exr(get_img_path(i))
        resized = resize(img, args.new_size)
        write_exr(get_img_path(i, args.output_dir), resized)
    shutil.copy(pickle_path, os.path.join(args.output_dir, pickle_name))


if __name__ == "__main__":
    main()
