import argparse
import os
import pickle
import shutil
from multiprocessing import Pool
from functools import partial

from tqdm import tqdm

from utils import resize, mkdirs
from data_utils import load_exr, write_exr
from constants import pickle_name, data_path, imgs_dir_name, get_img_path


def resize_move(i, args):
    img = load_exr(get_img_path(i))
    resized = resize(img, args.new_size)
    write_exr(get_img_path(i, args.output_dir), resized)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir')
    parser.add_argument('new_size', type=int)
    parser.add_argument('--pool-size', type=int, default=10)
    args = parser.parse_args()

    # TODO: fix path hard coding (constants file etc)

    pickle_path = os.path.join(data_path, pickle_name)

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    mkdirs(os.path.join(args.output_dir, imgs_dir_name))

    pool = Pool(args.pool_size)

    task = partial(resize_move, args=args)

    for _ in tqdm(pool.imap_unordered(task, range(len(data))),
                  total=len(data)):
        pass

    shutil.copy(pickle_path, os.path.join(args.output_dir, pickle_name))


if __name__ == "__main__":
    main()
