import argparse
import os
import pickle
import shutil
import zipfile
import glob

from tqdm import tqdm

from utils import resize, mkdirs
from load_data import load_exr, write_exr
from constants import pickle_name, get_img_path, imgs_dir_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    parser.add_argument('--resize-to', type=int, default=None)
    args = parser.parse_args()

    all_data = []

    pickle_path = os.path.join(args.input_dir, pickle_name)

    mkdirs(args.output_dir)
    mkdirs(os.path.join(args.output_dir, imgs_dir_name))

    files = glob.glob("{}/*.zip".format(args.input_dir))
    overall_index = 0
    for full_path in tqdm(files):
        with zipfile.ZipFile(full_path, 'r') as zip_ref:
            zip_ref.extractall(args.input_dir)

        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        all_data.extend(data)

        print(full_path)

        for i in range(len(data)):
            img_path = get_img_path(i, args.input_dir)
            output_path = get_img_path(overall_index, args.output_dir)

            overall_index += 1

            if args.resize_to is None:
                shutil.move(img_path, output_path)
            else:
                img = load_exr(img_path)
                print(img.max())
                print(img.min())
                # resized = resize(img, args.resize_to)
                write_exr(output_path, img)

    output_pickle_path = os.path.join(args.output_dir, pickle_name)

    with open(pickle_path, 'wb') as f:
        pickle.dump(all_data, f)


if __name__ == "__main__":
    main()
