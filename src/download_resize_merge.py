from multiprocessing import Pool
import argparse
import functools
import os
import pickle
import shutil
import zipfile

import dropbox
from tqdm import tqdm

from constants import pickle_name, get_img_path, imgs_dir_name
from dropbox_utils import download, list_folder
from utils import mkdirs, resize
from load_data import load_exr, write_exr


def copy_image(download_dir, overall_index, output_dir, args, i):
    img_path = get_img_path(i, download_dir)
    output_path = get_img_path(overall_index + i, output_dir)

    if args.resize_to is None:
        shutil.move(img_path, output_path)
    else:
        img = load_exr(img_path)
        resized = resize(img, args.resize_to)
        write_exr(output_path, resized)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('api_key')
    parser.add_argument('--resize-to', type=int, default=None)
    args = parser.parse_args()

    dbx = dropbox.Dropbox(args.api_key)

    download_dir = "downloads/"
    output_dir = "merged_downloads/"

    mkdirs(download_dir)
    mkdirs(output_dir)
    mkdirs(os.path.join(output_dir, imgs_dir_name))

    files = [data.name for data in list_folder(dbx, "/", "sphere_renders")]

    pickle_path = os.path.join(download_dir, pickle_name)

    all_data = []
    overall_index = 0

    p = Pool(16)

    for file in tqdm(files):
        shutil.rmtree(download_dir, ignore_errors=True)
        mkdirs(download_dir)

        output_file = "downloads/{}".format(file)
        download(dbx,
                 "/sphere_renders/{}".format(file),
                 output_file,
                 verbose=False)
        with zipfile.ZipFile(output_file, 'r') as zip_ref:
            zip_ref.extractall(download_dir)

        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        all_data.extend(data)

        p.map(
            functools.partial(copy_image, download_dir, overall_index,
                              output_dir, args), range(len(data)))
        overall_index += len(data)

    p.close()
    p.join()
    output_pickle_path = os.path.join(output_dir, pickle_name)

    with open(output_pickle_path, 'wb') as f:
        pickle.dump(all_data, f)


if __name__ == "__main__":
    main()
