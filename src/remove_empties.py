import argparse
import os
import pickle
import shutil

from tqdm import tqdm

from utils import resize, mkdirs
from load_data import load_exr, write_exr


# probably shouldn't need to be used in the future...
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    args = parser.parse_args()

    # TODO: fix path hard coding (constants file etc)
    pickle_name = 'scenes.p'

    pickle_path = os.path.join(args.dir, pickle_name)

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    new_data = []
    index = 0

    def file_path(i):
        return os.path.join(args.dir, 'imgs', 'img_{}.exr'.format(i))

    for i in range(len(data)):
        if data[i].size == 0:
            print("EMPTY")
        else:
            new_data.append(data[i])

            if i != index:
                shutil.move(file_path(i), file_path(index))
            index += 1

    with open(pickle_path, 'wb') as f:
        pickle.dump(new_data, f)


if __name__ == "__main__":
    main()
