import pickle

from generate import basic_setup, DisplayBlenderScene
from constants import pickle_path


def main():
    index = 100

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    basic_setup(True)

    DisplayBlenderScene(data[index])


if __name__ == "__main__":
    main()
