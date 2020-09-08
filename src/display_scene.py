import pickle

import numpy as np

from gen_utils import fill_to_size
from generate import basic_setup, DisplayBlenderScene
from constants import pickle_path


def main():
    index = 16

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    for i in range(100):
        print("i:", i, "shape:", data[i].shape)

    basic_setup(True)

    value = data[index]

    value = fill_to_size(value, 100)

    scene = DisplayBlenderScene(value)

    # print(scene.spheres[0].data.triangles[0].co)
    # print(scene.spheres[0].data.vertices[0].co)
    # print(scene.spheres[0].data.vertices[0].co.x)
    # print(scene.spheres[0].data.vertices[0].co.y)
    # print(scene.spheres[0].data.vertices[0].co.z)


if __name__ == "__main__":
    main()
