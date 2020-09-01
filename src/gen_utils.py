import numpy as np


def random_seed():
    return np.random.randint(0, 2**31 - 1)


# not quite invisible :(
def invisible_material():
    color = np.array([1.0, 0.0, 0.0])
    metallic = 0.0
    specular = 0.0
    roughness = 0.0
    transmission = 1.0
    emission = np.array([0.0, 0.0, 0.0])

    return np.concatenate(
        (color, [metallic], [specular], [roughness], [transmission], emission))


def filler_sphere():
    location = np.array([9.0, 3.0, 9.0])
    # W, X, Y, Z
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    scale = np.array([1.0, 1.0, 1.0])
    mat_params = invisible_material()

    return np.concatenate((location, quat, scale, mat_params))


def fill_to_size(value, size):
    return np.concatenate(
        (value, [filler_sphere() for _ in range(size - value.shape[0])]))
