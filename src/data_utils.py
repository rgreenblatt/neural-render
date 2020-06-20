import Imath
import imageio
import OpenEXR
import numpy as np


def load_exr(path):
    # imagio doesn't work to load exrs in PIZ format...
    img = OpenEXR.InputFile(path)

    dw = img.header()['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    C = img.channels("RGB", Imath.PixelType(Imath.PixelType.FLOAT))
    C = (np.fromstring(c, dtype=np.float32) for c in C)
    C = (np.reshape(c, isize) for c in C)
    img = np.concatenate([c[..., None] for c in C], axis=2)

    return img


def write_exr(path, img):
    return imageio.imwrite(path, img)
