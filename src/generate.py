from pathlib import Path
import os
import sys

from tqdm import tqdm
import bpy
import mathutils
import numpy as np


def mkdirs(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def render_image(path):
    bpy.context.scene.render.filepath = path

    logfile = 'blender_render.log'
    open(logfile, 'a').close()
    old = os.dup(1)
    sys.stdout.flush()
    os.close(1)
    os.open(logfile, os.O_WRONLY)

    bpy.ops.render.render(write_still=True)

    os.close(1)
    os.dup(old)
    os.close(old)

def main():
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    bpy.context.scene.eevee.taa_render_samples = 8

    save_dir = "data/"

    mkdirs(save_dir)

    cube = bpy.data.objects["Cube"]

    np.random.seed(100)

    count = 2**12

    transforms = np.random.uniform(low=-5.0, high=5.0, size=[count, 3])

    np.save("data/transforms.npy", transforms)

    for i in tqdm(range(count)):
        cube.location = mathutils.Vector((*transforms[i], ))

        render_image("{}/img_{}.png".format(save_dir, i))


if __name__ == "__main__":
    main()
