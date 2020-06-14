from pathlib import Path
import os
import sys
import argparse
import math
import pickle

from tqdm import tqdm
import bpy
import mathutils
import numpy as np
from scipy.spatial.transform import Rotation as R

from generate_config import GenerateConfig
from gen_utils import random_seed
from constants import pickle_name, get_img_path


def select_set(obj, value):
    # Only works in blender >= 2.80
    obj.select_set(value)


def mkdirs(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def remove_printing(thunk):
    logfile = '/dev/null'
    open(logfile, 'a').close()
    old = os.dup(1)
    sys.stdout.flush()
    os.close(1)
    os.open(logfile, os.O_WRONLY)

    thunk()

    os.close(1)
    os.dup(old)
    os.close(old)


def render_image(path):
    bpy.context.scene.render.filepath = path

    remove_printing(lambda: bpy.ops.render.render(write_still=True))


def random_location():
    base_loc = np.array([0.0, 15.0, 0.0])
    translate = np.random.uniform(low=-12.0, high=10.0, size=[3])

    return base_loc + translate


def random_rotation(scale=1.0):
    quat = R.from_rotvec(R.random().as_rotvec() * scale).as_quat()

    # switch (X, Y, Z, W) to (W, X, Y, Z)
    return np.concatenate((quat[3:], quat[:3]))


def random_scale(location):
    base_scale = np.exp(np.random.normal(loc=-3.0)) * location[1]
    components_scale = np.random.uniform(low=0.5, high=1.5, size=[3])

    return base_scale * components_scale


def random_material():
    # TODO: consider tuning distributions

    color = np.random.uniform(size=3)
    metallic = np.random.uniform()**4
    specular = np.random.uniform()**4
    roughness = np.random.uniform()

    prob_transparent = 0.5

    is_transparent = np.random.uniform() < prob_transparent

    transmission = 1 - np.random.uniform()**4

    prob_emmisive = 0.2

    is_emmisive = np.random.uniform() < prob_emmisive

    emission = np.exp(np.random.normal(
        size=[3])) if is_emmisive else np.zeros(3)

    return np.concatenate(
        (color, [metallic], [specular], [roughness], [transmission], emission))


def random_scene(num_objects, rotation_scale):
    spheres = []
    for i in range(num_objects):
        location = random_location()
        rotation = random_rotation(rotation_scale)
        scale = random_scale(location)
        mat_params = random_material()

        spheres.append(np.concatenate((location, rotation, scale, mat_params)))

    return np.array(spheres)


class DisplayBlenderScene():
    def __init__(self, params):
        self.spheres = []
        self.materials = []

        for i in range(params.shape[0]):
            sphere, material = self._display_sphere(params[i], i)

            self.spheres.append(sphere)
            self.materials.append(material)

    def _display_sphere(self, sphere_params, i):
        bpy.ops.object.select_all(action='DESELECT')

        bpy.ops.mesh.primitive_uv_sphere_add()

        # sphere is selected
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.faces_shade_smooth()
        bpy.ops.object.editmode_toggle()

        sphere = bpy.context.selected_objects[0]

        sphere.rotation_mode = 'QUATERNION'

        location = sphere_params[:3]
        rotation = sphere_params[3:7]
        scale = sphere_params[7:10]

        sphere.location = (*location, )
        sphere.rotation_quaternion = (*rotation, )
        sphere.scale = (*scale, )

        mat = bpy.data.materials.new(name="sphere_mat_{}".format(i))

        sphere.data.materials.append(mat)

        mat.use_nodes = True
        mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
        self._set_material(sphere_params[10:],
                           mat.node_tree.nodes["Principled BSDF"])

        return sphere, mat

    def _set_material(self, material_params, bsdf):
        # base color: 0
        # metallic: 4
        # specular: 5
        # roughness: 7
        # transmission 15
        # emmision 17

        color = material_params[:3]
        metallic = material_params[3]
        specular = material_params[4]
        roughness = material_params[5]
        transmission = material_params[6]
        emission = material_params[7:]

        bsdf.inputs[0].default_value = (*color, 1)
        bsdf.inputs[4].default_value = metallic
        bsdf.inputs[5].default_value = specular
        bsdf.inputs[7].default_value = roughness
        bsdf.inputs[15].default_value = transmission
        bsdf.inputs[17].default_value = (*emission, 1)

    def cleanup(self):
        bpy.ops.object.select_all(action='DESELECT')

        for sphere in self.spheres:
            select_set(sphere, True)

        remove_printing(lambda: bpy.ops.object.delete())

        for mat in self.materials:
            bpy.data.materials.remove(mat)


def basic_setup(use_gpu):
    camera = bpy.data.objects["Camera"]
    camera.location = mathutils.Vector((0.0, 0.0, 0.0))
    camera.scale = mathutils.Vector((1.0, 1.0, 1.0))
    camera.rotation_euler[0] = math.pi / 2
    camera.rotation_euler[1] = 0.0
    camera.rotation_euler[2] = 0.0

    # make background dark
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[
        0].default_value = (0, 0, 0, 1)

    # sometimes required to "refresh" devices (doesn't need to be printed)
    print("devices are:",
          bpy.context.preferences.addons['cycles'].preferences.get_devices())

    bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
    bpy.context.scene.render.image_settings.exr_codec = 'PIZ'
    bpy.context.scene.render.engine = 'CYCLES'

    if use_gpu:
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.preferences.addons[
            'cycles'].preferences.compute_device_type = "CUDA"
        bpy.context.scene.render.tile_x = 256
        bpy.context.scene.render.tile_y = 256


def main(in_blender_mode=False):
    if not in_blender_mode:
        argv = sys.argv
        try:
            argv = argv[argv.index("--") + 1:]  # get all args after "--"
        except ValueError:
            argv = []

        cfg = GenerateConfig(argv)

        bpy.context.scene.render.resolution_x = cfg.resolution
        bpy.context.scene.render.resolution_y = cfg.resolution

        bpy.context.scene.cycles.samples = cfg.samples

        count = cfg.count
        seed = cfg.seed
        use_gpu = not cfg.no_gpu
    else:
        count = 1
        seed = 0
        use_gpu = True

    save_dir = os.path.join("generated", "seed_{}".format(seed))

    mkdirs(save_dir)

    # delete default cube and light
    if not in_blender_mode:
        bpy.ops.object.select_all(action='DESELECT')

        if 'Cube' in bpy.data.objects:
            select_set(bpy.data.objects['Cube'], True)
        if 'Light' in bpy.data.objects:
            select_set(bpy.data.objects['Light'], True)

        remove_printing(lambda: bpy.ops.object.delete())

    basic_setup(use_gpu)

    np.random.seed(seed)

    scenes = []
    blender_seed = random_seed()

    for i in tqdm(range(count)):
        object_count = np.random.randint(1, 100)

        bpy.context.scene.cycles.seed = blender_seed
        blender_seed += 1

        params = random_scene(object_count, 1.0)
        scene = DisplayBlenderScene(params)

        if not in_blender_mode:
            render_image(get_img_path(i, save_dir))

            scene.cleanup()

        scenes.append(params)

    if not in_blender_mode:
        with open(os.path.join(save_dir, pickle_name), 'wb') as f:
            pickle.dump(scenes, f)


if __name__ == "__main__":
    main()
