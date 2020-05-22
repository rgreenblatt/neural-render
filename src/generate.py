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


class RandomScene():
    def __init__(self):
        # for now, just a bunch of random spheres
        # num_objects = np.random.randint(0, 100)
        num_objects = np.random.randint(0, 100)

        self.object_params = []
        self.spheres = []
        self.materials = []

        for i in range(num_objects):
            self.object_params.append(self._generate_random_sphere(i))

        self.object_params = np.array(self.object_params)

    def _random_location(self):
        base_loc = np.array([0.0, 15.0, 0.0])
        translate = np.random.uniform(low=-12.0, high=10.0, size=[3])

        return base_loc + translate

    def _random_rotation(self):
        values = np.random.uniform(size=[3])
        u = values[0]
        v = values[1]
        w = values[2]

        return np.array([
            np.sqrt(1 - u) * np.sin(2 * math.pi * v),
            np.sqrt(1 - u) * np.cos(2 * math.pi * v),
            np.sqrt(u) * np.sin(2 * math.pi * w),
            np.sqrt(u) * np.cos(2 * math.pi * w),
        ])

    def _random_scale(self, location):
        base_scale = np.exp(np.random.normal(loc=-3.0)) * location[1]
        components_scale = np.random.uniform(low=0.5, high=1.5, size=[3])

        return base_scale * components_scale

    def _random_material(self, bsdf):
        # base color: 0 (0-1)
        # metallic: 4 (0-1)
        # specular: 5 (0-++ - mostly 0-1)
        # roughness: 7 (0-1 ? maybe higher but this should be decent default)
        # transmission 15 (0-1)
        # emmision 17 (3 values (rgb), cluster near zero)

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

        emmision = np.exp(np.random.normal(
            size=[3])) if is_emmisive else np.zeros(3)

        bsdf.inputs[0].default_value = (*color, 1)
        bsdf.inputs[4].default_value = metallic
        bsdf.inputs[5].default_value = specular
        bsdf.inputs[7].default_value = roughness
        bsdf.inputs[15].default_value = transmission
        bsdf.inputs[17].default_value = (*emmision, 1)

        return np.concatenate((color, [metallic], [specular], [roughness],
                               [transmission], emmision))

    def _generate_random_sphere(self, i):
        bpy.ops.object.select_all(action='DESELECT')

        bpy.ops.mesh.primitive_uv_sphere_add()

        # sphere is selected
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.faces_shade_smooth()
        bpy.ops.object.editmode_toggle()

        sphere = bpy.context.selected_objects[0]

        self.spheres.append(sphere)

        location = self._random_location()
        rotation = self._random_rotation()
        scale = self._random_scale(location)

        sphere.rotation_mode = 'QUATERNION'

        sphere.location = (*location, )
        sphere.rotation_quaternion = (*rotation, )
        sphere.scale = (*scale, )

        mat = bpy.data.materials.new(name="sphere_mat_{}".format(i))

        self.materials.append(mat)

        sphere.data.materials.append(mat)

        mat.use_nodes = True
        mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")

        mat_params = self._random_material(
            mat.node_tree.nodes["Principled BSDF"])

        return np.concatenate([location, rotation, scale, mat_params])

    def as_arr(self):
        return self.object_params

    def count(self):
        return self.object_params.shape[0]

    def values_per(self):
        return self.object_params.shape[1]

    def cleanup(self):
        bpy.ops.object.select_all(action='DESELECT')

        for sphere in self.spheres:
            sphere.select_set(True)

        remove_printing(lambda: bpy.ops.object.delete())

        for mat in self.materials:
            bpy.data.materials.remove(mat)


def main():
    in_blender_mode = False

    if not in_blender_mode:
        argv = sys.argv
        argv = argv[argv.index("--") + 1:]  # get all args after "--"

        parser = argparse.ArgumentParser()
        parser.add_argument('--resolution', type=int, default=128)
        parser.add_argument('--samples', type=int, default=128)
        parser.add_argument('--count', type=int, default=128)
        parser.add_argument('--seed', type=int, default=0)
        args = parser.parse_args(argv)

        bpy.context.scene.render.resolution_x = args.resolution
        bpy.context.scene.render.resolution_y = args.resolution

        bpy.context.scene.cycles.samples = args.samples

        count = args.count
        seed = args.seed
    else:
        count = 1
        seed = 0

    save_dir = "data/"

    mkdirs(save_dir)

    camera = bpy.data.objects["Camera"]
    camera.location = mathutils.Vector((0.0, 0.0, 0.0))
    camera.scale = mathutils.Vector((1.0, 1.0, 1.0))
    camera.rotation_euler[0] = math.pi / 2
    camera.rotation_euler[1] = 0.0
    camera.rotation_euler[2] = 0.0

    # delete default cube and light
    if not in_blender_mode:
        bpy.ops.object.select_all(action='DESELECT')

        bpy.data.objects['Cube'].select_set(True)
        bpy.data.objects['Light'].select_set(True)

        remove_printing(lambda: bpy.ops.object.delete())


    # make background dark
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[
        0].default_value = (0, 0, 0, 1)

    np.random.seed(seed)

    scenes = []

    # sometimes required to "refresh" devices (doesn't need to be printed)
    print("devices are:",
          bpy.context.preferences.addons['cycles'].preferences.get_devices())

    bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.preferences.addons[
        'cycles'].preferences.compute_device_type = "OPTIX"

    for i in tqdm(range(count)):
        scene = RandomScene()
        if not in_blender_mode:
            scenes.append(scene.as_arr())

            render_image("{}/imgs/img_{}.exr".format(save_dir, i))

            scene.cleanup()

    with open("{}/scenes.p".format(save_dir), 'wb') as f:
        pickle.dump(scenes, f)


if __name__ == "__main__":
    main()
