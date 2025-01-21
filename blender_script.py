"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

Example usage:
    blender -b -P blender_script.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 12 \
        --camera_dist 1.2

Here, input_model_paths.json is a json file containing a list of paths to .glb.

Khuyen:
Install blender and add it into path of your computer (Window)
Run file download-dataset to get organ_dataset (3D models listed in as-per-organ.csv)

# blender -b -P blender_script.py -- --object_path "E:/Khuyen/Y4 sem 1/CV/Team Project/organ_dataset/" \
    --output_dir "E:/Khuyen/Y4 sem 1/CV/Team Project/images_dataset" \
    --engine CYCLES --scale 0.8 --num_images 2 --camera_dist 1.2
"""

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
import uuid
from typing import Tuple
from mathutils import Vector, Matrix
import numpy as np
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Set, Tuple

import bpy
from mathutils import Vector

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default="~/.objaverse/hf-objaverse-v1/views_whole_sphere")
parser.add_argument(
    "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--scale", type=float, default=0.8)
parser.add_argument("--num_images", type=int, default=8)
parser.add_argument("--camera_dist", type=float, default=1.2)
    
argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

print('===================', args.engine, '===================')

context = bpy.context
scene = context.scene
render = scene.render

cam = scene.objects["Camera"]
cam.location = (0, 1.2, 0)
cam.data.lens = 40
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"

# setup lighting
bpy.ops.object.light_add(type="AREA")
light2 = bpy.data.lights["Area"]
light2.energy = 3000
bpy.data.objects["Area"].location[2] = 0.5
bpy.data.objects["Area"].scale[0] = 100
bpy.data.objects["Area"].scale[1] = 100
bpy.data.objects["Area"].scale[2] = 100

def _create_light(
    name: str,
    light_type: Literal["POINT", "SUN", "SPOT", "AREA"],
    location: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    energy: float,
    use_shadow: bool = False,
    specular_factor: float = 1.0,
):
    """Creates a light object.

    Args:
        name (str): Name of the light object.
        light_type (Literal["POINT", "SUN", "SPOT", "AREA"]): Type of the light.
        location (Tuple[float, float, float]): Location of the light.
        rotation (Tuple[float, float, float]): Rotation of the light.
        energy (float): Energy of the light.
        use_shadow (bool, optional): Whether to use shadows. Defaults to False.
        specular_factor (float, optional): Specular factor of the light. Defaults to 1.0.

    Returns:
        bpy.types.Object: The light object.
    """

    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_object = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location
    light_object.rotation_euler = rotation
    light_data.use_shadow = use_shadow
    light_data.specular_factor = specular_factor
    light_data.energy = energy
    return light_object

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 512*2
render.resolution_y = 512*2
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 128
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True

bpy.context.preferences.addons["cycles"].preferences.get_devices()
# Set the device_type
bpy.context.preferences.addons[
    "cycles"
].preferences.compute_device_type = "CUDA" # or "OPENCL"

def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )

def sample_spherical(radius=3.0, maxz=3.0, minz=0.):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        vec[2] = np.abs(vec[2])
        vec = vec / np.linalg.norm(vec, axis=0) * radius
        if maxz > vec[2] > minz:
            correct = True
    return vec

def sample_spherical(radius_min=1.5, radius_max=2.0, maxz=1.6, minz=-0.75):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
#         vec[2] = np.abs(vec[2])
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec

def randomize_camera():
    elevation = random.uniform(-90., 90.)
    azimuth = random.uniform(0., 360)
    distance = random.uniform(0.8, 1.6)
    return set_camera_location(elevation, azimuth, distance)

def set_camera_location(elevation, azimuth, distance):
    # from https://blender.stackexchange.com/questions/18530/
    x, y, z = sample_spherical(radius_min=1.5, radius_max=2.2, maxz=2.2, minz=-2.2)
    camera = bpy.data.objects["Camera"]
    camera.location = x, y, z

    direction = - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    return camera

# def randomize_lighting() -> None:
#     light2.energy = random.uniform(300, 600)
#     bpy.data.objects["Area"].location[0] = random.uniform(-1., 1.)
#     bpy.data.objects["Area"].location[1] = random.uniform(-1., 1.)
#     bpy.data.objects["Area"].location[2] = random.uniform(0.5, 1.5)

#     bpy.data.objects["Area"].scale[0] = 100
#     bpy.data.objects["Area"].scale[1] = 100
#     bpy.data.objects["Area"].scale[2] = 100

def randomize_lighting() -> Dict[str, bpy.types.Object]:
    """Randomizes the lighting in the scene.

    Returns:
        Dict[str, bpy.types.Object]: Dictionary of the lights in the scene. The keys are
            "key_light", "fill_light", "rim_light", and "bottom_light".
    """

    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    # Create key light
    key_light = _create_light(
        name="Key_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, -0.785398),
        energy=random.choice([3, 4, 5]),
    )

    # Create fill light
    fill_light = _create_light(
        name="Fill_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, 2.35619),
        energy=random.choice([2, 3, 4]),
    )

    # Create rim light
    rim_light = _create_light(
        name="Rim_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(-0.785398, 0, -3.92699),
        energy=random.choice([3, 4, 5]),
    )

    # Create bottom light
    bottom_light = _create_light(
        name="Bottom_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(3.14159, 0, 0),
        energy=random.choice([1, 2, 3]),
    )

    return dict(
        key_light=key_light,
        fill_light=fill_light,
        rim_light=rim_light,
        bottom_light=bottom_light,
    )


def reset_lighting() -> None:
    light2.energy = 1000
    bpy.data.objects["Area"].location[0] = 0
    bpy.data.objects["Area"].location[1] = 0
    bpy.data.objects["Area"].location[2] = 0.5

    bpy.data.objects["Area"].scale[0] = 100
    bpy.data.objects["Area"].scale[1] = 100
    bpy.data.objects["Area"].scale[2] = 100


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

# function from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
def get_3x4_RT_matrix_from_blender(cam):

    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    T_world2bcam = -1*R_world2bcam @ location

    # put into 3x4 matrix
    RT = Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
        ))
    return RT

def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

# def save_images(object_file: str) -> None:
#     """Saves rendered images of the object in the scene."""
#     try:
#         # Normalize and create the main output directory
#         output_dir = os.path.normpath(args.output_dir)
#         os.makedirs(output_dir, exist_ok=True)

#         reset_scene()

#         # Load the object
#         load_object(object_file)
#         object_uid = os.path.basename(object_file).split(".")[0]

#         # Create a subdirectory for the current object
#         object_dir = os.path.join(output_dir, object_uid).replace("\\", "/")
#         os.makedirs(object_dir, exist_ok=True)

#         normalize_scene()

#         # Create an empty object to track
#         empty = bpy.data.objects.new("Empty", None)
#         scene.collection.objects.link(empty)
#         cam_constraint.target = empty

#         randomize_lighting()

#         # Set render engine and format
#         bpy.context.scene.render.engine = args.engine
#         bpy.context.scene.render.image_settings.file_format = 'PNG'

#         for i in range(args.num_images):
#             try:
#                 # Set camera
#                 camera = randomize_camera()

#                 # Set render file path
#                 render_path = os.path.join(object_dir, f"{i:03d}.png").replace("\\", "/")
#                 scene.render.filepath = render_path

#                 # Perform rendering
#                 bpy.ops.render.render(write_still=True)

#                 # Save camera RT matrix
#                 RT = get_3x4_RT_matrix_from_blender(camera)
#                 RT_path = os.path.join(object_dir, f"{i:03d}.npy").replace("\\", "/")
#                 np.save(RT_path, RT)

#                 print(f"Saved image: {render_path}")
#                 print(f"Saved RT matrix: {RT_path}")
#             except Exception as e:
#                 print(f"Failed to save render or RT matrix for {object_file} (image {i}): {e}")

#     except Exception as main_exception:
#         print(f"Error during rendering for {object_file}: {main_exception}")
def save_images(object_file: str) -> None:
    """Saves rendered images of the object in the scene."""
    os.makedirs(args.output_dir, exist_ok=True)

    reset_scene()

    # load the object
    load_object(object_file)
    object_uid = os.path.basename(object_file).split(".")[0]
    normalize_scene()

    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    randomize_lighting()
    for i in range(args.num_images):
        print(light2.energy)
        # set camera
        camera = randomize_camera()

        # render the image
        render_path = os.path.join(args.output_dir, object_uid, f"{i:03d}.png")
        scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)

        # save camera RT matrix
        RT = get_3x4_RT_matrix_from_blender(camera)
        RT_path = os.path.join(args.output_dir, object_uid, f"{i:03d}.npy")
        np.save(RT_path, RT)

def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    # uid = uuid.uuid4()
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path


# if __name__ == "__main__":
#     try:
#         start_i = time.time()
#         if args.object_path.startswith("http"):
#             local_path = download_object(args.object_path)
#         else:
#             local_path = args.object_path
#         save_images(local_path)
#         end_i = time.time()
#         print("Finished", local_path, "in", end_i - start_i, "seconds")
#         # delete the object if it was downloaded
#         if args.object_path.startswith("http"):
#             os.remove(local_path)
#     except Exception as e:
#         print("Failed to render", args.object_path)
#         print(e)
import os
import time

import os
import time

def process_all_objects(object_path: str):
    """Processes all objects in the given path (including subdirectories)."""
    # Check if the path is a directory
    if os.path.isdir(object_path):
        for root, dirs, files in os.walk(object_path):
            # Filter out directories that end with "done"
            dirs[:] = [d for d in dirs if not d.endswith("done")]
            
            for file in files:
                # Process only .glb files
                if file.endswith(".glb"):
                    file_path = os.path.join(root, file)
                    print(f"Processing {file_path}...")
                    try:
                        start_i = time.time()
                        save_images(file_path)
                        end_i = time.time()
                        print(f"Finished {file_path} in {end_i - start_i} seconds")
                    except Exception as e:
                        print(f"Failed to render {file_path}")
                        print(e)
    else:
        print(f"{object_path} is not a valid directory.")


if __name__ == "__main__":
    try:
        start_i = time.time()

        # Get the path for the object directory (organ_dataset)
        local_path = args.object_path  # This is the directory that contains subfolders

        # Process all objects within the directory
        process_all_objects(local_path)

        end_i = time.time()
        print("Finished processing all objects in", end_i - start_i, "seconds")
    except Exception as e:
        print("An error occurred.")
        print(e)


# blender -b -P blender_script.py -- --object_path "E:/Khuyen/Y4 sem 1/CV/Team Project/organ_dataset/" --output_dir "E:/Khuyen/Y4 sem 1/CV/Team Project/views" --engine CYCLES --scale 0.8 --num_images 2 --camera_dist 1.2