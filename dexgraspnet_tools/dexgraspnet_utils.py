import os
from os.path import join, isdir, isfile, expanduser, abspath, dirname
import random
import numpy as np
import trimesh
from tqdm import tqdm
from collections import defaultdict
import json
import sys

from trimesh.bounds import oriented_bounds
from trimesh.transformations import translation_from_matrix, euler_from_matrix, translation_matrix

HOME_DIR = expanduser("~")
absjoin = lambda *args, **kwargs: abspath(join(*args, **kwargs))

dexgraspnet_path = join(HOME_DIR, "Documents/DexGraspNet/grasp_generation")
sys.path.append(dexgraspnet_path)

temp_lib_path = join(HOME_DIR, "miniconda3/envs/dexgraspnet/lib/python3.7/site-packages")
sys.path.append(temp_lib_path)
import transforms3d

joint_names = [
    'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
    'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
    'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
    'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
    'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
]
translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
use_visual_mesh = False
hand_file = "mjcf/shadow_hand_vis.xml" if use_visual_mesh else "mjcf/shadow_hand_wrist_free.xml"

USE_TEST_MESHDATA = False
if USE_TEST_MESHDATA:
    mesh_path = absjoin(dexgraspnet_path, "../data/meshdata")
    data_path = absjoin(dexgraspnet_path, "../data/dataset")
else:
    mesh_path = absjoin(dexgraspnet_path, "../data/downloaded/meshdata")
    data_path = absjoin(dexgraspnet_path, "../data/downloaded/dexgraspnet")
filtered_grasp_path = data_path + '_filtered'
# os.makedirs(filtered_grasp_path, exist_ok=True)

rainbow_colors_rgb = [
    (255, 0, 0),        # Red
    (255, 127, 0),      # Orange
    (255, 255, 0),      # Yellow
    (0, 255, 0),        # Green
    (0, 0, 255),        # Blue
    (75, 0, 130),       # Indigo
    (148, 0, 211)       # Violet
]

DEXGRASPNET_CATEGORIES = ['Bottle', 'Bowl', 'Fruit', 'Pan', 'FoodItem', 'MilkCarton',
                          'PillBottle', 'SodaCan', 'SoapBottle', 'TissueBox', 'Vase']


def get_grasp_pose(qpos):
    trans = [qpos[name] for name in translation_names]
    euler = [qpos[name] for name in rot_names]
    return trans, euler


def get_hand_pose(qpos):
    trans, euler = get_grasp_pose(qpos)
    rot = np.array(transforms3d.euler.euler2mat(*euler))
    rot = rot[:, :2].T.ravel().tolist()
    conf = trans + rot + [qpos[name] for name in joint_names]
    return conf


def load_sem_assets():
    """ load selected mesh data"""
    asset_summary = 'sem_assets.json'

    if isfile(asset_summary):
        grasp_object_dict = json.load(open(asset_summary, 'r'))

    else:
        grasp_object_dict = defaultdict(list)

        if not USE_TEST_MESHDATA:
            prefix = [f"sem-{name}" for name in DEXGRASPNET_CATEGORIES]

            for code in os.listdir(data_path):
                for f in prefix:
                    if code.startswith(f):
                        grasp_object_dict[f[4:]].append(code[:-4])
        else:
            for code in os.listdir(data_path):
                grasp_object_dict['unknown'].append(code[:-4])

        with open(asset_summary, 'w') as f:
            json.dump(dict(grasp_object_dict), f, indent=3)

    total = sum([len(v) for v in grasp_object_dict.values()])
    print(f'found in total {len(grasp_object_dict)} categories and {total} grasps')
    return grasp_object_dict


def load_hand_model(hand_file, grasp_datapoint, color=[200, 200, 250, 255]):
    import torch
    from utils.hand_model_lite import HandModelMJCFLite
    hand_pose = get_hand_pose(grasp_datapoint['qpos'])
    hand_pose = torch.tensor(hand_pose, dtype=torch.float, device="cpu").unsqueeze(0)
    scale = 1 / grasp_datapoint["scale"]
    hand_model = HandModelMJCFLite(hand_file, "mjcf/meshes")
    hand_model.set_parameters(hand_pose)
    hand_mesh = hand_model.get_trimesh_data(0).apply_scale(scale)
    hand_mesh.visual.face_colors = color
    return hand_mesh


def load_plane(object_mesh, manual_rotation=True, color=[50, 50, 50, 255]):
    T, extend = oriented_bounds(object_mesh)
    center = translation_from_matrix(T)
    rotation = euler_from_matrix(T)

    if manual_rotation:
        plane_transform = -translation_matrix([0, 0, extend[2] / 2])
        plane = trimesh.primitives.Box(extents=[10, 10, 0], transform=plane_transform)
    else:
        plane_transform = -translation_matrix([0, extend[1] / 2, 0])
        plane = trimesh.primitives.Box(extents=[10, 0, 10], transform=plane_transform)

    plane.visual.face_colors = color
    return plane


def load_grasp_data(grasp_object, filtered=False):
    root = data_path + '_filtered' if filtered else data_path
    grasp_data_path = join(root, grasp_object + ".npy")
    return np.load(grasp_data_path, allow_pickle=True) if isfile(grasp_data_path) else None


def save_grasp_data(grasp_object, filtered_grasp_data):
    np.save(join(data_path + '_filtered', grasp_object + ".npy"), filtered_grasp_data, allow_pickle=True)


def load_object(category='unknown', grasp_object: str = None, grasp_object_dict=None, visualize=False):
    if grasp_object is None:
        ## randomly choose one object to grasp from
        if grasp_object_dict is None:
            grasp_object_dict = load_sem_assets()
        grasp_object = random.choice(grasp_object_dict[category])

    grasp_data = load_grasp_data(grasp_object, filtered=False)
    print(f'choosing object {grasp_object} with {len(grasp_data)} grasps')

    object_mesh_path = join(mesh_path, grasp_object, "coacd/decomposed.obj")
    print(object_mesh_path)
    object_mesh = trimesh.load(object_mesh_path)
    plane = load_plane(object_mesh)

    if visualize:
        trimesh.Scene([object_mesh, plane]).show()
    return object_mesh, plane, grasp_data, grasp_object


def grasp_collide_with_plane(cm, hand_mesh, debug=False):
    ## https://trimesh.org/trimesh.collision.html
    if debug:
        collisions = cm.in_collision_single(hand_mesh, return_names=True)
        print(f"collisions: {collisions}")
        return collisions
    return cm.in_collision_single(hand_mesh)


def filter_grasp(grasp_object, grasp_data, plane, object_mesh=None, visualize=False):
    """ among 240 grasps of sem-Bottle-a86d587f38569fdf394a7890920ef7fd, 90 are collision free from table plane """
    if load_grasp_data(grasp_object, filtered=True) is not None:
        return

    cm = trimesh.collision.CollisionManager()
    cm.add_object('plane', plane)

    filtered_grasp_data = []
    for grasp_datapoint in tqdm(grasp_data):
        hand_mesh = load_hand_model(hand_file, grasp_datapoint, color=[255, 0, 0, 255])
        if not grasp_collide_with_plane(cm, hand_mesh):
            filtered_grasp_data.append(grasp_datapoint)
            if len(filtered_grasp_data) == 1 and visualize:
                trimesh.Scene([object_mesh, plane, hand_mesh]).show()
    print(f"{len(filtered_grasp_data)} out of {len(grasp_data)} grasps of {grasp_object} are collision free from table plane")

    save_grasp_data(grasp_object, filtered_grasp_data)


def show_k_grasps(grasp_object, plane, object_mesh, k=7, visualize=False):
    filtered_grasp_data = load_grasp_data(grasp_object, filtered=True)
    filtered_grasp_data = filtered_grasp_data[:min(k, len(filtered_grasp_data))]
    meshes = [object_mesh, plane]
    for i, grasp_datapoint in enumerate(filtered_grasp_data):
        meshes.append(load_hand_model(hand_file, grasp_datapoint, color=rainbow_colors_rgb[i%7]))
    if visualize:
        trimesh.Scene(meshes).show()
    return meshes


## ------------------------------------------------------------------------------------------------


def fix_dexgraspnet_urdf(object_mesh_path):
    """ to be loaded into pybullet `<mass value="10.0"/>` must exist inside `<inertial>` tags. """
    new_file_name = object_mesh_path.replace('.urdf', '_fixed.urdf')
    if not isfile(new_file_name):
        lines = open(object_mesh_path, 'r').readlines()
        new_lines = []
        for line in lines:
            if '<inertia ' in line:
                new_lines.append(line[:line.index('<inertia ')]+'<mass value="10.0"/>\n')
            new_lines.append(line)

        with open(new_file_name, 'w') as f:
            f.write(''.join(new_lines))
    return new_file_name


def load_object_in_pybullet(grasp_object, scale, add_plane=True, verbose=True):
    from pybullet_tools.utils import HideOutput, load_model, get_aabb, create_box, TAN, Point, set_point, \
        draw_pose, draw_aabb, unit_pose, get_aabb_extent
    object_mesh_path = join(mesh_path, grasp_object, "coacd/coacd.urdf")
    new_file_name = fix_dexgraspnet_urdf(object_mesh_path)
    if verbose:
        print(f'loading (file exists {isfile(object_mesh_path)}) with scale={scale}\t{object_mesh_path}')
    with HideOutput(True):
        body = load_model(new_file_name, scale=scale)

    if add_plane:
        draw_pose(unit_pose())
        aabb = get_aabb(body)
        # draw_aabb(aabb)
        # print(aabb, get_aabb_extent(aabb))
        floor = create_box(4, 4, 0.001, color=TAN)
        set_point(floor, Point(z=-0.001 / 2. + aabb.lower[2]))
    return body


# ## ------------------------------------------------------------------------------------------------
#
#
# def add_asset_to_world_with_saved_grasps(world, grasp_object):
#
#
