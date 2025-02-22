import os
from os.path import join, isdir, isfile
import random
import numpy as np
import torch
import trimesh
from tqdm import tqdm
from collections import defaultdict
import json
import sys

from trimesh.bounds import oriented_bounds
from trimesh.transformations import translation_from_matrix, euler_from_matrix, translation_matrix


dexgraspnet_path = "/home/zhutianyang/Documents/DexGraspNet/grasp_generation"
sys.path.append(dexgraspnet_path)

temp_lib_path = "/home/zhutianyang/miniconda3/envs/dexgraspnet/lib/python3.7/site-packages"
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
    mesh_path = join(dexgraspnet_path, "../data/meshdata")
    data_path = join(dexgraspnet_path, "../data/dataset")
else:
    mesh_path = join(dexgraspnet_path, "../data/downloaded/meshdata")
    data_path = join(dexgraspnet_path, "../data/downloaded/dexgraspnet")
filtered_grasp_path = data_path + '_filtered'
os.makedirs(filtered_grasp_path, exist_ok=True)


def get_grasp_pose(qpos):
    trans = [qpos[name] for name in translation_names]
    euler = [qpos[name] for name in rot_names]
    return trans, euler


def get_hand_pose(qpos, tensorize=False):
    trans, euler = get_grasp_pose(qpos)
    rot = np.array(transforms3d.euler.euler2mat(*euler))
    rot = rot[:, :2].T.ravel().tolist()
    conf = trans + rot + [qpos[name] for name in joint_names]
    if tensorize:
        return torch.tensor(conf, dtype=torch.float, device="cpu").unsqueeze(0)
    return conf


def load_sem_assets():
    """ load selected mesh data"""
    asset_summary = 'sem_assets.json'

    if isfile(asset_summary):
        grasp_object_dict = json.load(open(asset_summary, 'r'))

    else:
        grasp_object_dict = defaultdict(list)

        if not USE_TEST_MESHDATA:
            categories = ['Bottle', 'Bowl', 'Fruit', 'Pan', 'FoodItem', 'MilkCarton',
                          'PillBottle', 'SodaCan', 'SoapBottle', 'TissueBox', 'Vase']
            prefix = [f"sem-{name}" for name in categories]

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
    from utils.hand_model_lite import HandModelMJCFLite
    hand_pose = get_hand_pose(grasp_datapoint['qpos'])
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
    return np.load(join(root, grasp_object + ".npy"), allow_pickle=True)


def save_grasp_data(grasp_object, filtered_grasp_data):
    np.save(os.path.join(data_path + '_filtered', grasp_object + ".npy"), filtered_grasp_data, allow_pickle=True)


def random_sample_object(category='unknown', grasp_object_dict=None):
    ## randomly choose one object to grasp from
    if grasp_object_dict is None:
        grasp_object_dict = load_sem_assets()
    grasp_object = random.choice(grasp_object_dict[category])

    grasp_data = load_grasp_data(grasp_object, filtered=False)
    print(f'choosing object {grasp_object} with {len(grasp_data)} grasps')
    return grasp_data, grasp_object


def random_load_object(category='unknown', grasp_object_dict=None, visualize=False):
    grasp_data, grasp_object = random_sample_object(category, grasp_object_dict)

    object_mesh_path = os.path.join(mesh_path, grasp_object, "coacd/decomposed.obj")
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
    cm = trimesh.collision.CollisionManager()
    cm.add_object('plane', plane)

    filtered_grasp_data = []
    for grasp_datapoint in tqdm(grasp_data):
        hand_mesh = load_hand_model(hand_file, grasp_datapoint, color=[255, 0, 0, 255])
        if not grasp_collide_with_plane(cm, hand_mesh):
            filtered_grasp_data.append(grasp_datapoint)
            if len(filtered_grasp_data) == 1 and visualize:
                trimesh.Scene([object_mesh, plane, hand_mesh]).show()
    print(f"among {len(grasp_data)} grasps of {grasp_object}, {len(filtered_grasp_data)} are collision free from table plane")

    save_grasp_data(grasp_object, filtered_grasp_data)
