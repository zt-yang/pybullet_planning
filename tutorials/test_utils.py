from __future__ import print_function
import os
import csv
from collections import defaultdict
import argparse
from os.path import isfile

import os
import random
import json
import shutil
import time
import math

import numpy as np
from os.path import join, abspath, dirname, isdir, isfile
from os import listdir

from pybullet_tools.utils import connect, draw_pose, unit_pose, link_from_name, load_pybullet, load_model, \
    sample_aabb, AABB, set_pose, quat_from_euler, HideOutput, get_aabb_extent, unit_quat, remove_body, \
    set_camera_pose, wait_unlocked, disconnect, wait_if_gui, add_text, get_aabb
from pybullet_tools.utils import connect, draw_pose, unit_pose, set_caching
from pybullet_tools.bullet_utils import nice
from pybullet_tools.pr2_problems import create_floor

from world_builder.world_utils import get_instances as get_instances_helper
from world_builder.asset_constants import MODEL_HEIGHTS, MODEL_SCALES

from robot_builder.robot_builders import build_skill_domain_robot

from tutorials.config import modify_file_by_project, ASSET_PATH

from pddlstream.algorithms.meta import create_parser


def init_experiment(exp_dir):
    from pybullet_tools.logging_utils import TXT_FILE
    if isfile(TXT_FILE):
        os.remove(TXT_FILE)


def get_test_world(robot='feg', semantic_world=False, use_gui=True, draw_origin=False,
                   width=1980, height=1238, **kwargs):
    connect(use_gui=use_gui, shadows=False, width=width, height=height)  ##  , width=360, height=270
    if draw_origin:
        draw_pose(unit_pose(), length=.5)
        create_floor()
    set_caching(cache=False)
    if semantic_world:
        from world_builder.world import World
        world = World()
    else:
        from lisdf_tools.lisdf_loader import World
        world = World()

    if robot is not None:
        build_skill_domain_robot(world, robot, **kwargs)
    return world


def get_test_base_parser(task_name=None, parallel=False, use_viewer=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, default=task_name)
    parser.add_argument('-p', action='store_true', default=parallel)
    parser.add_argument('-v', '--viewer', action='store_true', default=use_viewer,
                        help='When enabled, enables the PyBullet viewer.')
    return parser


def get_parser(exp_name=None):
    parser = create_parser()
    parser.add_argument('-test', type=str, default=exp_name, help='Name of the test case')
    parser.add_argument('-cfree', action='store_true', help='Disables collisions during planning')
    parser.add_argument('-enable', action='store_true', help='Enables rendering during planning')
    parser.add_argument('-teleport', action='store_true', help='Teleports between configurations')
    parser.add_argument('-simulate', action='store_true', help='Simulates the system')
    return parser


def get_args(exp_name=None):
    parser = get_parser(exp_name=exp_name)
    args = parser.parse_args()
    print('Arguments:', args)
    return args


###########################################################################


def get_data(categories):
    """ to add partnet mobility assets into the asset/models folder
    1. make sure the whole partnet_mobility dataset is located parallel to the project dir
        i.e. PARTNET_PATH = abs_join(workspace_path, '..', 'dataset')
    2. add the category and instances you want into world_builder/asset_constants.py
    """
    from world_builder.paths import PARTNET_PATH

    for category in categories:
        models = get_instances_helper(category)

        target_model_path = join(ASSET_PATH, 'models', category)
        if not isdir(target_model_path):
            os.mkdir(target_model_path)

        if isdir(PARTNET_PATH):
            for idx in models:
                old_path = join(PARTNET_PATH, idx)
                new_path = join(target_model_path, idx)
                if isdir(old_path) and not isdir(new_path):
                    shutil.copytree(old_path, new_path)
                    print(f'copying {old_path} to {new_path}')
        else:
            print(f"PARTNET_PATH not found {PARTNET_PATH}")


def load_body(path, scale, pose_2d=(0, 0), random_yaw=False):
    file = join(path, 'mobility.urdf')
    # if 'MiniFridge' in file:
    #     file = file[file.index('../')+2:]
    #     file = '/home/yang/Documents/cognitive-architectures/bullet' + file
    print('loading', file)
    with HideOutput(True):
        body = load_model(file, scale=scale)
        if isinstance(body, tuple): body = body[0]
    pose = pose_from_2d(body, pose_2d, random_yaw=random_yaw)
    # pose = (pose[0], unit_quat())
    set_pose(body, pose)
    return body, file


def get_instances(category, **kwargs):
    instances = get_instances_helper(category, **kwargs)
    if len(instances) == 0:
        cat_dir = join(ASSET_PATH, 'models', category)
        if not isdir(cat_dir):
            os.mkdir(cat_dir)
            get_data(categories=[category])
        instances = get_instances_helper(category, **kwargs)
    if category.lower() == 'food' and 'VeggieSweetPotato' in instances:
        instances.pop('VeggieSweetPotato')
    return instances


def filter_instances(cat, given_instances):
    instances = get_instances(cat)
    if given_instances is not None:
        instances = {k: v for k, v in instances.items() if k in instances}
    print('instances', instances)
    return instances


def get_z_on_floor(body):
    return get_aabb_extent(get_aabb(body))[-1]/2


def get_floor_aabb(custom_limits):
    x_min, x_max = custom_limits[0]
    y_min, y_max = custom_limits[1]
    return AABB(lower=(x_min, y_min), upper=(x_max, y_max))


def sample_pose_on_floor(body, custom_limits):
    x, y = sample_aabb(get_floor_aabb(custom_limits))
    z = get_z_on_floor(body)
    return ((x, y, z), quat_from_euler((0, 0, math.pi)))


def pose_from_2d(body, xy, random_yaw=False):
    z = get_z_on_floor(body)
    yaw = math.pi ## facing +x axis
    if random_yaw:
        yaw = random.uniform(-math.pi, math.pi)
    return ((xy[0], xy[1], z), quat_from_euler((0, 0, yaw)))


def get_y_gap(category: str) -> float:
    """ gaps to lay assets in a line along y-axis"""
    gap = 2
    if category == 'KitchenCounter':
        gap = 3
    if category == 'MiniFridge':
        gap = 2
    if category in ['Food', 'Stapler', 'BraiserBody']:
        gap = 0.5
    return gap


def get_model_path(category, id):
    models_path = join(ASSET_PATH, 'models')
    category = [c for c in listdir(models_path) if c.lower() == category.lower()][0]
    if isinstance(id, str) and not id.isdigit():
        id = [i for i in listdir(join(models_path, category)) if i.lower() == id.lower()][0]
    path = join(models_path, category, id)
    return path


def load_model_instance(category, id, scale=1, location = (0, 0)):
    from world_builder.world_utils import get_model_scale

    path = get_model_path(category, id)
    if category in MODEL_HEIGHTS:
        height = MODEL_HEIGHTS[category]['height']
        scale = get_model_scale(path, h=height)
    elif category in MODEL_SCALES:
        scale = MODEL_SCALES[category][id]

    body, file = load_body(path, scale, location)
    return file, body, scale


###########################################################################


def save_csv(csv_file, data):
    csv_file = modify_file_by_project(csv_file)
    col_names = list(data.keys())
    col_data = list(data.values())

    file_exists = isfile(csv_file)
    with open(csv_file, 'a') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(col_names)
        for row in zip(*col_data):
            writer.writerow(row)


def read_csv(csv_file, summarize=True):
    from tabulate import tabulate
    csv_file = modify_file_by_project(csv_file)
    keys = None
    data = {}
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for j, row in enumerate(reader):
            if j == 0:
                keys = row
                data = defaultdict(list)
            else:
                for i, elem in enumerate(row):
                    data[keys[i]].append(elem if i == 0 else eval(elem))
                    if i == 1 and elem == 0:  ## failed run
                        break

    ## summarize the average, min, max, and count of each column
    if summarize:
        stats = [["name", "avg", "min", "max", "count"]]
        for key, value in data.items():
            if key in ["date"]: continue
            numbers = [sum(value) / len(value), min(value), max(value), len(value)]
            stats.append([key] + [nice(n) for n in numbers])
        print(tabulate(stats, headers="firstrow"))

    return data

