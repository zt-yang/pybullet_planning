from __future__ import print_function
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
    set_camera_pose, wait_unlocked, disconnect, wait_if_gui, create_box, get_aabb
from pybullet_tools.camera_utils import set_camera_target_body
from world_builder.asset_constants import MODEL_HEIGHTS, MODEL_SCALES
from world_builder.paths import DATABASES_PATH

from tutorials.test_utils import get_test_world, load_body, get_instances, \
    load_model_instance, get_model_path, get_y_gap, get_data
from tutorials.config import ASSET_PATH


# ####################################


def test_texture(category, id):
    connect(use_gui=True, shadows=False, width=1980, height=1238)
    path = join(ASSET_PATH, 'models', category, id) ## , 'mobility.urdf'

    body = load_body(path, 0.2)
    set_camera_target_body(body, dx=0.5, dy=0.5, dz=0.5)
    set_camera_target_body(body, dx=0.5, dy=0.5, dz=0.5)

    # import untangle
    # content = untangle.parse(path).robot
    #
    # import xml.etree.ElementTree as gfg
    # root = gfg.Element("robot")
    # tree = gfg.ElementTree(content)
    # with open(path.replace('mobility', 'mobility_2'), "wb") as files:
    #     tree.write(files)


def test_vhacd(category, visualize=False):
    from pybullet_tools.utils import process_urdf
    TEMP_OBJ_DIR = 'vhacd'
    instances = get_instances(category)
    if visualize:
        world = get_test_world()
        x, y = 0, 0

    instances = ['100021']

    for idx in instances:
        urdf_path = f'../../assets/models/{category}/{idx}/mobility.urdf'

        if visualize:
            body = load_pybullet(urdf_path)
            set_pose(body, ((x, y, 0), unit_quat()))

        new_urdf_path = process_urdf(urdf_path)

        if visualize:
            body = load_pybullet(new_urdf_path)
            set_pose(body, ((x+2, y, 0), unit_quat()))

            set_camera_target_body(body, dx=1, dy=1, dz=1)
            wait_unlocked()
            y += 2
        else:
            shutil.move(new_urdf_path, new_urdf_path.replace('mobility.urdf', f'{idx}.urdf'))
            shutil.move(TEMP_OBJ_DIR, urdf_path.replace('mobility.urdf', TEMP_OBJ_DIR))


## ----------------------------------------------------------------------------------------


def save_partnet_aabbs():
    world = get_test_world()
    draw_pose(unit_pose(), length=1.5)

    skips = ['Kettle', 'Toaster', 'TrashCan', 'CabinetAboveOven', 'DeskStorage']
    categories = list(MODEL_SCALES.keys()) + list(MODEL_HEIGHTS.keys())
    skip_till = None
    if skip_till is not None:
        categories = categories[categories.index(skip_till)+1:]
    categories = [c for c in categories if c not in skips and c != c.lower()]
    shapes = {}  ## category: {id: (dlower, dupper)}
    # if isfile(shape_file):
    #     shapes = json.load(open(shape_file, 'r'))
    for category in categories:
        instances = get_instances(category)
        if category not in shapes:
            shapes[category] = {}
        bodies = []
        for idx in instances:
            path = join(ASSET_PATH, 'models', category, idx, 'mobility.urdf')
            if not isfile(path) or idx in shapes[category]:
                print('skipping', path)
                continue
            path, body, scale = load_model_instance(category, idx)
            set_pose(body, ([0, 0, 0], quat_from_euler([0, 0, math.pi])))

            aabb = get_aabb(body)
            shapes[category][idx] = (aabb.lower, aabb.upper)
            bodies.append(body)
        #     wait_for_duration(0.25)
        # wait_unlocked()
        for body in bodies:
            remove_body(body)

    with open(join(DATABASES_PATH, 'partnet_shapes.json'), 'w') as f:
        json.dump(shapes, f, indent=2, sort_keys=False)


## ----------------------------------------------------------------------------------------


if __name__ == '__main__':

    """ ------------------------ object categories -------------------------
        Kitchen Movable: 'Bottle', 'Food', 'BraiserLid', 'Sink', 'SinkBase', 'Faucet',
        Kitchen Furniture: 'MiniFridge', 'KitchenCounter', 'MiniFridgeBase',
                            'OvenCounter', 'OvenTop', 'MicrowaveHanging', 'MiniFridgeBase',
                            'CabinetLower', 'CabinetTall', 'CabinetUpper', 'DishwasherBox'
        Kitchen Cooking: 'KitchenFork', 
        Packing:    'Stapler', 'Camera', 'EyeGlasses', 'Knife', 'Tray',
    ------------------------------------------------------------------------ """

    """ --- models related --- """
    get_data(categories=['DinerTable', 'DinerChair'])
    # test_texture(category='CoffeeMachine', id='103127')
    # test_vhacd(category='BraiserBody')
    # save_partnet_aabbs()
