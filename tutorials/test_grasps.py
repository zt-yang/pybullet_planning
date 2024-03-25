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
    set_camera_pose, wait_unlocked, disconnect, wait_if_gui, create_box, get_aabb, get_pose, draw_aabb, multiply, \
    Pose, get_link_pose, get_joint_limits, WHITE, RGBA, set_all_color, RED, GREEN, set_renderer, add_text, \
    Point, set_random_seed, set_numpy_seed, reset_simulation, joint_from_name, \
    get_joint_name, get_link_name, dump_joint, set_joint_position, ConfSaver, pairwise_link_collision
from pybullet_tools.bullet_utils import nice, set_camera_target_body
from pybullet_tools.pr2_tests import visualize_grasps
from pybullet_tools.general_streams import get_grasp_list_gen

from world_builder.world import State
from world_builder.world_utils import load_asset, get_instance_name, get_partnet_doors, get_partnet_spaces
from world_builder.world_utils import get_instances as get_instances_helper
from world_builder.asset_constants import MODEL_HEIGHTS, MODEL_SCALES

from robot_builder.robot_builders import build_skill_domain_robot

from tutorials.test_utils import get_test_world
from tutorials.config import ASSET_PATH


def test_grasps(robot='feg', categories=[], skip_grasps=False,
                test_attachment=False, test_rotation_offset=False, **kwargs):
    from pybullet_tools.bullet_utils import enumerate_rotational_matrices as emu
    world = get_test_world(robot, **kwargs)
    draw_pose(unit_pose(), length=10)
    robot = world.robot
    problem = State(world, grasp_types=robot.grasp_types)  ## , 'side' , 'top'
    rotation_matrices = [None] if not test_rotation_offset else emu(return_list=True)

    i = -1
    for cat in categories:

        tpt = math.pi / 4 if cat in ['Knife'] else None ## , 'EyeGlasses', 'Plate'
        funk = get_grasp_list_gen(problem, collisions=True, visualize=True,
                                  retain_all=True, top_grasp_tolerance=tpt, verbose=True)

        def test_grasp(body):
            set_renderer(True)
            body_pose = get_pose(body)  ## multiply(get_pose(body), Pose(euler=Euler(math.pi/2, 0, -math.pi/2)))
            outputs = funk(body)
            if isinstance(outputs, list):
                print(f'grasps on body {body}:', outputs)
            visualize_grasps(problem, outputs, body_pose, retain_all=not test_attachment or True,
                             test_attachment=test_attachment)
            set_renderer(True)
            set_camera_target_body(body, dx=0.5, dy=0.5, dz=0.8)

        if cat == 'box':
            body = create_box(0.05, 0.05, 0.05, mass=0.2, color=GREEN)
            set_pose(body, ((1, 1, 0.9), unit_pose()[1]))
            test_grasp(body)
            continue

        i += 1
        instances = get_instances(cat)
        print('instances', instances)
        n = len(instances)
        locations = [(i, get_gap(cat) * n) for n in range(1, n+1)]
        j = -1
        for id, scale in instances.items():
            j += 1
            if isinstance(id, tuple):
                cat, id = id
            path, body, _ = load_model_instance(cat, id, scale=scale, location=locations[j])
            instance_name = get_instance_name(abspath(path))
            obj_name = f'{cat.lower()}#{id}'
            world.add_body(body, obj_name, instance_name)
            set_camera_target_body(body)
            text = id.replace('veggie', '').replace('meat', '')
            draw_text_label(body, text, offset=(0, -0.2, 0.1))

            if cat == 'BraiserBody':
                print('get_aabb_extent', nice(get_aabb_extent(get_aabb(body))))
                set_camera_target_body(body, dx=0.05, dy=0, dz=0.5)
                # draw_points(body, size=0.05)
                # set_camera_target_body(body, dx=0.5, dy=0.5, dz=0.5)
                # pose = get_pose(body)
                # _, body, _ = load_model_instance('BraiserLid', id, scale=scale, location=locations[j])
                # set_pose(body, pose)

            # draw_aabb(get_aabb(body))

            """ --- fixing texture issues ---"""
            # world.add_joints_by_keyword(obj_name)
            # world.open_all_doors()

            """ test others """
            # test_robot_rotation(body, world.robot)
            # test_spatial_algebra(body, world.robot)
            # draw_fitted_box(body, draw_centroid=True)
            # grasps = get_hand_grasps(world, body)

            """ test grasps """
            if skip_grasps:
                print('length', round(get_aabb_extent(get_aabb(body))[1], 3))
                print('height', round(get_aabb_extent(get_aabb(body))[2], 3))
                print('point', round(get_pose(body)[0][2], 3))
                wait_if_gui()
            else:
                test_grasp(body)
                wait_unlocked()

        if len(categories) > 1:
            wait_if_gui(f'------------- Next object category? finished ({i+1}/{len(categories)})')

        if cat == 'MiniFridge':
            set_camera_pose((3, 7, 2), (0, 7, 1))
        elif cat == 'Food':
            set_camera_pose((3, 3, 2), (0, 3, 1))
        elif cat == 'Stapler':
            set_camera_pose((3, 1.5, 2), (0, 1.5, 1))

    remove_body(robot)

    # set_camera_target_body(body, dx=0.5, dy=0.5, dz=0.5)
    set_renderer(True)
    wait_if_gui('Finish?')
    disconnect()


if __name__ == '__main__':
    """ ------------------------ object categories -------------------------
        Kitchen Movable: 'Bottle', 'Food', 'BraiserLid', 'Sink', 'SinkBase', 'Faucet',
        Kitchen Furniture: 'MiniFridge', 'KitchenCounter', 'MiniFridgeBase',
                            'OvenCounter', 'OvenTop', 'MicrowaveHanging', 'MiniFridgeBase',
                            'CabinetLower', 'CabinetTall', 'CabinetUpper', 'DishwasherBox'
        Kitchen Cooking: 'KitchenFork', 
        Packing:    'Stapler', 'Camera', 'EyeGlasses', 'Knife', 'Tray',
    ------------------------------------------------------------------------ """
    robot = 'rummy'

    """ --- grasps related --- """
    # test_grasps(robot, ['Salter'], skip_grasps=False, test_attachment=False)  ## 'Salter'
    test_grasps(robot, ['VeggieCabbage'], skip_grasps=False, test_attachment=False)
    # add_scale_to_grasp_file(robot, category='MiniFridge')
    # add_time_to_grasp_file()