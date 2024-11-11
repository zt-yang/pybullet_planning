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
    Point, PI, get_link_name, dump_joint, set_joint_position, ConfSaver, pairwise_link_collision
from pybullet_tools.bullet_utils import nice, colors, color_names, draw_fitted_box
from pybullet_tools.camera_utils import set_camera_target_body
from pybullet_tools.grasp_utils import get_grasp_db_file, get_hand_grasps
from pybullet_tools.pose_utils import change_pose_interactive
from pybullet_tools.stream_tests import visualize_grasps
from pybullet_tools.general_streams import get_grasp_list_gen, Position, \
    get_stable_list_gen, get_handle_grasp_gen, sample_joint_position_gen

from world_builder.world import State
from world_builder.loaders_nvidia_kitchen import load_kitchen_floor_plan, load_stove_knobs
from world_builder.world_utils import load_asset, get_instance_name, get_partnet_doors, draw_body_label

from tutorials.test_utils import get_test_world, get_instances, filter_instances, \
    load_model_instance, get_model_path, get_y_gap
from tutorials.config import ASSET_PATH


def test_grasp(problem, body, funk, test_attachment=False, test_offset=False, **kwargs):
    set_renderer(True)
    body_pose = get_pose(body)  ## multiply(get_pose(body), Pose(euler=Euler(math.pi/2, 0, -math.pi/2)))
    outputs = funk(body)
    if isinstance(outputs, list) and not test_offset:
        print(f'grasps on body {body} ({len(outputs)}):', outputs)

    visualize_grasps(problem, outputs, body_pose, retain_all=not test_attachment,
                     test_attachment=test_attachment, **kwargs)
    set_renderer(True)
    set_camera_target_body(body, dx=0.5, dy=0.5, dz=0.8)


def run_test_grasps(robot='feg', categories=[], given_instances=None, skip_grasps=False,
                    use_all_grasps=False, visualize=True, retain_all=True, verbose=False, test_attachment=False,
                    test_rotation_matrix=False, skip_rotation_index_until=None, rotation_matrices=None,
                    test_translation_matrix=False, skip_grasp_index_until=None, translation_matrices=None,
                    default_rotation=None, top_grasp_tolerance=None, side_grasp_tolerance=None, **kwargs):

    from pybullet_tools.grasp_utils import enumerate_rotational_matrices as emu, enumerate_translation_matrices

    world = get_test_world(robot, **kwargs)
    draw_pose(unit_pose(), length=10)
    robot = world.robot
    problem = State(world, grasp_types=robot.grasp_types)
    if rotation_matrices is None:
        rotation_matrices = emu(return_list=True) if test_rotation_matrix else [default_rotation]
    if translation_matrices is None:
        translation_matrices = enumerate_translation_matrices() if test_translation_matrix else [(0, 0, -0.13)]
    test_offset = test_rotation_matrix or test_translation_matrix
    color = RED

    for k1, r in enumerate(rotation_matrices):
        for k2, t in enumerate(translation_matrices):
            if test_offset:
                ## found it
                if r is not None:
                    if test_rotation_matrix and skip_rotation_index_until is not None and k1 < skip_rotation_index_until:
                        continue
                    problem.robot.change_tool_from_hand(Pose(point=t, euler=r))
                k = k1 * len(translation_matrices) + k2
                idx = k % len(colors)
                color = colors[idx]
                color_name = color_names[idx]

            for i, cat in enumerate(categories):
                tgt = top_grasp_tolerance
                sgt = side_grasp_tolerance
                if top_grasp_tolerance is None and cat in ['Knife']:
                    tgt = math.pi / 4  ## , 'EyeGlasses', 'Plate'
                funk = get_grasp_list_gen(problem, use_all_grasps=use_all_grasps, collisions=True,
                                          visualize=visualize, retain_all=retain_all,
                                          verbose=verbose, top_grasp_tolerance=tgt, side_grasp_tolerance=sgt,
                                          test_offset=test_offset, skip_grasp_index_until=skip_grasp_index_until)

                if cat == 'box':
                    body = create_box(0.05, 0.05, 0.05, mass=0.2, color=GREEN)
                    set_pose(body, ((1, 1, 0.9), unit_pose()[1]))
                    test_grasp(problem, body, funk, test_attachment)
                    continue

                instances = filter_instances(cat, given_instances)
                n = len(instances)
                locations = [(i, get_y_gap(cat) * n) for n in range(1, n+1)]
                j = -1
                for id, scale in instances.items():
                    j += 1
                    if isinstance(id, tuple):
                        cat, id = id
                    path, body, _ = load_model_instance(cat, id, scale=scale, location=locations[j])
                    instance_name = get_instance_name(abspath(path))
                    obj_name = f'{cat.lower()}#{id}'
                    world.add_body(body, obj_name, instance_name)
                    # set_camera_target_body(body)
                    text = id.replace('veggie', '').replace('meat', '')
                    draw_body_label(body, text, offset=(0, -0.2, 0.1))

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
                        if test_offset:
                            print(f'\n\n{k}/{len(rotation_matrices)*len(translation_matrices)}',
                                  f'\t{k1}/{len(rotation_matrices)} r = {nice(r)}',
                                  f'\t{k2}/{len(translation_matrices)} t = {nice(t)}\t', color_name, '\n\n')

                        test_grasp(problem, body, funk, test_attachment, test_offset, color=color)
                        # if test_rotation_matrix:
                        #     wait_unlocked()

                ## focus on objects
                if not test_offset:
                    if len(categories) > 1:
                        wait_if_gui(f'------------- Next object category? finished ({i+1}/{len(categories)})')

                    if cat == 'MiniFridge':
                        set_camera_pose((3, 7, 2), (0, 7, 1))
                    elif cat == 'Food':
                        set_camera_pose((3, 3, 2), (0, 3, 1))
                    elif cat == 'Stapler':
                        set_camera_pose((3, 1.5, 2), (0, 1.5, 1))

        if test_translation_matrix:
            wait_unlocked()

    remove_body(robot)

    # set_camera_target_body(body, dx=0.5, dy=0.5, dz=0.5)
    set_renderer(True)
    wait_if_gui('Finish?')
    disconnect()


def add_scale_to_grasp_file(robot, category):
    from pybullet_tools.logging_utils import dump_json
    from pprint import pprint

    ## find all instance, scale, and instance names
    instances = get_instances(category)
    instance_names = {}
    for id, scale in instances.items():
        path = get_model_path(category, id)
        name = get_instance_name(join(path, 'mobility.urdf'))
        instance_names[name] = (id, scale)

    ## find the grasp file of the robot and add scale
    db_file = get_grasp_db_file(robot)
    db = json.load(open(db_file, 'r'))
    updates = {}
    for key, data in db.items():
        for name in instance_names:
            if key.startswith(name):
                id, data['scale'] = instance_names[name]
                if data['name'] == 'None':
                    data['name'] = f"{category}/{id}"
                updates[key] = data
    pprint(updates)
    db.update(updates)
    if isfile(db_file): os.remove(db_file)
    dump_json(db, db_file, sort_dicts=False)


def add_time_to_grasp_file():
    from pybullet_tools.logging_utils import dump_json
    for robot in ['pr2', 'feg']:
        db_file = get_grasp_db_file(robot)
        db = json.load(open(db_file, 'r'))
        for key, data in db.items():
            data["datetime"] = "22" + data["datetime"]
            db[key] = data
        dump_json(db, db_file, sort_dicts=False)


## ----------------------------------------------------------------------


def run_test_handle_grasps(robot, category, skip_grasps=False):
    from pybullet_tools.pr2_streams import get_handle_pose

    world = get_test_world(robot, draw_base_limits=False)
    problem = State(world)
    funk = get_handle_grasp_gen(problem, visualize=False)
    funk2 = sample_joint_position_gen()

    ## load fridge
    instances = get_instances(category)
    n = len(instances)
    i = 0
    locations = [(0, 2*n) for n in range(1, n+1)]
    set_camera_pose((4, 3, 2), (0, 3, 0.5))

    def create_marker(xy):
        if category.lower() in ['cabinettop']:
            marker = create_box(0.05, 0.05, 0.07, color=RED)
            set_pose(marker, Pose(Point(x=xy[0], y=xy[1], z=1.2)))

    create_marker((0, 0))

    for id in instances:
        xy = locations[i]
        path, body, _ = load_model_instance(category, id, location=xy)
        i += 1
        instance_name = get_instance_name(path)
        world.add_body(body, f'{category.lower()}#{id}', instance_name)
        set_camera_target_body(body, dx=1, dy=1, dz=1)\

        if 'doorless' in category.lower():
            continue
        create_marker(xy)

        draw_body_label(body, id)

        ## color links corresponding to semantic labels
        body_joints = get_partnet_doors(path, body)
        world.add_joints(body, body_joints)

        for body_joint in body_joints:

            ## a few other tests here
            if skip_grasps:
                ## --- open door smartly when doing relaxed heuristic feasibility checking
                # open_joint(body, body_joint[1], hide_door=True)

                ## --- sample open positions during planning
                # pstn1 = Position(body_joint)
                # for (pstn, ) in funk2(body_joint, pstn1):
                #     pstn.assign()
                #     wait_if_gui('Next?')

                ## --- normalize joint positions for PIGINet
                b, j = body_joint
                # set_joint_position(b, j, 1.57)
                pstn1 = Position(body_joint)
                dump_joint(b, j)
                print('     positions', pstn1.value)

            else:
                outputs = funk(body_joint)
                body_pose = get_handle_pose(body_joint)

                set_renderer(True)
                set_camera_target_body(body, dx=2, dy=1, dz=1)
                visualize_grasps(problem, outputs, body_pose, retain_all=True)
                set_camera_target_body(body, dx=2, dy=1, dz=1)

        wait_if_gui('Next?')

    set_camera_pose((8, 8, 2), (0, 8, 1))
    wait_if_gui('Finish?')
    disconnect()


def run_test_handle_grasps_counter(robot='pr2', visualize=True, length_variants=True,
                                   skip_grasp_index_until=None,
                                   joint_types=('drawer', 'door', 'knob'), **kwargs):
    connect(use_gui=True, shadows=False, width=1980, height=1238)
    draw_pose(unit_pose(), length=2.)

    world = get_test_world(robot, semantic_world=True, draw_base_limits=True, **kwargs)
    robot = world.robot
    floor = load_kitchen_floor_plan(world, surfaces={}, plan_name='counter.svg')

    ## debug: inspect knob positions
    # set_camera_target_body(5, link=3, dx=0.3, dy=0.3, dz=0.4); set_renderer(True)

    if 'knob' in joint_types:
        load_stove_knobs(world)

    world.summarize_all_objects()

    joints = []
    for jp in joint_types:
        joints += world.cat_to_bodies(jp)
    # joints = [(4, 40)]

    for body_joint in joints:
        obj = world.BODY_TO_OBJECT[body_joint]
        link = obj.handle_link
        print(body_joint, get_link_name(obj.body, link))
        body, joint = body_joint
        set_camera_target_body(body, link=link, dx=0.5, dy=0.5, dz=0.5)
        draw_fitted_box(body, link=link, draw_centroid=True)
        get_hand_grasps(world, body, link=link, visualize=visualize, verbose=False,
                        retain_all=True, handle_filter=True, length_variants=length_variants,
                        skip_grasp_index_until=skip_grasp_index_until)
        set_camera_target_body(body, link=link, dx=0.5, dy=0.5, dz=0.5)

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
    robot = 'feg'  ## 'pr2'

    """ --- just load assets --- """
    # run_test_grasps(robot, ['BraiserLid'], skip_grasps=True)  ## 'Salter'

    """ --- grasps related --- """
    kwargs = dict(skip_grasps=False, test_attachment=False, side_grasp_tolerance=PI/4)
    run_test_grasps(robot, ['Bottle'], given_instances=['3616'], **kwargs)  ## 'Salter'
    # run_test_grasps(robot, ['VeggieCabbage'], **kwargs)

    # add_scale_to_grasp_file(robot, category='MiniFridge')
    # add_time_to_grasp_file()

    """ --- grasps for articulated storage units --- 
        IN: 'MiniFridge', 'MiniFridgeDoorless', 'CabinetTop'
    """
    # run_test_handle_grasps(robot, category='CabinetTop', skip_grasps=True)
    # run_test_handle_grasps_counter(robot)
