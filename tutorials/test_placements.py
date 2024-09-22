from __future__ import print_function
import os
import random
import json

import numpy as np
from os.path import join, abspath, dirname, isdir, isfile
from os import listdir

from pybullet_tools.utils import connect, draw_pose, unit_pose, link_from_name, load_pybullet, load_model, \
    sample_aabb, AABB, set_pose, quat_from_euler, HideOutput, get_aabb_extent, unit_quat, remove_body, \
    set_camera_pose, wait_unlocked, disconnect, wait_if_gui, create_box, get_aabb, get_pose, draw_aabb, multiply, \
    Pose, get_link_pose, get_joint_limits, WHITE, RGBA, set_all_color, RED, GREEN, set_renderer, add_text, \
    Point, set_random_seed, set_numpy_seed, reset_simulation, joint_from_name, \
    get_joint_name, get_link_name, dump_joint, set_joint_position, ConfSaver, pairwise_link_collision
from pybullet_tools.bullet_utils import nice, set_camera_target_body, draw_fitted_box, \
    open_joint, take_selected_seg_images, dump_json
from pybullet_tools.grasp_utils import get_hand_grasps, get_grasp_db_file
from pybullet_tools.general_streams import get_grasp_list_gen, get_contain_list_gen, Position, \
    get_stable_list_gen, get_handle_grasp_gen, sample_joint_position_gen

from robot_builder.robot_builders import build_skill_domain_robot

from world_builder.world import State
from world_builder.world_utils import load_asset, get_instance_name, get_partnet_doors, get_partnet_spaces, \
    draw_body_label

from tutorials.test_utils import get_test_world, load_body, get_instances, \
    load_model_instance, get_model_path, get_y_gap


def add_heights_to_pose_database(movable, surface, zs):
    from world_builder.world_utils import Z_CORRECTION_FILE as file
    from scipy.stats import norm
    if len(zs) == 0:
        print('         no samples for', movable, surface)
        return
    mina, maxa = min(zs), max(zs)
    while maxa - mina > 0.02:
        zs.remove(mina)
        zs.remove(maxa)
        print('       throwing away outliners', nice(mina), nice(maxa))
        mina, maxa = min(zs), max(zs)
    print('         length', len(zs))
    collection = json.load(open(file, 'r')) if isfile(file) else {}
    if movable not in collection:
        collection[movable] = {}
    mu, std = norm.fit(zs)
    collection[movable][surface] = [round(i, 6) for i in [mina, maxa, mu, std]]
    dump_json(collection, file, width=150)


def test_placement_in(robot, category, movable_category='Food', movable_instance=None,
                      seg_links=False, gen_z=False, **kwargs):

    world = get_test_world(robot)
    problem = State(world)
    funk = get_contain_list_gen(problem, collisions=True, verbose=False, **kwargs)

    ## load fridge
    if movable_instance is None:
        movable_instances = get_instances(movable_category)
        movable_instance = random.choice(list(movable_instances.keys()))

    instances = get_instances(category)
    n = len(instances)
    i = 0
    locations = [(0, get_y_gap(category) * n) for n in range(1, n + 1)]
    set_camera_pose((4, 3, 2), (0, 3, 0.5))
    for id in instances:
        (x, y) = locations[i]
        path, body, scale = load_model_instance(category, id, location=(x, y))
        # new_urdf_path, body = reload_after_vhacd(path, body, scale, id=id)

        name = f'{category.lower()}#{id}'
        if category in ['MiniFridge', 'Fridge', 'Cabinet', 'Microwave']:
            name += '_storage'
        world.add_body(body, name, path=path)
        set_camera_target_body(body, dx=1, dy=0, dz=1)

        ## color links corresponding to semantic labels
        spaces = get_partnet_spaces(path, body)
        world.add_spaces(body, spaces, path=path)

        doors = get_partnet_doors(path, body)
        for door in doors:
            open_joint(door[0], door[1])
        set_renderer(True)

        """ test taking images of link segments """
        if seg_links:
            name = category.lower()
            img_dir = join(dirname(__file__), 'pb_images', name)
            os.makedirs(img_dir, exist_ok=True)

            indices = {body: name}
            indices.update({(b, None, l): f"{name}::{get_link_name(b, l)}" for b, _, l in spaces})
            indices.update({(b, j): f"{name}::{get_joint_name(b, j)}" for b, j in doors})
            take_selected_seg_images(world, img_dir, body, indices, dx=1.5, dy=0, dz=1)

        else:

            for body_link in spaces:
                x += 1
                # space = clone_body(body, links=body_link[-1:], visual=True, collision=True)
                # world.add_body(space, f'{category.lower()}#{id}-{body_link}')

                cabbage, path = load_asset(movable_category, x=x, y=y, z=0, yaw=0,
                                           random_instance=movable_instance)[:2]
                cabbage_name = f'cabbage#{i}-{body_link}'
                world.add_body(cabbage, cabbage_name, path=path)

                outputs = funk(cabbage, body_link)
                if gen_z:
                    container_z = get_pose(body)[0][2]
                    zs = [output[0].value[0][2] for output in outputs if output is not None]
                    zs = [z - container_z for z in zs]
                    mov_mobility = f"{movable_category}/{movable_instance}"
                    sur_mobility = f"{category}/{id}"
                    add_heights_to_pose_database(mov_mobility, sur_mobility, zs)
                    continue

                set_pose(cabbage, outputs[0][0].value)
                markers = []
                for j in range(1, len(outputs)):
                    marker = load_asset(movable_category, x=x, y=y, z=0, yaw=0,
                                        random_instance=movable_instance)[0]
                    markers.append(marker)
                    set_pose(marker, outputs[j][0].value)

                set_renderer(True)
                set_camera_target_body(cabbage, dx=1, dy=0, dz=1)
                wait_if_gui('Next space?')
                for m in markers:
                    remove_body(m)
                reset_simulation()
                set_renderer(True)
        i += 1
        reset_simulation()

    # set_camera_pose((4, 3, 2), (0, 3, 0.5))
    if not gen_z:
        wait_if_gui('Finish?')
    disconnect()


def test_placement_on(robot, category, surface_name=None, seg_links=False, gen_z=False,
                      movable_category='Food', movable_instance=None, **kwargs):

    world = get_test_world(robot)
    problem = State(world)
    funk = get_stable_list_gen(problem, collisions=True, verbose=False, **kwargs)

    if movable_instance is None:
        movable_instances = get_instances(movable_category)
        movable_instance = random.choice(list(movable_instances.keys()))

    ###########################################################

    if category == 'box' and gen_z:
        cabbage, path = load_asset(movable_category, x=0, y=0, z=0, yaw=0,
                                   random_instance=movable_instance)[:2]
        z = get_pose(cabbage)[0][2] - get_aabb(cabbage).lower[2]
        zs = [z] * 20

        mov_mobility = f"{movable_category}/{movable_instance}"
        sur_mobility = f"{category}"
        add_heights_to_pose_database(mov_mobility, sur_mobility, zs)
        return

    surface_links = {
        'BraiserBody': 'braiser_bottom',
        'Sink': 'sink_bottom',
    }
    if category in surface_links and surface_name is None:
        surface_name = surface_links[category]

    ## load fridges
    instances = get_instances(category)
    n = len(instances)
    i = 0
    locations = [(0, get_y_gap(category) * n) for n in range(2, n + 2)]
    set_camera_pose((4, 3, 2), (0, 3, 0.5))
    for id in instances:
        (x, y) = locations[i]
        path, body, scale = load_model_instance(category, id, location=(x, y))
        # new_urdf_path, body = reload_after_vhacd(path, body, scale, id=id)

        ## ---- add surface
        name = f'{category.lower()}#{id}'
        world.add_body(body, name, path=path)
        set_camera_target_body(body, dx=1, dy=0, dz=1)
        nn = category.lower()
        indices = {body: nn}
        if surface_name is not None:
            link = link_from_name(body, surface_name)
            print('radius', round( get_aabb_extent(get_aabb(body, link=link))[0]/ scale / 2, 3))
            print('height', round( get_aabb_extent(get_aabb(body))[2]/ scale / 2, 3))
            body = (body, None, link)
            world.add_body(body, name, path=path)
            indices[body] = f"{nn}::{surface_name}"

        """ test taking images of link segments """
        if seg_links:
            name = category.lower()
            img_dir = join(dirname(__file__), 'pb_images', name)
            os.makedirs(img_dir, exist_ok=True)
            take_selected_seg_images(world, img_dir, body[0], indices, dx=0.2, dy=0, dz=1)
            # wait_unlocked()

        else:
            ## ---- add many cabbages
            # space = clone_body(body, links=body_link[-1:], visual=True, collision=True)
            # world.add_body(space, f'{category.lower()}#{id}-{body_link}')
            x += 1
            cabbage, path = load_asset(movable_category, x=x, y=y, z=0, yaw=0,
                                       random_instance=movable_instance)[:2]
            cabbage_name = f'{movable_category}#{i}-{body}'
            world.add_body(cabbage, cabbage_name, path=path)

            outputs = funk(cabbage, body)
            if gen_z:
                if isinstance(body, tuple):
                    container_z = get_pose(body[0])[0][2]
                else:
                    container_z = get_pose(body)[0][2]
                zs = [output[0].value[0][2] for output in outputs if output is not None]
                zs = [z - container_z for z in zs]
                mov_mobility = f"{movable_category}/{movable_instance}"
                sur_mobility = f"{category}/{id}"
                add_heights_to_pose_database(mov_mobility, sur_mobility, zs)
                continue

            set_pose(cabbage, outputs[0][0].value)
            markers = []
            for j in range(1, len(outputs)):
                marker = load_asset(movable_category, x=x, y=y, z=0, yaw=0,
                                    random_instance=movable_instance)[0]
                markers.append(marker)
                set_pose(marker, outputs[j][0].value)

            set_renderer(True)
            set_camera_target_body(cabbage, dx=1, dy=0, dz=1)
            wait_if_gui('Next surface?')
            for m in markers:
                remove_body(m)
            reset_simulation()
        i += 1

    # set_camera_pose((4, 3, 2), (0, 3, 0.5))
    if not gen_z:
        wait_if_gui('Finish?')
    disconnect()


############################################################################


def get_placement_z(robot='pr2'):
    from world_builder.world_utils import Z_CORRECTION_FILE as file
    kwargs = dict(num_samples=50, gen_z=True, learned_sampling=False)
    surfaces = ['box', 'Sink', 'Microwave', "OvenCounter"]
    storage = ['CabinetTop', 'MiniFridge']
    movables = {
        'Bottle': surfaces + storage,
        'Food': surfaces + storage + ['BraiserBody'],
        'Medicine': surfaces + storage + ['BraiserBody'],
        'BraiserLid': ['box'],
    }
    dic = json.load(open(file, 'r')) if isfile(file) else {}
    for mov, surfaces in movables.items():
        for sur in surfaces:
            num_sur_instances = len(get_instances(sur)) if sur != 'box' else 1
            for ins in get_instances(mov):
                mov_mibility = f"{mov}/{ins}"
                if mov_mibility in dic:
                    if sur == 'box' and sur in dic[mov_mibility]:
                        continue
                    elif len([i for i in dic[mov_mibility] if sur in i]) == num_sur_instances:
                        continue
                if sur in ['MiniFridge', 'CabinetTop']:
                    test_placement_in(robot, category=sur, movable_category=mov,
                                      movable_instance=ins, **kwargs)
                else:
                    test_placement_on(robot, category=sur, movable_category=mov,
                                      movable_instance=ins, **kwargs)


############################################################################


def test_placement_counter():
    from world_builder.loaders import load_floor_plan
    from world_builder.world import World

    connect(use_gui=True, shadows=False, width=1980, height=1238)
    draw_pose(unit_pose(), length=2.)

    surfaces = {
        'counter': {
            'back_left_stove': [],
            'back_right_stove': [],
            'front_left_stove': [],
            'front_right_stove': [],
            'hitman_countertop': [],
            'indigo_tmp': [],
        }
    }
    spaces = {
        'counter': {
            'sektion': [],
            'dagger': [],
            'hitman_drawer_top': [],
            # 'hitman_drawer_bottom': [],
            'indigo_drawer_top': [],
            # 'indigo_drawer_bottom': [],
            'indigo_tmp': []
        },
    }

    world = World()
    floor = load_floor_plan(world, plan_name='counter.svg', surfaces=surfaces, spaces=spaces)
    robot = build_skill_domain_robot(world, 'feg')

    world.open_all_doors_drawers()
    world.summarize_all_objects()
    state = State(world, grasp_types=robot.grasp_types)
    funk = get_grasp_list_gen(state, collisions=True, num_samples=3,
                              visualize=False, retain_all=False)

    surfaces = world.cat_to_bodies('surface')
    spaces = world.cat_to_bodies('space')
    regions = surfaces
    opened_poses = {}

    for rg in regions:
        r = world.BODY_TO_OBJECT[rg]
        draw_aabb(get_aabb(r.body, link=r.link))
        opened_poses[rg] = get_link_pose(r.body, r.link)

        if rg in surfaces:
            body = r.place_new_obj('OilBottle').body
            draw_fitted_box(body, draw_centroid=False)
            set_camera_target_body(body, dx=0.5, dy=0, dz=0)
            set_camera_target_body(body, dx=0.5, dy=0, dz=0)
            grasps = get_hand_grasps(world, body, visualize=True, retain_all=True)

        # if rg in spaces:
        #     body = r.place_new_obj('MeatTurkeyLeg').body
        #     draw_fitted_box(body, draw_centroid=False)
        #     set_camera_target_body(body, dx=0.1, dy=0, dz=0.5)
        #     set_camera_target_body(body, dx=0.1, dy=0, dz=0.5)
        #     # grasps = get_hand_grasps(world, body, visualize=False, retain_all=False)
        #
        #     outputs = funk(body)
        #     visualize_grasps(state, outputs, get_pose(body), retain_all=True, collisions=True)

        set_renderer(True)
        print(f'test_placement_counter | placed {body} on {r}')
    wait_if_gui('Finish?')
    disconnect()


def test_pick_place_counter(robot):
    from world_builder.loaders_partnet_kitchen import load_random_mini_kitchen_counter
    world = get_test_world(robot, semantic_world=True)
    load_random_mini_kitchen_counter(world)


if __name__ == '__main__':
    robot = 'rummy'

    """ --- placement for articulated storage units --- 
        IN: 'MiniFridge', 'MiniFridgeDoorless', 'CabinetTop'
    """
    # test_placement_in(robot, category='MiniFridge', seg_links=False,
    #                   movable_category='BraiserLid', learned_sampling=True)

    """ --- placement related for supporting surfaces --- 
        ON: 'KitchenCounter', 
            'Tray' (surface_name='tray_bottom'),
            'Sink' (surface_name='sink_bottom'),
            'BraiserBody' (surface_name='braiser_bottom'),
    """
    # test_placement_on(robot, category='BraiserBody', surface_name='braiser_bottom', seg_links=True)
    # test_placement_on(robot, category='Sink', surface_name='sink_bottom', seg_links=True)

    # get_placement_z()

    """ --- specific kitchen counter --- """
    test_placement_counter()  ## initial placement
    # test_pick_place_counter(robot)