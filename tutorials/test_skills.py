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
from pybullet_tools.bullet_utils import nice, \
    draw_fitted_box, open_joint, dump_json
from pybullet_tools.camera_utils import set_camera_target_body, take_selected_seg_images
from pybullet_tools.pose_utils import sample_random_pose
from pybullet_tools.grasp_utils import get_hand_grasps, get_grasp_db_file
from pybullet_tools.flying_gripper_utils import se3_ik, create_fe_gripper, set_se3_conf
from pybullet_tools.stream_tests import visualize_grasps
from pybullet_tools.general_streams import get_grasp_list_gen, get_contain_list_gen, Position, \
    get_stable_list_gen, get_handle_grasp_gen, sample_joint_position_gen

from world_builder.world import State
from world_builder.world_utils import draw_body_label
from world_builder.loaders import create_house_floor, create_table, create_movable
from world_builder.asset_constants import MODEL_HEIGHTS, MODEL_SCALES

from tutorials.test_utils import get_test_world, load_body, get_instances, \
    load_model_instance, get_model_path, get_y_gap
from tutorials.config import ASSET_PATH


DEFAULT_TEST = 'kitchen' ## 'blocks_pick'

seed = None
if seed is None:
    seed = random.randint(0, 10 ** 6 - 1)
set_random_seed(seed)
set_numpy_seed(seed)
print('Seed:', seed)

# ####################################


def test_robot_rotation(body, robot):
    pose = ((0.2,0.3,0), quat_from_euler((math.pi/4, math.pi/2, 1.2)))
    set_pose(body, pose)
    conf = se3_ik(robot, pose)
    set_se3_conf(robot, conf)
    set_camera_target_body(body, dx=0.5, dy=0.5, dz=0.5)


def test_spatial_algebra(body, robot):

    ## transformations
    O_T_G = ((0.5, 0, 0), unit_quat())
    O_R_G = ((0, 0, 0), quat_from_euler((0, -math.pi / 2, 0)))
    G = multiply(O_T_G, O_R_G)
    gripper = robot.create_gripper(color=RED)

    ## original object pose
    set_pose(body, unit_pose())
    set_camera_target_body(body, dx=0.5, dy=0.5, dz=0.5)
    W_X_G = multiply(get_pose(body), G)
    set_pose(gripper, W_X_G)
    set_camera_target_body(body, dx=0.5, dy=0.5, dz=0.5)

    ## new object pose given rotation
    # object_pose = ((0.4, 0.3, 0), quat_from_euler((-1.2, 0.3, 0)))
    object_pose = ((0, 0, 0), quat_from_euler((-1.2, 0, 0)))
    set_pose(body, object_pose)
    W_X_G = multiply(get_pose(body), G)
    draw_pose(W_X_G, length=0.3)
    set_pose(gripper, W_X_G)
    set_camera_target_body(gripper, dx=0.5, dy=0, dz=0.5)
    wait_if_gui()


def reload_after_vhacd(path, body, scale, id=None):
    from pybullet_tools.utils import process_urdf, TEMP_URDF_DIR

    pose = get_pose(body)
    remove_body(body)
    new_urdf_path = process_urdf(path)
    id_urdf_path = join(TEMP_URDF_DIR, f"{id}.urdf")
    os.rename(new_urdf_path, id_urdf_path)
    body = load_pybullet(id_urdf_path, scale=scale)
    set_pose(body, pose)
    return id_urdf_path, body

def test_gripper_joints():
    """ visualize ee link pose as conf changes """
    world = get_test_world(robot='feg')
    robot = world.robot

    set_se3_conf(robot, (0, 0, 0, 0, 0, 0))
    set_camera_target_body(robot, dx=0.5, dy=0.5, dz=0.5)
    for j in range(3, 6):
        limits = get_joint_limits(robot, j)
        values = np.linspace(limits[0], limits[1], num=36)
        for v in values:
            conf = [0, 0, 0, 0, math.pi/2, 0]
            conf[j] = v
            set_se3_conf(robot, conf)
            set_camera_target_body(robot, dx=0.5, dy=0.5, dz=0.5)
            time.sleep(0.1)

    wait_if_gui('Finish?')
    disconnect()


def test_gripper_range(IK=False):
    """ visualize all possible gripper orientation """
    world = get_test_world(robot='feg')
    robot = world.robot

    set_se3_conf(robot, (0, 0, 0, 0, 0, 0))
    set_camera_target_body(robot, dx=0.5, dy=0.5, dz=0.5)
    choices = np.linspace(-math.pi, math.pi, num=9)[:-1]
    bad = [choices[1], choices[3]]
    mi, ma = min(choices), max(choices)
    ra = ma - mi
    def get_color(i, j, k):
        color = RGBA((i-mi)/ra, (j-mi)/ra, (k-mi)/ra, 1)
        return color
    def mynice(tup):
        tup = nice(tup)
        if len(tup) == 2:
            return tup[-1]
        return tuple(list(tup)[-3:])
    for i in choices:
        for j in choices:
            for k in choices:
                if IK:
                    gripper = create_fe_gripper(init_q=[0, 0, 0, 0, 0, 0], POINTER=True)
                    pose = ((0,0,0), quat_from_euler((i,j,k)))
                    conf = se3_ik(robot, pose)
                    if conf == None:
                        remove_body(gripper)
                        print('failed IK at', nice(pose))
                        continue
                    else:
                        print('pose =', mynice(pose), '-->\t conf =', mynice(conf))
                        set_se3_conf(gripper, conf)
                        set_all_color(gripper, WHITE)
                        # if j in bad:
                        #     set_all_color(gripper, RED)
                        # else:
                        #     set_all_color(gripper, GREEN)
                else:
                    conf = [0, 0, 0, i, j, k]
                    gripper = create_fe_gripper(init_q=conf, POINTER=True)
                    set_all_color(gripper, WHITE)
                    pose = get_link_pose(gripper, link_from_name(gripper, 'panda_hand'))
                    print('conf =', mynice(conf), '-->\t pose =', mynice(pose))

                    # set_all_color(gripper, get_color(i,j,k))
            set_camera_target_body(robot, dx=0.5, dy=0.5, dz=0.5)
    set_camera_target_body(robot, dx=0.5, dy=0.5, dz=0.5)

    wait_if_gui('Finish?')
    disconnect()


def test_torso():
    world = get_test_world(robot='pr2')
    robot = world.robot
    torso_joint = joint_from_name(robot, 'torso_lift_joint')
    l = get_joint_limits(robot, torso_joint)
    robot.set_joint_positions([torso_joint], [1.5])
    print(l)
    # x, y, z, yaw = robot.get_positions('base-torso')
    # robot.set_positions_by_group('base-torso', (x, y, 0.9, yaw))
    wait_unlocked()
    print(robot)


def test_reachability(robot):
    world = get_test_world(robot=robot, semantic_world=True, custom_limits=((-4, -4), (4, 4)))
    robot = world.robot
    state = State(world, grasp_types=robot.grasp_types)

    for w, xy in [(0.3, (0, 0)), (0.5, (2, 2))]:
        table1 = create_table(world, w=w, xy=xy)
        movable1 = create_movable(world, table1, xy=xy)
        result = robot.check_reachability(movable1, state)
        print('w', w, result)

    wait_unlocked()


def test_tracik(robot, verbose=False):
    from pybullet_tools.tracik import IKSolver
    from robot_builder.spot_utils import solve_leg_conf
    world = get_test_world(robot=robot, width=1200, height=1200,
                           semantic_world=True, draw_origin=True,
                           custom_limits=((-3, -3), (3, 3)))
    robot = world.robot
    set_camera_target_body(robot.body, distance=0.5)
    set_joint_position(robot.body, joint_from_name(robot.body, 'arm0.f1x'), -1.5)
    # compute_link_lengths(robot.body)

    box = create_box(0.05, 0.05, 0.075, color=(1, 0, 0, 1))
    grasp_pose = ((0, 0, 0.2), quat_from_euler((0, math.pi/2, 0)))

    tool_link = robot.get_tool_link()
    body_solver = IKSolver(robot, tool_link=tool_link, first_joint='torso_lift_joint',
                           custom_limits=robot.custom_limits)  ## using 13 joints

    while True:
        box_pose = sample_random_pose(AABB(lower=[-0.3, -0.3, 0.2], upper=[0.3, 0.3, 1.2])) ## ((0, 0, 0.1), (0, 0, 0, 1))
        gripper_pose = multiply(box_pose, grasp_pose)
        print('\n', nice(gripper_pose))
        for conf in body_solver.generate(gripper_pose):
            joint_state = dict(zip(body_solver.joints, conf))
            joint_values = {}
            for i, value in joint_state.items():
                if i == 0:
                    continue
                joint_name = get_joint_name(robot.body, i)
                joint_values[i] = (joint_name, value)

            collided = False
            with ConfSaver(robot.body):
                body_solver.set_conf(conf)
                body_link = link_from_name(robot.body, 'body_link')
                for i in ['arm0.link_sh1', 'arm0.link_hr0', 'arm0.link_el0',
                          'arm0.link_el1', 'arm0.link_wr0', 'arm0.link_wr1']:
                    link = link_from_name(robot.body, i)
                    if pairwise_link_collision(robot.body, link, robot.body, body_link):
                        collided = True
                        break
            if collided:
                if verbose:
                    print('\n\n self-collision!')
                break

            leg_conf = solve_leg_conf(robot.body, joint_state[0], verbose=False)
            if leg_conf is None:
                if verbose:
                    print('\n\n failed leg ik!')
                break

            for i in range(len(leg_conf.values)):
                index = leg_conf.joints[i]
                value = leg_conf.values[i]
                joint_values[index] = (get_joint_name(robot.body, index), value)

            joint_values = dict(sorted(joint_values.items()))
            for i, (joint_name, value) in joint_values.items():
                print('\t', i, '\t', joint_name, '\t', round(value, 3))

            set_pose(box, box_pose)
            body_solver.set_conf(conf)
            leg_conf.assign()
            break


############################################################################

if __name__ == '__main__':

    robot = 'rummy'  ## 'spot' | 'feg' | 'pr2'
    # test_gripper_joints()
    # test_gripper_range()
    # test_torso()
    # test_reachability(robot)
    # test_tracik(robot)
