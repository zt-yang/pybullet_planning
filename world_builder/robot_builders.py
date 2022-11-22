import json
from os.path import join

import numpy as np
import math
from world_builder.entities import Camera
from world_builder.robots import PR2Robot, FEGripper


def get_robot_builder(builder_name):
    if builder_name == 'build_fridge_domain_robot':
        return build_fridge_domain_robot
    elif builder_name == 'build_table_domain_robot':
        return build_table_domain_robot
    return None

############################################


def maybe_add_robot(world, template_dir=None):
    config_file = join(template_dir, 'planning_config.json')
    planning_config = json.load(open(config_file, 'r'))
    if 'robot_builder' not in planning_config:
        return
    custom_limits = planning_config['base_limits']
    robot_name = planning_config['robot_name']
    robot_builder = get_robot_builder(planning_config['robot_builder'])
    robot_builder(world, robot_name=robot_name, custom_limits=custom_limits)


#######################################################

from pybullet_tools.pr2_problems import create_pr2
from pybullet_tools.pr2_primitives import get_base_custom_limits
from pybullet_tools.pr2_utils import draw_viewcone, get_viewcone, get_group_conf, set_group_conf, get_other_arm, \
    get_carry_conf, set_arm_conf, open_arm, close_arm, arm_conf, REST_LEFT_ARM
from pybullet_tools.bullet_utils import set_pr2_ready, BASE_LINK, BASE_RESOLUTIONS, BASE_VELOCITIES, BASE_JOINTS, \
    draw_base_limits, BASE_LIMITS, CAMERA_FRAME, CAMERA_MATRIX, EYE_FRAME, collided
from pybullet_tools.utils import LockRenderer, HideOutput, draw_base_limits, PI


def set_pr2_ready(pr2, arm='left', grasp_type='top', DUAL_ARM=False):
    other_arm = get_other_arm(arm)
    if not DUAL_ARM:
        initial_conf = get_carry_conf(arm, grasp_type)
        set_arm_conf(pr2, arm, initial_conf)
        open_arm(pr2, arm)
        set_arm_conf(pr2, other_arm, arm_conf(other_arm, REST_LEFT_ARM))
        close_arm(pr2, other_arm)
    else:
        for a in [arm, other_arm]:
            initial_conf = get_carry_conf(a, grasp_type)
            set_arm_conf(pr2, a, initial_conf)
            open_arm(pr2, a)


def create_pr2_robot(world, base_q=(0, 0, 0),
                     DUAL_ARM=False, USE_TORSO=True,
                     custom_limits=BASE_LIMITS,
                     resolutions=BASE_RESOLUTIONS,
                     DRAW_BASE_LIMITS=False,
                     max_velocities=BASE_VELOCITIES, robot=None):

    if robot is None:
        with LockRenderer(lock=True):
            with HideOutput(enable=True):
                robot = create_pr2()
                set_pr2_ready(robot, DUAL_ARM=DUAL_ARM)
        if len(base_q) == 3:
            set_group_conf(robot, 'base', base_q)
        elif len(base_q) == 4:
            set_group_conf(robot, 'base-torso', base_q)

    with np.errstate(divide='ignore'):
        weights = np.reciprocal(resolutions)

    if isinstance(custom_limits, dict):
        custom_limits = np.asarray(list(custom_limits.values())).T.tolist()

    if DRAW_BASE_LIMITS:
        draw_base_limits(custom_limits)
    robot = PR2Robot(robot, base_link=BASE_LINK, joints=BASE_JOINTS,
                     DUAL_ARM=DUAL_ARM, USE_TORSO=USE_TORSO,
                     custom_limits=get_base_custom_limits(robot, custom_limits),
                     resolutions=resolutions, weights=weights)
    world.add_robot(robot, max_velocities=max_velocities)

    # print('initial base conf', get_group_conf(robot, 'base'))
    # set_camera_target_robot(robot, FRONT=True)

    camera = Camera(robot, camera_frame=CAMERA_FRAME, camera_matrix=CAMERA_MATRIX, max_depth=2.5, draw_frame=EYE_FRAME)
    robot.cameras.append(camera)

    ## don't show depth and segmentation data yet
    # if args.camera: robot.cameras[-1].get_image(segment=args.segment)

    return robot


#######################################################


from pybullet_tools.flying_gripper_utils import create_fe_gripper, plan_se3_motion, Problem, \
    get_free_motion_gen, set_gripper_positions, get_se3_joints, set_gripper_positions, \
    set_se3_conf ## se3_from_pose,


def create_gripper_robot(world, custom_limits, initial_q=(0, 0, 0, 0, 0, 0), robot=None):
    from pybullet_tools.flying_gripper_utils import BASE_RESOLUTIONS, BASE_VELOCITIES, BASE_LINK

    if robot is None:
        with LockRenderer(lock=True):
            with HideOutput(enable=True):
                robot = create_fe_gripper()
        set_se3_conf(robot, initial_q)

    with np.errstate(divide='ignore'):
        weights = np.reciprocal(BASE_RESOLUTIONS)
    robot = FEGripper(robot, base_link=BASE_LINK, joints=get_se3_joints(robot),
                  custom_limits=custom_limits, resolutions=BASE_RESOLUTIONS, weights=weights)
    world.add_robot(robot, max_velocities=BASE_VELOCITIES)

    return robot


#######################################################


def build_table_domain_robot(world, robot_name, custom_limits=None, initial_q=None):
    from world_builder.builders import create_gripper_robot, create_pr2_robot
    """ simplified cooking domain """
    if robot_name == 'feg':
        if custom_limits is None:
            custom_limits = {0: (0, 4), 1: (3, 12), 2: (0, 2)}
        if initial_q is None:
            initial_q = [0.9, 8, 0.7, 0, -math.pi / 2, 0]
        robot = create_gripper_robot(world, custom_limits, initial_q=initial_q)
    else:
        if custom_limits is None:
            custom_limits = ((0, 0), (8, 8))
        if initial_q is None:
            initial_q = (1.79, 6, PI / 2 + PI / 2)
        robot = create_pr2_robot(world, base_q=initial_q, custom_limits=custom_limits,
                                 USE_TORSO=False, DRAW_BASE_LIMITS=True)
    return robot


def build_fridge_domain_robot(world, robot_name, custom_limits=None):
    """ counter and fridge in the (6, 6) range """
    x, y = (5, 3)
    if robot_name == 'feg':
        if custom_limits is None:
            custom_limits = {0: (0, 6), 1: (0, 6), 2: (0, 2)}
        robot = create_gripper_robot(world, custom_limits, initial_q=[x, y, 0.7, 0, -math.pi / 2, 0])
        robot.set_spawn_range(((2.5, 2, 0.5), (3.8, 3.5, 1.9)))
    else:
        if custom_limits is None:
            custom_limits = ((0, 0, 0), (6, 6, 1.5))
        robot = create_pr2_robot(world, custom_limits=custom_limits, base_q=(x, y, PI / 2 + PI / 2))
        robot.set_spawn_range(((4.2, 2, 0.5), (5, 3.5, 1.9)))
    return robot


def build_robot_from_args(world, robot_name, custom_limits, **kwargs):
    if robot_name == 'feg':
        robot = create_gripper_robot(world, custom_limits, **kwargs)
    else:
        robot = create_pr2_robot(world, custom_limits=custom_limits, **kwargs)
    return robot