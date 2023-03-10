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
    return build_robot_from_args


#######################################################

from pybullet_tools.pr2_problems import create_pr2
from pybullet_tools.pr2_primitives import get_base_custom_limits
from pybullet_tools.pr2_utils import draw_viewcone, get_viewcone, get_group_conf, set_group_conf, get_other_arm, \
    get_carry_conf, set_arm_conf, open_arm, close_arm, arm_conf, REST_LEFT_ARM
from pybullet_tools.bullet_utils import set_pr2_ready, BASE_LINK, BASE_RESOLUTIONS, BASE_VELOCITIES, BASE_JOINTS, \
    draw_base_limits, BASE_LIMITS, CAMERA_FRAME, CAMERA_MATRIX, EYE_FRAME, collided
from pybullet_tools.utils import LockRenderer, HideOutput, PI


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


def create_pr2_robot(world, base_q=(0, 0, 0), DUAL_ARM=False, USE_TORSO=True,
                     custom_limits=BASE_LIMITS, resolutions=BASE_RESOLUTIONS,
                     DRAW_BASE_LIMITS=False, max_velocities=BASE_VELOCITIES,
                     robot=None):

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


from pybullet_tools.flying_gripper_utils import create_fe_gripper, get_se3_joints, set_se3_conf


def create_gripper_robot(world=None, custom_limits=((0, 0, 0), (6, 12, 2)),
                         initial_q=(0, 0, 0, 0, 0, 0), DRAW_BASE_LIMITS=False, robot=None):
    from pybullet_tools.flying_gripper_utils import BASE_RESOLUTIONS, BASE_VELOCITIES, \
        BASE_LINK, get_se3_custom_limits

    if robot is None:
        with LockRenderer(lock=True):
            with HideOutput(enable=True):
                robot = create_fe_gripper()
        set_se3_conf(robot, initial_q)

    with np.errstate(divide='ignore'):
        weights = np.reciprocal(BASE_RESOLUTIONS)

    if DRAW_BASE_LIMITS:
        draw_base_limits(custom_limits)
    custom_limits = get_se3_custom_limits(custom_limits)

    robot = FEGripper(robot, base_link=BASE_LINK, joints=get_se3_joints(robot),
                  custom_limits=custom_limits, resolutions=BASE_RESOLUTIONS, weights=weights)
    if world is not None:
        world.add_robot(robot, max_velocities=BASE_VELOCITIES)

    return robot


#######################################################


def build_skill_domain_robot(world, robot_name, **kwargs):
    """ simplified cooking domain, pr2 no torso """
    if 'initial_xy' not in kwargs:
        kwargs['initial_xy'] = (0, 0)
    if 'DRAW_BASE_LIMITS' not in kwargs:
        kwargs['DRAW_BASE_LIMITS'] = False
    if 'custom_limits' not in kwargs:
        if robot_name == 'feg':
            kwargs['custom_limits'] = ((0, 0, 0), (2, 10, 2))
        elif robot_name == 'pr2':
            kwargs['custom_limits'] = ((0, 0, 0), (3, 10, 2.4))
    return build_robot_from_args(world, robot_name, **kwargs)


def build_table_domain_robot(world, robot_name, **kwargs):
    """ testing basic pick and place """
    kwargs['initial_xy'] = (0, 0)
    if 'custom_limits' not in kwargs:
        kwargs['custom_limits'] = ((-4, -4, 0), (4, 4, 2))
        if robot_name == 'pr2':
            kwargs['USE_TORSO'] = True
    return build_robot_from_args(world, robot_name, **kwargs)


def build_fridge_domain_robot(world, robot_name, **kwargs):
    """ counter and fridge in the (6, 6) range """
    kwargs['spawn_range'] = ((4.2, 2, 0.5), (5, 3.5, 1.9))
    if 'custom_limits' not in kwargs:
        kwargs['custom_limits'] = ((0, 0, 0), (6, 6, 2))
    if 'initial_xy' not in kwargs:
        kwargs['initial_xy'] = (5, 3)
    return build_robot_from_args(world, robot_name, **kwargs)


def build_oven_domain_robot(world, robot_name, **kwargs):
    """ kitchen svg, focus on the oven and cabinets """
    kwargs['initial_xy'] = (1.5, 8)
    if 'custom_limits' not in kwargs:
        kwargs['custom_limits'] = ((0, 4, 0), (2.5, 10, 2))
    return build_robot_from_args(world, robot_name, **kwargs)


def build_robot_from_args(world, robot_name, **kwargs):
    spawn_range = None
    if 'spawn_range' in kwargs:
        spawn_range = kwargs['spawn_range']
        del kwargs['spawn_range']

    if robot_name == 'feg':
        if 'initial_q' not in kwargs and 'initial_xy' in kwargs:
            x, y = kwargs['initial_xy']
            del kwargs['initial_xy']
            kwargs['initial_q'] = [x, y, 0.7, 0, -PI / 2, 0]
    else:
        if 'base_q' not in kwargs and 'initial_xy' in kwargs:
            x, y = kwargs['initial_xy']
            del kwargs['initial_xy']
            kwargs['base_q'] = (x, y, PI)

        if robot_name == 'spot':
            robot = load_spot_robot(world, **kwargs)
        else:
            robot = create_pr2_robot(world, **kwargs)

    if spawn_range is not None:
        robot.set_spawn_range(spawn_range)
    return robot
