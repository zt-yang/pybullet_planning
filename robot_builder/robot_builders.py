import numpy as np

from pybullet_tools.bullet_utils import BASE_LINK, BASE_RESOLUTIONS, BASE_VELOCITIES, BASE_JOINTS, \
    draw_base_limits as draw_base_limits_bb, BASE_LIMITS, CAMERA_MATRIX
from pybullet_tools.utils import LockRenderer, HideOutput, PI, unit_pose

from robot_builder.robots import PR2Robot, FEGripper, SpotRobot
from robot_builder.robot_utils import create_mobile_robot, BASE_GROUP, BASE_TORSO_GROUP


def get_robot_builder(builder_name):
    if builder_name == 'build_fridge_domain_robot':
        return build_fridge_domain_robot
    elif builder_name == 'build_table_domain_robot':
        return build_table_domain_robot
    return build_robot_from_args


#######################################################

from pybullet_tools.pr2_problems import create_pr2
from pybullet_tools.pr2_primitives import get_base_custom_limits
from pybullet_tools.pr2_utils import set_group_conf, get_other_arm, \
    get_carry_conf, set_arm_conf, open_arm, close_arm, arm_conf, REST_LEFT_ARM
from pybullet_tools.logging_utils import print_red


def set_pr2_ready(pr2, arm='left', grasp_type='top', dual_arm=False):
    other_arm = get_other_arm(arm)
    if not dual_arm:
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


def create_pr2_robot(world, base_q=(0, 0, 0), dual_arm=False, use_torso=True,
                     custom_limits=BASE_LIMITS, resolutions=BASE_RESOLUTIONS,
                     draw_base_limits=False, max_velocities=BASE_VELOCITIES, robot=None, **kwargs):

    if robot is None:
        robot = create_pr2()
        set_pr2_ready(robot, arm=PR2Robot.arms[0], dual_arm=dual_arm)
        if len(base_q) == 3:
            set_group_conf(robot, 'base', base_q)
        elif len(base_q) == 4:
            set_group_conf(robot, 'base-torso', base_q)

    with np.errstate(divide='ignore'):
        weights = np.reciprocal(resolutions)

    if isinstance(custom_limits, dict):
        custom_limits = np.asarray(list(custom_limits.values())).T.tolist()

    if draw_base_limits:
        draw_base_limits_bb(custom_limits)

    robot = PR2Robot(robot, base_link=BASE_LINK,
                     dual_arm=dual_arm, use_torso=use_torso,
                     custom_limits=get_base_custom_limits(robot, custom_limits),
                     resolutions=resolutions, weights=weights, **kwargs)
    world.add_robot(robot, max_velocities=max_velocities)

    # print('initial base conf', get_group_conf(robot, 'base'))
    # set_camera_target_robot(robot, FRONT=True)

    robot.add_cameras(max_depth=2.5, camera_matrix=CAMERA_MATRIX, verbose=True)

    ## don't show depth and segmentation data yet
    # if args.camera: robot.cameras[-1].get_image(segment=args.segment)

    return robot


#######################################################

from robot_builder.spot_utils import load_spot, SPOT_JOINT_GROUPS


def create_spot_robot(world, **kwargs):
    return create_mobile_robot(world, load_spot, SpotRobot, BASE_TORSO_GROUP, SPOT_JOINT_GROUPS, **kwargs)


#######################################################


from pybullet_tools.flying_gripper_utils import create_fe_gripper, get_se3_joints, set_se3_conf


def create_gripper_robot(world=None, custom_limits=((0, 0, 0), (6, 12, 2)),
                         initial_q=(0, 0, 0, 0, 0, 0), draw_base_limits=False, robot=None, **kwargs):
    from pybullet_tools.flying_gripper_utils import BASE_RESOLUTIONS, BASE_VELOCITIES, \
        BASE_LINK, get_se3_custom_limits

    if robot is None:
        with LockRenderer(lock=True):
            with HideOutput(enable=True):
                robot = create_fe_gripper()
        set_se3_conf(robot, initial_q)

    with np.errstate(divide='ignore'):
        weights = np.reciprocal(BASE_RESOLUTIONS)

    if draw_base_limits:
        draw_base_limits_bb(custom_limits)
    custom_limits = get_se3_custom_limits(custom_limits)

    robot = FEGripper(robot, base_link=BASE_LINK, joints=get_se3_joints(robot), custom_limits=custom_limits,
                      resolutions=BASE_RESOLUTIONS, weights=weights, **kwargs)
    if world is not None:
        world.add_robot(robot, max_velocities=BASE_VELOCITIES)

    return robot


#######################################################


def build_skill_domain_robot(world, robot_name, **kwargs):
    """ simplified cooking domain, pr2 no torso """
    if 'initial_xy' not in kwargs:
        kwargs['initial_xy'] = (0, 0)
    if 'draw_base_limits' not in kwargs:
        kwargs['draw_base_limits'] = False
    if 'custom_limits' not in kwargs:
        if robot_name == 'feg':
            kwargs['custom_limits'] = ((0, 0, 0), (2, 10, 2))
        elif robot_name in ['spot', 'pr2']:
            kwargs['custom_limits'] = ((0, 0, 0), (3, 10, 2.4))
    return build_robot_from_args(world, robot_name, **kwargs)


def build_table_domain_robot(world, robot_name, **kwargs):
    """ testing basic pick and place """
    if 'base_q' not in kwargs:
        kwargs['initial_xy'] = (0, 0)
    if 'custom_limits' not in kwargs:
        kwargs['custom_limits'] = ((-4, -4, 0), (4, 4, 2))
    if robot_name != 'feg':
        kwargs['use_torso'] = True
    return build_robot_from_args(world, robot_name, **kwargs)


def build_namo_domain_robot(world, robot_name, **kwargs):
    """ testing pull in Navigation Among Movable Obstacles (NAMO) """
    if 'base_q' not in kwargs:
        kwargs['initial_xy'] = (1.5, 0)
    if 'custom_limits' not in kwargs:
        kwargs['custom_limits'] = ((-2.75, -3.5, 0), (5.5, 4.5, 2))
    if robot_name != 'feg':
        kwargs['use_torso'] = True
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


def build_robot_from_args(world, robot_name, create_robot_fn=None, **kwargs):
    """ call upon different robot classes """
    spawn_range = None
    if 'spawn_range' in kwargs:
        spawn_range = kwargs['spawn_range']
        del kwargs['spawn_range']

    if 'z_range' in kwargs:
        x_min, y_min = kwargs['custom_limits'][0][:2]
        x_max, y_max = kwargs['custom_limits'][1][:2]
        z_min, z_max = kwargs['z_range']
        kwargs['custom_limits'] = ((x_min, y_min, z_min), (x_max, y_max, z_max))
        del kwargs['z_range']

    if robot_name == 'feg':
        if 'initial_q' not in kwargs and 'initial_xy' in kwargs:
            x, y = kwargs['initial_xy']
            kwargs['initial_q'] = [x, y, 0.7, 0, -PI / 2, 0]
        for key in ['base_q', 'initial_xy']:
            if key in kwargs:
                kwargs.pop(key)
        robot = create_gripper_robot(world, **kwargs)

    else:
        if 'base_q' not in kwargs and 'initial_xy' in kwargs:
            x, y = kwargs['initial_xy']
            kwargs['base_q'] = (x, y, PI)
            if 'use_torso' in kwargs and kwargs['use_torso']:
                kwargs['base_q'] = (x, y, 0, PI)
        elif isinstance(kwargs['base_q'], str):
            kwargs['base_q'] = eval(kwargs['base_q'])

        if 'initial_xy' in kwargs:
            del kwargs['initial_xy']

        if create_robot_fn is not None:
            robot = create_robot_fn(world, **kwargs)
        elif robot_name == 'spot':
            robot = create_spot_robot(world, **kwargs)
        elif robot_name == 'pr2':
            robot = create_pr2_robot(world, **kwargs)
        else:
            print_red(f'Robot not found = {robot_name}, did you forget to provide create_robot_fn?')
            return None

    if spawn_range is not None:
        robot.set_spawn_range(spawn_range)
    robot.world = world
    return robot
