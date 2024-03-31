import numpy as np
import copy

from pybullet_tools.utils import joint_from_name, get_link_subtree, link_from_name, clone_body, \
    set_all_color, TRANSPARENT, get_max_limit, get_min_limit, get_extend_fn, get_moving_links, \
    set_joint_positions, pairwise_collision, get_link_pose, multiply, set_pose, \
    RED, set_color, get_link_name, get_joints, is_movable

from pybullet_tools.bullet_utils import BASE_LINK, BASE_RESOLUTIONS, BASE_VELOCITIES, BASE_JOINTS, \
    draw_base_limits as draw_base_limits_bb, BASE_LIMITS, CAMERA_FRAME, CAMERA_MATRIX, EYE_FRAME


BASE_GROUP = 'base'
TORSO_GROUP = 'torso'
BASE_TORSO_GROUP = 'base-torso'
HEAD_GROUP = 'head'
ARM_GROUP = 'arm'
GRIPPER_GROUP = 'gripper'

BASE_JOINTS = ['x', 'y', 'theta']

ROBOTIQ_JOINTS = [
    # Works for 85 and 140
    'finger_joint',  # [0, 0.8757]
    'left_inner_finger_joint', ## mimics - 'left_arm_finger_joint'
    'left_inner_knuckle_joint', ## mimics 'left_arm_finger_joint'
    'right_outer_knuckle_joint', ## mimics 'left_arm_finger_joint'
    'right_inner_finger_joint', ## mimics - 'left_arm_finger_joint'
    'right_inner_knuckle_joint', ## mimics 'left_arm_finger_joint'
]

#####################################


def create_robot_gripper(robot, link_name, visual=True, color=None):
    links = get_link_subtree(robot, link_from_name(robot, link_name))
    gripper = clone_body(robot, links=links, visual=False, collision=True)  # TODO: joint limits
    if not visual:
        set_all_color(gripper, TRANSPARENT)
    if color is not None:
        set_all_color(gripper, color)
    return gripper


def get_joints_by_names(robot, names):
    return [joint_from_name(robot, name) for name in names]


def get_robot_group_joints(robot, group, joint_groups):
    assert group in joint_groups
    return get_joints_by_names(robot, joint_groups[group])


def set_robot_group_conf(robot, group, joint_groups, positions):
    set_joint_positions(robot, get_robot_group_joints(robot, group, joint_groups), positions)


def get_robot_gripper_link(robot, arm, tool_frames):
    assert arm in tool_frames
    return link_from_name(robot, tool_frames[arm])


def get_robot_gripper_joints(robot, arm, joint_groups):
    return get_robot_group_joints(robot, gripper_group_from_arm(arm), joint_groups)


def get_cloned_gripper_joints(gripper_cloned):
    return [joint for joint in get_joints(gripper_cloned) if is_movable(gripper_cloned, joint)]


def gripper_group_from_arm(arm):
    return f"{arm}_gripper"


#####################################


def close_until_collision(robot, gripper_joints, bodies=[], open_conf=None, closed_conf=None, num_steps=25, **kwargs):
    if not gripper_joints:
        return None
    if open_conf is None:
        open_conf = [get_max_limit(robot, joint) for joint in gripper_joints]
    if closed_conf is None:
        closed_conf = [get_min_limit(robot, joint) for joint in gripper_joints]
    resolutions = np.abs(np.array(open_conf) - np.array(closed_conf)) / num_steps
    extend_fn = get_extend_fn(robot, gripper_joints, resolutions=resolutions)
    close_path = [open_conf] + list(extend_fn(open_conf, closed_conf))
    collision_links = frozenset(get_moving_links(robot, gripper_joints))

    for i, conf in enumerate(close_path):
        set_joint_positions(robot, gripper_joints, conf)
        if any(pairwise_collision((robot, collision_links), body, **kwargs) for body in bodies):
            if i == 0:
                return close_path[i-1][0]  ## None
            return close_path[i-1][0]
    return close_path[-1][0]


#####################################


def get_robot_base_custom_limits_dict(robot, base_limits, yaw_limit=None, torso_joint_name='z'):
    x, y, theta = BASE_JOINTS

    if isinstance(base_limits, dict):
        return base_limits
    if len(base_limits[0]) == 2:
        x_limits, y_limits = zip(*base_limits)
    if len(base_limits[0]) == 3:
        x_limits, y_limits, z_limits = zip(*base_limits)

    custom_limits = {
        joint_from_name(robot, x): x_limits,
        joint_from_name(robot, y): y_limits,
    }
    if yaw_limit is not None:
        custom_limits.update({
            joint_from_name(robot, theta): yaw_limit,
        })
    if len(base_limits[0]) == 3:
        custom_limits[joint_from_name(robot, torso_joint_name)] = z_limits
    return custom_limits


def create_mobile_robot(world, load_robot_urdf_fn, robot_class, base_group, joint_groups,
                        base_q=None, custom_limits=BASE_LIMITS, use_torso=True,
                        draw_base_limits=False, max_velocities=BASE_VELOCITIES, robot=None, **kwargs):

    if base_q is None:
        base_q = [0] * 4 if use_torso else [0] * 3

    if robot is None:
        robot = load_robot_urdf_fn()
        set_robot_group_conf(robot, base_group, joint_groups, base_q)

    if isinstance(custom_limits, dict):
        custom_limits_dict = copy.deepcopy(custom_limits)
        custom_limits = np.asarray(list(custom_limits.values())).T.tolist()
    else:
        torso_joint_name = 'z' if base_group == BASE_GROUP else joint_groups[BASE_TORSO_GROUP][-2]
        custom_limits_dict = get_robot_base_custom_limits_dict(robot, custom_limits, torso_joint_name=torso_joint_name)

    if draw_base_limits:
        draw_base_limits_bb(custom_limits)

    robot = robot_class(robot, custom_limits=custom_limits_dict, use_torso=use_torso, **kwargs)
    world.add_robot(robot, max_velocities=max_velocities)

    return robot
