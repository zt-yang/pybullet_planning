from collections import defaultdict

import numpy as np
import copy
import math
import time

from pybullet_tools.utils import joint_from_name, get_link_subtree, link_from_name, clone_body, \
    set_all_color, TRANSPARENT, get_max_limit, get_min_limit, get_extend_fn, get_moving_links, \
    set_joint_positions, pairwise_collision, get_link_pose, multiply, set_pose, euler_from_quat, \
    RED, set_color, get_link_name, get_joints, is_movable, wait_for_user, quat_from_euler, set_renderer, \
    get_movable_joints, get_joint_name, get_joint_position, get_aabb, get_aabb_center, draw_aabb, YELLOW, \
    AABB, GREEN, pairwise_link_collision
from pybullet_tools.pr2_primitives import Conf
from pybullet_tools.bullet_utils import BASE_LINK, BASE_RESOLUTIONS, BASE_VELOCITIES, BASE_JOINTS, \
    draw_base_limits as draw_base_limits_bb, BASE_LIMITS, nice
from pybullet_tools.grasp_utils import enumerate_rotational_matrices, \
    enumerate_translation_matrices, test_transformations_template
from pybullet_tools.logging_utils import bcolors, print_debug

BASE_GROUP = 'base'
TORSO_GROUP = 'torso'
BASE_TORSO_GROUP = 'base-torso'
HEAD_GROUP = 'head'
ARM_GROUP = 'arm'
GRIPPER_GROUP = 'gripper'

BASE_JOINTS = ['x', 'y', 'theta']
BASE_TORSO_JOINTS = ['x', 'y', 'torso_lift_joint', 'theta']

ROBOTIQ_JOINTS = [
    # Works for 85 and 140
    'finger_joint',  # [0, 0.8]
    'left_inner_knuckle_joint', ## mimics + / -
    'left_inner_finger_joint', ## mimics - / +
    'right_outer_knuckle_joint', ## mimics + / -
    'right_inner_knuckle_joint', ## mimics + / -
    'right_inner_finger_joint', ## mimics - / +
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
    """ return joints """
    assert group in joint_groups
    joint_names = joint_groups[group]
    if isinstance(joint_names[0], int):
        return joint_names
    return get_joints_by_names(robot, joint_names)


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


def test_tool_from_root_transformations(tool_from_root, get_attachment_fn, test_rotation=False, test_translation=False):
    if not test_rotation and not test_translation:
        wait_for_user('\t\t visualized attachment')
        return

    rotations = [euler_from_quat(tool_from_root[1])]
    translations = [tool_from_root[0]]
    if test_rotation:
        rotations = enumerate_rotational_matrices(return_list=True)
    if test_translation:
        translations = enumerate_translation_matrices(x=0.2)

    def funk(t, r):
        tool_from_root = (t, quat_from_euler(r))
        attachment = get_attachment_fn(tool_from_root)
        attachment.assign()

    title = 'robot_utils.test_tool_from_root_transformations'
    test_transformations_template(rotations, translations, funk, title)


def close_until_collision(robot, gripper_joints, bodies=[], open_conf=None, closed_conf=None,
                          num_steps=25, **kwargs):
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
                        draw_base_limits=False, max_velocities=BASE_VELOCITIES, robot=None,
                        return_body=False, debug_joint_names=False, **kwargs):

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
    if return_body:
        return robot.body

    ## print joints by potential joint groups, set use_joint_groups=False for finding the joint_groups
    if debug_joint_names:
        print_potential_joint_groups(robot, use_joint_groups=True)

    world.add_robot(robot, max_velocities=max_velocities)

    for arm in robot.get_all_arms():
        robot.open_arm(arm)
    return robot


def print_potential_joint_groups(robot, use_joint_groups=False):
    """ for loading a new robot, the joint_groups in the template robot class is likely wrong """
    joints = {get_joint_name(robot, j): j for j in get_movable_joints(robot)}
    joints_by_group = defaultdict(list)

    if use_joint_groups:
        joints_by_group.update(robot.joint_groups)
        for name, joint in joints.items():
            found = False
            for k, v in joints_by_group.items():
                if name in v:
                    found = True
                    break
            if not found:
                joints_by_group['unknown'].append(name)
    else:
        keywords = ['left', 'right', 'head', 'waist', 'torso']
        for name, joint in joints.items():
            found = False
            for k in keywords:
                if k in name:
                    joints_by_group[k].append(name)
                    found = True
                    break
            if not found:
                joints_by_group['unknown'].append(name)

    styles = list(bcolors.mapping)
    for i, (group, joints) in enumerate(joints_by_group.items()):
        style = styles[i]
        default_positions = [get_joint_position(robot, joint_from_name(robot, j)) for j in joints]
        line = f"\n{group} ({len(joints)})\t{default_positions}"
        print_debug(line, style=style)
        for name in joints:
            line = "\t"+name
            if group in ['leg']:
                line += f'\tchild = {get_link_name(robot, joint_from_name(robot, name))}'
            print_debug(line, style=style)


## -------------------------------------------------------------------------


def solve_leg_conf(body, torso_lift_value, zO, lA, lB, joint_groups, leg_group_name='leg',
                   get_leg_positions_from_hip_knee_angles=lambda x, y: [x, y], return_positions=True, verbose=True):
    """ return hip_pitch_joint, knee_joint values """
    from sympy import Symbol, Eq, solve, cos, sin

    aA = Symbol('aA', real=True)
    # aB = Symbol('aB', real=True)
    aBB = Symbol('aBB', real=True)  ## pi = (pi/2 - aA) + aBB - aB
    e1 = Eq(lA * cos(aA) + lB * sin(aBB), zO + torso_lift_value)
    e2 = Eq(lA * sin(aA), lB * cos(aBB))

    start = time.time()
    solutions = solve([e1, e2], aA, aBB)  ## , e2, e3, e4, e5
    if verbose:
        print(f'solve_leg_conf in : {round(time.time() - start, 2)} sec')

    solutions = [r for r in solutions if r[0] > 0 and r[1] > 0]
    if len(solutions) == 0:
        return None

    aA, aBB = solutions[0]
    aB = - math.pi/2 - aA + aBB
    joint_values = get_leg_positions_from_hip_knee_angles(aA, aB)

    if return_positions:
        return joint_values

    joints = get_robot_group_joints(body, leg_group_name, joint_groups)
    conf = Conf(body, joints, joint_values)
    return conf


def compute_link_lengths(body, hip_link_name, toe_link_name, upper_leg_link_name, lower_leg_link_name):
    """
    -- o O -----------  body of robot (zO)
       |\
     aA  \ lA
          o A   ------  knee of robot
         / \  aB
     lB / aBB
    B o -------------  toe of robot
    """

    hip_link = link_from_name(body, hip_link_name)
    hip_aabb = get_aabb(body, link=hip_link)
    draw_aabb(hip_aabb, color=GREEN)
    zO = get_aabb_center(hip_aabb)[2]

    toe_link = link_from_name(body, toe_link_name)
    toe_aabb = get_aabb(body, link=toe_link)
    draw_aabb(toe_aabb, color=GREEN)
    zB = get_aabb_center(toe_aabb)[2]

    upper_leg_link = link_from_name(body, upper_leg_link_name)
    upper_leg_aabb = get_aabb(body, link=upper_leg_link)
    draw_aabb(upper_leg_aabb, color=RED)
    zA1 = upper_leg_aabb.lower[2]

    lower_leg_link = link_from_name(body, lower_leg_link_name)
    lower_leg_aabb = get_aabb(body, link=lower_leg_link)
    draw_aabb(lower_leg_aabb, color=RED)
    zA2 = lower_leg_aabb.upper[2]

    zA = (zA1 + zA2) / 2

    (x1, y1, _), (x2, y2, _) = lower_leg_aabb
    found_upper_aabb = AABB(lower=(x1, y1, zA+0.01), upper=(x2, y2, zO))
    found_lower_aabb = AABB(lower=(x1, y1, zB), upper=(x2, y2, zA-0.01))
    draw_aabb(found_upper_aabb, color=YELLOW)
    draw_aabb(found_lower_aabb, color=YELLOW)

    lA = zO - zA
    lB = zA - zB
    zO = lA + lB - 0.05
    print(f"## computed upper and lower leg lengths\n\tzO = {zO}\n\tlA = {lA}\n\tlB = {lB}")
    return zO, lA, lB


def check_arm_body_collisions(body, arm_links=[], body_link_names=[], verbose=False):
    collided = False
    for body_link_name in body_link_names:
        body_link = link_from_name(body, body_link_name)
        for i in arm_links:
            link = link_from_name(body, i)
            if pairwise_link_collision(body, link, body, body_link):
                collided = True
                break
            if verbose: print(f"cfree {(body_link_name, i)}")
    return collided
