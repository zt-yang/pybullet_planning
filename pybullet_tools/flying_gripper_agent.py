from __future__ import print_function

import os
import time

from pybullet_tools.pr2_streams import get_handle_grasp_gen, get_ik_ir_grasp_handle_gen, \
    get_marker_grasp_gen, get_bconf_in_region_test, \
    get_bconf_in_region_gen, get_pose_in_region_gen, get_motion_wconf_gen, get_update_wconf_p_two_gen, \
    get_marker_pose_gen, get_pull_marker_to_pose_motion_gen, get_pull_marker_to_bconf_motion_gen,  \
    get_pull_marker_random_motion_gen, get_ik_ungrasp_handle_gen, get_pose_in_region_test, \
    get_cfree_btraj_pose_test, get_joint_position_open_gen, get_ik_ungrasp_mark_gen, \
    get_ik_ir_wconf_gen, get_ik_ir_wconf_gen, get_pose_in_space_test
from pybullet_tools.pr2_primitives import get_stable_gen, get_group_joints, Conf, \
    get_base_custom_limits, Pose, Conf, \
    get_ik_ir_gen, get_motion_gen, get_cfree_pose_pose_test, get_cfree_traj_pose_test, \
    Attach, Detach, Clean, Cook, control_commands, \
    get_gripper_joints, GripperCommand, apply_commands, State, Trajectory
from pybullet_tools.general_streams import get_cfree_approach_pose_test, get_grasp_list_gen, get_stable_list_gen, \
    sample_joint_position_open_list_gen, get_update_wconf_pst_gen, get_update_wconf_p_gen, get_sample_wconf_list_gen, \
    Position, get_contain_list_gen, get_pose_from_attachment

from pybullet_tools.bullet_utils import summarize_facts, print_plan, print_goal, save_pickle, set_camera_target_body, \
    set_camera_target_robot, nice, BASE_LIMITS, get_file_short_name
from pybullet_tools.pr2_problems import create_pr2
from pybullet_tools.pr2_utils import PR2_TOOL_FRAMES, create_gripper, set_group_conf
from pybullet_tools.utils import connect, disconnect, wait_if_gui, LockRenderer, HideOutput, get_client, \
    joint_from_name, WorldSaver, sample_placement, PI, add_parameter, add_button, Pose, Point, Euler, \
    euler_from_quat, get_joint, get_joints, PoseSaver, get_pose, get_link_pose, get_aabb, \
    get_joint_position, aabb_overlap, add_text, remove_handles, get_com_pose, get_closest_points,\
    set_color, RED, YELLOW, GREEN, multiply, get_unit_vector, unit_quat, get_bodies, BROWN, \
    pairwise_collision, connect, get_pose, point_from_pose, \
    disconnect, get_joint_positions, enable_gravity, save_state, restore_state, HideOutput, remove_body, \
    get_distance, LockRenderer, get_min_limit, get_max_limit, has_gui, WorldSaver, wait_if_gui, add_line, SEPARATOR, \
    BROWN, BLUE, WHITE, TAN, GREY, YELLOW, GREEN, BLACK, RED, CLIENTS

from os.path import join, isfile
from pddlstream.algorithms.algorithm import parse_problem, reset_globals
from pddlstream.algorithms.constraints import PlanConstraints
from pddlstream.language.generator import from_gen_fn, from_list_fn, from_fn, fn_from_constant, empty_gen, from_test
from pddlstream.language.constants import Equal, AND, PDDLProblem, is_plan
from pddlstream.utils import read, INF, get_file_path, find_unique, Profiler
from pddlstream.language.function import FunctionInfo
from pddlstream.language.stream import StreamInfo, PartialInputs
from pddlstream.language.object import SharedOptValue
from pddlstream.language.external import defer_shared, never_defer
from collections import namedtuple

from .flying_gripper_utils import get_ik_fn, get_free_motion_gen, get_pull_door_handle_motion_gen, get_reachable_test

def get_stream_map(p, c, l, t):
    # p = problem
    # c = collisions
    # l = custom_limits
    # t = teleport
    stream_map = {
        'sample-pose-on': from_list_fn(get_stable_list_gen(p, collisions=c)),
        'sample-pose-in': from_list_fn(get_contain_list_gen(p, collisions=c, verbose=False)),
        'sample-grasp': from_list_fn(get_grasp_list_gen(p, collisions=True)),

        'inverse-kinematics-hand': from_fn(get_ik_fn(p, collisions=c, teleport=t, custom_limits=l, verbose=False)),
        'test-cfree-pose-pose': from_test(get_cfree_pose_pose_test(collisions=c)),
        'test-cfree-approach-pose': from_test(get_cfree_approach_pose_test(p, collisions=c)),
        'test-cfree-traj-pose': from_test(get_cfree_traj_pose_test(p.robot, collisions=c, verbose=False)),

        'plan-free-motion-hand': from_fn(get_free_motion_gen(p, collisions=c, teleport=t, custom_limits=l)),

        'get-joint-position-open': from_list_fn(sample_joint_position_open_list_gen(p)),
        'sample-handle-grasp': from_list_fn(get_handle_grasp_gen(p, collisions=c, verbose=False)),

        'inverse-kinematics-grasp-handle': from_fn(get_ik_fn(p, collisions=c, teleport=t, custom_limits=l, verbose=False)),

        # 'plan-base-pull-drawer-handle': from_fn(
        #     get_pull_drawer_handle_motion_gen(p, collisions=c, teleport=t, custom_limits=l)),
        'plan-base-pull-door-handle': from_fn(
            get_pull_door_handle_motion_gen(p, collisions=c, teleport=t, custom_limits=l)),
        'get-pose-from-attachment': from_fn(get_pose_from_attachment(p)),

        # 'plan-arm-turn-knob-handle': from_fn(
        #     get_turn_knob_handle_motion_gen(p, collisions=c, teleport=t, custom_limits=l)),
        #
        # 'sample-marker-grasp': from_list_fn(get_marker_grasp_gen(p, collisions=c)),
        # 'inverse-kinematics-grasp-marker': from_gen_fn(
        #     get_ik_ir_grasp_handle_gen(p, collisions=True, teleport=t, custom_limits=l,
        #                                learned=False, verbose=False)),
        # 'inverse-kinematics-ungrasp-marker': from_fn(
        #     get_ik_ungrasp_mark_gen(p, collisions=True, teleport=t, custom_limits=l)),
        # 'plan-base-pull-marker-random': from_gen_fn(
        #     get_pull_marker_random_motion_gen(p, collisions=c, teleport=t, custom_limits=l,
        #                                       learned=False)),
        #
        # 'sample-marker-pose': from_list_fn(get_marker_pose_gen(p, collisions=c)),
        # 'plan-base-pull-marker-to-bconf': from_fn(get_pull_marker_to_bconf_motion_gen(p, collisions=c, teleport=t)),
        # 'plan-base-pull-marker-to-pose': from_fn(get_pull_marker_to_pose_motion_gen(p, collisions=c, teleport=t)),
        # 'test-bconf-in-region': from_test(get_bconf_in_region_test(p.robot)),
        # 'test-pose-in-region': from_test(get_pose_in_region_test()),
        # 'test-pose-in-space': from_test(get_pose_in_space_test()),  ##
        #
        # # 'sample-bconf-in-region': from_gen_fn(get_bconf_in_region_gen(p, collisions=c, visualize=False)),
        # 'sample-bconf-in-region': from_list_fn(get_bconf_in_region_gen(p, collisions=c, visualize=False)),
        # 'sample-pose-in-region': from_list_fn(get_pose_in_region_gen(p, collisions=c, visualize=False)),
        #
        # 'update-wconf-p': from_fn(get_update_wconf_p_gen()),
        # 'update-wconf-p-two': from_fn(get_update_wconf_p_two_gen()),
        'update-wconf-pst': from_fn(get_update_wconf_pst_gen()),
        'test-reachable-pose': from_test(get_reachable_test(p, custom_limits=l)),
        # 'update-wconf-pst-for-reachability': from_list_fn(get_sample_wconf_list_gen(p)),

        'MoveCost': move_cost_fn,

        # 'TrajPoseCollision': fn_from_constant(False),
        # 'TrajArmCollision': fn_from_constant(False),
        # 'TrajGraspCollision': fn_from_constant(False),
    }
    return stream_map

from pybullet_tools.pr2_agent import opt_move_cost_fn, opt_pose_fn, opt_ik_fn, opt_ik_wconf_fn, opt_motion_fn, \
    move_cost_fn

# def get_stream_info(partial=False, defer=False):
#     stream_info = {
#         # 'test-cfree-pose-pose': StreamInfo(p_success=1e-3, verbose=verbose),
#         # 'test-cfree-approach-pose': StreamInfo(p_success=1e-2, verbose=verbose),
#         # 'test-cfree-traj-pose': StreamInfo(p_success=1e-1, verbose=verbose),
#
#         'MoveCost': FunctionInfo(opt_move_cost_fn),
#     }
#     stream_info.update({
#         'sample-pose-on': StreamInfo(opt_gen_fn=PartialInputs('?r')),
#         'sample-pose-in': StreamInfo(opt_gen_fn=PartialInputs('?r')),
#         'inverse-kinematics': StreamInfo(opt_gen_fn=PartialInputs('?p')),
#         'plan-base-motion': StreamInfo(opt_gen_fn=PartialInputs('?q1 ?q2'),
#                                       defer_fn=defer_shared if defer else never_defer),
#         'plan-base-motion-wconf': StreamInfo(opt_gen_fn=PartialInputs('?q1 ?q2 ?w'),
#                                       defer_fn=defer_shared if defer else never_defer),
#                        } if partial else {
#         'sample-pose': StreamInfo(opt_gen_fn=from_fn(opt_pose_fn)),
#         'inverse-kinematics': StreamInfo(opt_gen_fn=from_fn(opt_ik_fn)),
#         'plan-base-motion': StreamInfo(opt_gen_fn=from_fn(opt_motion_fn)),
#     })
#     return stream_info
