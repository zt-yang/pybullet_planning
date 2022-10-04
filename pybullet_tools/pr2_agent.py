from __future__ import print_function

import os
import random
import sys
import time
import numpy as np
import types
import json

from pybullet_tools.pr2_streams import get_pull_door_handle_motion_gen as get_pull_drawer_handle_motion_gen
from pybullet_tools.pr2_streams import get_pull_door_handle_motion_gen as get_turn_knob_handle_motion_gen
from pybullet_tools.pr2_streams import get_stable_gen, Position, get_handle_grasp_gen, \
    get_ik_ir_grasp_handle_gen, get_update_wconf_p_gen, get_pose_in_space_test, \
    get_marker_grasp_gen, get_bconf_in_region_test, get_pull_door_handle_motion_gen, \
    get_bconf_in_region_gen, get_pose_in_region_gen, get_motion_wconf_gen, get_update_wconf_p_two_gen, \
    get_marker_pose_gen, get_pull_marker_to_pose_motion_gen, get_pull_marker_to_bconf_motion_gen,  \
    get_pull_marker_random_motion_gen, get_ik_ungrasp_handle_gen, get_pose_in_region_test, \
    get_cfree_btraj_pose_test, get_joint_position_open_gen, get_ik_ungrasp_mark_gen, \
    sample_joint_position_open_list_gen, get_update_wconf_pst_gen, get_ik_ir_wconf_gen, get_ik_gen, get_ik_fn

from pybullet_tools.pr2_primitives import get_group_joints, Conf, get_base_custom_limits, Pose, Conf, \
    get_ik_ir_gen, get_motion_gen, get_cfree_approach_pose_test, get_cfree_pose_pose_test, get_cfree_traj_pose_test, \
    move_cost_fn, Attach, Detach, Clean, Cook, control_commands, \
    get_gripper_joints, GripperCommand, apply_commands, State, Trajectory, Simultaneous, create_trajectory
from pybullet_tools.general_streams import get_grasp_list_gen, get_contain_list_gen
from pybullet_tools.bullet_utils import summarize_facts, print_plan, print_goal, save_pickle, set_camera_target_body, \
    set_camera_target_robot, nice, BASE_LIMITS, get_file_short_name, get_root_links
from pybullet_tools.pr2_problems import create_pr2
from pybullet_tools.pr2_utils import PR2_TOOL_FRAMES, create_gripper, set_group_conf
from pybullet_tools.utils import connect, disconnect, wait_if_gui, LockRenderer, HideOutput, get_client, \
    joint_from_name, WorldSaver, sample_placement, PI, add_parameter, add_button, Pose, Point, Euler, \
    euler_from_quat, get_joint, get_joints, PoseSaver, get_pose, get_link_pose, get_aabb, \
    get_joint_position, aabb_overlap, add_text, remove_handles, get_com_pose, get_closest_points,\
    set_color, RED, YELLOW, GREEN, multiply, get_unit_vector, unit_quat, get_bodies, BROWN, \
    pairwise_collision, connect, get_pose, point_from_pose, set_renderer, \
    disconnect, get_joint_positions, enable_gravity, save_state, restore_state, HideOutput, remove_body, \
    get_distance, LockRenderer, get_min_limit, get_max_limit, has_gui, WorldSaver, wait_if_gui, add_line, SEPARATOR, \
    BROWN, BLUE, WHITE, TAN, GREY, YELLOW, GREEN, BLACK, RED, CLIENTS, wait_unlocked, get_movable_joints, set_all_color, \
    TRANSPARENT, apply_alpha, get_all_links, get_color, get_texture, dump_body, clear_texture, get_link_name, get_joint_name
from pybullet_tools.flying_gripper_utils import get_se3_joints

from os.path import join, isfile
from pddlstream.algorithms.focused import solve_focused
from pddlstream.algorithms.algorithm import parse_problem, reset_globals
from pddlstream.algorithms.constraints import PlanConstraints, WILD
from pddlstream.algorithms.downward import set_cost_scale
from pddlstream.language.generator import from_gen_fn, from_list_fn, from_fn, fn_from_constant, empty_gen, from_test
from pddlstream.language.constants import Equal, AND, PDDLProblem, is_plan
from pddlstream.utils import read, INF, get_file_path, find_unique, Profiler
from pddlstream.language.function import FunctionInfo
from pddlstream.language.stream import StreamInfo, PartialInputs, universe_test
from pddlstream.language.object import SharedOptValue
from pddlstream.language.external import defer_shared, never_defer
from pddlstream.language.conversion import params_from_objects
from collections import namedtuple

from world_builder.entities import Object
from world_builder.actions import get_primitive_actions
from world_builder.world_generator import get_pddl_from_list

def get_stream_map(p, c, l, t, movable_collisions=True, motion_collisions=True,
                   pull_collisions=True, base_collisions=True):
    # p = problem
    # c = collisions
    # l = custom_limits
    # t = teleport
    movable_collisions &= c
    motion_collisions &= c
    base_collisions &= c
    pull_collisions &= c
    # print('\n------------ STREAM MAP -------------')
    # print('Movable collisions:', movable_collisions)
    # print('Motion collisions:', motion_collisions)
    # print('Pull collisions:', pull_collisions)
    # print('Base collisions:', base_collisions)
    # print('Teleport:', t)

    stream_map = {
        'sample-pose': from_gen_fn(get_stable_gen(p, collisions=c)),
        'sample-pose-inside': from_gen_fn(get_contain_list_gen(p, collisions=c, verbose=False)),  ##
        'sample-grasp': from_gen_fn(get_grasp_list_gen(p, collisions=True, visualize=False)), # TODO: collisions

        'inverse-kinematics': from_gen_fn(get_ik_ir_gen(p, collisions=c, teleport=t, custom_limits=l,
                                                        learned=False, max_attempts=60, verbose=False)),
        'inverse-reachability-wconf': from_gen_fn(  ## get_ik_ir_wconf_gen
            get_ik_gen(p, collisions=c, teleport=t, ir_only=True, custom_limits=l, WCONF=True,
                       learned=False, verbose=False, visualize=False)),
        'inverse-kinematics-wconf': from_fn(get_ik_fn(p, collisions=motion_collisions, teleport=t, verbose=False, ACONF=False)),

        'plan-base-motion': from_fn(get_motion_gen(p, collisions=base_collisions, teleport=t, custom_limits=l)),
        'plan-base-motion-wconf': from_fn(get_motion_wconf_gen(p, collisions=base_collisions, teleport=t, custom_limits=l)),

        'test-cfree-pose-pose': from_test(get_cfree_pose_pose_test(collisions=c)),
        'test-cfree-approach-pose': from_test(get_cfree_approach_pose_test(p, collisions=c)),
        'test-cfree-traj-pose': from_test(get_cfree_traj_pose_test(p.robot, collisions=c)),

        'test-cfree-traj-position': from_test(universe_test),

        'test-cfree-btraj-pose': from_test(get_cfree_btraj_pose_test(p.robot, collisions=c)),

        # 'get-joint-position-open': from_fn(get_joint_position_open_gen(p)),
        'get-joint-position-open': from_gen_fn(sample_joint_position_open_list_gen(p)),

        'sample-handle-grasp': from_gen_fn(get_handle_grasp_gen(p, collisions=c)),

        # TODO: apply motion_collisions to pulling?
        'inverse-kinematics-grasp-handle': from_gen_fn(  ## get_ik_ir_grasp_handle_gen
            get_ik_gen(p, collisions=pull_collisions, teleport=t, custom_limits=l,
                        learned=False, verbose=False, ACONF=True, WCONF=False)),
        'inverse-kinematics-ungrasp-handle': from_gen_fn(
            get_ik_ungrasp_handle_gen(p, collisions=pull_collisions, teleport=t, custom_limits=l,
                                      verbose=False, WCONF=False)),
        # 'inverse-kinematics-grasp-handle-wconf': from_gen_fn(
        #     get_ik_ir_grasp_handle_gen(p, collisions=c, teleport=t, custom_limits=l,
        #                                learned=False, verbose=False, ACONF=True, WCONF=True)),
        # 'inverse-kinematics-ungrasp-handle-wconf': from_gen_fn(
        #     get_ik_ungrasp_handle_gen(p, collisions=c, teleport=t, custom_limits=l,
        #                               verbose=False, WCONF=True)),

        'plan-base-pull-drawer-handle': from_fn(  ## get_pull_drawer_handle_motion_gen
            get_pull_door_handle_motion_gen(p, collisions=c, teleport=t, custom_limits=l)),
        'plan-base-pull-door-handle': from_fn(
            get_pull_door_handle_motion_gen(p, collisions=pull_collisions, teleport=t, custom_limits=l)),
        'plan-arm-turn-knob-handle': from_fn(  ## get_turn_knob_handle_motion_gen
            get_pull_door_handle_motion_gen(p, collisions=c, teleport=t, custom_limits=l)),

        'sample-marker-grasp': from_list_fn(get_marker_grasp_gen(p, collisions=c)),
        'inverse-kinematics-grasp-marker': from_gen_fn(
            get_ik_ir_grasp_handle_gen(p, collisions=c, teleport=t, custom_limits=l,
                                       learned=False, verbose=False)),
        'inverse-kinematics-ungrasp-marker': from_fn(
            get_ik_ungrasp_mark_gen(p, collisions=c, teleport=t, custom_limits=l)),
        'plan-base-pull-marker-random': from_gen_fn(
            get_pull_marker_random_motion_gen(p, collisions=c, teleport=t, custom_limits=l,
                                              learned=False)),

        'sample-marker-pose': from_list_fn(get_marker_pose_gen(p, collisions=c)),
        'plan-base-pull-marker-to-bconf': from_fn(get_pull_marker_to_bconf_motion_gen(p, collisions=c, teleport=t)),
        'plan-base-pull-marker-to-pose': from_fn(get_pull_marker_to_pose_motion_gen(p, collisions=c, teleport=t)),
        'test-bconf-in-region': from_test(get_bconf_in_region_test(p.robot)),
        'test-pose-in-region': from_test(get_pose_in_region_test()),
        'test-pose-in-space': from_test(get_pose_in_space_test()),  ##

        # 'sample-bconf-in-region': from_gen_fn(get_bconf_in_region_gen(p, collisions=c, visualize=False)),
        'sample-bconf-in-region': from_list_fn(get_bconf_in_region_gen(p, collisions=c, visualize=False)),
        'sample-pose-in-region': from_list_fn(get_pose_in_region_gen(p, collisions=c, visualize=False)),

        'update-wconf-p': from_fn(get_update_wconf_p_gen()),
        'update-wconf-p-two': from_fn(get_update_wconf_p_two_gen()),
        'update-wconf-pst': from_fn(get_update_wconf_pst_gen()),

        'MoveCost': move_cost_fn,

        # 'TrajPoseCollision': fn_from_constant(False),
        # 'TrajArmCollision': fn_from_constant(False),
        # 'TrajGraspCollision': fn_from_constant(False),
    }

    if not movable_collisions:
        # TODO(caelan): predicate disabling in domain.pddl
        stream_map.update({
            'test-cfree-pose-pose': from_test(universe_test),
            'test-cfree-approach-pose': from_test(universe_test),
            'test-cfree-traj-pose': from_test(universe_test),
            'test-cfree-traj-position': from_test(universe_test),
            'test-cfree-btraj-pose': from_test(universe_test),

            #'test-bconf-in-region': from_test(universe_test),
            #'test-pose-in-region': from_test(universe_test),
            #'test-pose-in-space': from_test(universe_test),
        })

    return stream_map

# def get_stream_info(partial, defer):
#     stream_info = {
#         # 'test-cfree-pose-pose': StreamInfo(p_success=1e-3, verbose=verbose),
#         # 'test-cfree-approach-pose': StreamInfo(p_success=1e-2, verbose=verbose),
#         # 'test-cfree-traj-pose': StreamInfo(p_success=1e-1, verbose=verbose),
#
#         'MoveCost': FunctionInfo(opt_move_cost_fn),
#     }
#     stream_info.update({
#                            'sample-pose': StreamInfo(opt_gen_fn=PartialInputs('?r')),
#                            'inverse-kinematics': StreamInfo(opt_gen_fn=PartialInputs('?p')),
#                            'plan-base-motion': StreamInfo(opt_gen_fn=PartialInputs('?q1 ?q2'),
#                                                           defer_fn=defer_shared if defer else never_defer),
#                        } if partial else {
#         'sample-pose': StreamInfo(opt_gen_fn=from_fn(opt_pose_fn)),
#         'inverse-kinematics': StreamInfo(opt_gen_fn=from_fn(opt_ik_fn)),
#         'plan-base-motion': StreamInfo(opt_gen_fn=from_fn(opt_motion_fn)),
#     })
#     return stream_info


def get_stream_info(unique=False):
    stream_info = {
        # 'test-cfree-pose-pose': StreamInfo(p_success=1e-3, verbose=verbose),
        # 'test-cfree-approach-pose': StreamInfo(p_success=1e-2, verbose=verbose),
        # 'test-cfree-traj-pose': StreamInfo(p_success=1e-1, verbose=verbose),
        'MoveCost': FunctionInfo(opt_move_cost_fn),
    }
    stream_info.update({
        'sample-pose': StreamInfo(opt_gen_fn=from_fn(opt_pose_fn)),
        'sample-pose-inside': StreamInfo(opt_gen_fn=from_fn(opt_pose_inside_fn)),
        'inverse-kinematics': StreamInfo(opt_gen_fn=from_fn(opt_ik_fn)),
        # 'inverse-kinematics-wconf': StreamInfo(opt_gen_fn=from_fn(opt_ik_wconf_fn)),
        'plan-base-motion': StreamInfo(opt_gen_fn=from_fn(opt_motion_fn)),
        # 'plan-base-motion-wconf': StreamInfo(opt_gen_fn=from_fn(opt_motion_wconf_fn)),
        'sample-joint-position': StreamInfo(opt_gen_fn=from_fn(opt_position_fn)),
        # 'inverse-kinematics-grasp-handle': StreamInfo(opt_gen_fn=from_fn(opt_ik_grasp_fn)),
    })

    # TODO: automatically populate using stream_map
    opt_gen_fn = PartialInputs(unique=unique)
    stream_info = {
        'MoveCost': FunctionInfo(opt_fn=opt_move_cost_fn),

        # 'test-cfree-pose-pose': StreamInfo(p_success=1e-3, verbose=verbose),
        # 'test-cfree-approach-pose': StreamInfo(p_success=1e-2, verbose=verbose),
        #'test-cfree-traj-pose': StreamInfo(p_success=1e-1),
        #'test-cfree-approach-pose': StreamInfo(p_success=1e-1),

        'sample-grasp': StreamInfo(opt_gen_fn=opt_gen_fn),
        'sample-handle-grasp': StreamInfo(opt_gen_fn=opt_gen_fn),

        'get-joint-position-open': StreamInfo(opt_gen_fn=opt_gen_fn),
        'sample-joint-position': StreamInfo(opt_gen_fn=opt_gen_fn),
        'update-wconf-pst': StreamInfo(opt_gen_fn=PartialInputs(unique=False)), # TODO(caelan): limited depth

        # TODO: still not re-ordering quite right
        'sample-pose': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e-1),
        'sample-pose-inside': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e-1),

        'inverse-kinematics': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e0),
        'inverse-kinematics-wconf': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e0),

        'inverse-kinematics-grasp-handle': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e0),
        'inverse-kinematics-ungrasp-handle': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e0),
        'plan-base-pull-door-handle': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e0),

        'plan-base-motion': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e1),
        'plan-base-motion-wconf': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e1),
    }

    return stream_info


#######################################################

CustomValue = namedtuple('CustomValue', ['stream', 'values'])

def opt_move_cost_fn(t):
    # q1, q2 = t.values
    # distance = get_distance(extract_point2d(q1), extract_point2d(q2))
    #return BASE_CONSTANT + distance / BASE_VELOCITY
    return 1

def opt_pose_fn(o, r):
    p = CustomValue('p-sp', (r,))
    return p,

def opt_ik_fn(a, o, p, g):
    q = CustomValue('q-ik', (p,))
    t = CustomValue('t-ik', tuple())
    return q, t

def opt_ik_wconf_fn(a, o, p, g, w):
    return opt_ik_fn(a, o, p, g)

def opt_motion_fn(q1, q2):
    t = CustomValue('t-pbm', (q1, q2))
    return t,

def opt_pose_inside_fn(o, r):
    p = CustomValue('p-spi', (r,))
    return p,

def opt_position_fn(o, r):
    p = CustomValue('pstn-end', (r,))
    return p,

def opt_ik_grasp_fn(a, o, p, g):
    q = CustomValue('q-ik-hg', (p,))
    aq = CustomValue('aq-ik-hg', (p,))
    t = CustomValue('t-ik-hg', tuple())
    return q, aq, t

def opt_ik_wconf_fn(a, o, p, g, w):
    q = CustomValue('q-ik', (p,))
    t = CustomValue('t-ik', tuple())
    return q, t


def opt_motion_wconf_fn(q1, q2, w):
    t = CustomValue('t-pbm', (q1, q2))
    return t,


#######################################################

def post_process(problem, plan, teleport=False, use_commands=False, verbose=False):
    if plan is None:
        return None
    commands = []
    for i, action in enumerate(plan):
        if use_commands:
            new_commands = get_primitive_commands(action, problem.robot, teleport)
        else:
            new_commands = get_primitive_actions(action, problem.world, teleport)
        commands += new_commands
        if verbose:
            print(i, action)
    return commands

def get_close_command(robot, arm, grasp, teleport=False):
    return GripperCommand(robot, arm, grasp.grasp_width, teleport=teleport)

def get_open_command(robot, arm, teleport=False):
    gripper_joint = get_gripper_joints(robot, arm)[0]
    position = get_max_limit(robot, gripper_joint)
    return GripperCommand(robot, arm, position, teleport=teleport)

def get_primitive_commands(action, robot, teleport=False):
    name, args = action
    if name in ['move_base']: #, 'move_base_wconf']:
        c = args[-1]
        new_commands = c.commands
    elif name == 'move_base_wconf':
        q1, q2, c, w = args
        new_commands = c.commands
    elif name == 'pick':
        a, b, p, g, _, c = args[:6]
        [t] = c.commands
        close_gripper = get_close_command(robot, a, g, teleport=teleport)
        attach = Attach(robot, a, g, b)
        new_commands = [t, close_gripper, attach, t.reverse()]
    elif name == 'place':
        a, b, p, g, _, c = args[:6]
        [t] = c.commands
        open_gripper = get_open_command(robot, a, teleport=teleport)
        detach = Detach(robot, a, b)
        new_commands = [t, detach, open_gripper, t.reverse()]
    elif name in 'grasp_handle':
        a, o, p, g, q, aq1, aq2, c = args
        close_gripper = get_close_command(robot, a, g, teleport=teleport)
        new_commands = list(c.commands) + [close_gripper]
    elif name in 'ungrasp_handle':
        a, o, p, g, q, aq1, aq2, c = args
        open_gripper = get_open_command(robot, a, teleport=teleport)
        new_commands = list(c.reverse().commands) + [open_gripper]
    elif name == 'pull_door_handle':
        a, o, p1, p2, g, q1, q2, bt, aq1, aq2, at = args
        #new_commands = at.commands
        #new_commands = bt.commands
        dt = create_trajectory(robot=p1.body, joints=[p1.joint],
                               path=np.linspace([p1.value], [p2.value], num=len(bt.commands[0].path), endpoint=True))
        new_commands = [Simultaneous(commands=[bt, at, dt])]
    elif name == 'clean':  # TODO: add text or change color?
        body, sink = args
        new_commands = [Clean(body)]
    elif name == 'cook':
        body, stove = args
        new_commands = [Cook(body)]
    elif name == 'press_clean':
        body, sink, arm, button, bq, c = args
        [t] = c.commands
        new_commands = [t, Clean(body), t.reverse()]
    elif name == 'press_cook':
        body, sink, arm, button, bq, c = args
        [t] = c.commands
        new_commands = [t, Cook(body), t.reverse()]
    else:
        raise ValueError(name)

    return new_commands

#######################################################

def place_movable(certified):
    for literal in certified:
        if literal[0] != 'not':
            continue
        fact = literal[1]
        if fact[0] == 'trajposecollision':
            _, b, p = fact[1:]
            p.assign()
        if fact[0] == 'trajarmcollision':
            _, a, q = fact[1:]
            q.assign()
        if fact[0] == 'trajgraspcollision':
            _, a, o, g = fact[1:]
            # TODO: finish this

def move_cost_fn(c):
    [t] = c.commands
    distance = t.distance(distance_fn=lambda q1, q2: get_distance(q1[:2], q2[:2]))
    #return BASE_CONSTANT + distance / BASE_VELOCITY
    return 1

#######################################################

def extract_point2d(v):
    if isinstance(v, Conf):
        return v.values[:2]
    if isinstance(v, Pose):
        return point_from_pose(v.value)[:2]
    if isinstance(v, SharedOptValue):
        if v.stream == 'sample-pose':
            r, = v.values
            return point_from_pose(get_pose(r))[:2]
        if v.stream == 'inverse-kinematics':
            p, = v.values
            return extract_point2d(p)
    if isinstance(v, CustomValue):
        if v.stream == 'p-sp':
            r, = v.values
            return point_from_pose(get_pose(r))[:2]
        if v.stream == 'q-ik':
            p, = v.values
            return extract_point2d(p)
    raise ValueError(v.stream)

def opt_move_cost_fn(t):
    # q1, q2 = t.values
    # distance = get_distance(extract_point2d(q1), extract_point2d(q2))
    #return BASE_CONSTANT + distance / BASE_VELOCITY
    return 1

#######################################################

CustomValue = namedtuple('CustomValue', ['stream', 'values'])

def opt_pose_fn(o, r):
    p = CustomValue('p-sp', (r,))
    return p,

def opt_ik_fn(a, o, p, g):
    q = CustomValue('q-ik', (p,))
    t = CustomValue('t-ik', tuple())
    return q, t

def opt_motion_fn(q1, q2):
    t = CustomValue('t-pbm', (q1, q2))
    return t,

def opt_pose_inside_fn(o, r):
    p = CustomValue('p-spi', (r,))
    return p,

def opt_position_fn(o, r):
    p = CustomValue('pstn-end', (r,))
    return p,

def opt_ik_grasp_fn(a, o, p, g):
    q = CustomValue('q-ik-hg', (p,))
    aq = CustomValue('aq-ik-hg', (p,))
    t = CustomValue('t-ik-hg', tuple())
    return q, aq, t

def opt_ik_wconf_fn(a, o, p, g, w):
    q = CustomValue('q-ik', (p,))
    t = CustomValue('t-ik', tuple())
    return q, t

def opt_motion_wconf_fn(q1, q2, w):
    t = CustomValue('t-pbm', (q1, q2))
    return t,

#######################################################

class Problem(object):
    def __init__(self, robot, movable=tuple(), openable=tuple(),
                 surfaces=tuple(), spaces=tuple(), floors=tuple(),
                 grasp_types=tuple(['top']), arms=tuple(['left']),  ## 'side',
                 costs=False, body_names={}, body_types=[], base_limits=None):
        self.robot = robot
        self.arms = arms
        self.movable = movable
        self.openable = openable
        self.grasp_types = grasp_types
        self.surfaces = surfaces
        self.spaces = spaces
        self.floors = floors

        # self.sinks = sinks
        # self.stoves = stoves
        # self.buttons = buttons

        # self.goal_conf = goal_conf
        # self.goal_holding = goal_holding
        # self.goal_on = goal_on
        # self.goal_cleaned = goal_cleaned
        # self.goal_cooked = goal_cooked
        self.costs = costs
        self.body_names = body_names
        self.body_types = body_types
        self.base_limits = base_limits
        all_movable = [self.robot] + list(self.movable)
        self.fixed = list(filter(lambda b: b not in all_movable, get_bodies()))
        self.gripper = None
    def get_gripper(self, arm='left', visual=True):
        # upper = get_max_limit(problem.robot, get_gripper_joints(problem.robot, 'left')[0])
        # set_configuration(gripper, [0]*4)
        # dump_body(gripper)
        if self.gripper is None:
            self.gripper = create_gripper(self.robot, arm=arm, visual=visual)
        return self.gripper
    def remove_gripper(self):
        if self.gripper is not None:
            remove_body(self.gripper)
            self.gripper = None
    def __repr__(self):
        return repr(self.__dict__)

#######################################################

def pddlstream_from_state_goal(state, goals, domain_pddl='pr2_kitchen.pddl',
                               stream_pddl='pr2_stream.pddl',
                               custom_limits=BASE_LIMITS,
                               init_facts=[], ## avoid duplicates
                               facts=[],  ## completely overwrite
                               collisions=True, teleport=False, PRINT=True, **kwargs):
    from pybullet_tools.logging import myprint as print

    robot = state.robot
    world = state.world
    problem = state
    if not isinstance(custom_limits, dict):
        custom_limits = get_base_custom_limits(robot, custom_limits)

    world.summarize_all_objects()

    if len(facts) == 0:
        facts = state.get_facts(init_facts)
    init = facts

    print(f'pr2_agent.pddlstream_from_state_goal(\n'
          f'\tdomain = {domain_pddl}, \n'
          f'\tstream = {stream_pddl}, \n'
          f'\tcustom_limits = {custom_limits}')

    if isinstance(goals, tuple): ## debugging
        test, name = goals
        # test_initial_region(state, init)
        # test_marker_pull_bconfs(state, init)
        if test == 'test_handle_grasps':
            goals = test_handle_grasps(state, name)
        elif test == 'test_grasps':
            goals = test_grasps(state, name)
        elif test == 'test_grasp_ik':
            goals = test_grasp_ik(state, init, name)
        # test_pulling_handle_ik(state)
        # test_drawer_open(state, goals)
        elif test == 'test_pose_gen':
            goals, ff = test_pose_gen(state, init, name[0], name[1])
            init += ff
        elif test == 'test_update_wconf_pst':
            goals, ff = test_update_wconf_pst(state, init, name)
            init += ff
        elif test == 'test_door_pull_traj':
            goals = test_door_pull_traj(state, init, name)
        elif test == 'test_reachable_pose':
            goals = test_reachable_pose(state, init, name)
        elif test == 'test_sample_wconf':
            goals, ff = test_sample_wconf(state, init, name)
            init += ff
        elif test == 'test_at_reachable_pose':
            goals = test_at_reachable_pose(init, name)
        elif test == 'test_new_wconf':
            goals = test_new_wconf(init, name)
        else:
            print('\n\n\npr2_agent.pddlstream_from_state_goal | didnt implement', goals)
            sys.exit()

    goal = [AND]
    goal += goals
    if len(goals) > 0:
        if goals[0][0] == 'AtBConf':
            init += [('BConf', goals[0][1])]
        elif goals[0][0] == 'AtSEConf':
            init += [('SEConf', goals[0][1])]
        elif goals[0][0] == 'AtPosition':
            init += [('Position', goals[0][1], goals[0][2]), ('IsOpenedPosition', goals[0][1], goals[0][2])]
        elif goals[0][0] == 'AtGrasp':
            init += [('Grasp', goals[0][2], goals[0][3])]
        elif goals[0][0] == 'AtHandleGrasp':
            init += [('HandleGrasp', goals[0][2], goals[0][3])]
        elif goals[0][0] == 'AtMarkerGrasp':
            init += [('MarkerGrasp', goals[0][2], goals[0][3])]

        if goal[-1] == ("not", ("AtBConf", "")):
            atbconf = [i for i in init if i[0].lower() == "AtBConf".lower()][0]
            goal[-1] = ("not", atbconf)

    if PRINT:
        summarize_facts(init, world, name='Facts extracted from observation')

    ## make all pred lower case
    new_init = []
    for fact in init:
        new_tup = [fact[0].lower()]
        new_tup.extend(fact[1:])
        new_init.append(tuple(new_tup))
    init = new_init

    init_added = [n for n in init_facts if n not in init]
    if len(init_facts) != 0:  ## only print the world facts the first time
        summarize_facts(init_added, world, name='Added facts from PDDLStream preimage')
        init = init + init_added

    domain_pddl = read(domain_pddl)
    stream_pddl = read(stream_pddl)
    constant_map = {k: k for k in state.constants}
    goal = [g for g in goal if not (g[0] == 'not' and g[1][0] == '=')]

    if PRINT:
        print_goal(goal, world=world)

    stream_map = robot.get_stream_map(problem, collisions, custom_limits, teleport, **kwargs)
    # get_press_gen(problem, teleport=teleport)
    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)


def is_plan_abstract(plan):
    for step in plan:
        if '--no-' in step.name:
            return True
    return False


from pddlstream.algorithms.meta import solve, DEFAULT_ALGORITHM
from pybullet_tools.utils import disconnect, LockRenderer, has_gui, WorldSaver, wait_if_gui, \
    SEPARATOR, get_aabb, wait_for_duration, safe_remove, ensure_dir, reset_simulation
from pddlstream.utils import read, INF, get_file_path, find_unique, Profiler, str_from_object, TmpCWD


def solve_one(pddlstream_problem, stream_info, fc=None, diverse=False, lock=False,
              max_time=INF, downward_time=10, evaluation_time=10, max_plans=100,
              visualize=False):
    if diverse:
        # max_plans = 200 ## 100
        plan_dataset = []
        max_skeletons = 1
        use_feedback = False
    else:
        max_plans = 1
        plan_dataset = None
        max_skeletons = INF
        use_feedback = True

    # with Profiler():
    set_cost_scale(cost_scale=1)
    with LockRenderer(lock=lock):
        solution = solve_focused(pddlstream_problem, stream_info=stream_info,
                                 planner='ff-astar1', max_planner_time=downward_time, debug=False,
                                 unit_costs=False, success_cost=INF, initial_complexity=5,
                                 max_time=max_time, verbose=True, visualize=visualize,
                                 unit_efforts=True, effort_weight=None,
                                 unique_optimistic=True, use_feedback=use_feedback,
                                 forbid=True, max_plans=max_plans, fc=fc,
                                 bind=True, max_skeletons=max_skeletons,
                                 plan_dataset=plan_dataset, evaluation_time=evaluation_time, max_solutions=1,
                                 search_sample_ratio=0)
        # solution = solve(pddlstream_problem, algorithm=DEFAULT_ALGORITHM, unit_costs=False,
        #                  stream_info=stream_info, success_cost=INF, verbose=True, debug=False,
        #                  visualize=visualize, feasibility_checker=fc)
    return solution


def solve_multiple(pddlstream_problem, stream_info, lock=True, **kwargs):
    reset_globals()
    # profiler = Profiler(field='tottime', num=25) ## , enable=profile # cumtime | tottime
    # profiler.save()

    temp_dir = '/tmp/pddlstream-{}-{}/'.format(os.getpid(), int(time.time()))
    print(f'\n\n\n\nsolve_multiple at temp dir {temp_dir} \n\n\n\n')
    safe_remove(temp_dir)
    ensure_dir(temp_dir)
    cwd_saver = TmpCWD(temp_cwd=temp_dir)  # TODO: multithread
    cwd_saver.save()  # TODO: move to the constructor
    lock_saver = LockRenderer(lock=lock)

    solution = solve_one(pddlstream_problem, stream_info, lock=lock, **kwargs)

    # profiler.restore()
    return solution, join(cwd_saver.tmp_cwd, 'visualizations')


def get_named_colors(kind='tablaeu', alpha=1.):
    # from matplotlib._color_data import BASE_COLORS, TABLEAU_COLORS, XKCD_COLORS, CSS4_COLORS
    # from pybullet_planning.pybullet_tools.utils import CHROMATIC_COLORS, COLOR_FROM_NAME
    # TODO: colors.py from past projects
    from matplotlib.colors import BASE_COLORS, TABLEAU_COLORS, XKCD_COLORS, CSS4_COLORS, to_rgba
    from pybullet_planning.pybullet_tools.utils import RGBA, apply_alpha
    if kind == 'base':
        return {name: apply_alpha(rgb, alpha=alpha) for name, rgb in BASE_COLORS.items()} # TODO: single character
    elif kind == 'tablaeu':
        colors = TABLEAU_COLORS
    elif kind == 'xkcd':
        colors = XKCD_COLORS
    elif kind == 'css4':
        colors = CSS4_COLORS
    else:
        raise NotImplementedError(kind)
    return {name.split(':')[-1].replace(' ', '-').lower(): RGBA(*to_rgba(hex, alpha=alpha))
            for name, hex in colors.items()}


def colorize_world(world, color_types=['brown', 'tan'], transparency=0.5):
    named_colors = get_named_colors(kind='xkcd')
    colors = [color for name, color in named_colors.items()
              if any(color_type in name for color_type in color_types)] # TODO: convex combination
    for body in world.fixed:
        joints = get_movable_joints(body)
        if not joints:
            continue
        # dump_body(body)
        #body_color = apply_alpha(WHITE, alpha=0.5)
        #body_color = random.choice(colors)
        body_color = apply_alpha(0.9*np.ones(3))
        links = get_all_links(body)
        rigid = get_root_links(body)

        #links = set(links) - set(rigid)
        for link in links:
            #print('Body: {} | Link: {} | Joints: {}'.format(body, link, joints))
            #print(get_color(body, link=link))
            #print(get_texture(body, link=link))
            #clear_texture(body, link=link)
            #link_color = body_color
            link_color = np.array(body_color) + np.random.normal(0, 1e-2, 4) # TODO: clip
            link_color = apply_alpha(link_color, alpha=1.0 if link in rigid else transparency)
            set_color(body, link=link, color=link_color)


def serial_checker(checker):
    return lambda plans: list(map(checker, plans))


def get_debug_checker(world):
    def debug_checker(plans):
        for i, plan in enumerate(plans):
            renamed_plan = []
            for action in plan:
                action_name, args = action
                args = params_from_objects(args)
                body, joint = None, None
                if action_name == 'move_base_wconf':
                    continue
                elif action_name == 'pick':
                    arm, body = args[:2]
                elif action_name == 'place':
                    arm, body = args[:2]
                elif action_name == 'grasp_handle':
                    arm, (body, joint) = args[:2]
                    continue
                elif action_name == 'pull_door_handle':
                    arm, (body, joint) = args[:2]
                elif action_name == 'ungrasp_handle':
                    arm, (body, joint) = args[:2]
                    continue
                else:
                    raise NotImplementedError(action_name)
                action_name = action_name.split('_')[0]
                obj = world.get_object(body)
                obj_name = obj.name
                if joint is None:
                    instance_name = f'{action_name}({obj_name})'
                else:
                    joint_name = get_joint_name(body, joint) if joint is not None else None
                    instance_name = f'{action_name}({obj_name}, {joint_name})'
                renamed_plan.append(instance_name)
            plan_name = f'[{", ".join(renamed_plan)}]'
            print(f'{i+1}/{len(plans)}) {plan_name} ({len(renamed_plan)})')
        wait_unlocked()
        return [True]*len(plans)
    return debug_checker


def solve_pddlstream(pddlstream_problem, state, domain_pddl=None, visualization=False,
                     collect_dataset=False, max_cost=INF,
                     profile=True, lock=False, max_time=5*50, preview=False, **kwargs):
    # from examples.pybullet.utils.pybullet_tools.utils import CLIENTS
    from pybullet_tools.logging import myprint as print

    start_time = time.time()

    CLIENTS[get_client()] = True # TODO: hack
    saver = WorldSaver()
    world = state.world
    objects = world.objects
    print(f'Robot: {world.robot} | Objects: {world.objects}\n'
          f'Movable: {world.movable} | Fixed: {world.fixed} | Floor: {world.floors}')
    stream_info = world.robot.get_stream_info()

    colorize_world(world)
    #wait_unlocked()

    #########################

    print(SEPARATOR)

    plan_dataset = None
    if collect_dataset:
        #plan_dataset = PlanDataset()
        plan_dataset = []

    # TODO: more general version will just sort and return a subset the plans
    feasibility_checker = None
    # feasibility_checker = lambda *args: False # Reject all
    # feasibility_checker = lambda *args: True # Accept all
    # feasibility_checker = lambda *args: np.random.random() # Randomize

    skeleton = [
        ('grasp_handle', [WILD, WILD, WILD, WILD, WILD, WILD, WILD, WILD]),
        ('pull_door_handle', [WILD, WILD, WILD, WILD, WILD, WILD, WILD, WILD, WILD, WILD, WILD]),
        ('ungrasp_handle', [WILD, WILD, WILD, WILD, WILD, WILD, WILD, WILD]),
        ('pick', [WILD, WILD, WILD, WILD, WILD, WILD, WILD]),
        ('place', [WILD, WILD, WILD, WILD, WILD, WILD, WILD]),
    ]
    #constraints = PlanConstraints(skeletons=[skeleton], exact=False, max_cost=max_cost + 1)
    constraints = PlanConstraints(max_cost=max_cost + 1) # TODO: plus 1 in action costs?

    if collect_dataset:
        max_plans, max_planner_time, max_skeletons = 100, 10, 1
    else:
        max_plans, max_planner_time, max_skeletons = 1, 10, INF

    max_solutions = 6
    evaluation_time = 120

    # profiler = Profiler(field='cumtime' if profile else None, num=25) # cumtime | tottime
    # profiler.save()
    if True:
        with LockRenderer(lock=lock):
            # solution = solve(pddlstream_problem, algorithm='adaptive', unit_costs=True, visualize=False,
            #                  stream_info=stream_info, success_cost=INF, verbose=True, debug=False)
            solution = solve_focused(pddlstream_problem, stream_info=stream_info, constraints=constraints,
                                     planner='ff-astar1', max_planner_time=max_planner_time,
                                     debug=False,
                                     initial_complexity=5,
                                     unit_costs=False, success_cost=INF,
                                     max_time=max_time, verbose=True, visualize=visualization,
                                     #unit_efforts=True, effort_weight=1,
                                     unit_efforts=True, effort_weight=None,
                                     bind=True,
                                     unique_optimistic=True, # NOTE(caelan): cannot use update-wconf-pst
                                     use_feedback=True, # plan_dataset
                                     forbid=True,
                                     max_plans=max_plans, max_skeletons=max_skeletons,
                                     fc=feasibility_checker,
                                     plan_dataset=plan_dataset, evaluation_time=evaluation_time,
                                     max_solutions=max_solutions,
                                     search_sample_ratio=0, **kwargs)

    else:
        solution = solve_one(pddlstream_problem, stream_info, fc=feasibility_checker, visualize=visualization)
    saver.restore()
    # profiler.restore()

    if plan_dataset is not None:
        from mamao_tools.data_utils import get_plan_skeleton
        indices = world.get_indices()
        solutions_log = []
        for i, (opt_solution, real_solution) in enumerate(plan_dataset):
            stream_plan, (opt_plan, preimage), opt_cost = opt_solution
            plan = None
            if real_solution is not None:
                plan, cost, certificate = real_solution
                solution = real_solution # TODO: first solution
            skeleton = get_plan_skeleton(opt_plan, indices=indices)
            print(f'\n{i+1}/{len(plan_dataset)}) Optimistic Plan: {opt_plan}\n'
                  f'Skeleton: {skeleton}\nPlan: {plan}')
            log = {
                'optimistic_plan': str(opt_plan),
                'skeleton': str(skeleton),
                'plan': [str(a) for a in plan] if plan is not None else None,
            }
            solutions_log.append(log)
        with open('multiple_solutions.json', 'w') as f:
            json.dump(solutions_log, f, indent=3)

    # PARALLEL = True
    #
    # if PARALLEL:
    #     solution, log_dir = solve_multiple(pddlstream_problem, stream_info, visualization)
    # else:
    #     solution = solve_one(pddlstream_problem, stream_info, visualization)
    #     log_dir = 'visualizations'

    knowledge = parse_problem(pddlstream_problem, stream_info=stream_info,
                              constraints=PlanConstraints(), unit_costs=True, unit_efforts=True)

    plan, cost, evaluations = solution
    solved = is_plan(plan)

    print('Solved: {}'.format(solved))
    print('Cost: {:.3f}'.format(cost))
    print_plan(plan, world)

    time_log = {'planning': round(time.time()-start_time, 4)}
    start_time = time.time()
    if plan != None:
        preimage = evaluations.preimage_facts

        ## ------ debug why can't find action skeleton
        ## test_grasp_ik(state, state.get_facts()+preimage, name='eggblock')

        ## save_pickle(pddlstream_problem, plan, preimage) ## discarded

        summarize_facts(preimage, world, name='Preimage generated by PDDLStream')

        if is_plan_abstract(plan):
            from bullet.leap.hierarchical import check_preimage
            env = check_preimage(pddlstream_problem, plan, preimage, init=pddlstream_problem.init,
                                 objects=objects, domain_pddl=domain_pddl)
        else:
            env = None  ## preimage
        plan_str = [str(a) for a in plan]
    else:
        env = None
        plan_str = 'FAILED'
        preimage = []

    time_log['preimage'] = round(time.time() - start_time, 4)
    time_log['goal'] = [f'{g[0]}({g[1:]})' for g in pddlstream_problem.goal[1:]]
    time_log['plan'] = plan_str
    time_log['plan_len'] = len(plan) if plan != None else 0

    ## for collecting data
    if visualization:
        time_log['init'] = [[str(a) for a in f] for f in preimage]

    if preview:
        from lisdf_tools.lisdf_planning import Problem as LISDFProblem
        state.assign()
        lisdf_problem = LISDFProblem(world)
        commands = post_process(lisdf_problem, plan, use_commands=False)

        state.assign()
        wait_if_gui('Begin?')
        from world_builder.actions import apply_actions
        apply_actions(lisdf_problem, commands, time_step=5e-2)
        wait_if_gui('Finish?')
        state.assign()

    reset_globals()  ## reset PDDLStream solutions

    return plan, env, knowledge, time_log, preimage ##, log_dir


def test_initial_region(state, init):
    world = state.world
    if world.name_to_body('hallway') != None:
        robot = state.robot
        marker = world.name_to_body('marker')
        funk = get_bconf_in_region_test(robot)
        funk2 = get_pose_in_region_test(robot)
        bq = [i for i in init if i[0].lower() == "AtBConf".lower()][0][-1]
        p = [i for i in init if i[0].lower() == "AtPose".lower() and i[1] == marker][0][-1]

        for location in ['hallway', 'storage', 'kitchen']:
            answer1 = funk(bq, world.name_to_body(location))
            answer2 = funk2(marker, p, world.name_to_body(location))
            print(f"RobInRoom({location}) = {answer1}\tInRoom({marker}, {location}) = {answer2}")
        print('---------------\n')


def test_marker_pull_bconfs(state, init):
    world = state.world
    funk = get_pull_marker_random_motion_gen(state)  ## is a from_fn
    o = world.name_to_body('marker')
    p1 = [i for i in init if i[0].lower() == "AtPose".lower() and i[1] == o][0][2]
    g = test_marker_pull_grasps(state, o)
    bq1 = [i for i in init if i[0].lower() == "AtBConf".lower()][0][1]
    p2, bq2, t = funk('left', o, p1, g, bq1)
    rbb = create_pr2()
    set_group_conf(rbb, 'base', bq2.values)


def test_marker_pull_grasps(state, marker, visualize=False):
    funk = get_marker_grasp_gen(state)
    grasps = funk(marker) ## funk(cart) ## is a previous version
    if visualize:
        robot = state.robot
        cart = state.world.BODY_TO_OBJECT[marker].grasp_parent
        for grasp in grasps:
            gripper_grasp = visualize_grasp(robot, get_pose(marker), grasp[0].value)
            set_camera_target_body(gripper_grasp, dx=0, dy=-1, dz=0)
            print('collision with marker', pairwise_collision(gripper_grasp, marker))
            print('collision with cart', pairwise_collision(gripper_grasp, cart))
            remove_body(gripper_grasp)
    print('test_marker_pull_grasps:', grasps)
    return grasps

def test_handle_grasps(state, name='hitman_drawer_top_joint', visualize=False, verbose=False):
    if isinstance(name, str):
        body_joint = state.world.name_to_body(name)
    else: ##if isinstance(name, Object):
        body_joint = name
        name = state.world.BODY_TO_OBJECT[body_joint].shorter_name

    funk = get_handle_grasp_gen(state, visualize=visualize, verbose=verbose)
    outputs = funk(body_joint)
    if visualize:
        name_to_object = state.world.name_to_object
        body_pose = name_to_object(name).get_handle_pose()
        visualize_grasps_by_quat(state, outputs, body_pose, verbose=verbose)
    print(f'test_handle_grasps ({len(outputs)}): {outputs}')
    arm = state.robot.arms[0]
    goals = [("AtHandleGrasp", arm, body_joint, outputs[0][0])]
    return goals

def test_grasps(state, name='cabbage', visualize=True):
    title = 'pr2_agent.test_grasps | '
    if isinstance(name, str):
        body = state.world.name_to_body(name)
    else: ## if isinstance(name, Object):
        body = name
    robot = state.robot

    funk = get_grasp_list_gen(state, visualize=False, RETAIN_ALL=False)
    outputs = funk(body)

    if 'left' in robot.joint_groups:
        if visualize:
            body_pose = get_pose(body)
            visualize_grasps(state, outputs, body_pose)
        print(f'{title}grasps:', outputs)
        goals = [("AtGrasp", 'left', body, outputs[1][0])]

    elif 'hand' in robot.joint_groups:
        from pybullet_tools.bullet_utils import collided

        g = outputs[0][0]
        gripper = robot.visualize_grasp(g.body, g.approach, width=g.grasp_width)
        gripper_conf = get_cloned_se3_conf(robot, gripper)
        goals = [("AtGrasp", 'hand', body, g)]
        if visualize:
            set_renderer(True)
            goals = [("AtSEConf", Conf(robot, get_se3_joints(robot), gripper_conf))]
            # body_collided = len(get_closest_points(gripper, 2)) != 0
            # body_collided_2 = collided(gripper, [2], world=state.world, verbose=True, tag=title)
            # print(f'{title}collision between gripper {gripper} and object {g.body}: {body_collided}')
        else:
            remove_body(gripper)
    return goals


def visualize_grasps(state, outputs, body_pose, RETAIN_ALL=False):
    robot = state.robot
    colors = [BROWN, BLUE, WHITE, TAN, GREY, YELLOW, GREEN, BLACK, RED]

    def visualize_grasp(grasp, index=0):
        w = grasp.grasp_width
        if RETAIN_ALL:
            gripper_grasp = robot.visualize_grasp(body_pose, grasp.value, body=grasp.body,
                                                  color=colors[i%len(colors)], width=w)
            # set_camera_target_body(gripper_grasp, dx=0.5, dy=0.5, dz=0.5)
        else:
            gripper_grasp = robot.visualize_grasp(body_pose, grasp.value, body=grasp.body, color=GREEN, width=w)
            gripper_approach = robot.visualize_grasp(body_pose, grasp.approach, color=BROWN)
            # set_camera_target_body(gripper_approach, dx=0, dy=-1, dz=0)
            remove_body(gripper_grasp)
            remove_body(gripper_approach)

        return gripper_grasp

    # if not isinstance(outputs, types.GeneratorType):
    #     for i in range(len(outputs)):
    #         visualize_grasp(outputs[i][0], index=i)
    # else:

    i = 0
    gripper_grasp = 0
    for grasp in outputs:
        gripper_grasp = visualize_grasp(grasp[0], index=i)
        i += 1
    set_camera_target_body(gripper_grasp, dx=0.5, dy=0.5, dz=0.5)

    # if RETAIN_ALL:
    #     wait_if_gui()

def visualize_grasps_by_quat(state, outputs, body_pose, verbose=False):
    robot = state.robot
    colors = [BROWN, BLUE, WHITE, TAN, GREY, YELLOW, GREEN, BLACK, RED]
    all_grasps = {}
    for i in range(len(outputs)):
        grasp = outputs[i][0]
        quat = nice(grasp.value[1])
        if quat not in all_grasps:
            all_grasps[quat] = []
        all_grasps[quat].append(grasp)

    j = 0
    set_renderer(True)
    for k, v in all_grasps.items():
        print(f'{len(v)} grasps of quat {k}')
        visuals = []
        color = colors[j%len(colors)]
        for grasp in v:
            gripper = robot.visualize_grasp(body_pose, grasp.value, color=color, verbose=verbose,
                                            width=grasp.grasp_width, body=grasp.body)
            visuals.append(gripper)
        j += 1
        wait_if_gui()
        # set_camera_target_body(gripper, dx=0, dy=0, dz=1)
        for visual in visuals:
            remove_body(visual)

from pybullet_tools.flying_gripper_utils import get_cloned_se3_conf, se3_ik
def test_grasp_ik(state, init, name='cabbage', visualize=True):
    goals = test_grasps(state, name, visualize=False)
    body, grasp = goals[0][-2:]
    robot = state.robot
    custom_limits = robot.get_custom_limits()
    pose = [i for i in init if i[0].lower() == "AtPose".lower() and i[1] == body][0][-1]
    wconfs = [i for i in init if "NewWConf".lower() in i[0].lower()]

    if len(wconfs) == 0:
        funk = get_ik_ir_gen(state, verbose=visualize, custom_limits=custom_limits
                             )('left', body, pose, grasp)
        print('test_grasp_ik', body, pose, grasp)
    else:
        wconf = wconfs[0][-1]
        print('test_grasp_ik', body, pose, grasp, wconf)
        wconf.printout()
        funk = get_ik_ir_wconf_gen(state, verbose=visualize, custom_limits=custom_limits
                                   )('left', body, pose, grasp, wconf)
    next(funk)

    return goals

def test_pulling_handle_ik(problem):
    name_to_body = problem.name_to_body

    # ## --------- test pulling handle ik
    # for bq in [(1.583, 6.732, 0.848), (1.379, 7.698, -2.74), (1.564, 7.096, 2.781)]:
    #     custom_limits = get_base_custom_limits(problem.robot, BASE_LIMITS)
    #     robot = problem.robot
    #     drawer = name_to_body('hitman_drawer_top_joint')
    #     funk = get_handle_grasp_gen(problem)
    #     grasp = funk(drawer)[0][0]
    #     funk = get_pull_handle_motion_gen(problem)
    #     position1 = Position(name_to_body('hitman_drawer_top_joint'))
    #     position2 = Position(name_to_body('hitman_drawer_top_joint'), 'max')
    #     bq1 = Conf(robot, get_group_joints(robot, 'base'), bq)
    #     t = funk('left', drawer, position1, position2, grasp, bq1)

    ## --------- test modified pulling handle ik
    for bq in [(1.583, 6.732, 0.848), (1.379, 7.698, -2.74), (1.564, 7.096, 2.781)]:
        custom_limits = get_base_custom_limits(problem.robot, BASE_LIMITS)
        robot = problem.robot
        drawer = name_to_body('hitman_drawer_top_joint')
        funk = get_handle_grasp_gen(problem)
        grasp = funk(drawer)[0][0]
        funk = get_pull_drawer_handle_motion_gen(problem, extent='max')
        position1 = Position(name_to_body('hitman_drawer_top_joint'))
        position2 = Position(name_to_body('hitman_drawer_top_joint'), 'max')
        bq1 = Conf(robot, get_group_joints(robot, 'base'), bq)
        t = funk('left', drawer, grasp, bq1)
        break

def test_pose_gen(problem, init, o, s):
    pose = [i for i in init if i[0].lower() == "AtPose".lower() and i[1] == o][0][-1]
    if isinstance(o, Object):
        o = o.body
    funk = get_stable_gen(problem)(o, s)
    p = next(funk)[0]
    print(f'test_pose_gen({o}, {s}) | {p}')
    pose.assign()
    return [('AtPose', o, p)], [('Pose', o, p), ('Supported', o, p, s)]

def test_update_wconf_pst(problem, init, o):
    pst1 = [f[2] for f in init if f[0].lower() == 'atposition' and f[1] == o][0]
    funk1 = sample_joint_position_open_list_gen(problem)
    pst2 = funk1(o, pst1)[0][0]

    w1 = [f[1] for f in init if f[0].lower() == 'inwconf'][0]
    funk2 = get_update_wconf_pst_gen()
    [w2] = funk2(w1, o, pst2)
    return [('InWConf', w2)], [('WConf', w2)]

def test_door_pull_traj(problem, init, o):
    from pybullet_tools.flying_gripper_utils import get_pull_door_handle_motion_gen
    pst1 = [f[2] for f in init if f[0].lower() == 'atposition' and f[1] == o][0]
    funk1 = sample_joint_position_open_list_gen(problem)
    pst2 = funk1(o, pst1)[0][0]

    funk2 = get_handle_grasp_gen(problem, visualize=True)
    # g = funk2(o)[0][0]
    grasps = funk2(o)

    q1 = [f[1] for f in init if f[0].lower() == 'atseconf'][0]
    funk3 = get_pull_door_handle_motion_gen(problem, visualize=False, verbose=False)
    for i in range(len(grasps)):
        (g,) = grasps[i]
        print(f'\n!!!! pr2_agent.test_door_pull_traj | grasp {i}: {nice(g.value)}')
        result = funk3('hand', o, pst1, pst2, g, q1)
        if result != None:
            [q2, cmd] = result
            return [("AtPosition", o, pst2)]
    print('\n\n!!!! cant find any handle grasp that works for', o)
    sys.exit()

def test_reachable_pose(state, init, o):
    from pybullet_tools.flying_gripper_utils import get_reachable_test
    robot = state.robot
    funk = get_reachable_test(state, custom_limits=robot.custom_limits)
    p = [f[2] for f in init if f[0].lower() == "AtPose".lower() and f[1] == o][0]
    q = [f[1] for f in init if f[0].lower() == 'AtSEConf'.lower()][0]
    w = [f[1] for f in init if f[0].lower() == 'InWConf'.lower()][0]

    outputs = get_grasp_list_gen(state)(o)
    for (g,) in outputs:
        result = funk(o, p, g, q, w)
        if result: return True
    return False

def test_sample_wconf(state, init, o):
    from pybullet_tools.general_streams import get_sample_wconf_list_gen
    funk = get_sample_wconf_list_gen(state)
    w1 = [f[1] for f in init if f[0].lower() == 'InWConf'.lower()][0]
    outputs = funk(o, w1)
    p2, w2 = outputs[0]
    joint_o = (p2.body, p2.joint)
    return [('InWConf', w2)], [('WConf', w2), ('Position', joint_o, p2), ('NewWConfPst', w1, joint_o, p2, w2)]

def test_at_reachable_pose(init, o):
    p = [f[2] for f in init if f[0].lower() == "AtPose".lower() and f[1] == o][0]
    return [('AtReachablePose', o, p)]

def test_new_wconf(init, j):
    wconf = [w[1] for w in init if w[0].lower() == 'atwconf']
    new_wconfs = [w[1] for w in init if w[0].lower() == 'wconf' and w[1] not in wconf]
    for new_wconf in new_wconfs:
        if new_wconf.positions[j].value > 0:
            new_pstn = [f[3] for f in init if f[0].lower() == 'newwconfpst' and f[2] == j and f[4] == new_wconf][0]
            # return [('AtPosition', j, new_pstn), ('InWConf', new_wconf)]
            # return [('AtPosition', j, new_pstn)]
            return [('InWConf', new_wconf)]
    sys.exit()