from __future__ import print_function

import copy
import os
import sys
import time
import numpy as np
from pprint import pprint, pformat

from pybullet_tools.pr2_streams import get_stable_gen, Position, get_pose_in_space_test, \
    get_marker_grasp_gen, get_bconf_in_region_test, \
    get_bconf_in_region_gen, get_pose_in_region_gen, get_base_motion_gen, \
    get_marker_pose_gen, get_pull_marker_to_pose_motion_gen, get_pull_marker_to_bconf_motion_gen,  \
    get_pull_marker_random_motion_gen, get_pose_in_region_test, get_cfree_btraj_pose_test, \
    sample_joint_position_gen
from pybullet_tools.mobile_streams import get_ik_fn_old, get_ik_gen_old, get_ik_rel_gen_old, \
    get_ik_rel_fn_old, get_pull_door_handle_motion_gen, get_pull_door_handle_with_link_motion_gen, \
    get_ik_ungrasp_gen, get_ik_ungrasp_mark_gen, get_ik_pull_gen, get_ik_pull_with_link_gen

from pybullet_tools.pr2_primitives import get_group_joints, get_base_custom_limits, Pose, Conf, \
    get_ik_ir_gen, move_cost_fn, Attach, Detach, Clean, Cook, \
    get_gripper_joints, GripperCommand, Simultaneous, create_trajectory
from pybullet_tools.general_streams import get_grasp_list_gen, get_contain_list_gen, get_handle_grasp_list_gen, \
    get_handle_grasp_gen, get_compute_pose_kin, get_compute_pose_rel_kin, get_above_pose_gen, \
    get_cfree_approach_pose_test, get_cfree_pose_pose_test, get_cfree_traj_pose_test, \
    get_bconf_close_to_surface, sample_joint_position_closed_gen, get_cfree_rel_pose_pose_test, \
    get_cfree_approach_rel_pose_test, get_reachable_test, get_stable_list_gen, get_cfree_pose_between_test, \
    get_nudge_grasp_gen, get_cfree_traj_pose_at_bconf_at_joint_position_test, \
    get_cfree_traj_pose_at_bconf_at_joint_position_at_link_pose_test
from pybullet_tools.camera_utils import set_camera_target_body
from pybullet_tools.bullet_utils import BASE_LIMITS, initialize_collision_logs, clean_preimage, check_joint_state
from pybullet_tools.logging_utils import summarize_facts, print_plan, print_goal, summarize_bconfs, \
    summarize_poses, print_dict, print_domain
from pybullet_tools.pr2_utils import create_pr2_gripper, set_group_conf
from pybullet_tools.pr2_streams import DEFAULT_RESOLUTION
from pybullet_tools.utils import get_client, get_joint_limits, \
    Pose, get_bodies, pairwise_collision, get_pose, point_from_pose, set_renderer, get_joint_name, \
    remove_body, LockRenderer, WorldSaver, wait_if_gui, SEPARATOR, safe_remove, ensure_dir, \
    get_distance, get_max_limit, BROWN, BLUE, WHITE, TAN, GREY, YELLOW, GREEN, BLACK, RED, CLIENTS, wait_unlocked
from pybullet_tools.stream_tests import process_debug_goals
from pybullet_tools.logging_utils import myprint as print, print_debug

from world_builder.entities import Object
from world_builder.actions import get_primitive_actions, repair_skeleton, apply_commands, \
    PULL_UNTIL, NUDGE_UNTIL

from lisdf_tools.lisdf_planning import Problem as LISDFProblem

from pddl_domains.pddl_utils import remove_stream_by_name

from pddlstream.utils import read, INF, TmpCWD
from pddlstream.algorithms.focused import solve_focused
from pddlstream.algorithms.algorithm import parse_problem, reset_globals
from pddlstream.algorithms.constraints import PlanConstraints
from pddlstream.algorithms.downward import set_cost_scale
from pddlstream.language.generator import from_gen_fn, from_list_fn, from_fn, from_test
from pddlstream.language.constants import AND, PDDLProblem, is_plan
from pddlstream.language.function import FunctionInfo
from pddlstream.language.stream import StreamInfo, PartialInputs, universe_test
from pddlstream.language.object import SharedOptValue
from pddlstream.language.external import defer_unique
from pddlstream.language.conversion import params_from_objects
from collections import namedtuple

pull_kwargs = dict(ACONF=True, learned=True, verbose=False, visualize=False)
ir_kwargs = dict(ir_only=True, max_attempts=60, learned=True)
FAILED = 'FAILED'


def get_stream_map(p, c, l, t, movable_collisions=True, motion_collisions=True,
                   pull_collisions=True, base_collisions=True, debug=False, verbose=False,
                   use_all_grasps=False, top_grasp_tolerance=None, side_grasp_tolerance=None, ir_max_attempts=60,
                   use_learned_ir=True, resolution=DEFAULT_RESOLUTION, num_grasps=20):
    """ p = problem, c = collisions, l = custom_limits, t = teleport
        add the kwargs to config yaml files in problem_sets.problem_utils.
    """
    from pybullet_tools.logging_utils import myprint as print

    movable_collisions &= c
    motion_collisions &= c
    base_collisions &= c
    pull_collisions &= c
    print('\n------------ STREAM MAP -------------')
    print(f'\tMovable collisions: {movable_collisions}')
    print(f'\tMotion collisions: {motion_collisions}')
    print(f'\tPull collisions: {pull_collisions}')
    print(f'\tBase collisions: {base_collisions}')
    print(f'\tTeleport: {t}')
    print('-------------------------------------')

    tc = dict(teleport=t, custom_limits=l)
    ptc = dict(teleport=t, custom_limits=l, collisions=pull_collisions)
    pp = dict(collisions=c, num_samples=30, verbose=False)
    go = dict(collisions=True, top_grasp_tolerance=top_grasp_tolerance, side_grasp_tolerance=side_grasp_tolerance,
              use_all_grasps=use_all_grasps, num_samples=num_grasps, visualize=False, verbose=True, debug=False)
    gg = dict(collisions=c, max_samples=None)
    ik = dict(collisions=motion_collisions, ACONF=False, teleport=t, resolution=resolution)

    ir = copy.deepcopy(ir_kwargs)
    ir.update(dict(collisions=True, max_attempts=ir_max_attempts, learned=use_learned_ir))

    pull = copy.deepcopy(pull_kwargs)
    pull.update(dict(collisions=True))

    stream_map = {

        ## ---------------------------------------------------
        ##                    poses
        ## ---------------------------------------------------
        'sample-pose': from_gen_fn(get_stable_list_gen(p, **pp)),
        'sample-relpose': from_gen_fn(get_stable_list_gen(p, relpose=True, **pp)),
        'sample-pose-inside': from_gen_fn(get_contain_list_gen(p, **pp)),
        'sample-relpose-inside': from_gen_fn(get_contain_list_gen(p, relpose=True, **pp)),
        'sample-pose-sprinkle': from_gen_fn(get_above_pose_gen(p, collisions=c)),

        ## ---------------------------------------------------
        ##                    positions
        ## ---------------------------------------------------
        'get-joint-position-open': from_gen_fn(sample_joint_position_gen(p, num_samples=6, p_max=PULL_UNTIL)),
        'get-joint-position-closed': from_gen_fn(sample_joint_position_closed_gen()),
        # 'get-joint-position-closed': from_gen_fn(sample_joint_position_gen(num_samples=6, closed=True)),
        'get-joint-position-nudged-open': from_gen_fn(sample_joint_position_gen(p, num_samples=6, p_max=NUDGE_UNTIL)),

        ## ---------------------------------------------------
        ##                    grasps
        ## ---------------------------------------------------
        'sample-grasp': from_gen_fn(get_grasp_list_gen(p, **go)),
        'sample-handle-grasp': from_gen_fn(get_handle_grasp_gen(p, **gg)),
        'sample-nudge-grasp': from_gen_fn(get_nudge_grasp_gen(p, **gg)),
        'sample-nudge-back-grasp': from_gen_fn(get_nudge_grasp_gen(p, nudge_back=True, **gg)),

        ## ---------------------------------------------------
        ##                    configurations
        ## ---------------------------------------------------

        'compute-pose-kin': from_fn(get_compute_pose_kin()),
        'compute-pose-rel-kin': from_fn(get_compute_pose_rel_kin()),

        'inverse-reachability': from_gen_fn(get_ik_gen_old(p, verbose=True, visualize=False, **ir_kwargs, **tc)),
        'inverse-kinematics': from_fn(get_ik_fn_old(p, verbose=True, visualize=False, **ik)),

        'inverse-reachability-rel': from_gen_fn(get_ik_rel_gen_old(p, verbose=False, visualize=False, **ir_kwargs, **tc)),
        'inverse-kinematics-rel': from_fn(get_ik_rel_fn_old(p, verbose=False, visualize=False, **ik)),

        ## pddl_domains/extensions/_pull_decomposed_stream.pddl
        'inverse-kinematics-grasp-handle': from_gen_fn(get_ik_gen_old(p, **pull, **tc)),
        'inverse-kinematics-ungrasp-handle': from_gen_fn(get_ik_ungrasp_gen(p, verbose=False, **ptc)),

        ## pddl_domains/extensions/_nudge_v1b.pddl
        'inverse-kinematics-nudge-door': from_gen_fn(get_ik_gen_old(p, **pull, **tc)),
        'inverse-kinematics-nudge-door-back': from_gen_fn(get_ik_gen_old(p, **pull, **tc)),

        ## pddl_domains/extensions/_pull_stream.pddl
        'inverse-kinematics-pull': from_gen_fn(get_ik_pull_gen(p, **pull, **tc)),
        'inverse-kinematics-pull-with-link': from_gen_fn(get_ik_pull_with_link_gen(p, **pull, **tc)),

        ## ---------------------------------------------------
        ##                    motion
        ## ---------------------------------------------------

        'plan-base-motion': from_fn(get_base_motion_gen(p, collisions=base_collisions, **tc)),
        # 'plan-base-motion-with-obj': from_fn(get_base_motion_with_obj_gen(p, collisions=base_collisions, teleport=t, custom_limits=l)),

        'plan-arm-turn-knob': from_fn(get_pull_door_handle_motion_gen(p, **ptc, verbose=True)),

        ## pddl_domains/extensions/_pull_decomposed_stream.pddl
        'plan-base-pull-handle': from_fn(get_pull_door_handle_motion_gen(p, **ptc, verbose=True)),
        'plan-base-pull-handle-with-link': from_fn(get_pull_door_handle_with_link_motion_gen(p, **ptc)),

        ## pddl_domains/extensions/_nudge_v1b.pddl
        'plan-base-nudge-door': from_fn(get_pull_door_handle_motion_gen(p, **ptc)),
        'plan-base-nudge-door-back': from_fn(get_pull_door_handle_motion_gen(p, **ptc)),
        # 'plan-arm-turn-knob-handle': from_fn(get_turn_knob_handle_motion_gen(p, visualize=False, **ptc)),

        'MoveCost': move_cost_fn,

        ## ---------------------------------------------------
        ##                    tests
        ## ---------------------------------------------------

        'test-cfree-pose-pose': from_test(get_cfree_pose_pose_test(p, collisions=c, visualize=False)),
        'test-cfree-approach-pose': from_test(get_cfree_approach_pose_test(p, collisions=c)),
        'test-cfree-rel-pose-pose': from_test(get_cfree_rel_pose_pose_test(p, collisions=c)),
        'test-cfree-approach-rel-pose': from_test(get_cfree_approach_rel_pose_test(p, collisions=c)),

        'test-cfree-pose-between': from_test(get_cfree_pose_between_test(p.robot, collisions=c)),

        'test-cfree-traj-pose': from_test(get_cfree_traj_pose_test(p, collisions=c)),
        'test-cfree-traj-position': from_test(get_cfree_traj_pose_test(p, collisions=c)),

        ## pddl_domains/extensions/_pull_stream.pddl
        'test-cfree-traj-pose-at-bconf-at-joint-position': from_test(
            get_cfree_traj_pose_at_bconf_at_joint_position_test(p, collisions=c)),
        'test-cfree-traj-position-at-bconf-at-joint-position': from_test(
            get_cfree_traj_pose_at_bconf_at_joint_position_test(p, collisions=c)),
        'test-cfree-traj-pose-at-bconf-at-joint-position-at-link-pose': from_test(
            get_cfree_traj_pose_at_bconf_at_joint_position_at_link_pose_test(p, collisions=c)),
        'test-cfree-traj-position-at-bconf-at-joint-position-at-link-pose': from_test(
            get_cfree_traj_pose_at_bconf_at_joint_position_at_link_pose_test(p, collisions=c)),

        'test-cfree-btraj-pose': from_test(get_cfree_btraj_pose_test(p.robot, collisions=c)),

        'test-bconf-close-to-surface': from_test(get_bconf_close_to_surface(p)),
        'test-inverse-reachability': from_test(get_reachable_test()),

        ## ---------------------------------------------------
        ##                    not in use
        ## ---------------------------------------------------

        'sample-marker-grasp': from_list_fn(get_marker_grasp_gen(p, collisions=c)),
        'inverse-kinematics-grasp-marker': from_gen_fn(get_ik_gen_old(p, learned=False, verbose=False, **ptc)),
        'inverse-kinematics-ungrasp-marker': from_fn(get_ik_ungrasp_mark_gen(p, collisions=c, **tc)),
        'plan-base-pull-marker-random': from_gen_fn(get_pull_marker_random_motion_gen(p, collisions=c, learned=False, **tc)),

        'sample-marker-pose': from_list_fn(get_marker_pose_gen(p, collisions=c)),
        'plan-base-pull-marker-to-bconf': from_fn(get_pull_marker_to_bconf_motion_gen(p, collisions=c, teleport=t)),
        'plan-base-pull-marker-to-pose': from_fn(get_pull_marker_to_pose_motion_gen(p, collisions=c, teleport=t)),
        'test-bconf-in-location': from_test(get_bconf_in_region_test(p.robot)),
        'test-pose-in-location': from_test(get_pose_in_region_test()),
        'test-pose-in-space': from_test(get_pose_in_space_test()),

        # 'sample-bconf-in-location': from_gen_fn(get_bconf_in_region_gen(p, collisions=c, visualize=False)),
        'sample-bconf-in-location': from_list_fn(get_bconf_in_region_gen(p, collisions=c, visualize=False)),
        'sample-pose-in-location': from_list_fn(get_pose_in_region_gen(p, collisions=c, visualize=False)),

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

            #'test-bconf-in-location': from_test(universe_test),
            #'test-pose-in-location': from_test(universe_test),
            #'test-pose-in-space': from_test(universe_test),
        })

    return stream_map


def get_stream_info(unique=False, defer_fn=defer_unique):
    # TODO: automatically populate using stream_map
    opt_gen_fn = PartialInputs(unique=unique)
    stream_info = {
        'MoveCost': FunctionInfo(opt_fn=opt_move_cost_fn),

        # 'test-cfree-pose-pose': StreamInfo(p_success=1e-3, verbose=verbose),
        # 'test-cfree-approach-pose': StreamInfo(p_success=1e-2, verbose=verbose),
        # 'test-cfree-traj-pose': StreamInfo(p_success=1e-1),
        # 'test-cfree-approach-pose': StreamInfo(p_success=1e-1),

        'sample-grasp': StreamInfo(opt_gen_fn=opt_gen_fn),
        'sample-handle-grasp': StreamInfo(opt_gen_fn=opt_gen_fn),

        'get-joint-position-open': StreamInfo(opt_gen_fn=opt_gen_fn),
        'get-joint-position-closed': StreamInfo(opt_gen_fn=opt_gen_fn),

        # TODO: still not re-ordering quite right
        'sample-pose': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e-1),
        'sample-pose-inside': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e-1),
        'sample-relpose': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e-1),
        'sample-relpose-inside': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e-1),

        'inverse-reachability': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e0),
        'inverse-kinematics': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e0),
        'inverse-reachability-rel': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e0),
        'inverse-kinematics-rel': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e0),
        'inverse-kinematics-grasp-handle': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e0),
        'inverse-kinematics-ungrasp-handle': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e0),

        'plan-base-pull-handle': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e0),
        'plan-base-pull-handle-with-link': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e0),

        'plan-base-motion': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e1, defer_fn=defer_fn),
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


#######################################################

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
            self.gripper = create_pr2_gripper(self.robot, arm=arm, visual=visual)
        return self.gripper

    def remove_gripper(self):
        if self.gripper is not None:
            remove_body(self.gripper)
            self.gripper = None

    def __repr__(self):
        return repr(self.__dict__)


#######################################################


def fix_init_given_goals(goals, init):
    add_init = []
    for goal in goals:
        pred = goal[0].lower()
        if pred == 'AtBConf'.lower():
            add_init += [('BConf', goal[1])]

        elif pred == 'AtSEConf'.lower():
            add_init += [('SEConf', goal[1])]

        elif pred == 'AtPose'.lower():
            add_init += [('Pose', goal[1], goal[2])]

        elif pred == 'AtPosition'.lower():
            body_joint = goal[1]
            position = goal[2]
            old_position = [f[-1] for f in init if f[0].lower() == 'atposition' and f[1] == body_joint][0]
            # pred = 'IsOpenedPosition'
            # if position.value == min([abs(n) for n in get_joint_limits(body_joint[0], body_joint[1])]):
            #     pred = 'IsClosedPosition'
            add_init += [('Position', body_joint, position), ('IsSampledPosition', body_joint, old_position, position)]

        elif pred == 'AtGrasp'.lower():
            add_init += [('Grasp', goal[2], goal[3])]

        elif pred == 'AtHandleGrasp'.lower():
            add_init += [('HandleGrasp', goal[2], goal[3])]

        elif pred == 'AtMarkerGrasp'.lower():
            add_init += [('MarkerGrasp', goal[2], goal[3])]

        elif pred == 'on':
            add_init += [('Stackable', goal[1], goal[2]), ('Graspable', goal[1]), ('Surface', goal[2])]
            find_movable_link = [f for f in init if f[0].lower() == 'movablelink' and f[1] == goal[2]]
            if len(find_movable_link) == 0:
                add_init += [('StaticLink', goal[2])]

        elif pred == 'in':
            add_init += [('Containble', goal[1], goal[2]), ('Graspable', goal[1]), ('Space', goal[2])]

    to_add = [f for f in add_init if f not in init and tuple([f[0].lower()] + list(f[1:])) not in init]
    if len(to_add) > 0:
        print(f'[stream_agent.fix_init_given_goals] adding {to_add}')
        init += to_add

    if goals[-1] == ("not", ("AtBConf", "")):
        atbconf = [i for i in init if i[0].lower() == "AtBConf".lower()][0]
        goals[-1] = ("not", atbconf)
    return init


def pddlstream_from_state_goal(state, goals, domain_pddl='pr2_kitchen.pddl',
                               stream_pddl='pr2_stream.pddl',
                               custom_limits=BASE_LIMITS,
                               init_facts=[], ## avoid duplicates
                               facts=[],  ## completely overwrite
                               objects=None,  ## only some objects included in planning
                               collisions=True, teleport=False, verbose=True,
                               print_fn=None, problem_dict=None, **kwargs):
    if print_fn is None:
        from pybullet_tools.logging_utils import myprint as print_fn

    robot = state.robot
    world = state.world
    problem = state
    if not isinstance(custom_limits, dict):
        custom_limits = get_base_custom_limits(robot, custom_limits)
    if 'cfree' in kwargs:
        collisions = not kwargs['cfree']
        kwargs.pop('cfree')
    print_domain(domain_pddl, stream_pddl, custom_limits)

    world.summarize_all_objects(print_fn=print_fn)

    if len(facts) == 0:
        facts = state.get_facts(init_facts=init_facts, objects=objects)
    init = facts

    goal = process_debug_goals(state, goals, init)
    if len(goals) > 0:
        init = fix_init_given_goals(goal[1:], init)

    summarize_facts(init, world, name='Facts extracted from observation', print_fn=print_fn)
    init = make_init_lower_case(init)

    ## only print the world facts the first time
    init_added = [n for n in init_facts if n not in init]
    if len(init_facts) != 0:
        summarize_facts(init_added, world, name='Added facts from PDDLStream preimage', print_fn=print_fn)
        init = init + init_added

    stream_map = robot.get_stream_map(problem, collisions, custom_limits, teleport,
                                      domain_pddl=domain_pddl, verbose=verbose, **kwargs)
    domain_pddl = read(domain_pddl)
    stream_pddl = read(stream_pddl)
    constant_map = {k: k for k in state.world.constants}

    goal = [g for g in goal if not (g[0] == 'not' and g[1][0] == '=')]

    if problem_dict is not None and 'english_goal' in problem_dict:
        print_goal([AND, [problem_dict['english_goal']]], world=world, print_fn=print_fn)
    else:
        print_goal(goal, world=world, print_fn=print_fn)

    world.summarize_body_indices(print_fn=print_fn)
    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)


def is_plan_abstract(plan):
    for step in plan:
        if '--no-' in step.name:
            return True
    return False


def get_diverse_kwargs(kwargs, diverse=True, max_plans=None):
    if diverse:
        max_plans = 100 if max_plans is None else max_plans
        plan_dataset = []
        max_skeletons = 1
        use_feedback = False
    else:
        max_plans = 1
        plan_dataset = None
        max_skeletons = INF
        use_feedback = True
    kwargs.update(dict(max_plans=max_plans, plan_dataset=plan_dataset,
                       max_skeletons=max_skeletons, use_feedback=use_feedback))
    return kwargs, plan_dataset


def get_test_subgoals(init):
    arms = [fact[1] for fact in init if fact[0] == 'arm']
    objects = [fact[1] for fact in init if fact[0] == 'graspable']
    print(f'Arms: {arms} | Objects: {objects}')
    arm = arms[0]
    obj = objects[0]
    return [('Holding', arm, obj),]


def get_test_skeleton():
    from pddlstream.algorithms.constraints import WILD
    return [
        ('grasp_handle', [WILD, WILD, WILD, WILD, WILD, WILD, WILD, WILD]),
        ('pull_door_handle', [WILD, WILD, WILD, WILD, WILD, WILD, WILD, WILD, WILD, WILD, WILD]),
        ('ungrasp_handle', [WILD, WILD, WILD, WILD, WILD, WILD, WILD, WILD]),
        ('pick', [WILD, WILD, WILD, WILD, WILD, WILD, WILD]),
        ('place', [WILD, WILD, WILD, WILD, WILD, WILD, WILD]),
    ]


def solve_one(pddlstream_problem, stream_info, diverse=False, lock=False, visualize=True,
              fc=None, domain_modifier=None, skeleton=None, subgoals=None, soft_subgoals=False, max_time=INF,
              downward_time=10, evaluation_time=10, stream_planning_timeout=30, total_planning_timeout=180,
              max_cost=INF, collect_dataset=False, max_plans=None, max_solutions=0, verbose_outside=True, **kwargs):

    # skeleton = get_test_skeleton()
    # subgoals = get_test_subgoals(pddlstream_problem.init)

    if skeleton is not None and len(skeleton) > 0:
        print_skeleton(skeleton)
        skeleton = repair_skeleton(skeleton, pddlstream_problem.domain_pddl)
        constraints = PlanConstraints(skeletons=[skeleton], exact=False, max_cost=max_cost + 1)

    elif subgoals is None or len(subgoals) == 0:
        constraints = PlanConstraints(max_cost=max_cost + 1)

    else:
        if subgoals is None:
            subgoals = []
        if len(subgoals) > 0:
            print('-' * 40 + f' soft_subgoals: {soft_subgoals} ' + '-' * 40)
            print('\n'.join([str(s) for s in subgoals]))
            print('-' * 100)
        subgoal_costs = len(subgoals) * [100] if soft_subgoals else None
        constraints = PlanConstraints(subgoals=subgoals, subgoal_costs=subgoal_costs, max_cost=max_cost + 1)  # TODO: plus 1 in action costs?

    if collect_dataset:
        max_solutions = 6 if max_solutions == 0 else max_solutions
    diverse = diverse or collect_dataset

    ## the other kwargs are provided in cogarch_utils.get_pddlstream_kwargs()
    planner_kwargs_default = dict(planner='ff-astar1', unit_costs=False, success_cost=INF, verbose=True,
                                  unique_optimistic=True, forbid=True, bind=True)
    planner_kwargs = dict(max_planner_time=downward_time, max_time=max_time, evaluation_time=evaluation_time,
                          stream_planning_timeout=stream_planning_timeout, total_planning_timeout=total_planning_timeout,
                          initial_complexity=5, visualize=visualize, fc=fc, domain_modifier=domain_modifier,
                          # unit_efforts=True, effort_weight=None,
                          max_solutions=max_solutions, search_sample_ratio=0, **kwargs)
    planner_dict, plan_dataset = get_diverse_kwargs(planner_kwargs, diverse=diverse, max_plans=max_plans)

    if verbose_outside:
        print_dict(planner_dict, 'planner_kwargs')

    # with Profiler():
    initialize_collision_logs()
    set_cost_scale(cost_scale=1)
    with LockRenderer(lock=lock):
        solution = solve_focused(pddlstream_problem, stream_info=stream_info, constraints=constraints,
                                 **planner_kwargs_default, **planner_kwargs)
    if collect_dataset:
        return solution, plan_dataset
    if solution is None:
        solution = None, 0, []
    return solution


def create_cwd_saver():
    # temp_dir = '/tmp/pddlstream-{}/'.format(os.getpid())
    temp_dir = '/tmp/pddlstream-{}-{}/'.format(os.getpid(), int(time.time()))
    print(f'\n\n\n\nsolve_multiple at temp dir {temp_dir} \n\n\n\n')
    safe_remove(temp_dir)
    ensure_dir(temp_dir)
    cwd_saver = TmpCWD(temp_cwd=temp_dir)  # TODO: multithread
    cwd_saver.save()  # TODO: move to the constructor
    return cwd_saver


def solve_multiple(pddlstream_problem, stream_info, lock=True, cwd_saver=None, **kwargs):
    reset_globals()
    # profiler = Profiler(field='tottime', num=25) ## , enable=profile # cumtime | tottime
    # profiler.save()

    if cwd_saver is None:
        cwd_saver = create_cwd_saver()
    lock_saver = LockRenderer(lock=lock)

    solution = solve_one(pddlstream_problem, stream_info, lock=lock, **kwargs)

    # profiler.restore()
    return solution, cwd_saver


def get_named_colors(kind='tablaeu', alpha=1.):
    # from matplotlib._color_data import BASE_COLORS, TABLEAU_COLORS, XKCD_COLORS, CSS4_COLORS
    # from pybullet_planning.pybullet_tools.utils import CHROMATIC_COLORS, COLOR_FROM_NAME
    # TODO: colors.py from past projects
    from matplotlib.colors import BASE_COLORS, TABLEAU_COLORS, XKCD_COLORS, CSS4_COLORS, to_rgba
    from pybullet_tools.utils import RGBA, apply_alpha
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
                if action_name == 'move_base':
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
                     collect_dataset=False, domain_modifier=None,
                     profile=True, lock=False, max_time=5*50, preview=False, **kwargs):
    from pybullet_tools.logging_utils import myprint as print

    start_time = time.time()

    CLIENTS[get_client()] = True # TODO: hack
    saver = WorldSaver()
    world = state.world
    objects = world.objects
    stream_info = world.robot.get_stream_info()

    # if has_gui():
    #     world.make_doors_transparent()
        # set_renderer(True)
        # time.sleep(0.1)
        # set_renderer(False)

    #########################

    print(SEPARATOR)

    # profiler = Profiler(field='cumtime' if profile else None, num=25) # cumtime | tottime
    # profiler.save()

    with LockRenderer(lock=lock):
        solution = solve_one(pddlstream_problem, stream_info,
                             domain_modifier=domain_modifier,
                             collect_dataset=collect_dataset, world=world,
                             visualize=visualization, **kwargs)

    ## collect data of multiple solutions
    if collect_dataset:
        from pigi_tools.data_utils import save_multiple_solutions
        solution, plan_dataset = solution

        indices = world.get_indices()
        solution = save_multiple_solutions(plan_dataset, indices=indices)

    saver.restore()
    # profiler.restore()
    reset_globals()
    state.remove_gripper()
    world.remove_redundant_bodies()

    knowledge = parse_problem(pddlstream_problem, stream_info=stream_info,
                              constraints=PlanConstraints(), unit_costs=True,
                              unit_efforts=True)

    plan, cost, evaluations = solution
    solved = is_plan(plan)

    print('Solved: {}'.format(solved))
    print('Cost: {:.3f}'.format(cost))
    print_plan(plan, world)

    time_log = {'planning': round(time.time()-start_time, 4)}
    start_time = time.time()
    if plan is not None:
        preimage = clean_preimage(evaluations.preimage_facts)

        ## ------ debug why can't find action skeleton
        ## test_grasp_ik(state, state.get_facts()+preimage, name='eggblock')

        ## save_pickle(pddlstream_problem, plan, preimage) ## discarded

        summarize_facts(preimage, world, name='Preimage generated by PDDLStream')
        summarize_poses(preimage)
        summarize_bconfs(preimage, plan)

        if is_plan_abstract(plan):
            from leap_tools.hierarchical import check_preimage
            cache = state.world.remove_unpickleble_attributes()
            env, plan = check_preimage(pddlstream_problem, plan, preimage, domain_pddl,
                                       init=pddlstream_problem.init, objects=objects,
                                       domain_modifier=domain_modifier)
            state.world.recover_unpickleble_attributes(cache)
        else:
            env = None  ## preimage
    else:
        env = None
        preimage = []

    time_log['preimage'] = round(time.time() - start_time, 4) if time.time() - start_time > 0.1 else 0
    time_log.update(log_goal_plan_init(pddlstream_problem.goal[1:], plan, preimage))

    if preview:
        state.assign()
        lisdf_problem = LISDFProblem(world)
        commands = post_process(lisdf_problem, plan)

        state.assign()
        wait_if_gui('Begin?')
        apply_commands(lisdf_problem, commands, time_step=5e-2)
        wait_if_gui('Finish?')
        state.assign()

    return plan, env, knowledge, time_log, preimage


def log_goal_plan_init(goal, plan, preimage):
    """ goal is a sequence of predicates, excluding AND """
    def get_plan_skeleton(action):
        args = tuple([a for a in action.args if isinstance(a, int) or isinstance(a, tuple)])
        return f"{action.name}{args}".replace('),)', '))')

    goal_no_obj = [[elem.pybullet_name if isinstance(elem, Object) else elem for elem in g] for g in goal]

    return {
        'goal': [f'{g[0]}({g[1:]})' for g in goal],
        'goal_original': goal_no_obj,
        'plan': [str(a) for a in plan] if plan is not None else FAILED,
        'plan_skeleton': [get_plan_skeleton(a) for a in plan] if plan is not None else FAILED,
        'plan_len': len(plan) if plan is not None else 0,
        'init': [[str(a) for a in f] for f in preimage]
    }


def print_skeleton(skeleton):
    if skeleton is not None and len(skeleton) > 0:
        print('-' * 40 + f' skeleton ' + '-' * 40)
        print('\n'.join([str(s) for s in skeleton]))
        print('-' * 90)


def heuristic_modify_stream(pddlstream_problem, world):
    from pddlstream.language.constants import PDDLProblem

    domain_pddl, constant_map, stream_pddl, stream_map, init, goal = pddlstream_problem
    title = '[pddlstream_agent._heuristic_modify_stream] \t'

    def check_joint_open(joint_body):
        body, joint = joint_body
        category, status = check_joint_state(body, joint)
        return 'OPEN' in status

    sample_open = True
    sample_close = True
    found_relevant = False
    reason_not_open = None
    reason_not_close = None
    for goal_item in goal[1:]:
        goal_pred = goal_item[0].lower()
        if goal_pred in ['openedjoint']:
            sample_close = False
            reason_not_close = 'goal is to open joint'
        elif goal_pred in ['closedjoint']:
            sample_open = False
            reason_not_open = 'goal is to close joint'
        elif goal_pred in ['handlegrasped', 'openedjoint', 'graspedhandle', 'pulled', 'pulledoneaction']:
            if isinstance(goal_item[1], tuple):
                if check_joint_open(goal_item[1]):
                    sample_open = False
                    reason_not_open = 'goal is to manipulate a joint that is open'
                else:
                    sample_close = False
                    reason_not_close = 'goal is to manipulate a joint that is closed'
            else:
                print(f'{title} trying to use a movable as joint in {goal}')
        else:
            continue
        found_relevant = True
        break

    if not found_relevant:
        joint_is_open = [int(check_joint_open(f[1])) for f in init if f[0] == 'joint']
        if len(joint_is_open) > 0:
            if sum(joint_is_open) == len(joint_is_open):
                sample_open = False
                reason_not_open = 'all joints are open at the beginning'
            if sum(joint_is_open) == 0:
                sample_close = False
                reason_not_close = 'all joints are closed at the beginning'

    stream_name = 'get-joint-position-open'
    if not sample_open:
        print_debug(f'{title}remove_stream_by_name({stream_name}) because {reason_not_open}')
        stream_pddl = remove_stream_by_name(stream_pddl, stream_name)
    elif not sample_close:
        print_debug(f'{title} keeping stream {stream_name} in domain')

    stream_name = 'get-joint-position-closed'
    if not sample_close:
        print_debug(f'{title}remove_stream_by_name({stream_name}) because {reason_not_close}')
        stream_pddl = remove_stream_by_name(stream_pddl, stream_name)
    elif not sample_open:
        print_debug(f'{title} keeping stream {stream_name} in domain')

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)


## -----------------------------------------------------------------------------------


def post_process(problem, plan, simulate=False, teleport=False, verbose=False):
    if plan is None:
        return None
    commands = []
    print('\nstream_agent.post_process ...')
    for i, action in enumerate(plan):
        new_commands = get_primitive_actions(action, problem.world, teleport, simulate=simulate)
        commands += new_commands
        if verbose:
            print(i, action)
    return commands


def make_init_lower_case(init):
    return [tuple([x[0].lower()] + list(x[1:])) for x in init]
