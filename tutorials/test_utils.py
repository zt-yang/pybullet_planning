from __future__ import print_function
import os
import json
from os.path import join, abspath, dirname, isdir, isfile
from config import EXP_PATH


from pybullet_tools.utils import connect, draw_pose, unit_pose, link_from_name, load_pybullet, load_model, \
    sample_aabb, AABB, set_pose, get_aabb, get_aabb_center, quat_from_euler, Euler, HideOutput, get_aabb_extent, \
    set_camera_pose, wait_unlocked, disconnect, wait_if_gui, create_box, wait_for_duration, \
    SEPARATOR, get_aabb, get_pose, approximate_as_prism, draw_aabb, multiply, unit_quat, remove_body, invert, \
    Pose, get_link_pose, get_joint_limits, WHITE, RGBA, set_all_color, RED, GREEN, set_renderer, clone_body, \
    add_text, joint_from_name, set_caching, Point, set_random_seed, set_numpy_seed, reset_simulation, \
    get_joint_name, get_link_name, dump_joint, set_joint_position, ConfSaver, pairwise_link_collision
from pybullet_tools.pr2_problems import create_floor

from world_builder.robot_builders import build_skill_domain_robot

from pddlstream.algorithms.meta import solve, create_parser


def init_experiment(exp_dir):
    from pybullet_tools.logging import TXT_FILE
    if isfile(TXT_FILE):
        os.remove(TXT_FILE)


def get_test_world(robot='feg', semantic_world=False, draw_origin=False,
                   width=1980, height=1238, **kwargs):
    connect(use_gui=True, shadows=False, width=width, height=height)  ##  , width=360, height=270
    if draw_origin:
        draw_pose(unit_pose(), length=.5)
        create_floor()
    set_caching(cache=False)
    if semantic_world:
        from world_builder.world import World
        world = World()
    else:
        from lisdf_tools.lisdf_loader import World
        world = World()
    build_skill_domain_robot(world, robot, **kwargs)
    return world


def get_args(exp_name=None):
    parser = create_parser()
    parser.add_argument('-test', type=str, default=exp_name, help='Name of the test case')
    parser.add_argument('-cfree', action='store_true', help='Disables collisions during planning')
    parser.add_argument('-enable', action='store_true', help='Enables rendering during planning')
    parser.add_argument('-teleport', action='store_true', help='Teleports between configurations')
    parser.add_argument('-simulate', action='store_true', help='Simulates the system')
    args = parser.parse_args()
    print('Arguments:', args)
    return args