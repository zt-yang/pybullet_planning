import os
import sys
from os.path import join, abspath, dirname, isdir, isfile
R = abspath(join(dirname(__file__), os.pardir, os.pardir))
sys.path.extend([R] + [join(R, name) for name in ['pybullet_planning', 'lisdf', 'pddlstream']])
# print('\n'.join([p for p in sys.path if 'vlm-tamp' in p]))

import math

from pybullet_tools.utils import connect, draw_pose, unit_pose, link_from_name, load_pybullet, load_model, \
    sample_aabb, AABB, set_pose, quat_from_euler, HideOutput, get_aabb_extent, unit_quat, remove_body, \
    set_camera_pose, wait_unlocked, disconnect, wait_if_gui, create_box, get_aabb, get_pose, draw_aabb, multiply, \
    Pose, get_link_pose, get_joint_limits, WHITE, RGBA, set_all_color, RED, GREEN, set_renderer, add_text, \
    Point, set_random_seed, set_numpy_seed, reset_simulation, joint_from_name, PI, \
    get_joint_name, get_link_name, dump_joint, set_joint_position, ConfSaver, pairwise_link_collision
from pybullet_tools.bullet_utils import nice, \
    draw_fitted_box, open_joint, dump_json
from pybullet_tools.camera_utils import set_camera_target_body, take_selected_seg_images


from tutorials.test_utils import get_test_world, load_body, get_instances, \
    load_model_instance, get_model_path, get_y_gap
from tutorials.config import ASSET_PATH

from tutorials.test_grasps import run_test_grasps, run_test_handle_grasps_counter

from cogarch_tools.cogarch_run import run_agent
from cogarch_tools.processes.teleop_agent import TeleOpAgent

from world_builder.paths import EXP_PATH

from robot_builder.spot_utils import solve_spot_leg_conf
from robot_builder.robot_tests import test_whole_body_ik

from leap_tools.hierarchical_agent import HierarchicalAgent


namo_kwargs = dict(domain='pddl_domains/mobile_namo_domain.pddl', stream='pddl_domains/mobile_namo_stream.pddl')


def test_load_spot_robot(verbose=False):
    world = get_test_world(robot='spot', base_q=(0, 0, 0, 3.14159), width=1200, height=1200,
                           semantic_world=True, draw_origin=True,
                           custom_limits=((-3, -3), (3, 3)))
    robot = world.robot
    set_camera_target_body(robot.body, distance=0.5)

    ## open gripper
    set_joint_position(robot.body, joint_from_name(robot.body, 'arm0.f1x'), -1.5)

    ## compute once, while robot is floored while
    # solve_spot_leg_conf(robot.body, 0.5, compute_params=True)

    tool_link = robot.get_tool_link()
    box_range_aabb = AABB(lower=[-0.3, -0.3, 0.2], upper=[0.3, 0.3, 1.2])
    test_whole_body_ik(robot, tool_link, box_range_aabb)


def test_spot_grasps():
    kwargs = dict(categories=['VeggieCabbage'], skip_grasps=False, base_q=(0, 0, 0, 0))
    kwargs['categories'] = ['Food']

    ## --- step 1: find tool_from_hand transformation
    debug_kwargs = dict(verbose=True, test_rotation_matrix=True, skip_grasp_index=1,
                        test_translation_matrix=False)

    ## --- step 2: find tool_from_root transformation (multiple rotations may look correct,
    #               but only one works after running IR - IK)
    ## (1.571, 3.142, -1.571) (1.571, 3.142, 1.571) (1.571, 3.142, 0) (-1.571, 0, 0)
    debug_kwargs = dict(verbose=True, test_attachment=True, visualize=False, retain_all=False)

    ## --- step 3: verify all grasps generated for one object
    # debug_kwargs = dict(verbose=True, test_attachment=False, visualize=True, retain_all=True)

    ## --- step 4: verify top_grasp_tolerance filtering
    debug_kwargs = dict(verbose=True, visualize=False, retain_all=False, top_grasp_tolerance=PI/4)

    run_test_grasps('spot', **debug_kwargs, **kwargs)
    # run_test_grasps('pr2', **debug_kwargs, **kwargs)
    # run_test_grasps('feg', **debug_kwargs, **kwargs)


def test_pr2_grasps():
    world_builder_args = dict(movable_category=None)
    # world_builder_args = dict()
    run_agent(config='config_dev.yaml', problem='test_pick', world_builder_args=world_builder_args)


def test_cart_domain_pr2():
    from pddl_domains.pddl_utils import update_namo_pddl
    update_namo_pddl()

    problem = ['test_navigation', 'test_cart_pull'][1]
    run_agent(config='config_dev.yaml', problem=problem, **namo_kwargs)


def test_office_chair_domain_spot():
    problem = ['test_spot_pick', 'test_office_chairs'][1]
    run_agent(config='config_spot.yaml', problem=problem)


if __name__ == '__main__':
    test_load_spot_robot()
    # test_spot_grasps()
    # test_pr2_grasps()
    # test_cart_domain_pr2()
    # test_office_chair_domain_spot()
