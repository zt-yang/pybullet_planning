from __future__ import print_function

from pybullet_tools.general_streams import get_cfree_approach_pose_test, get_grasp_list_gen, \
    get_stable_list_gen, sample_joint_position_gen, get_cfree_pose_pose_test, \
    get_cfree_traj_pose_test, get_handle_grasp_gen, \
    Position, get_contain_list_gen, get_pose_from_attachment, get_stable_gen, get_contain_gen

from pddlstream.language.generator import from_gen_fn, from_list_fn, from_fn, from_test
from pddlstream.language.function import FunctionInfo
from pddlstream.language.stream import StreamInfo, PartialInputs

from pybullet_tools.flying_gripper_utils import get_ik_fn, get_free_motion_gen, \
    get_pull_handle_motion_gen, get_reachable_test
from pybullet_tools.stream_agent import opt_move_cost_fn


def get_stream_map(p, c, l, t, **kwargs):
    """ p = problem, c = collisions, l = custom_limits, t = teleport """

    stream_map = {
        'sample-pose-on': from_gen_fn(get_stable_gen(p, collisions=c)),
        'sample-pose-in': from_gen_fn(get_contain_gen(p, collisions=c, verbose=False)),
        # debug weiyu
        'sample-grasp': from_list_fn(get_grasp_list_gen(p, collisions=c, visualize=False, top_grasp_tolerance=None)), #math.pi/4)),

        'inverse-kinematics-hand': from_fn(get_ik_fn(p, collisions=c, teleport=t, custom_limits=l, verbose=False)),
        'test-cfree-pose-pose': from_test(get_cfree_pose_pose_test(p, collisions=c)),
        'test-cfree-approach-pose': from_test(get_cfree_approach_pose_test(p, collisions=c)),
        'test-cfree-traj-pose': from_test(get_cfree_traj_pose_test(p, collisions=c, verbose=False)),

        'plan-free-motion-hand': from_fn(get_free_motion_gen(p, collisions=c, teleport=t, custom_limits=l)),

        'get-joint-position-open': from_gen_fn(sample_joint_position_gen(p, num_samples=6)),
        'sample-handle-grasp': from_gen_fn(get_handle_grasp_gen(p, collisions=c, verbose=False)),

        'inverse-kinematics-grasp-handle': from_fn(get_ik_fn(p, collisions=c, teleport=t,
                                                             custom_limits=l, verbose=False)),
        'plan-grasp-pull-handle': from_fn(get_pull_handle_motion_gen(p, collisions=c, teleport=t)),
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
        # 'test-bconf-in-location': from_test(get_bconf_in_region_test(p.robot)),
        # 'test-pose-in-location': from_test(get_pose_in_region_test()),
        # 'test-pose-in-space': from_test(get_pose_in_space_test()),  ##
        #
        # # 'sample-bconf-in-location': from_gen_fn(get_bconf_in_region_gen(p, collisions=c, visualize=False)),
        # 'sample-bconf-in-location': from_list_fn(get_bconf_in_region_gen(p, collisions=c, visualize=False)),
        # 'sample-pose-in-location': from_list_fn(get_pose_in_region_gen(p, collisions=c, visualize=False)),

        # 'MoveCost': move_cost_fn,

        # 'TrajPoseCollision': fn_from_constant(False),
        # 'TrajArmCollision': fn_from_constant(False),
        # 'TrajGraspCollision': fn_from_constant(False),
    }
    return stream_map


def get_stream_info(unique=False):
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

        # TODO: still not re-ordering quite right
        'sample-pose-on': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e-1),
        'sample-pose-in': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e-1),

        'inverse-kinematics-hand': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e0),

        'inverse-kinematics-grasp-handle': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e0),
        'plan-base-pull-door-handle': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e0),

        'plan-free-motion-hand': StreamInfo(opt_gen_fn=opt_gen_fn, overhead=1e1),
    }

    return stream_info
