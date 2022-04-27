import numpy as np
from itertools import product

from .utils import create_box, set_base_values, set_point, set_pose, get_pose, \
    get_bodies, z_rotation, load_model, load_pybullet, HideOutput, create_body, assign_link_colors, \
    get_box_geometry, get_cylinder_geometry, create_shape_array, unit_pose, Pose, \
    Point, LockRenderer, FLOOR_URDF, TABLE_URDF, add_data_path, TAN, set_color, BASE_LINK, remove_body,\
    add_data_path, connect, dump_body, disconnect, wait_for_user, get_movable_joints, get_sample_fn, \
    set_joint_positions, get_joint_name, LockRenderer, link_from_name, get_link_pose, \
    multiply, Pose, Point, interpolate_poses, HideOutput, draw_pose, set_camera_pose, load_pybullet, \
    assign_link_colors, add_line, point_from_pose, remove_handles, BLUE, INF

from .pr2_utils import DRAKE_PR2_URDF

from .ikfast.utils import IKFastInfo
from .ikfast.ikfast import * # For legacy purposes

FE_GRIPPER_URDF = "models/franka_description/robots/hand.urdf"
#FRANKA_URDF = "models/franka_description/robots/panda_arm.urdf"
FRANKA_URDF = "models/franka_description/robots/panda_arm_hand.urdf"

PANDA_INFO = IKFastInfo(module_name='franka_panda.ikfast_panda_arm', base_link='panda_link0',
                        ee_link='panda_link8', free_joints=['panda_joint7'])

def create_franka():
    with LockRenderer():
        with HideOutput(True):
            robot = load_model(FRANKA_URDF, fixed_base=True)
            assign_link_colors(robot, max_colors=3, s=0.5, v=1.)
            # set_all_color(robot, GREEN)
    return robot

def create_fe_gripper():
    with LockRenderer():
        with HideOutput(True):
            robot = load_model(FE_GRIPPER_URDF, fixed_base=False)
            # assign_link_colors(robot, max_colors=3, s=0.5, v=1.)
            # set_all_color(robot, GREEN)
    return robot

#####################################################################

def test_retraction(robot, info, tool_link, distance=0.1, **kwargs):
    ik_joints = get_ik_joints(robot, info, tool_link)
    start_pose = get_link_pose(robot, tool_link)
    end_pose = multiply(start_pose, Pose(Point(z=-distance)))
    handles = [add_line(point_from_pose(start_pose), point_from_pose(end_pose), color=BLUE)]
    #handles.extend(draw_pose(start_pose))
    #hand®les.extend(draw_pose(end_pose))
    path = []
    pose_path = list(interpolate_poses(start_pose, end_pose, pos_step_size=0.01))
    for i, pose in enumerate(pose_path):
        print('Waypoint: {}/{}'.format(i+1, len(pose_path)))
        handles.extend(draw_pose(pose))
        conf = next(either_inverse_kinematics(robot, info, tool_link, pose, **kwargs), None)
        if conf is None:
            print('Failure!')
            path = None
            wait_for_user()
            break
        set_joint_positions(robot, ik_joints, conf)
        path.append(conf)
        wait_for_user()
        # for conf in islice(ikfast_inverse_kinematics(robot, info, tool_link, pose, max_attempts=INF, max_distance=0.5), 1):
        #    set_joint_positions(robot, joints[:len(conf)], conf)
        #    wait_for_user()
    remove_handles(handles)
    return path

def test_ik(robot, info, tool_link, tool_pose):
    draw_pose(tool_pose)
    # TODO: sort by one joint angle
    # TODO: prune based on proximity
    ik_joints = get_ik_joints(robot, info, tool_link)
    for conf in either_inverse_kinematics(robot, info, tool_link, tool_pose, use_pybullet=False,
                                          max_distance=INF, max_time=10, max_candidates=INF):
        # TODO: profile
        set_joint_positions(robot, ik_joints, conf)
        wait_for_user()

def sample_ik_tests(robot):
    joints = get_movable_joints(robot)
    tool_link = link_from_name(robot, 'panda_hand')

    info = PANDA_INFO
    check_ik_solver(info)

    sample_fn = get_sample_fn(robot, joints)
    for i in range(10):
        print('Iteration:', i)
        conf = sample_fn()
        set_joint_positions(robot, joints, conf)
        wait_for_user()
        # test_ik(robot, info, tool_link, get_link_pose(robot, tool_link))
        test_retraction(robot, info, tool_link, use_pybullet=False,
                        max_distance=0.1, max_time=0.05, max_candidates=100)

######################################################

from .utils import clone_body, irange
from .bullet_utils import nice
#
# def plan_cartesian_motion(robot, waypoint_poses, custom_limits=(), target_link=BASE_LINK,
#                           max_iterations=200, max_time=INF, **kwargs):
#     """ a quick implementation based on plan_cartesian_motion from utils.py """
#
#     lower_limits, upper_limits = custom_limits
#
#     sub_robot = clone_body(robot, visual=False, collision=False)
#     null_space = None
#
#     solutions = []
#     for target_pose in waypoint_poses:
#         print(f'franka_utils.plan_cartesian_motion | target_pose {nice(target_pose)}')
#         start_time = time.time()
#         for iteration in irange(max_iterations):
#             if elapsed_time(start_time) >= max_time:
#                 remove_body(sub_robot)
#                 return None
#             sub_kinematic_conf = inverse_kinematics_helper(sub_robot, sub_target_link, target_pose, null_space=null_space)
#             if sub_kinematic_conf is None:
#                 remove_body(sub_robot)
#                 return None
#
#             # print(f'pddlstream.plan_cartesian_motion | {iteration}\t {nice(sub_kinematic_conf)}')
#             set_joint_positions(sub_robot, sub_joints, sub_kinematic_conf)
#             # print(f'\t link pose {get_link_pose(sub_robot, sub_target_link)}')
#             if is_pose_close(get_link_pose(sub_robot, sub_target_link), target_pose, **kwargs):
#                 # print(f'\t is close {get_link_pose(sub_robot, sub_target_link)}')
#                 set_joint_positions(robot, selected_joints, sub_kinematic_conf)
#                 kinematic_conf = get_configuration(robot)
#                 if not all_between(lower_limits, kinematic_conf, upper_limits):
#                     # print(f'\t out of limits {kinematic_conf} lower {lower_limits} upper {upper_limits}')
#
#                     #movable_joints = get_movable_joints(robot)
#                     #print([(get_joint_name(robot, j), l, v, u) for j, l, v, u in
#                     #       zip(movable_joints, lower_limits, kinematic_conf, upper_limits) if not (l <= v <= u)])
#                     #print("Limits violated")
#                     #wait_if_gui()
#                     remove_body(sub_robot)
#                     return None
#                 #print("IK iterations:", iteration)
#                 solutions.append(kinematic_conf)
#                 break
#         else:
#             remove_body(sub_robot)
#             return None
#     # TODO: finally:
#     remove_body(sub_robot)
#     return solutions
