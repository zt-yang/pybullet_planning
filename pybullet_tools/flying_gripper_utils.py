import numpy as np
from itertools import product

from .utils import create_box, set_base_values, set_point, set_pose, get_pose, \
    get_bodies, z_rotation, load_model, load_pybullet, HideOutput, create_body, assign_link_colors, \
    get_box_geometry, get_cylinder_geometry, create_shape_array, unit_pose, Pose, \
    Point, LockRenderer, FLOOR_URDF, TABLE_URDF, add_data_path, TAN, set_color, BASE_LINK, remove_body,\
    add_data_path, connect, dump_body, disconnect, wait_for_user, get_movable_joints, get_sample_fn, \
    set_joint_positions, get_joint_name, LockRenderer, link_from_name, get_link_pose, \
    multiply, Pose, Point, interpolate_poses, HideOutput, draw_pose, set_camera_pose, load_pybullet, \
    assign_link_colors, add_line, point_from_pose, remove_handles, BLUE, INF, create_shape, \
    approximate_as_prism

from .pr2_utils import DRAKE_PR2_URDF

from .ikfast.utils import IKFastInfo
from .ikfast.ikfast import * # For legacy purposes

FE_GRIPPER_URDF = "models/franka_description/robots/hand_se3.urdf"
#FRANKA_URDF = "models/franka_description/robots/panda_arm.urdf"
FRANKA_URDF = "models/franka_description/robots/panda_arm_hand.urdf"

PANDA_INFO = IKFastInfo(module_name='franka_panda.ikfast_panda_arm', base_link='panda_link0',
                        ee_link='panda_link8', free_joints=['panda_joint7'])

class Problem():
    def __init__(self, robot, obstacles):
        self.robot = robot
        self.fixed = obstacles

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

from pybullet_tools.utils import plan_joint_motion, create_flying_body, SE3, euler_from_quat, BodySaver, \
    intrinsic_euler_from_quat, quat_from_euler, wait_for_duration, get_aabb, get_aabb_extent, \
    joint_from_name
from pybullet_tools.pr2_primitives import Trajectory, Commands, State

SE3_GROUP = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']

def get_joints_by_group(robot, group):
    return [joint_from_name(robot, j) for j in group]

def get_se3_joints(robot):
    return get_joints_by_group(robot, SE3_GROUP)

def pose_to_se3(p):
    # return list(p[0]) + list(euler_from_quat(p[1]))
    print('\n\n franka_utils.pose_to_se3 | deprecated! \n\n')
    return np.concatenate([np.asarray(p[0]), np.asarray(euler_from_quat(p[1]))])

def se3_to_pose(conf):
    print('\n\n franka_utils.se3_to_pose | deprecated! \n\n')
    return (conf[:3], quat_from_euler(conf[3:]))

def se3_from_pose(p):
    return np.concatenate([np.asarray(p[0]), np.asarray(euler_from_quat(p[1]))])

def pose_from_se3(conf):
    return (conf[:3], quat_from_euler(conf[3:]))

def approximate_as_box(robot):
    pose = get_pose(robot)
    set_pose(robot, unit_pose())
    aabb = get_aabb(robot)
    set_pose(robot, pose)
    return get_aabb_extent(aabb)

def plan_se3_motion(robot, initial_conf, final_conf, obstacles=[], custom_limits={}):
    joints = get_se3_joints(robot)
    set_joint_positions(robot, joints, initial_conf)
    path = plan_joint_motion(robot, joints, final_conf, obstacles=obstacles,
                             self_collisions=False, custom_limits=custom_limits)
    return path

from pybullet_tools.pr2_primitives import Conf

def get_free_motion_gen(problem, custom_limits={}, collisions=True, visualize=False, teleport=False):
    robot = problem.robot
    saver = BodySaver(robot)
    obstacles = problem.fixed if collisions else []
    def fn(q1, q2, fluents=[]):
        saver.restore()
        q1.assign()
        raw_path = plan_se3_motion(robot, q1.values, q2.values, obstacles=obstacles, custom_limits=custom_limits)
        if raw_path == None:
            return []
        path = [Conf(robot, get_se3_joints(robot), conf) for conf in raw_path]
        if visualize:
            for q in subsample_path(path, order=3):
                q.assign()
                draw_pose(pose_from_se3(q.values), length=0.02)
                wait_for_duration(0.5)
            wait_for_user('finished')
            return raw_path
            q1.assign()
        t = Trajectory(path)
        cmd = Commands(State(), savers=[BodySaver(robot)], commands=[t])
        return (cmd,)
    return fn

def subsample_path(path, order=2, max_len=10, min_len=3):
    return path[::order]