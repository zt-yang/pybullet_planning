import copy
import random

from os.path import join
import numpy as np
from itertools import product
import pybullet as p
from math import radians as rad
import math

from pybullet_tools.utils import create_box, set_base_values, set_point, set_pose, get_pose, GREY, \
    get_bodies, z_rotation, load_model, load_pybullet, HideOutput, create_body, assign_link_colors, \
    get_box_geometry, get_cylinder_geometry, create_shape_array, unit_pose, unit_quat, Pose, \
    Point, LockRenderer, FLOOR_URDF, TABLE_URDF, add_data_path, TAN, set_color, remove_body,\
    add_data_path, connect, dump_body, disconnect, wait_for_user, get_movable_joints, get_sample_fn, \
    set_joint_positions, get_joint_name, LockRenderer, link_from_name, get_link_pose, \
    multiply, interpolate_poses, HideOutput, draw_pose, set_camera_pose, load_pybullet, \
    assign_link_colors, add_line, point_from_pose, remove_handles, BLUE, BROWN, INF, create_shape, \
    approximate_as_prism, set_renderer, plan_joint_motion, create_flying_body, SE3, euler_from_quat, BodySaver, \
    intrinsic_euler_from_quat, quat_from_euler, wait_for_duration, get_aabb, get_aabb_extent, \
    joint_from_name, get_joint_limits, irange, is_pose_close, CLIENT, set_all_color, GREEN, RED, \
    wait_unlocked, dump_joint, VideoSaver
from pybullet_tools.pr2_primitives import Conf, Grasp, Trajectory, Commands, State
from pybullet_tools.general_streams import Position, get_grasp_list_gen, get_handle_link, \
    process_motion_fluents
from pybullet_tools.bullet_utils import collided, nice
from pybullet_tools.camera_utils import set_camera_target_body
from pybullet_tools.grasp_utils import add_to_rc2oc

from pybullet_tools.ikfast.utils import IKFastInfo
from pybullet_tools.ikfast.ikfast import * # For legacy purposes

from robot_builder.robot_utils import get_joints_by_names

FE_GRIPPER_URDF = "models/franka_description/robots/hand_se3.urdf"
FE_POINTER_URDF = "models/franka_description/robots/pointer_se3.urdf"
#FRANKA_URDF = "models/franka_description/robots/panda_arm.urdf"
FRANKA_URDF = "models/franka_description/robots/panda_arm_hand.urdf"

PANDA_INFO = IKFastInfo(module_name='franka_panda.ikfast_panda_arm', base_link='panda_link0',
                        ee_link='panda_link8', free_joints=['panda_joint7'])

BASE_VELOCITIES = np.array([1., 1., 1, rad(180), rad(180), rad(180)]) / 1.
BASE_RESOLUTIONS = np.array([0.05, 0.05, 0.05, rad(10), rad(10), rad(10)])

FEG_ARM_NAME = 'hand'
FEG_TOOL_LINK = 'panda_hand'
FEG_JOINT_GROUPS = {
    'hand_gripper': [f'panda_finger_joint{k}' for k in [1, 2]]
}
CACHE = {}


def get_se3_custom_limits(custom_limits):
    if isinstance(custom_limits, dict):
        return custom_limits
    x_limits, y_limits, z_limits = zip(*custom_limits)
    return { 0: x_limits, 1: y_limits, 2: z_limits }


def create_franka():
    with LockRenderer():
        with HideOutput(True):
            robot = load_model(FRANKA_URDF, fixed_base=True)
            assign_link_colors(robot, max_colors=3, s=0.5, v=1.)
            # set_all_color(robot, GREEN)
    return robot


def create_fe_gripper(init_q=None, POINTER=False, scale=1, color=None):
    path = FE_GRIPPER_URDF
    if POINTER:
        path = FE_POINTER_URDF
        scale = 0.01
    with LockRenderer():
        with HideOutput(True):
            robot = load_model(path, fixed_base=False, scale=scale)
            set_gripper_positions(robot, w=0.08)
            if init_q != None:
                set_se3_conf(robot, init_q)
            if color != None:
                set_all_color(robot, color)
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
    tool_link = link_from_name(robot, FEG_TOOL_LINK)

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


SE3_GROUP = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
PANDA_FINGERS_GROUP = ['panda_finger_joint1', 'panda_finger_joint2']
BASE_LINK = 'world_link' ## 'panda_hand' ##


def get_gripper_positions(robot):
    joints = get_joints_by_names(robot, PANDA_FINGERS_GROUP)
    return get_joint_positions(robot, joints)


def set_gripper_positions(robot, w=0.0):
    joints = get_joints_by_names(robot, PANDA_FINGERS_GROUP)
    set_joint_positions(robot, joints, [w/2, w/2])


def open_gripper(robot):
    set_gripper_positions(robot, w=0.08)


def open_cloned_gripper(robot, gripper, w=0.12): ## 0.08 is the limit
    """ because link and joint names aren't cloned """
    joints = get_joints_by_names(robot, PANDA_FINGERS_GROUP)
    w = min(w, 0.12)
    set_joint_positions(gripper, joints, [w / 2, w / 2])


def close_cloned_gripper(robot, gripper):
    """ because link and joint names aren't cloned """
    joints = get_joints_by_names(robot, PANDA_FINGERS_GROUP)
    set_joint_positions(gripper, joints, [0, 0])


def set_cloned_se3_conf(robot, gripper, conf):
    joints = get_joints_by_names(robot, SE3_GROUP)
    return set_joint_positions(gripper, joints, conf)


def get_cloned_se3_conf(robot, gripper):
    joints = get_joints_by_names(robot, SE3_GROUP)
    return get_joint_positions(gripper, joints)


def get_cloned_gripper_positions(robot, gripper):
    joints = get_joints_by_names(robot, PANDA_FINGERS_GROUP)
    return get_joint_positions(gripper, joints)


def get_cloned_hand_pose(robot, gripper):
    link = link_from_name(robot, FEG_TOOL_LINK)
    return get_link_pose(gripper, link)


def get_hand_pose(robot):
    link = link_from_name(robot, FEG_TOOL_LINK)
    return get_link_pose(robot, link)


def set_se3_conf(robot, se3):
    set_joint_positions(robot, get_se3_joints(robot), se3)
    # pose = pose_from_se3(se3)
    # set_pose(robot, pose)


def get_se3_joints(robot):
    return get_joints_by_names(robot, SE3_GROUP)


def get_se3_conf(robot):
    return get_joint_positions(robot, get_se3_joints(robot))
    # pose = get_pose(robot)
    # return se3_from_pose(pose)

# def pose_to_se3(p):
#     # return list(p[0]) + list(euler_from_quat(p[1]))
#     print('\n\n franka_utils.pose_to_se3 | deprecated! \n\n')
#     return np.concatenate([np.asarray(p[0]), np.asarray(euler_from_quat(p[1]))])
#
# def se3_to_pose(conf):
#     print('\n\n franka_utils.se3_to_pose | deprecated! \n\n')
#     return (conf[:3], quat_from_euler(conf[3:]))


def se3_from_pose(p):
    print('Deprecated se3_from_pose, please use se3_ik()')
    return list(np.concatenate([np.asarray(p[0]), np.asarray(euler_from_quat(p[1]))]))


def pose_from_se3(conf):
    # print('Deprecated pose_from_se3, please use se3_fk()')
    return (conf[:3], quat_from_euler(conf[3:]))


def se3_ik(robot, target_pose, max_iterations=200, max_time=5, verbose=False, mod_target=None):
    report_failure = False
    debug = False

    if mod_target is not None:
        actual_target = copy.deepcopy(target_pose)
        target_pose = mod_target

    title = f'   se3_ik | for pose {nice(target_pose)}'
    if nice(target_pose) in CACHE:
        if verbose: print(f'{title} found in cache')
        return CACHE[nice(target_pose)]
    start_time = time.time()
    link = link_from_name(robot, FEG_TOOL_LINK)
    target_point, target_quat = target_pose

    sub_joints = get_se3_joints(robot)
    sub_robot = robot.get_gripper()  ## color=BLUE ## for debugging
    limits = [get_joint_limits(robot, j) for j in sub_joints]
    lower_limits = [l[0] for l in limits]
    upper_limits = [l[1] for l in limits]
    # lower_limits[3:] = [-1.5*math.pi] * 3
    # upper_limits[3:] = [1.5*math.pi] * 3

    for iteration in irange(max_iterations):
        if not verbose and elapsed_time(start_time) >= max_time:
            # remove_body(sub_robot)
            if verbose or report_failure: print(f'{title} failed after {max_time} sec')
            return None
        try:
            sub_kinematic_conf = p.calculateInverseKinematics(sub_robot, link, target_point, target_quat,
                                                              lowerLimits=lower_limits, upperLimits=upper_limits,
                                                              physicsClientId=CLIENT)
        except p.error:
            return None
        sub_kinematic_conf = sub_kinematic_conf[:-2] ##[3:-2]
        # conf = list(sub_kinematic_conf[:3])
        # for v in sub_kinematic_conf[3:]:
        #     if v > 2*math.pi:
        #         v -= 2*math.pi
        #     if v < -2*math.pi:
        #         v += 2*math.pi
        #     conf.append(v)
        # sub_kinematic_conf = tuple(conf)

        set_joint_positions(sub_robot, sub_joints, sub_kinematic_conf)
        if verbose and debug: print(f'   se3_ik iter {iteration} | {nice(sub_kinematic_conf, 4)}')
        if is_pose_close(get_link_pose(sub_robot, link), target_pose):
            if verbose:
                print(f'{title} found after {iteration} trials and '
                    f'{nice(elapsed_time(start_time))} sec', nice(sub_kinematic_conf))
                # set_camera_target_body(sub_robot, dx=0.5, dy=0.5, dz=0.5)
            # remove_body(sub_robot)
            if mod_target != None:
                sub_kinematic_conf = list(actual_target[0]) + list(sub_kinematic_conf)[3:]
                CACHE[nice(actual_target)] = sub_kinematic_conf
            else:
                CACHE[nice(target_pose)] = sub_kinematic_conf
            return sub_kinematic_conf
    if verbose or report_failure: print(f'{title} failed after {max_iterations} iterations')
    print(f'\n{title} se3 ik failed \n')
    return None


def approximate_as_box(robot):
    pose = get_pose(robot)
    set_pose(robot, unit_pose())
    aabb = get_aabb(robot)
    set_pose(robot, pose)
    return get_aabb_extent(aabb)


def plan_se3_motion(robot, initial_conf, final_conf, obstacles=[],
                    custom_limits={}, attachments=[], visualize=False):
    joints = get_se3_joints(robot)
    set_joint_positions(robot, joints, initial_conf)
    path = plan_joint_motion(robot, joints, final_conf, obstacles=obstacles,
                             weights=[1, 1, 1, 0.2, 0.2, 0.2],
                             restarts=4, iterations=40, smooth=100,
                             attachments=attachments,
                             self_collisions=False, custom_limits=custom_limits)
    if visualize and path is None:
        start = robot.create_gripper('hand', color=GREEN)
        set_cloned_se3_conf(robot, start, initial_conf)
        end = robot.create_gripper('hand', color=RED)
        set_cloned_se3_conf(robot, end, final_conf)
        set_renderer(True)
        wait_unlocked()
        set_renderer(False)
        remove_body(start)
        remove_body(end)
    return path


def get_free_motion_gen(problem, custom_limits={}, collisions=True, teleport=False,
                        visualize=False, time_step=0.05):
    robot = problem.robot
    saver = BodySaver(robot)
    obstacles = problem.fixed if collisions else []
    title = '   !!!   flying_gripper_utils.get_free_motion_gen with obstacles'

    def fn(q1, q2, fluents=[]):
        if fluents:
            process_motion_fluents(fluents, robot)

        saver.restore()
        q1.assign()
        # set_renderer(visualize)

        raw_path = plan_se3_motion(robot, q1.values, q2.values, obstacles=obstacles,
                                   custom_limits=custom_limits)
        if raw_path is None:
            print(title, obstacles)
            return None
        path = [Conf(robot, get_se3_joints(robot), conf) for conf in raw_path]
        if visualize:
            set_renderer(True)
            for q in subsample_path(path, order=3):
                # wait_for_user('start?')
                q.assign()
                draw_pose(pose_from_se3(q.values), length=0.02)
                wait_for_duration(time_step)
            return raw_path
            q1.assign()
        t = Trajectory(path)
        cmd = Commands(State(), savers=[BodySaver(robot)], commands=[t])
        return (cmd,)
    return fn


def subsample_path(path, order=2, max_len=10, min_len=3):
    return path[::order]


APPROACH_PATH = {}


def get_approach_path(robot, o, g, obstacles=[], verbose=False, custom_limits={}):
    title = 'flying_gripper_utils.get_approach_path |'
    body_pose = robot.get_body_pose(o, verbose=False)
    key = f"{str(g)}-{nice(body_pose)}"
    if key in APPROACH_PATH:
        if verbose:
            print(f'{title} | found in cache for key {key}')
        return APPROACH_PATH[key]

    joints = get_se3_joints(robot)
    approach_pose = multiply(body_pose, g.approach)
    grasp_pose = multiply(body_pose, g.value)
    if verbose:
        print(f'{title} | body_pose = {nice(body_pose)}')
        print(f'{title} | grasp_pose = {nice(grasp_pose)}')
        print(f'{title} | approach_pose = {nice(approach_pose)}')

    seconf1 = se3_ik(robot, approach_pose, verbose=(o==(10,3)))
    seconf2 = se3_ik(robot, grasp_pose, verbose=(o==(10,3)))
    q1 = Conf(robot, joints, seconf1)
    q2 = Conf(robot, joints, seconf2)
    q1.assign()
    if verbose:
        set_renderer(True)
    raw_path = plan_se3_motion(robot, q1.values, q2.values, obstacles=obstacles,
                               custom_limits=custom_limits, visualize=False)
    ## , attachments=attachments.values()
    if raw_path is None or seconf2 is None:
        APPROACH_PATH[key] = None
        return None
    ## to prevent the strange twisting motion
    # path = [Conf(robot, get_se3_joints(robot), conf) for conf in raw_path]
    path = []
    for conf in raw_path:
        conf = list(conf)
        conf[3:] = list(seconf2)[3:]
        path += [Conf(robot, get_se3_joints(robot), conf)]

    APPROACH_PATH[key] = path
    # if verbose:
    #     print(f'{title} | put in cache for key {key}')
    return path


def get_ik_fn(problem, teleport=False, verbose=False,
              custom_limits={}, **kwargs):
    robot = problem.robot
    obstacles = problem.fixed

    def fn(a, o, p, g, fluents=[]):
        if fluents:
            process_motion_fluents(fluents, robot)
        # set_renderer(False)
        p.assign()
        attachments = {}
        path = get_approach_path(robot, o, g, obstacles, verbose=verbose, custom_limits=custom_limits)
        if path == None:
            return None
        t = Trajectory(path)
        q1 = path[0]
        q2 = path[-1]
        cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[t])
        # if end_conf:
        #     return (q1, q2, cmd)
        return (q1, cmd)
    return fn


def get_pull_handle_motion_gen(problem, collisions=True, teleport=False,
                               num_intervals=12, around_to=4, visualize=False, verbose=True):
    if teleport:
        num_intervals = 1
    robot = problem.robot
    world = problem.world
    saver = BodySaver(robot)
    obstacles = problem.fixed if collisions else []
    title = '   !!!    flying_gripper_utils.get_pull_door_handle_motion_gen | step'

    def fn(a, o, pst1, pst2, g, q1, q2, fluents=[]):
        if fluents:
            process_motion_fluents(fluents, robot)
        if pst1.value == pst2.value:
            return None

        # set_renderer(False)
        saver.restore()
        pst1.assign()
        q1.assign()

        handle_link = get_handle_link(o)
        old_pose = get_link_pose(o[0], handle_link)

        # print(f'flying_gripper_utils.get_pull_door_handle_motion_gen | handle_link of body_joint {o} is {joint_object.handle_link}')

        if visualize:
            set_renderer(enable=True)
            gripper = robot.visualize_grasp(old_pose, g.value, verbose=verbose,
                                            width=g.grasp_width, body=g.body)
            set_camera_target_body(gripper, dx=0.2, dy=0, dz=1) ## look top down
            remove_body(gripper)

        ## saving the mapping between robot bconf to object pst for execution
        mapping = {}
        rpose_rounded = tuple([round(n, around_to) for n in q1.values])
        mapping[rpose_rounded] = pst1.value

        path = []
        for i in range(num_intervals):
            step_str = f"{title} {i}/{num_intervals}\t"
            value = (i + 1) / num_intervals * (pst2.value - pst1.value) + pst1.value
            pst_after = Position((pst1.body, pst1.joint), value)
            pst_after.assign()
            # new_pose = get_link_pose(joint_object.body, joint_object.handle_link)
            new_pose = get_link_pose(o[0], handle_link)

            ## somehow without mod_target, ik would fail
            mod_pose = Pose(euler=euler_from_quat(new_pose[1]))
            mod_grasp_pose = multiply(mod_pose, g.value)

            ## just visualizing
            if visualize:
                gripper = robot.visualize_grasp(new_pose, g.value, color=BROWN, verbose=True,
                                        width=1, body=g.body, mod_target=mod_grasp_pose)  ## g.grasp_width
                if gripper is None:
                    if verbose:
                        print(step_str + f"se3_ik failed with g", nice(g.value))
                    break
                set_camera_target_body(gripper, dx=0.2, dy=0, dz=1) ## look top down
                set_camera_target_body(gripper, dx=0.2, dy=0, dz=1) ## look top down
                remove_body(gripper)

            ## actual computation
            gripper_after = multiply(new_pose, g.value)
            gripper_conf = se3_ik(robot, gripper_after, verbose=False, mod_target=mod_grasp_pose)
            if gripper_conf is None:
                if verbose:
                    print(step_str + f"se3_ik failed with g", nice(g.value))
                break
            q_after = Conf(robot, get_se3_joints(robot), gripper_conf)

            q_after.assign()
            if collided(robot, obstacles, world=world, verbose=verbose, tag='handle pull'):
                print(f'{step_str} hand collided', nice(gripper_conf))
                if len(path) > 1:
                    path[-1].assign()
                break
            else:
                path.append(q_after)
                # if verbose: print(f'{step_str} : {nice(q_after.values)}')

            rpose_rounded = tuple([round(n, around_to) for n in q_after.values])
            mapping[rpose_rounded] = value

        if len(path) < num_intervals: ## * 0.75:
            return None

        add_to_rc2oc(robot, a, o, mapping)

        add_data_path()
        t = Trajectory(path)
        cmd = Commands(State(), savers=[BodySaver(robot)], commands=[t])
        q2 = t.path[-1]
        step_str = f"flying_gripper_utils.get_pull_door_handle_motion_gen | step {len(path)}/{num_intervals}\t"
        if not verbose:
            print(f'{step_str} : {nice(q2.values)}')
        return (cmd,)
        # return (q2, cmd)

    return fn


def get_reachable_test(problem, custom_limits={}, visualize=False):
    robot = problem.robot
    obstacles = problem.fixed

    def test(o, p, g, q, fluents=[]):
        if fluents:
            process_motion_fluents(fluents, robot)
        # set_renderer(False)
        with ConfSaver(robot):
            p.assign()
            q.assign()

            body_pose = robot.get_body_pose(g.body)
            approach_pose = multiply(body_pose, g.approach)
            conf = se3_ik(robot, approach_pose)

            result = True
            if conf is None:
                result = False
            else:
                raw_path = plan_se3_motion(robot, q.values, conf, obstacles=obstacles,
                                           custom_limits=custom_limits)
                if raw_path is None:
                    result = False

                elif visualize:
                    set_renderer(True)
                    gripper = robot.create_gripper('hand', color=GREY)
                    set_cloned_se3_conf(robot, gripper, conf)
                    set_camera_target_body(gripper, dx=0.5, dy=-0.5, dz=0.5)  ## look top down
                    remove_body(gripper)
                    # set_renderer(False)

            print(f'       flying_gripper_utils.get_reachable_test({o}, {p}, {q}, {g}) ->\t {result}')
        return result

    return test


class Problem():
    def __init__(self, robot, obstacles):
        self.robot = robot
        self.fixed = obstacles


def quick_demo(world):
    from pybullet_tools.utils import set_all_static
    from world_builder.world import State

    robot = world.robot
    custom_limits = robot.custom_limits

    set_all_static()
    state = State(world)
    world.summarize_all_objects()

    problem = Problem(robot, state.fixed)
    lid = world.name_to_body('lid')

    ## sample grasp
    g_sampler = get_grasp_list_gen(state)
    outputs = g_sampler(lid)
    g = outputs[0][0]
    body_pose = robot.get_body_pose(lid, verbose=False)
    approach_pose = multiply(body_pose, g.approach)

    ## set approach pose as goal pose

    joints = get_se3_joints(robot)
    seconf1 = [0.9, 8, 0.7, 0, -math.pi/2, 0] ## [0.9, 8, 0.4, 0, 0, 0]
    seconf2 = [0.9, 8, 1.4, 0, 0, 0]
    seconf2 = se3_from_pose(approach_pose)
    q1 = Conf(robot, joints, seconf1)
    q2 = Conf(robot, joints, seconf2)
    q1.assign()

    ## plan and execute path, saving the first depth map and all se3 confs
    funk = get_free_motion_gen(problem, custom_limits, visualize=True, time_step=0.1)

    video_path = join('visualizations', 'video_tmp.mp4')
    with VideoSaver(video_path):
        raw_path = funk(q1, q2)
    state.world.visualize_image(((1.7, 8.1, 1), (0.5, 0.5, -0.5, -0.5)))

    ## output to json
    print('len(raw_path)', len(raw_path))

    with open('gripper_traj.txt', 'w') as f:
        f.write('\n'.join([str(nice(p)) for p in raw_path]))

    # wait_if_gui('end?')


def quick_demo_debug(world):
    from pybullet_tools.utils import set_all_static
    from world_builder.world import State

    class Problem():
        def __init__(self, robot, obstacles):
            self.robot = robot
            self.fixed = obstacles

    robot = world.robot
    custom_limits = robot.custom_limits

    set_all_static()
    state = State(world)
    world.summarize_all_objects()

    problem = Problem(robot, state.fixed)
    # bottle = random.choice(world.cat_to_bodies('bottle'))
    bowl = random.choice(world.cat_to_bodies('bowl'))

    ## sample grasp
    g_sampler = get_grasp_list_gen(state, visualize=True)
    outputs = g_sampler(bowl)
    g = outputs[0][0]
    body_pose = robot.get_body_pose(bowl, verbose=False)
    approach_pose = multiply(body_pose, g.approach)

    ## set approach pose as goal pose

    joints = get_se3_joints(robot)
    seconf1 = [0.9, 8, 0.7, 0, -math.pi/2, 0] ## [0.9, 8, 0.4, 0, 0, 0]
    seconf2 = [0.9, 8, 1.4, 0, 0, 0]
    seconf2 = se3_from_pose(approach_pose)
    q1 = Conf(robot, joints, seconf1)
    q2 = Conf(robot, joints, seconf2)
    q1.assign()

    ## plan and execute path, saving the first depth map and all se3 confs
    funk = get_free_motion_gen(problem, custom_limits, visualize=True, time_step=0.1)

    video_path = join('visualizations', 'video_tmp.mp4')
    with VideoSaver(video_path):
        raw_path = funk(q1, q2)
    state.world.visualize_image(((1.7, 8.1, 1), (0.5, 0.5, -0.5, -0.5)))

    ## output to json
    print('len(raw_path)', len(raw_path))

    with open('gripper_traj.txt', 'w') as f:
        f.write('\n'.join([str(nice(p)) for p in raw_path]))


#########################################################################


def tests():
    connect(use_gui=True)
    add_data_path()
    draw_pose(Pose(), length=1.)
    set_camera_pose(camera_point=[1, -1, 1])

    ## create the world
    obstacles = []
    # obstacles = [p.loadURDF("plane.urdf")]
    # obstacles.append(create_box(w=SIZE, l=SIZE, h=SIZE, color=RED))

    ## create thr gripper robot
    robot = create_fe_gripper()
    # dump_body(robot)
    tool_link = link_from_name(robot, 'tool_link')
    draw_pose(Pose(), parent=robot, parent_link=tool_link)

    problem = Problem(robot, obstacles)
    custom_limits = {n: (-1, 1) for n in [0, 1, 2]}

    # test_feg_joints(robot, custom_limits)
    test_feg_motion_planning(problem, custom_limits)
    disconnect()


def test_feg_joints(robot, custom_limits):
    se3_joints = get_se3_joints(robot)
    for joint in se3_joints:
        dump_joint(robot, joint)

    seconf  = [0.6, -0.4, 0, math.pi/2, 0, 0]
    set_joint_positions(robot, se3_joints, seconf)
    print(get_joint_positions(robot, se3_joints))

    ## don't mix them up! won't update the positions when you set pose
    seconf[1] = -seconf[1]
    pose = pose_from_se3(seconf)
    set_pose(robot, pose)
    print(get_pose(robot))

    wait_for_user()


def test_feg_motion_planning(problem, custom_limits):
    from pybullet_tools.pr2_primitives import Conf

    funk = get_free_motion_gen(problem, custom_limits, visualize=True)

    robot = problem.robot
    joints = get_se3_joints(robot)
    seconf1 = [0.6, -0.4, 0, math.pi / 4, math.pi / 2, 0]
    seconf2 = [-0.2, 0.2, 0.8, math.pi / 2, 0, 0]
    q1 = Conf(robot, joints, seconf1)
    q2 = Conf(robot, joints, seconf2)
    q1.assign()
    set_camera_pose(camera_point=[1, -1, 1])

    funk(q1, q2)

    wait_for_user('end?')


if __name__ == '__main__':
    tests()