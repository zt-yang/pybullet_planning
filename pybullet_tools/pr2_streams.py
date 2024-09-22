from __future__ import print_function

import copy
import time

from pybullet_tools.utils import invert, get_all_links, get_name, set_pose, get_link_pose, is_placement, \
    pairwise_collision, set_joint_positions, get_joint_positions, sample_placement, get_pose, waypoints_from_path, \
    unit_quat, plan_base_motion, plan_joint_motion, base_values_from_pose, pose_from_base_values, \
    uniform_pose_generator, add_fixed_constraint, remove_debug, remove_fixed_constraint, \
    disable_real_time, enable_gravity, joint_controller_hold, get_distance, Point, Euler, set_joint_position, \
    get_min_limit, user_input, step_simulation, get_body_name, get_bodies, BASE_LINK, get_joint_position, \
    add_segments, get_max_limit, link_from_name, BodySaver, get_aabb, interpolate_poses, wait_for_user, \
    plan_direct_joint_motion, has_gui, create_attachment, wait_for_duration, get_extend_fn, set_renderer, \
    get_custom_limits, all_between, get_unit_vector, wait_if_gui, create_box, set_point, quat_from_euler, \
    set_base_values, euler_from_quat, INF, elapsed_time, get_moving_links, flatten_links, get_relative_pose, \
    get_joint_limits, unit_pose, point_from_pose, clone_body, set_all_color, GREEN, BROWN, get_link_subtree, \
    RED, remove_body, aabb2d_from_aabb, aabb_overlap, aabb_contains_point, get_aabb_center, get_link_name, \
    get_links, check_initial_end, get_collision_fn, BLUE, WHITE, TAN, GREY, YELLOW, aabb_contains_aabb

from pybullet_tools.bullet_utils import multiply, has_tracik, visualize_bconf
from pybullet_tools.camera_utils import set_camera_target_body
from pybullet_tools.pose_utils import bconf_to_pose, pose_to_bconf, add_pose, sample_new_bconf
from pybullet_tools.grasp_utils import check_cfree_gripper, add_to_rc2oc
from pybullet_tools.ikfast.pr2.ik import pr2_inverse_kinematics
from pybullet_tools.ikfast.utils import USE_CURRENT
from pybullet_tools.pr2_primitives import Grasp, iterate_approach_path, \
    APPROACH_DISTANCE, get_tool_from_root, Conf, Commands, create_trajectory, \
    Trajectory, State
from pybullet_tools.pr2_problems import create_pr2
from pybullet_tools.pr2_utils import open_arm, arm_conf, get_gripper_link, get_arm_joints, SIDE_HOLDING_LEFT_ARM, \
    learned_pose_generator, PR2_TOOL_FRAMES, get_group_joints, get_group_conf, TOOL_POSE, MAX_GRASP_WIDTH, GRASP_LENGTH, \
    SIDE_HEIGHT_OFFSET, approximate_as_prism, set_group_conf, TOP_HOLDING_LEFT_ARM
from pybullet_tools.utils import wait_unlocked, WorldSaver, ConfSaver
from pybullet_tools.general_streams import *


BASE_EXTENT = 3.5 # 2.5
BASE_LIMITS = (-BASE_EXTENT*np.ones(2), BASE_EXTENT*np.ones(2))
GRASP_LENGTH = 0.03
APPROACH_DISTANCE = 0.1 + GRASP_LENGTH
SELF_COLLISIONS = False
DEFAULT_RESOLUTION = math.radians(3) # 0.05


########################################################################


def pr2_grasp(body, value, grasp_type=None):
    if grasp_type is None:
        euler = euler_from_quat(value[1])
        grasp_type = 'top'
        if euler[0] == euler[1] == 0:
            grasp_type = 'side'
    approach_vector = APPROACH_DISTANCE * get_unit_vector([1, 0, 0])
    return Grasp(grasp_type, body, value, multiply((approach_vector, unit_quat()), value),
                 TOP_HOLDING_LEFT_ARM)


# def get_handle_grasps(body_joint, tool_pose=TOOL_POSE, body_pose=unit_pose(),
#                       max_width=MAX_GRASP_WIDTH, grasp_length=GRASP_LENGTH,
#                       robot=None, obstacles=[], full_name=None, world=None):
#     from pybullet_tools.utils import Pose
#     from bullet_utils import find_grasp_in_db, add_grasp_in_db
#
#     DEBUG = True
#
#     PI = math.pi
#     body = body_joint[0]
#     handle_link = get_handle_link(body_joint)
#
#     found, db = find_grasp_in_db('handle_grasps.json', full_name)
#     if found is not None: return found
#
#     handle_pose = get_handle_pose(body_joint)
#
#     # TODO: compute bounding box width wrt tool frame
#     center, (w, l, h) = approximate_as_prism(body, body_pose=body_pose, link=handle_link)
#     translate_center = Pose(point=point_from_pose(body_pose)-center)
#     num_h = 4
#     top_offset = h / num_h
#     gl = grasp_length / 3
#
#     grasps = []
#     rots = [[0,0,0], [0,0,PI/2], [0,0,PI], [0,PI/2,0], [0,PI,0], [PI/2,0,0], [PI,0,0]]
#     title = f'pr2_streams.get_handle_grasps({full_name}): '
#     all_rots = []
#     for i in range(len(rots)):
#         for j in range(len(rots)):
#             r1 = Pose(euler=rots[i])
#             r2 = Pose(euler=rots[j])
#             all_rots.append(multiply(r1, r2))
#             index = f'{i*len(rots)+j}/{len(rots)**2}'
#     #
#     # all_rots = list(set(all_rots))
#     # for i in range(len(all_rots)):
#     #     rot = all_rots[i]
#     #     for gh in range(0, num_h):
#     #         index = f"{i}/{len(all_rots)}"
#
#             for gh in range(0, num_h):
#                 translate_z = Pose(point=[gh*top_offset, 0, w / 2 - gl])
#                 grasp = multiply(tool_pose, translate_z, r1, r2, translate_center, body_pose)
#                 result = check_cfree_gripper(grasp, world, handle_pose, obstacles, visualize=DEBUG) ##
#                 print(f'{title}test grasp ({index}, gh={gh}), {result}')
#                 if result:
#                     if grasp not in grasps:
#                         grasps += [grasp]
#                 else:
#                     break
#
#     ## lastly store the newly sampled grasps
#     add_grasp_in_db(db, db_file, full_name, grasps)
#     return grasps

## -------- moved to robot.
# def iterate_approach_path(robot, arm, gripper, pose_value, grasp, body=None):
#     tool_from_root = get_tool_from_root(robot, arm)
#     grasp_pose = multiply(pose_value, invert(grasp.value))
#     approach_pose = multiply(pose_value, invert(grasp.approach))
#     for tool_pose in interpolate_poses(grasp_pose, approach_pose):
#         set_pose(gripper, multiply(tool_pose, tool_from_root))
#         # if body is not None:
#         #     set_pose(body, multiply(tool_pose, grasp.value))
#         yield

##################################################


def solve_nearby_ik(robot, arm, approach_pose, custom_limits={}):
    arm_joints = get_arm_joints(robot, arm)
    grasp_conf = get_joint_positions(robot, arm_joints)
    if has_tracik():
        from pybullet_tools.tracik import IKSolver
        tool_link = robot.get_tool_link(arm)
        ik_solver = IKSolver(robot, tool_link=tool_link, first_joint=arm_joints[0],
                             custom_limits=custom_limits)  # TODO: cache
        approach_conf = ik_solver.solve(approach_pose, seed_conf=grasp_conf)
    else:
        approach_conf = pr2_inverse_kinematics(robot, arm, approach_pose, custom_limits=custom_limits,
                                               upper_limits=USE_CURRENT, nearby_conf=USE_CURRENT)
    return approach_conf


# def get_ik_fn(problem, custom_limits={}, collisions=True, teleport=False, verbose=True, visualize=False,
#               pick_up=True, given_grasp_conf=False, given_base_conf=True):
#     """ pick_up: whether to move up the arm after attaching to the project
#         if pick_up is False for picking/grasping, then placing/releasing requires given_grasp_conf=True
#     """
#     robot = problem.robot
#     world = problem.world
#     obstacles = problem.fixed if collisions else []
#     ignored_pairs = world.ignored_pairs
#     world_saver = WorldSaver()
#     title = f'\t\tpr2_streams.get_ik_fn (pick_up={pick_up}, given_grasp_conf={given_grasp_conf}, ' \
#             f'given_base_conf={given_base_conf}):\t'
#
#     def helper_before(arm, obj, pose, grasp, base_conf, start_aconf, fluents=[]):
#         if isinstance(obj, tuple):
#             obj = obj[0]
#         obstacles_here = [o for o in obstacles if o != obj]
#         ignored_pairs_here = copy.deepcopy(ignored_pairs)
#
#         if fluents:
#             attachments = process_motion_fluents(fluents, robot)
#             attachments = {a.child: a for a in attachments}
#             obstacles_here.extend([p[1] for p in fluents if p[0] == 'atpose' and p[1] != obj])
#         else:
#             world_saver.restore()
#         collided_kwargs = dict(obstacles=obstacles_here, articulated=False, world=world, verbose=verbose,
#                                ignored_pairs=ignored_pairs_here, min_num_pts=3)
#
#         """ attached objects are only used for collision checking during holding up
#             so they are not used for operator pick_half """
#         if pick_up and False:
#             attachment = grasp.get_attachment(problem.robot, arm, visualize=False)
#             attachments = {attachment.child: attachment}
#         else:
#             attachments = {}
#
#         """ holding joint handles v.s. holding objects """
#         if 'pstn' in str(pose):  ## isinstance(pose, Position):
#             pose_value = linkpose_from_position(pose)
#         else:
#             pose_value = pose.value
#
#         # ## TODO: change to world.get_grasp_parent
#         # addons = [obj]
#         # if hasattr(world, 'BODY_TO_OBJECT') and obj in world.BODY_TO_OBJECT and \
#         #         world.BODY_TO_OBJECT[obj].grasp_parent is not None:
#         #     addons.append(world.BODY_TO_OBJECT[obj].grasp_parent)
#
#         ## see the number of points in object
#         surfaces = [obst for obst in obstacles_here if is_placement(obj, obst)]
#         if collided(obj, surfaces, articulated=False, world=world, verbose=verbose, min_num_pts=0):
#             print(f'{title} !!! object collided with {surfaces}')
#         # ignored_pairs_here.extend([[(body, obst), (obst, body)] for obst in surfaces])
#         # approach_obstacles = problem.world.refine_marker_obstacles(obj, approach_obstacles)  ## for steerables
#
#         tool_from_root = get_tool_from_root(robot, arm)
#         gripper_pose = multiply(robot.get_grasp_pose(pose_value, grasp.value, body=obj), invert(tool_from_root))
#         approach_pose = multiply(robot.get_grasp_pose(pose_value, grasp.approach, body=obj), invert(tool_from_root))
#
#         # sample_fn = get_sample_fn(robot, arm_joints)
#         pose.assign()
#         base_conf.assign()
#         # open_arm(robot, arm)
#
#         """ ======== return directly if default_conf is in collision ================= """
#         arm_joints = get_arm_joints(robot, arm)
#         set_joint_positions(robot, arm_joints, start_aconf)
#
#         if collided(robot.body, **collided_kwargs):
#             if verbose:
#                 print(title, 'default_conf collided with obstacles at bconf')
#             return None
#
#         """ ============ visualize gripper at grasp pose ========== """
#         gripper_grasp = None
#         if visualize:
#             print(title, 'showing grasp_value', nice(grasp.value))
#             gripper_grasp = robot.visualize_grasp(pose_value, grasp.approach, body=obj)
#             set_camera_target_body(gripper_grasp)
#             wait_unlocked()
#
#         return approach_pose, collided_kwargs, attachments, gripper_grasp
#
#     def helper_after(arm, obj, pose, grasp, base_conf, grasp_conf,
#                      approach_pose, collided_kwargs, attachments, gripper_grasp):
#         arm_joints = get_arm_joints(robot, arm)
#         default_conf = arm_conf(arm, grasp.carry)
#
#         """ ============= then get approach conf =========== """
#         if has_tracik():
#             from pybullet_tools.tracik import IKSolver
#             tool_link = robot.get_tool_link(arm)
#             ik_solver = IKSolver(robot, tool_link=tool_link, first_joint=arm_joints[0],
#                                  custom_limits=custom_limits)  # TODO: cache
#             approach_conf = ik_solver.solve(approach_pose, seed_conf=grasp_conf)
#         else:
#             approach_conf = pr2_inverse_kinematics(robot, arm, approach_pose, custom_limits=custom_limits,
#                                                    upper_limits=USE_CURRENT, nearby_conf=USE_CURRENT)
#
#         if (approach_conf is None) or collided(robot, **collided_kwargs):
#             if verbose:
#                 if approach_conf is not None:
#                     approach_conf = nice(approach_conf)
#                 print(f'{title}Approach IK failure, approach_conf =', approach_conf)
#             if visualize:
#                 remove_body(gripper_grasp)
#             return None
#         # elif verbose:
#         #     print(f'{title}Approach IK success | ik({robot} at {nice(base_conf.values)}, '
#         #           f'{arm}, {nice(approach_pose[0])}) | pose = {pose}, grasp = {nice(grasp.approach)} -> {nice(approach_conf)}')
#
#         """ ======== get grasp_path -> approach_conf to grasp_conf ======= """
#         if teleport:
#             path = [default_conf, approach_conf, grasp_conf]
#         else:
#
#             resolutions = math.radians(1)  ## DEFAULT_RESOLUTION
#             resolutions = resolutions * np.ones(len(arm_joints))
#             motion_planning_kwargs = dict(
#                 self_collisions=SELF_COLLISIONS, obstacles=collided_kwargs['obstacles'],
#                 use_aabb=True, cache=True, ignored_pairs=collided_kwargs['ignored_pairs'],
#                 custom_limits=custom_limits, resolutions=resolutions / 2.
#             )
#
#             if is_top_grasp(robot, arm, obj, grasp) or True:
#                 grasp_path = robot.plan_joint_motion(approach_conf, grasp_conf, plan_direct_joint_motion,
#                                                      attachments, arm, pose=pose, debug=False,
#                                                      verbose=verbose, title=title, **motion_planning_kwargs)
#                 if grasp_path is None:
#                     if visualize:
#                         remove_body(gripper_grasp)
#                     if verbose:
#                         print(f'{title}Approach path planning failure')
#                     return None
#                 dest_conf = approach_conf
#             else:
#                 grasp_path = []
#                 dest_conf = grasp_conf
#
#             """ ============ get approach_path -> default_conf to approach_conf ========== """
#             motion_planning_kwargs.update(dict(restarts=2, iterations=25, smooth=0))  # smooth=25
#             approach_path = robot.plan_joint_motion(default_conf, dest_conf, plan_joint_motion, attachments, arm,
#                                                     pose=pose, verbose=verbose, title=title, **motion_planning_kwargs)
#             if approach_path is None:
#                 if visualize:
#                     remove_body(gripper_grasp)
#                 if verbose:
#                     print(f'{title}Grasp path planning failure')
#                 return None
#             path = approach_path + grasp_path
#
#         mt = create_trajectory(robot, arm_joints, path)
#         cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt])
#
#         set_joint_positions(robot, arm_joints, default_conf)  # default_conf | sample_fn()
#
#         if visualize:
#             remove_body(gripper_grasp)
#
#         return cmd
#
#     if not given_grasp_conf and given_base_conf:
#         """ ================================================================
#          case 1: picking up and placing down objects (pick_up=True), grasping handle (pick_up=False)
#          ================================================================ """
#         def fn(arm, obj, pose, grasp, base_conf, fluents=[]):
#             default_conf = arm_conf(arm, grasp.carry)
#
#             result = helper_before(arm, obj, pose, grasp, base_conf, default_conf, fluents=fluents)
#             if result is None:
#                 return None
#             approach_pose, collided_kwargs, attachments, gripper_grasp = result
#
#             """ ============== compute grasp conf ========================= """
#             arm_joints = get_arm_joints(robot, arm)
#             if base_conf.joint_state is not None:
#                 grasp_conf = list(map(base_conf.joint_state.get, arm_joints))
#                 set_joint_positions(robot, arm_joints, grasp_conf)
#             else:
#                 grasp_conf = robot.compute_grasp_aconf(arm, obj, pose, grasp, custom_limits=custom_limits,
#                                                        title=title, **collided_kwargs)
#                 if grasp_conf is None:
#                     if visualize:
#                         remove_body(gripper_grasp)
#                     return None
#
#             """ ========= plan motion of the pick / place ============ """
#             cmd = helper_after(arm, obj, pose, grasp, base_conf, grasp_conf,
#                                approach_pose, collided_kwargs, attachments, gripper_grasp)
#
#             # ## replay sink trajectory
#             # if hasattr(world, 'BODY_TO_OBJECT'):
#             #     mov_name = world.BODY_TO_OBJECT[body].name
#             #     print('       robot base conf during planning, ', nice(robot.get_base_conf().values))
#             #     print('         length of trajectory, ', cmd, len(mt.path))
#             #     if 'bottle' in mov_name:
#             #         robot.test_trajectory(arm, body, pose, grasp, base_conf, cmd)
#
#             if cmd is None:
#                 return None
#
#             if not pick_up:
#                 aconf = cmd.commands[0].path[-1]
#
#                 """ ========= compute the X_base^object for ungrasping ============ """
#                 x, y, z, theta = base_conf.values
#                 bpose = ((x, y, z), quat_from_euler(Euler(yaw=theta)))
#                 ## aconf.x_base_to_object = multiply(bpose, invert(pose.value))
#
#                 return (aconf, cmd)
#
#             return (cmd,)
#
#     else:
#
#         """ ================================================================
#          constrained placing, with a starting arm configuration
#          ================================================================ """
#         def plan_motion(arm, obj, pose, grasp, base_conf, grasp_conf, fluents=[]):
#             grasp_conf = grasp_conf.values
#
#             result = helper_before(arm, obj, pose, grasp, base_conf, grasp_conf, fluents=fluents)
#             if result is None:
#                 return None
#             approach_pose, collided_kwargs, attachments, gripper_grasp = result
#
#             cmd = helper_after(arm, obj, pose, grasp, base_conf, grasp_conf,
#                                approach_pose, collided_kwargs, attachments, gripper_grasp)
#             if cmd is None:
#                 return None
#             return (cmd.commands[0].path[-1], cmd)
#
#         if given_grasp_conf and given_base_conf:
#
#             """ ================================================================
#              case 2: release handle (pick_up=False)
#              ================================================================ """
#             fn = plan_motion
#
#         else:
#
#             """ ================================================================
#              case 3： placing down objects (pick_up=False)
#              ================================================================ """
#             def fn(arm, obj, pose, grasp, grasp_conf, end_aconf, fluents=[]):
#
#                 funk = get_base_from_ik_fn(robot)
#                 base_conf = funk(arm, obj, pose, grasp, grasp_conf)
#                 result = plan_motion(arm, obj, pose, grasp, base_conf, grasp_conf, fluents)
#                 if result is None:
#                     return None
#                 _, cmd = result
#                 return (base_conf, cmd)
#
#     return fn


def get_base_from_ik_fn(robot, visualize=True):

    def gen(arm, body, pose, grasp, arm_conf):
        x_base_to_object = arm_conf.x_base_to_object
        bpose = multiply(x_base_to_object, pose.value)
        (x, y, z), quat = bpose
        theta = euler_from_quat(quat)[-1]
        base_conf = Conf(robot, robot.get_base_joints(), [x, y, z, theta])

        if visualize:
            base_conf.assign()
            arm_conf.assign()
            pose.assign()
            wait_unlocked()

        return base_conf
    return gen

##################################################

# def get_marker_grasp_gen(problem, collisions=False, randomize=True, visualize=False):
#     collisions = True
#     obstacles = problem.fixed if collisions else []
#     world = problem.world
#     def fn(body):
#         grasps = []
#         markers = world.BODY_TO_OBJECT[body].grasp_markers
#         obs = copy.deepcopy(obstacles)
#         if body in obs: obs.remove(body) ## grasp can collide with the object
#         for marker in markers:
#             approach_vector = APPROACH_DISTANCE * get_unit_vector([1, 0, 0]) ##[2, 0, -1])
#             grasps.extend(HandleGrasp('side', marker, g, multiply((approach_vector, unit_quat()), g), SIDE_HOLDING_LEFT_ARM)
#                           for g in get_marker_grasps(marker, grasp_length=GRASP_LENGTH, robot=world.robot, obstacles=obs))  ## , body_pose=body_pose
#             for grasp in grasps:
#                 grasp.grasp_width = 1
#         return [(g,) for g in grasps]
#     return fn


def get_marker_grasp_gen(problem, collisions=False, randomize=False, visualize=True, retrain_all=True):
    world = problem.world
    robot = world.robot
    collisions = True
    obstacles = problem.fixed if collisions else []

    # box_grasp_gen = get_box_grasp_gen(problem, grasp_length=-0.1, grasp_types=['side'],
    #                                   collisions=collisions, randomize=randomize)

    def fn(marker):

        def check_cfree_gripper(grasp):
            gripper_grasp = robot.visualize_grasp(get_pose(marker), grasp.value, new_gripper=visualize)
            if visualize:
                set_camera_target_body(gripper_grasp, dx=0, dy=-1, dz=1)

            for b in obstacles:
                if pairwise_collision(gripper_grasp, b):
                    print('making marker grasp collide with', b)
                    # remove_body(gripper_grasp)
                    return False

            if not retrain_all:
                remove_body(gripper_grasp)
            return True

        # approach_vector = APPROACH_DISTANCE * get_unit_vector([1, 0, 0]) ##[2, 0, -1])
        # approach_pose = (approach_vector, unit_quat())
        # carry = SIDE_HOLDING_LEFT_ARM
        # grasps = get_marker_grasps(marker, grasp_length=GRASP_LENGTH, robot=world.robot, obstacles=obstacles)
        # for grasp in grasps:
        #     grasp.grasp_width = 1
        # grasps = [HandleGrasp('side', marker, g, multiply(approach_pose, g), carry) for g in grasps]

        grasps = get_hand_grasps(world, marker)
        if collisions:
            grasps = [g for g in grasps if check_cfree_gripper(g[0])]

        return [(g,) for g in grasps]
    return fn


def get_marker_grasps(body, under=False, tool_pose=TOOL_POSE, body_pose=unit_pose(),
                    max_width=MAX_GRASP_WIDTH, grasp_length=GRASP_LENGTH, top_offset=SIDE_HEIGHT_OFFSET,
                    robot=None, obstacles=[]):

    from pybullet_tools.utils import Pose



    # TODO: compute bounding box width wrt tool frame
    center, (w, l, h) = approximate_as_prism(body, body_pose=body_pose)
    translate_center = Pose(point=point_from_pose(body_pose)-center)
    grasps = []
    x_offset = h/2 - top_offset
    under = True
    for j in range(1 + under):
        swap_xz = Pose(euler=[0, -math.pi / 2 + j * math.pi, 0])
        if l <= max_width:
            for gl in [grasp_length]:  ## [grasp_length/3, grasp_length/2, grasp_length]:
                translate_z = Pose(point=[x_offset, 0, w / 2 - gl])
                for i in range(1, 2):
                    rotate_z = Pose(euler=[0, 0, i * math.pi/2])
                    grasp = multiply(tool_pose, translate_z, rotate_z, swap_xz, translate_center, body_pose)
                    if check_cfree_gripper(grasp):
                        grasps += [grasp]  # , np.array([l])
    return grasps

# def visualize_grasp(robot, body_pose, grasp, arm='left', color=GREEN):
#     link_name = PR2_GRIPPER_ROOTS[arm]
#     links = get_link_subtree(robot, link_from_name(robot, link_name))
#     gripper_grasp = clone_body(robot, links=links, visual=True, collision=True)
#     open_cloned_gripper(gripper_grasp)
#
#     set_all_color(gripper_grasp, color)
#     tool_from_root = get_tool_from_root(robot, arm)
#     grasp_pose = multiply(multiply(body_pose, invert(grasp)), tool_from_root)
#     set_pose(gripper_grasp, grasp_pose)
#
#     direction = get_gripper_direction(grasp_pose)
#     print('\n', nice(grasp_pose), direction)
#     if direction == None:
#         print('new direction')
#         return gripper_grasp, True
#     if 'down' in direction:
#         print('pointing down')
#         return gripper_grasp, True
#     return gripper_grasp, False


##################################################

def sample_points_along_line(body, marker, num_intervals=None, bq=None,
                             limit=(2.5, 3), learned=False):

    x1, y1, z1 = get_pose(body)[0]
    (x2, y2, z2), quat = get_pose(marker)
    k = (y2 - y1) / (x2 - x1)

    def sample_point():
        lo, hi = limit
        dx = np.random.uniform(lo, hi)
        if learned: dx = lo
        return dx, (x2 + dx, y2 + k * dx)

    dx, (x, y) = sample_point()
    pose2 = ((x, y, z2), quat)

    if num_intervals != None and bq != None:
        rx, ry, ryaw = bq
        bqs = []  ## base config of robot
        for n in range(num_intervals):
            x = rx + dx * ((n+1)/num_intervals)
            y = ry + k * dx * ((n+1)/num_intervals)
            bqs.append((x, y, ryaw))
        return pose2, bqs

    return pose2


def get_bqs_given_p2(marker, parent, bq, pose2, num_intervals):
    x1, y1, z1 = get_pose(parent)[0]
    (x2, y2, z2), quat = get_pose(marker)
    k = (y2 - y1) / (x2 - x1)

    ## reverse engineer the dx
    ((x, y, z2), quat) = pose2
    dx = x - x2

    rx, ry, ryaw = bq
    bqs = []  ## base config of robot
    for n in range(num_intervals):
        x = rx + dx * ((n + 1) / num_intervals)
        y = ry + k * dx * ((n + 1) / num_intervals)
        bqs.append((x, y, ryaw))

    return bqs


def get_bqs_given_bq2(marker, parent, bq1, bq2, num_intervals):
    x1, y1, z1 = get_pose(parent)[0]
    (x2, y2, z2), quat = get_pose(marker)
    k = (y2 - y1) / (x2 - x1)

    ## reverse engineer the dx
    dx = bq2[0] - bq1[0]
    x = x2 + dx
    y = y2 + k * dx
    pose2 = ((x, y, z2), quat)

    rx, ry, ryaw = bq1
    bqs = []
    for n in range(num_intervals):
        x = rx + dx * ((n + 1) / num_intervals)
        y = ry + k * dx * ((n + 1) / num_intervals)
        bqs.append((x, y, ryaw))

    return pose2, bqs


def get_marker_pose_gen(problem, num_samples=70, collisions=False, visualize=False):
    from pybullet_tools.pr2_primitives import Pose
    collisions = True
    world = problem.world
    def fn(o, p1):
        poses = []
        parent = world.BODY_TO_OBJECT[o].grasp_parent
        p1.assign()

        for i in range(num_samples):
            pose2 = sample_points_along_line(parent, o)
            if visualize:
                visualize_point(pose2[0], world)
            p2 = Pose(o, pose2)
            poses.append((p2,))
        return poses
    return fn


def get_parent_new_pose(p1, p2, p3):
    x1, y1, _ = p1[0]
    x2, y2, _ = p2[0]
    (x3, y3, z3), quat = p3
    return ((x3+x2-x1, y3+y2-y1, z3), quat)


def get_pull_marker_random_motion_gen(problem, custom_limits={}, collisions=True, max_attempts=30,
                               teleport=False, num_intervals=30, learned=False):
    from pybullet_tools.pr2_primitives import Pose

    if teleport:
        num_intervals = 1
    robot = problem.robot
    saver = BodySaver(robot)
    obstacles = problem.fixed if collisions else []
    world = problem.world
    def fn(a, o, p1, g, bq1, o2, p3, fluents=[]):

        parent = world.BODY_TO_OBJECT[o].grasp_parent
        approach_obstacles = copy.deepcopy(obstacles)
        if parent in approach_obstacles: approach_obstacles.remove(parent)

        attempts = 0
        while attempts < max_attempts:
            saver.restore()
            p1.assign()
            bq1.assign()

            ## sample a p2 along the direction and deduce bconf
            pose2, bqs = sample_points_along_line(parent, o, num_intervals, bq1.values, learned=learned)
            p2 = Pose(o, pose2)
            p4 = Pose(o2, get_parent_new_pose(p1.value, pose2, p3.value))

            bpath = [Conf(robot, robot.get_base_joints(), bq) for bq in bqs]
            collided = False

            ## TODO: do collision checking with other streams
            for bq in bpath:
                bq.assign()
                if any(pairwise_collision(robot, b) or pairwise_collision(parent, b) for b in approach_obstacles):
                    collided = True
            if not collided:
                bt = Trajectory(bpath)
                cmd = Commands(State(), savers=[BodySaver(robot.body)], commands=[bt])
                yield (p2, bpath[-1], p4, cmd)
    return fn


def get_pull_marker_to_pose_motion_gen(problem, custom_limits={}, collisions=True,
                               teleport=False, num_intervals=30):
    from pybullet_tools.pr2_primitives import Pose

    if teleport:
        num_intervals = 1
    robot = problem.robot
    saver = BodySaver(robot)
    obstacles = problem.fixed if collisions else []
    world = problem.world
    def fn(a, o, p1, p2, g, bq1, o2, p3, fluents=[]):

        parent = world.BODY_TO_OBJECT[o].grasp_parent
        approach_obstacles = copy.deepcopy(obstacles)
        if parent in approach_obstacles: approach_obstacles.remove(parent)

        saver.restore()
        p1.assign()
        bq1.assign()

        ## get list of bconf sample along the direction given p2
        bqs = get_bqs_given_p2(o, parent, bq1.values, p2.value, num_intervals)
        p4 = Pose(o2, get_parent_new_pose(p1.value, p2.value, p3.value))

        bpath = [Conf(robot, robot.get_base_joints(), bq) for bq in bqs]
        collided = False

        ## TODO: do collision checking with other streams
        for bq in bpath:
            bq.assign()
            if any(pairwise_collision(robot, b) or pairwise_collision(parent, b) for b in approach_obstacles):
                collided = True
        if not collided:
            bt = Trajectory(bpath)
            cmd = Commands(State(), savers=[BodySaver(robot.body)], commands=[bt])
            return (bpath[-1], p4, cmd)
        return None

    return fn


def get_pull_marker_to_bconf_motion_gen(problem, custom_limits={}, collisions=True,
                               teleport=False, num_intervals=30):
    from pybullet_tools.pr2_primitives import Pose

    if teleport:
        num_intervals = 1
    robot = problem.robot
    saver = BodySaver(robot)
    obstacles = problem.fixed if collisions else []
    world = problem.world
    def fn(a, o, p1, g, bq1, bq2, o2, p3, fluents=[]):

        parent = world.BODY_TO_OBJECT[o].grasp_parent
        approach_obstacles = copy.deepcopy(obstacles)
        if parent in approach_obstacles: approach_obstacles.remove(parent)

        saver.restore()
        p1.assign()
        bq1.assign()

        ## get list of bconf sample along the direction given p2
        pose2, bqs = get_bqs_given_bq2(o, parent, bq1.values, bq2.values, num_intervals)
        p2 = Pose(o, pose2)
        p4 = Pose(o2, get_parent_new_pose(p1.value, pose2, p3.value))

        bpath = [Conf(robot, get_group_joints(robot, 'base'), bq) for bq in bqs]
        collided = False

        ## TODO: do collision checking with other streams
        for bq in bpath:
            bq.assign()
            if any(pairwise_collision(robot, b) or pairwise_collision(parent, b) for b in approach_obstacles):
                collided = True
        if not collided:
            bt = Trajectory(bpath)
            cmd = Commands(State(), savers=[BodySaver(robot.body)], commands=[bt])
            return (p2, p4, cmd)
        return None

    return fn


##################################################


def get_pose_in_region_test():
    def test(o, p, r):
        p.assign()
        obj_aabb = aabb2d_from_aabb(get_aabb(o))
        region_aabb = aabb2d_from_aabb(get_aabb(r))
        # return aabb_overlap(obj_aabb, region_aabb)

        obj_center = get_aabb_center(obj_aabb)
        return aabb_contains_point(obj_aabb, region_aabb)
    return test

def get_bconf_in_region_test(robot):
    rob = robot
    def test(bq, r):
        bq.assign()
        ## needs to be only rob base because arm may stick over in the air
        rob_aabb = aabb2d_from_aabb(get_aabb(rob, link_from_name(rob, "base_link")))
        region_aabb = aabb2d_from_aabb(get_aabb(r))
        # return aabb_overlap(rob_aabb, region_aabb)

        rob_center = get_aabb_center(rob_aabb)
        return aabb_contains_point(rob_center, region_aabb)
    return test


def get_bconf_in_region_gen(problem, collisions=True, max_attempts=10, verbose=False, visualize=False):
    obstacles = problem.fixed if collisions else []
    robot = problem.robot
    yaw = 0

    def gen(region):
        lower, upper = get_aabb(region)
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            x = np.random.uniform(lower[0], upper[0])
            y = np.random.uniform(lower[1], upper[1])
            bq = Conf(robot, robot.get_base_joints(), (x, y, yaw))
            bq.assign()
            if not any(pairwise_collision(robot, obst) for obst in obstacles):
                if visualize:
                    rbb = create_pr2()
                    set_group_conf(rbb, 'base', bq.values)
                yield (bq,)

    def list_fn(region):
        lower, upper = get_aabb(region)
        attempts = 0
        bqs = []
        while attempts < max_attempts:
            attempts += 1
            positions = list(robot.get_base_positions())
            x = np.random.uniform(lower[0], upper[0])
            y = np.random.uniform(lower[1], upper[1])
            yaw = np.random.uniform(0, PI*2)
            positions[:2] = [x, y]
            positions[-1] = yaw
            bq = Conf(robot, robot.get_base_joints(), positions)
            with ConfSaver(robot):
                bq.assign()
                if not any(pairwise_collision(robot, obst) for obst in obstacles):
                    if visualize:
                        rbb = create_pr2()
                        set_group_conf(rbb, robot.base_group, bq.values)
                    bqs.append((bq,))
        return bqs
    return list_fn ## gen


def get_pose_in_region_gen(problem, collisions=True, max_attempts=40, verbose=False, visualize=False):
    from pybullet_tools.pr2_primitives import Pose
    obstacles = problem.fixed if collisions else []
    robot = problem.robot
    def gen(o, r):
        ((_, _, z), quat) = get_pose(o)
        lower, upper = get_aabb(r)
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            x = np.random.uniform(lower[0], upper[0])
            y = np.random.uniform(lower[1], upper[1])
            pose = (x, y, z), quat
            # x, y, z, yaw = sample_obj_in_body_link_space(robot, region,
            #     PLACEMENT_ONLY=True, XY_ONLY=True, verbose=verbose, **kwargs)
            p = Pose(o, pose)
            p.assign()
            if not any(pairwise_collision(o, obst) for obst in obstacles):
                yield (p,)
    # return gen

    def list_fn(o, r):
        ((_, _, z), quat) = get_pose(o)
        lower, upper = get_aabb(r)
        attempts = 0
        poses = []
        while attempts < max_attempts:
            attempts += 1
            x = np.random.uniform(lower[0], upper[0])
            y = np.random.uniform(lower[1], upper[1])
            pose = (x, y, z), quat
            # x, y, z, yaw = sample_obj_in_body_link_space(robot, region,
            #     PLACEMENT_ONLY=True, XY_ONLY=True, verbose=verbose, **kwargs)
            p = Pose(o, pose)
            p.assign()
            if not any(pairwise_collision(o, obst) for obst in obstacles):
                poses.append((p,))
        return poses
    return list_fn ## gen

##################################################


def get_base_motion_gen(problem, custom_limits={}, collisions=True, teleport=False, debug=False):
    robot = problem.robot
    saver = BodySaver(robot)
    obstacles = problem.fixed if collisions else []
    ignored_pairs = problem.world.ignored_pairs

    def fn(bq1, bq2, fluents=[]):
        obstacles_here = copy.deepcopy(obstacles)
        saver.restore()
        # print(f'get_base_motion_gen({nice(bq1.values)}, {nice(bq2.values)})')
        attachments = process_motion_fluents(fluents, robot, verbose=False)
        if len(attachments) > 0:
            # print('get_base_motion_gen, attachments', attachments)
            obstacles_here = [o for o in obstacles if o not in attachments]

        ## add the object to be grasped and placement surface in ignored_pairs
        ignored_pairs_here = copy.deepcopy(ignored_pairs)
        for a in attachments:
            if a.parent == robot:
                surface = robot.world.body_to_object(1).supporting_surface
                if surface is not None:
                    pair = (surface.body, a.child)
                    if pair not in ignored_pairs_here:
                        ignored_pairs_here.extend([pair, (a.child, surface.body)])

        bq1.assign()
        # TODO: did base motion planning fail?
        # TODO: add objects to obstacles

        # bconf = get_joint_positions(robot, robot.get_base_joints())
        # aconf = get_joint_positions(robot, get_arm_joints(robot, 'left'))

        # arm = robot.arms[0]
        # arm_joints = get_arm_joints(robot, arm) # TODO(caelan): should set the arms conf
        # default_conf = arm_conf(arm, grasp.carry)
        # set_joint_positions(robot, arm_joints, default_conf)

        params = [(4, 75)] ## (4, 50), (10, 100)
        num_trials = len(params)  ## sometimes it can't find a path to get around the open door
        while num_trials > 0:
            param = params[-num_trials]
            raw_path = plan_joint_motion(robot, bq2.joints, bq2.values, attachments=attachments,
                                         obstacles=obstacles_here, self_collisions=robot.self_collisions,
                                         custom_limits=custom_limits, resolutions=None, # TODO: base resolutions
                                         use_aabb=True, cache=True,
                                         restarts=param[0], iterations=param[1], smooth=50,
                                         ignored_pairs=ignored_pairs_here
                                         ) # smooth=50
                                         # restarts=4, iterations=50, smooth=50)
            # break
            num_trials -= 1
            if raw_path != None:
                break
            bq1.assign()

        if raw_path is None:
            print(f'Failed motion plan (with world config)! obstacles_here = {obstacles_here}, obstacles = {obstacles}')
            if debug:
                for i, bq in enumerate([bq1, bq2]):
                    bq.assign()
                    print('{}) Base conf: {}'.format(i, bq))
                    wait_unlocked()
            return None
        path = [Conf(robot, bq2.joints, q) for q in raw_path]
        bt = Trajectory(path)
        cmd = Commands(State(), savers=[BodySaver(robot.body)], commands=[bt])
        return (cmd,)
    return fn

##################################################


def get_cfree_btraj_pose_test(robot, collisions=True, verbose=True):
    def test(c, b2, p2):
        # TODO: infer robot from c
        if not collisions:
            return True
        state = c.assign()
        if b2 in state.attachments:
            return True
        p2.assign()

        if verbose:
            robot_pose = robot.get_pose()
            print('    get_cfree_btraj_pose_test   \    pose of robot', nice(robot_pose))
        for _ in c.apply(state):
            state.assign()
            for b1 in state.attachments:
                if pairwise_collision(b1, b2):
                    if verbose:
                        print(f'      collision with {b1}, {b2}')
                        print(f'         pose of {b1}', nice(get_pose(b1)))
                        print(f'         pose of {b2}', nice(get_pose(b2)))
                    #wait_for_user()
                    return False
            if pairwise_collision(robot, b2):
                if verbose:
                    print(f'      collision {robot}, {b2}')
                    print(f'         pose of robot', nice(robot.get_pose()))
                    print(f'         pose of {b2}', nice(get_pose(b2)))
                return False
        # TODO: just check collisions with moving links
        return True
    return test


