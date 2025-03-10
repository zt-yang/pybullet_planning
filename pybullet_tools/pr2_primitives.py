from __future__ import print_function

import copy
import pybullet as p
import random
import time
from itertools import islice, count

import numpy as np

from .ikfast.pr2.ik import is_ik_compiled, pr2_inverse_kinematics
from .ikfast.utils import USE_CURRENT, USE_ALL
from pybullet_tools.pr2_problems import get_fixed_bodies
from pybullet_tools.pr2_utils import GET_GRASPS, get_gripper_joints, \
    get_carry_conf, get_top_grasps, get_side_grasps, open_arm, arm_conf, get_gripper_link, get_arm_joints, \
    learned_pose_generator, PR2_TOOL_FRAMES, get_x_presses, PR2_GROUPS, joints_from_names, \
    is_drake_pr2, get_group_joints, get_group_conf, compute_grasp_width, PR2_GRIPPER_ROOTS
from pybullet_tools.utils import invert, multiply, get_name, set_pose, get_link_pose, is_placement, \
    pairwise_collision, set_joint_positions, get_joint_positions, sample_placement, get_pose, waypoints_from_path, \
    unit_quat, plan_base_motion, plan_joint_motion, base_values_from_pose, pose_from_base_values, \
    uniform_pose_generator, sub_inverse_kinematics, add_fixed_constraint, remove_debug, remove_fixed_constraint, \
    disable_real_time, enable_gravity, joint_controller_hold, get_distance, get_links, get_link_name, \
    get_min_limit, user_input, step_simulation, get_body_name, get_bodies, BASE_LINK, \
    add_segments, get_max_limit, link_from_name, BodySaver, get_aabb, interpolate_poses, \
    plan_direct_joint_motion, has_gui, wait_for_duration, get_extend_fn, set_renderer, \
    get_custom_limits, all_between, get_unit_vector, wait_if_gui, joint_from_name, create_box, set_point, \
    set_base_values, euler_from_quat, INF, elapsed_time, get_moving_links, flatten_links, get_relative_pose, \
    wait_unlocked, child_link_from_joint, get_rigid_clusters, link_pairs_collision

from pybullet_tools.bullet_utils import nice
from pybullet_tools.camera_utils import set_camera_target_robot, set_camera_target_body


BASE_EXTENT = 3.5 # 2.5
BASE_LIMITS = (-BASE_EXTENT*np.ones(2), BASE_EXTENT*np.ones(2))
GRASP_LENGTH = 0.03
APPROACH_DISTANCE = 0.1 + GRASP_LENGTH
SELF_COLLISIONS = False

##################################################

def get_base_limits(robot):
    if is_drake_pr2(robot):
        joints = get_group_joints(robot, 'base')[:2]
        lower = [get_min_limit(robot, j) for j in joints]
        upper = [get_max_limit(robot, j) for j in joints]
        return lower, upper
    return BASE_LIMITS

##################################################

class Pose(object):
    num = count()
    def __init__(self, body, value=None, support=None, init=False, index=None):
        self.body = body
        if value is None:
            value = get_pose(self.body)
        self.value = tuple(value)
        self.support = support
        self.init = init
        if index is None:
            index = next(self.num)
        self.index = index
        self.confs = tuple([])
    @property
    def bodies(self):
        return flatten_links(self.body)
    def assign(self):
        set_pose(self.body, self.value)
    def iterate(self):
        yield self
    def to_base_conf(self):
        values = base_values_from_pose(self.value)
        return Conf(self.body, range(len(values)), values)
    def __repr__(self):
        index = self.index
        #index = id(self) % 1000
        return 'p{}={}'.format(index, nice(self.value))
        # return 'p{}'.format(index)

class Grasp(object):
    def __init__(self, grasp_type, body, value, approach, carry, index=None):
        """ don't use grasp.carry because it should be arm independent """
        self.grasp_type = grasp_type
        self.body = body
        self.value = tuple(value) # gripper_from_object
        self.approach = tuple(approach)
        self.carry = tuple(carry)
        if index == None:
            index = id(self)
        self.index = index

    def get_attachment(self, robot, arm, **kwargs):
        return robot.make_attachment(self, arm, **kwargs)

    def __repr__(self):
        return 'g{}={}'.format(self.index % 1000, nice(self.value))
        # return 'g{}'.format(id(self) % 1000)


class Conf(object):
    def __init__(self, body, joints, values=None, init=False, index=None, joint_state=None):
        self.body = body if isinstance(body, int) else body.body
        self.joints = joints
        if values is None:
            values = get_joint_positions(self.body, self.joints)
        self.values = tuple(values)
        self.init = init
        if index is None:
            index = id(self)
        self.index = index
        self.joint_state = joint_state
        self.x_base_to_object = None
    @property
    def bodies(self): # TODO: misnomer
        return flatten_links(self.body, get_moving_links(self.body, self.joints))
    def assign(self):
        set_joint_positions(self.body, self.joints, self.values)
        if self.joint_state is not None:
            joints = self.joint_state.keys()
            values = self.joint_state.values()
            set_joint_positions(self.body, joints, values)
    def iterate(self):
        yield self
    def __repr__(self):
        if len(self.joints) == 7:
            return 'aq{}={}'.format(self.index % 1000, nice(self.values))
        return 'q{}={}'.format(self.index % 1000, nice(self.values))
        # return 'q{}'.format(id(self) % 1000)

#####################################

class Command(object):
    def control(self, dt=0):
        raise NotImplementedError()
    def apply(self, state, **kwargs):
        raise NotImplementedError()
    def iterate(self):
        raise NotImplementedError()

class Commands(Command):
    def __init__(self, state, savers=[], commands=[], index=None):
        self.state = state
        self.savers = tuple(savers)
        self.commands = tuple(commands)
        if index == None:
            index = id(self)
        self.index = index
    def assign(self):
        for saver in self.savers:
            saver.restore()
        return copy.copy(self.state)
    def apply(self, state, **kwargs):
        for command in self.commands:
            for result in command.apply(state, **kwargs):
                yield result
    def reverse(self):
        return self.__class__(self.state, savers=self.savers, index=self.index,
                              commands=[command.reverse() for command in reversed(self.commands)])
    def __repr__(self):
        commands = self.commands
        if len(commands) == 1:
            commands = commands[0]  ## YANG: to prevent error in prolog inference
        return 'c{}={}'.format(self.index % 1000, commands)
        # return 'c{}'.format(id(self) % 1000)

class Simultaneous(Command): # Parallel
    def __init__(self, commands=[]):
        self.commands = tuple(commands)
    def apply(self, state, **kwargs):
        iterators = [command.apply(state, **kwargs) for command in self.commands]
        while iterators:
            remaining_iterators = []
            outputs = []
            for iterator in iterators:
                try:
                    outputs.append(next(iterator)) # TODO: return False for the rest
                    remaining_iterators.append(iterator)
                except StopIteration:
                    pass
            yield outputs
            iterators = remaining_iterators
    def reverse(self):
        return self.__class__([command.reverse() for command in reversed(self.commands)])
    def __repr__(self):
        return '{{{}}}'.format(', '.join(map(repr, self.commands)))

#####################################

class Trajectory(Command):
    _draw = False
    def __init__(self, path):
        self.path = tuple(path)
        # TODO: constructor that takes in this info
    def apply(self, state, sample=1):
        handles = add_segments(self.to_points()) if self._draw and has_gui() else []
        for conf in self.path[::sample]:
            conf.assign()
            yield
        end_conf = self.path[-1]
        if isinstance(end_conf, Pose):
            state.poses[end_conf.body] = end_conf
        for handle in handles:
            remove_debug(handle)
    def control(self, dt=0, **kwargs):
        # TODO: just waypoints
        for conf in self.path:
            if isinstance(conf, Pose):
                conf = conf.to_base_conf()
            for _ in joint_controller_hold(conf.body, conf.joints, conf.values):
                step_simulation()
                time.sleep(dt)
    def to_points(self, link=BASE_LINK):
        # TODO: this is computationally expensive
        points = []
        for conf in self.path:
            with BodySaver(conf.body):
                conf.assign()
                #point = np.array(point_from_pose(get_link_pose(conf.body, link)))
                point = np.array(get_group_conf(conf.body, 'base'))
                point[2] = 0
                point += 1e-2*np.array([0, 0, 1])
                if not (points and np.allclose(points[-1], point, atol=1e-3, rtol=0)):
                    points.append(point)
        points = get_target_path(self)
        return waypoints_from_path(points)
    def distance(self, distance_fn=get_distance):
        total = 0.
        for q1, q2 in zip(self.path, self.path[1:]):
            total += distance_fn(q1.values, q2.values)
        return total
    def iterate(self):
        for conf in self.path:
            yield conf
    def reverse(self):
        return Trajectory(reversed(self.path))
    #def __repr__(self):
    #    return 't{}'.format(id(self) % 1000)
    def __repr__(self):
        d = 0
        if self.path:
            conf = self.path[0]
            d = 3 if isinstance(conf, Pose) else len(conf.joints)
        return 't({}, {})'.format(d, len(self.path))

def create_trajectory(robot, joints, path):
    return Trajectory(Conf(robot, joints, q) for q in path)

##################################################

class GripperCommand(Command):
    def __init__(self, robot, arm, position, teleport=False):
        self.robot = robot
        self.arm = arm
        self.position = position
        self.teleport = teleport
    def apply(self, state, **kwargs):
        joints = get_gripper_joints(self.robot, self.arm)
        start_conf = get_joint_positions(self.robot, joints)
        end_conf = [self.position] * len(joints)
        if self.teleport:
            path = [start_conf, end_conf]
        else:
            extend_fn = get_extend_fn(self.robot, joints)
            path = [start_conf] + list(extend_fn(start_conf, end_conf))
        for positions in path:
            set_joint_positions(self.robot, joints, positions)
            yield positions
    def control(self, **kwargs):
        joints = get_gripper_joints(self.robot, self.arm)
        positions = [self.position]*len(joints)
        for _ in joint_controller_hold(self.robot, joints, positions):
            yield

    def __repr__(self):
        return '{}({},{},{})'.format(self.__class__.__name__, get_body_name(self.robot),
                                     self.arm, self.position)

class Attach(Command):
    vacuum = True
    def __init__(self, robot, arm, grasp, body):
        self.robot = robot
        self.arm = arm
        self.grasp = grasp
        self.body = body
        self.link = link_from_name(self.robot, PR2_TOOL_FRAMES.get(self.arm, self.arm))
        #self.attachment = None
    def assign(self):
        gripper_pose = get_link_pose(self.robot, self.link)
        body_pose = multiply(gripper_pose, self.grasp.value)
        set_pose(self.body, body_pose)
    def apply(self, state, **kwargs):
        from pybullet_tools.pose_utils import create_attachment
        state.attachments[self.body] = create_attachment(self.robot, self.link, self.body)
        state.grasps[self.body] = self.grasp
        if self.body not in state.poses:
            print('Warning: attaching body without pose')
        del state.poses[self.body]
        yield
    def control(self, dt=0, **kwargs):
        if self.vacuum:
            add_fixed_constraint(self.body, self.robot, self.link)
            #add_fixed_constraint(self.body, self.robot, self.link, max_force=1) # Less force makes it easier to pick
        else:
            # TODO: the gripper doesn't quite work yet
            gripper_name = '{}_gripper'.format(self.arm)
            joints = joints_from_names(self.robot, PR2_GROUPS[gripper_name])
            values = [get_min_limit(self.robot, joint) for joint in joints] # Closed
            for _ in joint_controller_hold(self.robot, joints, values):
                step_simulation()
                time.sleep(dt)
    def __repr__(self):
        return '{}({},{},{})'.format(self.__class__.__name__, get_body_name(self.robot),
                                     self.arm, get_name(self.body))

class Detach(Command):
    def __init__(self, robot, arm, body):
        self.robot = robot
        self.arm = arm
        self.body = body
        self.link = link_from_name(self.robot, PR2_TOOL_FRAMES.get(self.arm, self.arm))
        # TODO: pose argument to maintain same object
    def apply(self, state, **kwargs):
        del state.attachments[self.body]
        state.poses[self.body] = Pose(self.body, get_pose(self.body))
        del state.grasps[self.body]
        yield
    def control(self, **kwargs):
        remove_fixed_constraint(self.body, self.robot, self.link)
    def __repr__(self):
        return '{}({},{},{})'.format(self.__class__.__name__, get_body_name(self.robot),
                                     self.arm, get_name(self.body))

##################################################

class Clean(Command):
    def __init__(self, body):
        self.body = body
    def apply(self, state, **kwargs):
        state.cleaned.add(self.body)
        self.control()
        yield
    def control(self, **kwargs):
        p.addUserDebugText('Cleaned', textPosition=(0, 0, .25), textColorRGB=(0,0,1), #textSize=1,
                           lifeTime=0, parentObjectUniqueId=self.body)
        #p.setDebugObjectColor(self.body, 0, objectDebugColorRGB=(0,0,1))
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.body)

class Cook(Command):
    # TODO: global state here?
    def __init__(self, body):
        self.body = body
    def apply(self, state, **kwargs):
        state.cleaned.remove(self.body)
        state.cooked.add(self.body)
        self.control()
        yield
    def control(self, **kwargs):
        # changeVisualShape
        # setDebugObjectColor
        #p.removeUserDebugItem # TODO: remove cleaned
        p.addUserDebugText('Cooked', textPosition=(0, 0, .5), textColorRGB=(1,0,0), #textSize=1,
                           lifeTime=0, parentObjectUniqueId=self.body)
        #p.setDebugObjectColor(self.body, 0, objectDebugColorRGB=(1,0,0))
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.body)

##################################################

##################################################

def accelerate_gen_fn(gen_fn, max_attempts=1):
    def new_gen_fn(*inputs):
        generator = gen_fn(*inputs)
        while True:
            for i in range(max_attempts):
                try:
                    output = next(generator)
                except StopIteration:
                    return
                if output is not None:
                    print(gen_fn.__name__, i)
                    yield output
                    break
    return new_gen_fn

def get_stable_gen(problem, collisions=True, **kwargs):
    obstacles = problem.fixed if collisions else []
    def gen(body, surface):
        # TODO: surface poses are being sampled in pr2_belief
        if surface is None:
            surfaces = problem.surfaces
        else:
            surfaces = [surface]
        while True:
            surface = random.choice(surfaces) # TODO: weight by area
            body_pose = sample_placement(body, surface, **kwargs)
            if body_pose is None:
                break
            p = Pose(body, body_pose, surface)
            p.assign()
            if not any(pairwise_collision(body, obst) for obst in obstacles if obst not in {body, surface}):
                yield (p,)
    # TODO: apply the acceleration technique here
    return gen

##################################################

def get_tool_from_root(robot, arm):
    root_link = link_from_name(robot, PR2_GRIPPER_ROOTS[arm])
    tool_link = link_from_name(robot, PR2_TOOL_FRAMES[arm])
    return get_relative_pose(robot, root_link, tool_link)

def iterate_approach_path(robot, arm, gripper, pose, grasp, body=None):
    tool_from_root = get_tool_from_root(robot, arm)
    grasp_pose = multiply(pose.value, invert(grasp.value))
    approach_pose = multiply(pose.value, invert(grasp.approach))
    for tool_pose in interpolate_poses(grasp_pose, approach_pose):
        set_pose(gripper, multiply(tool_pose, tool_from_root))
        if body is not None:
            set_pose(body, multiply(tool_pose, grasp.value))
        yield

# def get_ir_sampler(problem, custom_limits={}, max_attempts=25, collisions=True, learned=True, verbose=False):
#     robot = problem.robot
#     obstacles = problem.fixed if collisions else []
#     gripper = problem.get_gripper()
#
#     def gen_fn(arm, obj, pose, grasp):
#         pose.assign()
#         link = None
#         if isinstance(obj, tuple):  ## YANG: (body, link)
#             obj, link = obj
#         approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}
#                               # and not is_containment(obj, obst, link)}
#         iter = 0
#         for _ in iterate_approach_path(robot, arm, gripper, pose, grasp, body=obj):
#             if verbose:
#                 for b in approach_obstacles:
#                     if pairwise_collision(gripper, b):
#                         print(f'       get_ir_sampler  |  iterate_approach_path {iter}  |  collision between {gripper} and {b}')
#                         for l in get_links(b):
#                             if pairwise_collision((gripper, None), (b, [l])):
#                                 print('                colliding with', (b, l), get_link_name(b, l), get_aabb(b, l))
#                         return
#                     elif pairwise_collision(obj, b):
#                         print(f'       get_ir_sampler  |  iterate_approach_path {iter} |  collision between {obj} and {b}')
#                         return
#             else:
#                 if any(pairwise_collision(gripper, b) or pairwise_collision(obj, b) for b in approach_obstacles):
#                     return
#             iter += 1
#         gripper_pose = multiply(pose.value, invert(grasp.value)) # w_f_g = w_f_o * (g_f_o)^-1
#         default_conf = arm_conf(arm, grasp.carry)
#         arm_joints = get_arm_joints(robot, arm)
#         base_joints = get_group_joints(robot, 'base')
#         # if pose_to_xyzyaw(pose.value) == (1.093, 7.088, 0.696, 2.8):
#         #     yield (Conf(robot, base_joints, (1.241, 6.672, 1.874)),)
#
#         if learned:
#             base_generator = learned_pose_generator(robot, gripper_pose, arm=arm, grasp_type=grasp.grasp_type)
#         else:
#             base_generator = uniform_pose_generator(robot, gripper_pose)
#         lower_limits, upper_limits = get_custom_limits(robot, base_joints, custom_limits)
#         while True:
#             count = 0
#             for base_conf in islice(base_generator, max_attempts):
#                 count += 1
#                 if not all_between(lower_limits, base_conf, upper_limits):
#                     continue
#                 bq = Conf(robot, base_joints, base_conf)
#                 pose.assign()
#                 bq.assign()
#                 set_joint_positions(robot, arm_joints, default_conf)
#                 if verbose:
#                     for b in obstacles + [obj]:
#                         if pairwise_collision(robot, b):
#                             print(f'       get_ir_sampler  |  base_generator  |  collision between {robot} and {b}')
#                             continue
#                 else:
#                     if any(pairwise_collision(robot, b) for b in obstacles + [obj]):
#                         continue
#                 if verbose: print('IR attempts:', count)
#
#                 # print('IR checked:', bq, obstacles + [obj], ' | ', arm_joints, default_conf)  ## YANG
#                 yield (bq,)
#                 break
#             else:
#                 yield None
#     return gen_fn

##################################################


def get_ik_fn(problem, custom_limits={}, collisions=True, teleport=False):
    robot = problem.robot
    obstacles = problem.fixed if collisions else []
    if is_ik_compiled():
        print('Using ikfast for inverse kinematics')
    else:
        print('Using pybullet for inverse kinematics')

    def fn(arm, obj, pose, grasp, base_conf):
        approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}
        gripper_pose = multiply(pose.value, invert(grasp.value)) # w_f_g = w_f_o * (g_f_o)^-1
        #approach_pose = multiply(grasp.approach, gripper_pose)
        approach_pose = multiply(pose.value, invert(grasp.approach))
        arm_link = get_gripper_link(robot, arm)
        arm_joints = get_arm_joints(robot, arm)

        default_conf = arm_conf(arm, grasp.carry)
        #sample_fn = get_sample_fn(robot, arm_joints)
        pose.assign()
        base_conf.assign()
        open_arm(robot, arm)
        set_joint_positions(robot, arm_joints, default_conf) # default_conf | sample_fn()
        grasp_conf = pr2_inverse_kinematics(robot, arm, gripper_pose, custom_limits=custom_limits) #, upper_limits=USE_CURRENT)
                                            #nearby_conf=USE_CURRENT) # upper_limits=USE_CURRENT,
        if (grasp_conf is None) or any(pairwise_collision(robot, b) for b in obstacles): # [obj]
            #print('Grasp IK failure', grasp_conf)
            #if grasp_conf is not None:
            #    print(grasp_conf)
            #    #wait_if_gui()
            return None
        #approach_conf = pr2_inverse_kinematics(robot, arm, approach_pose, custom_limits=custom_limits,
        #                                       upper_limits=USE_CURRENT, nearby_conf=USE_CURRENT)
        approach_conf = sub_inverse_kinematics(robot, arm_joints[0], arm_link, approach_pose, custom_limits=custom_limits)
        if (approach_conf is None) or any(pairwise_collision(robot, b) for b in obstacles):
            #print('Approach IK failure', approach_conf)
            #wait_if_gui()
            return None
        approach_conf = get_joint_positions(robot, arm_joints)
        attachment = grasp.get_attachment(problem.robot, arm)
        attachments = {attachment.child: attachment}
        if teleport:
            path = [default_conf, approach_conf, grasp_conf]
        else:
            resolutions = 0.05**np.ones(len(arm_joints))
            grasp_path = plan_direct_joint_motion(robot, arm_joints, grasp_conf, attachments=attachments.values(),
                                                  obstacles=approach_obstacles, self_collisions=SELF_COLLISIONS,
                                                  custom_limits=custom_limits, resolutions=resolutions/2.)
            if grasp_path is None:
                print('Grasp path failure')
                return None
            set_joint_positions(robot, arm_joints, default_conf)
            approach_path = plan_joint_motion(robot, arm_joints, approach_conf, attachments=attachments.values(),
                                              obstacles=obstacles, self_collisions=SELF_COLLISIONS,
                                              custom_limits=custom_limits, resolutions=resolutions,
                                              restarts=2, max_iterations=25, smooth=25)
            if approach_path is None:
                print('Approach path failure')
                return None
            path = approach_path + grasp_path
        mt = create_trajectory(robot, arm_joints, path)
        cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt])
        return (cmd,)
    return fn

##################################################

def get_ik_ir_gen(problem, max_attempts=25, learned=True, teleport=False, verbose=True, **kwargs):
    # TODO: compose using general fn
    # ir_sampler = get_ir_sampler(problem, learned=learned, max_attempts=1, **kwargs)
    ir_sampler = get_ir_sampler(problem, learned=learned, max_attempts=40, verbose=verbose, **kwargs)
    ik_fn = get_ik_fn(problem, teleport=teleport, **kwargs)
    def gen(*inputs):
        #set_renderer(enable=verbose)
        a, o, p, g = inputs
        ir_generator = ir_sampler(*inputs)
        attempts = 0
        while True:
            if max_attempts <= attempts:
                if not p.init:
                    return
                attempts = 0
                yield None
            attempts += 1
            if verbose: print(f'   {attempts} | get_ik_ir_gen | inputs = {inputs}')

            try:
                ir_outputs = next(ir_generator)
            except StopIteration:
                if verbose: print('    stopped ir_generator in', attempts, 'attempts')
                return

            if ir_outputs is None:
                continue
            inp = ir_generator.gi_frame.f_locals
            inp = [inp[k] for k in ['pose', 'grasp', 'custom_limits']]
            if verbose:
                print(f'           ir_generator  |  inputs = {inp}  |  ir_outputs = {ir_outputs}')
                samp = create_box(.1, .1, .1, mass=1, color=(1, 0, 1, 1))
                x,y,_ = ir_outputs[0].values
                set_point(samp, (x,y,0.2))

            ik_outputs = ik_fn(*(inputs + ir_outputs))
            if ik_outputs is None:
                continue
            # print('                         ik_outputs = ik_fn(*(inputs + ir_outputs)) =', ik_outputs, ' | commands =', ik_outputs[0].commands)
            if verbose: print('succeed after IK attempts:', attempts)
            yield ir_outputs + ik_outputs
            return
            #if not p.init:
            #    return
    return gen

##################################################

def get_motion_gen(problem, custom_limits={}, collisions=True, teleport=False):
    # TODO: include fluents
    robot = problem.robot
    saver = BodySaver(robot)
    obstacles = problem.fixed if collisions else []
    def fn(bq1, bq2, fluents=[]):
        saver.restore()
        bq1.assign()
        print('Discarded pr2_primitives.get_motion_gen\tobstacles:', obstacles)
        if teleport:
            path = [bq1, bq2]
        elif is_drake_pr2(robot):
            raw_path = plan_joint_motion(robot, bq2.joints, bq2.values, attachments=[],
                                         obstacles=obstacles, custom_limits=custom_limits, self_collisions=SELF_COLLISIONS,
                                         restarts=1, max_iterations=50, smooth=50)
            if raw_path is None:
                print('Failed motion plan!')
                #set_renderer(True)
                #for bq in [bq1, bq2]:
                #    bq.assign()
                #    wait_if_gui()
                return None
            path = [Conf(robot, bq2.joints, q) for q in raw_path]
        else:
            goal_conf = base_values_from_pose(bq2.value)
            raw_path = plan_base_motion(robot, goal_conf, BASE_LIMITS, obstacles=obstacles)
            if raw_path is None:
                print('Failed motion plan!')
                return None
            path = [Pose(robot, pose_from_base_values(q, bq1.value)) for q in raw_path]
        bt = Trajectory(path)
        cmd = Commands(State(), savers=[BodySaver(robot)], commands=[bt])
        return (cmd,)
    return fn

##################################################

def get_press_gen(problem, max_attempts=25, learned=True, teleport=False):
    robot = problem.robot
    fixed = get_fixed_bodies(problem)

    def gen(arm, button):
        fixed_wo_button = list(filter(lambda b: b != button, fixed))
        pose = get_pose(button)
        grasp_type = 'side'

        link = get_gripper_link(robot, arm)
        default_conf = get_carry_conf(arm, grasp_type)
        joints = get_arm_joints(robot, arm)

        presses = get_x_presses(button)
        approach = ((APPROACH_DISTANCE, 0, 0), unit_quat())
        while True:
            for _ in range(max_attempts):
                press_pose = random.choice(presses)
                gripper_pose = multiply(pose, invert(press_pose)) # w_f_g = w_f_o * (g_f_o)^-1
                #approach_pose = gripper_pose # w_f_g * g_f_o * o_f_a = w_f_a
                approach_pose = multiply(gripper_pose, invert(multiply(press_pose, approach)))

                if learned:
                    base_generator = learned_pose_generator(robot, gripper_pose, arm=arm, grasp_type=grasp_type)
                else:
                    base_generator = uniform_pose_generator(robot, gripper_pose)
                set_joint_positions(robot, joints, default_conf)
                set_pose(robot, next(base_generator))
                raise NotImplementedError('Need to change this')
                if any(pairwise_collision(robot, b) for b in fixed):
                    continue

                approach_movable_conf = sub_inverse_kinematics(robot, joints[0], link, approach_pose)
                #approach_movable_conf = inverse_kinematics(robot, link, approach_pose)
                if (approach_movable_conf is None) or any(pairwise_collision(robot, b) for b in fixed):
                    continue
                approach_conf = get_joint_positions(robot, joints)

                gripper_movable_conf = sub_inverse_kinematics(robot, joints[0], link, gripper_pose)
                #gripper_movable_conf = inverse_kinematics(robot, link, gripper_pose)
                if (gripper_movable_conf is None) or any(pairwise_collision(robot, b) for b in fixed_wo_button):
                    continue
                grasp_conf = get_joint_positions(robot, joints)
                bp = Pose(robot, get_pose(robot)) # TODO: don't use this

                if teleport:
                    path = [default_conf, approach_conf, grasp_conf]
                else:
                    control_path = plan_direct_joint_motion(robot, joints, approach_conf,
                                                     obstacles=fixed_wo_button, self_collisions=SELF_COLLISIONS)
                    if control_path is None: continue
                    set_joint_positions(robot, joints, approach_conf)
                    retreat_path = plan_joint_motion(robot, joints, default_conf,
                                                     obstacles=fixed, self_collisions=SELF_COLLISIONS)
                    if retreat_path is None: continue
                    path = retreat_path[::-1] + control_path[::-1]
                mt = Trajectory(Conf(robot, joints, q) for q in path)
                yield (bp, mt)
                break
            else:
                yield None
    return gen

#####################################

def control_commands(commands, **kwargs):
    wait_if_gui('Control?')
    disable_real_time()
    enable_gravity()
    for i, command in enumerate(commands):
        print(i, command)
        command.control(*kwargs)


class State(object):
    def __init__(self, attachments={}, cleaned=set(), cooked=set()):
        self.poses = {body: Pose(body, get_pose(body))
                      for body in get_bodies() if body not in attachments}
        self.grasps = {}
        self.attachments = attachments
        self.cleaned = cleaned
        self.cooked = cooked

    def assign(self):
        for attachment in self.attachments.values():
            # attach.attachment.assign()
            attachment.assign()


def apply_commands(state, commands, time_step=None, pause=False, **kwargs):
    # wait_if_gui('Apply?')
    for i, command in enumerate(commands):
        print(i, command)
        for j, _ in enumerate(command.apply(state, **kwargs)):
            state.assign()
            if j == 0:
                continue
            if time_step is None:
                wait_for_duration(1e-2)
                wait_if_gui('Command {}, Step {}) Next?'.format(i, j))
            else:
                wait_for_duration(time_step)
        if pause:
            wait_if_gui()

#####################################

def get_target_point(conf):
    # TODO: use full body aabb
    robot = conf.body
    link = link_from_name(robot, 'torso_lift_link')
    #link = BASE_LINK
    # TODO: center of mass instead?
    # TODO: look such that cone bottom touches at bottom
    # TODO: the target isn't the center which causes it to drift
    with BodySaver(conf.body):
        conf.assign()
        lower, upper = get_aabb(robot, link)
        center = np.average([lower, upper], axis=0)
        point = np.array(get_group_conf(conf.body, 'base'))
        #point[2] = upper[2]
        point[2] = center[2]
        #center, _ = get_center_extent(conf.body)
        return point


def get_target_path(trajectory):
    # TODO: only do bounding boxes for moving links on the trajectory
    return [get_target_point(conf) for conf in trajectory.path]

#######################################################


BASE_CONSTANT = 1
BASE_VELOCITY = 0.25


def get_base_custom_limits(robot, base_limits, yaw_limit=None):
    if isinstance(base_limits, dict):
        return base_limits
    if len(base_limits[0]) == 2:
        x_limits, y_limits = zip(*base_limits)
    if len(base_limits[0]) == 3:
        x_limits, y_limits, z_limits = zip(*base_limits)

    custom_limits = {
        joint_from_name(robot, 'x'): x_limits,
        joint_from_name(robot, 'y'): y_limits,
    }
    if yaw_limit is not None:
        custom_limits.update({
            joint_from_name(robot, 'theta'): yaw_limit,
        })
    if len(base_limits[0]) == 3:
        custom_limits[joint_from_name(robot, 'torso_lift_joint')] = z_limits
    return custom_limits


def distance_fn(q1, q2):
    distance = get_distance(q1.values[:2], q2.values[:2])
    return BASE_CONSTANT + distance / BASE_VELOCITY


def move_cost_fn(t):
    distance = t.distance(distance_fn=lambda q1, q2: get_distance(q1[:2], q2[:2]))
    return BASE_CONSTANT + distance / BASE_VELOCITY
