from __future__ import print_function

import random
from itertools import product
from os.path import isfile, dirname, abspath, join
import numpy as np
import math
import pybullet as p
from pprint import pprint

import os
import json
from pybullet_tools.logging import dump_json

from pybullet_tools.pr2_utils import draw_viewcone, get_viewcone, get_group_conf, set_group_conf, get_other_arm, \
    get_carry_conf, set_arm_conf, open_arm, close_arm, arm_conf, REST_LEFT_ARM, get_group_joints

from pybullet_tools.utils import unit_pose, get_collision_data, get_links, LockRenderer, pairwise_collision, get_link_name, \
    set_pose, get_movable_joints, draw_pose, pose_from_pose2d, set_velocity, set_joint_states, get_bodies, \
    flatten, INF, inf_generator, get_time_step, get_all_links, get_visual_data, pose2d_from_pose, multiply, invert, \
    get_sample_fn, pairwise_collisions, sample_placement, is_placement, aabb_contains_point, point_from_pose, \
    aabb2d_from_aabb, is_center_stable, aabb_contains_aabb, get_model_info, get_name, get_pose, dump_link, \
    dump_joint, dump_body, PoseSaver, get_aabb, add_text, GREEN, AABB, remove_body, HideOutput, \
    stable_z, Pose, Point, create_box, load_model, get_joints, set_joint_position, BROWN, Euler, PI, \
    set_camera_pose, TAN, RGBA, sample_aabb, get_min_limit, get_max_limit, get_joint_position, get_joint_name, \
    euler_from_quat, get_client, JOINT_TYPES, get_joint_type, get_link_pose, get_closest_points, \
    body_collision, is_placed_on_aabb, joint_from_name, body_from_end_effector, flatten_links, \
    get_link_subtree, quat_from_euler, euler_from_quat, create_box, set_pose, Pose, Point, get_camera_matrix, \
    YELLOW, add_line, draw_point, RED, BROWN, BLACK, BLUE, GREY, remove_handles, apply_affine, vertices_from_rigid, \
    aabb_from_points, get_aabb_extent, get_aabb_center, get_aabb_edges, unit_quat, set_renderer


OBJ = '?obj'
LINK_STR = '::'  ## for lisdf object names

BASE_LINK = 'base_link'
BASE_JOINTS = ['x', 'y', 'theta']
BASE_VELOCITIES = np.array([1., 1., math.radians(180)]) / 1. # per second
BASE_RESOLUTIONS = np.array([0.05, 0.05, math.radians(10)])

zero_limits = 0 * np.ones(2)
half_limits = 12 * np.ones(2)
BASE_LIMITS = (-half_limits, +half_limits) ## (zero_limits, +half_limits) ##
BASE_LIMITS = ((-1, 3), (6, 13))

CAMERA_FRAME = 'high_def_optical_frame'
EYE_FRAME = 'wide_stereo_gazebo_r_stereo_camera_frame'
CAMERA_MATRIX = get_camera_matrix(width=640, height=480, fx=525., fy=525.) # 319.5, 239.5 | 772.55, 772.5


def set_pr2_ready(pr2, arm='left', grasp_type='top', DUAL_ARM=False):
    other_arm = get_other_arm(arm)
    if not DUAL_ARM:
        initial_conf = get_carry_conf(arm, grasp_type)
        set_arm_conf(pr2, arm, initial_conf)
        open_arm(pr2, arm)
        set_arm_conf(pr2, other_arm, arm_conf(other_arm, REST_LEFT_ARM))
        close_arm(pr2, other_arm)
    else:
        for a in [arm, other_arm]:
            initial_conf = get_carry_conf(a, grasp_type)
            set_arm_conf(pr2, a, initial_conf)
            open_arm(pr2, a)

def load_asset(obj, **kwargs):
    from world_builder.utils import load_asset as helper
    return helper(obj, **kwargs)


def add_body(body, pose=unit_pose()):
    set_pose(body, pose)
    return body


def Pose2d(x=0., y=0., yaw=0.):
    return np.array([x, y, yaw])


def place_body(body, pose2d=Pose2d(), z=None):
    if z is None:
        lower, upper = body.get_aabb()
        z = -lower[2]
        # z = stable_z_on_aabb(body, region) # TODO: don't worry about epsilon differences
    return add_body(body, pose_from_pose2d(pose2d, z=z))


def load_texture(path):
    import pybullet
    return pybullet.loadTexture(path)


#######################################################

def set_zero_state(body, zero_pose=True, zero_conf=True):
    if zero_pose:
        set_pose(body, unit_pose())
        set_velocity(body, *unit_pose())
    if zero_conf:
        joints = get_movable_joints(body)
        # set_joint_positions(body, joints, np.zeros(len(joints)))
        set_joint_states(body, joints, np.zeros(len(joints)), np.zeros(len(joints)))


def set_zero_world(bodies=None, **kwargs):
    if bodies is None:
        bodies = get_bodies()
    for body in bodies:
        set_zero_state(body, **kwargs)


def write_yaml():
    raise NotImplementedError()


def draw_pose2d(pose2d, z=0., **kwargs):
    return draw_pose(pose_from_pose2d(pose2d, z=z), **kwargs)


def draw_pose2d_path(path, z=0., **kwargs):
    # TODO: unify with open-world-tamp, namo, etc.
    # return list(flatten(draw_point(np.append(pose2d[:2], [z]), **kwargs) for pose2d in path))
    return list(flatten(draw_pose2d(pose2d, z=z, **kwargs) for pose2d in path))

def draw_pose3d_path(path, **kwargs):
    from .flying_gripper_utils import pose_from_se3
    return list(flatten(draw_pose(pose_from_se3(q), **kwargs) for q in path))

def get_indices(sequence):
    return range(len(sequence))


def clip_delta(difference, max_velocities, time_step):
    # TODO: self.max_delta
    durations = np.divide(np.absolute(difference), max_velocities)
    max_duration = np.linalg.norm(durations, ord=INF)
    if max_duration == 0.:
        return np.zeros(len(difference))
    return min(max_duration, time_step) / max_duration * np.array(difference)


def sample_bernoulli_step(events_per_sec, time_step):
    p_event = events_per_sec * time_step
    return random.random() <= p_event


def constant_controller(value):
    return (value for _ in inf_generator())


def timeout_controller(controller, timeout=INF, time_step=None):
    if time_step is None:
        time_step = get_time_step()
    time_elapsed = 0.
    for output in controller:
        if time_elapsed > timeout:
            break
        yield output
        time_elapsed += time_step


def set_collisions(body1, enable=False):
    import pybullet
    # pybullet.setCollisionFilterGroupMask()
    for body2 in get_bodies():
        for link1, link2 in product(get_all_links(body1), get_all_links(body2)):
            pybullet.setCollisionFilterPair(body1, body2, link1, link2, enable)


def get_color(body):  # TODO: unify with open-world-tamp
    # TODO: average over texture
    visual_data = get_visual_data(body)
    if not visual_data:
        # TODO: no viewer implies no visual data
        return None
    return visual_data[0].rgbaColor


def multiply2d(*pose2ds):
    poses = list(map(pose_from_pose2d, pose2ds))
    return pose2d_from_pose(multiply(*poses))


def invert2d(pose2d):
    # return -np.array(pose2d)
    return pose2d_from_pose(invert(pose_from_pose2d(pose2d)))


def project_z(point, z=2e-3):
    return np.append(point[:2], [z])


#######################################################

MIN_DISTANCE = 1e-2


def sample_conf(robot, obstacles=[], min_distance=MIN_DISTANCE):
    sample_fn = get_sample_fn(robot, robot.joints, custom_limits=robot.custom_limits)
    while True:
        conf = sample_fn()
        robot.set_positions(conf)
        if not pairwise_collisions(robot, obstacles, max_distance=min_distance):
            return conf


def sample_safe_placement(obj, region, obstacles=[], min_distance=MIN_DISTANCE):
    obstacles = set(obstacles) - {obj, region}
    while True:
        pose = sample_placement(obj, region)
        if pose is None:
            break
        if not pairwise_collisions(obj, obstacles, max_distance=min_distance):
            set_pose(obj, pose)
            return pose


def check_placement(obj, region):
    return is_center_stable(obj, region, above_epsilon=INF, below_epsilon=INF)  # is_center_stable | is_placement


def is_on(obj_aabb, region_aabb):
    return aabb_contains_aabb(aabb2d_from_aabb(obj_aabb), aabb2d_from_aabb(region_aabb))


def is_above(robot, aabb):
    # return is_center_stable(robot, self.button)
    return aabb_contains_point(point_from_pose(robot.get_pose())[:2], aabb2d_from_aabb(aabb))


#######################################################

def nice_float(ele, round_to=3):
    if isinstance(ele, int) or ele.is_integer():
        return int(ele)
    else:
        return round(ele, round_to)


def nice_tuple(tup, round_to=3):
    new_tup = []
    for ele in tup:
        new_tup.append(nice_float(ele, round_to))
    return tuple(new_tup)


def nice(tuple_of_tuples, round_to=3):
    ## float, int
    if isinstance(tuple_of_tuples, float) or isinstance(tuple_of_tuples, int):
        return nice_float(tuple_of_tuples, round_to)

    elif len(tuple_of_tuples) == 0:
        return []

    ## position, pose
    elif isinstance(tuple_of_tuples[0], tuple) or isinstance(tuple_of_tuples[0], np.ndarray):

        ## pose = (point, quat) -> (point, euler)
        if len(tuple_of_tuples[0]) == 3 and len(tuple_of_tuples[1]) == 4:
            return nice( (tuple_of_tuples[0], euler_from_quat(tuple_of_tuples[1])), round_to)
            ## pose = (point, quat) -> (x, y, z, yaw)
            # return pose_to_xyzyaw(tuple_of_tuples)

        new_tuple = []
        for tup in tuple_of_tuples:
            new_tuple.append(nice_tuple(tup, round_to))
        return tuple(new_tuple)

    ## AABB
    elif isinstance(tuple_of_tuples, AABB):
        lower, upper = tuple_of_tuples
        return AABB(nice_tuple(lower, round_to), nice_tuple(upper, round_to))

    ## point, euler, conf
    return nice_tuple(tuple_of_tuples, round_to)

#######################################################

def collided(obj, obstacles, world=None, tag='', verbose=False, visualize=False, min_num_pts=0):
    result = False

    if verbose:
        ## first find the bodies that collides with obj
        bodies = []
        for b in obstacles:
            if pairwise_collision(obj, b):
                result = True
                bodies.append(b)
        ## then find the exact links
        body_links = {}
        total = 0
        for b in bodies:
            key = world.get_debug_name(b) if (world != None) else b
            d = {}
            for l in get_links(b):
                pts = get_closest_points(b, obj, link1=l, link2=None)
                if len(pts) > 0:
                    link = get_link_name(b, l)
                    d[link] = len(pts)

                    if visualize:  ## visualize collision points for debugging
                        points = []
                        for point in pts:
                            points.append(visualize_point(point.positionOnA))
                        print(f'visualized {len(pts)} collision points')
                        for point in points:
                            remove_body(point)

                total += len(pts)
            d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
            body_links[key] = d

        ## when debugging, give a threshold for oven
        if total <= min_num_pts:
            result = False
        else:
            prefix = 'pr2_streams.collided '
            if len(tag) > 0: prefix += f'( {tag} )'
            print(f'{prefix} | {obj} with {body_links}')
    else:
        if any(pairwise_collision(obj, b) for b in obstacles):
            result = True
    return result

#######################################################

OBJ_SCALES = {
    'OilBottle': 0.25, 'VinegarBottle': 0.25, 'Salter': 0.1, 'Knife': 0.1, 'Fork': 0.2,
    'Microwave': 0.7, 'Pan': 0.3, 'Pot': 0.3, 'Kettle': 0.3,
    'Egg': 0.1, 'Veggie': 0.3, 'VeggieLeaf': 0.3, 'VeggieStem': 0.3,
    'MilkBottle': 0.2, 'Toaster': 0.2, 'Bucket': 0.7, 'Cart': 1.1,
    'PotBody': 0.3, 'BraiserBody': 0.37, 'BraiserLid': 0.37, 'Faucet': 0.35,
    'VeggieCabbage': 0.005, 'MeatTurkeyLeg': 0.0007, 'VeggieTomato': 0.005,
    'VeggieZucchini': 0.016, 'VeggiePotato': 0.015, 'VeggieCauliflower': 0.008,
    'VeggieGreenPepper': 0.0003, 'VeggieArtichoke': 0.017, 'MeatChicken': 0.0008,
}
OBJ_SCALES = {k.lower(): v * 0.7 for k, v in OBJ_SCALES.items()}
OBJ_YAWS = {
    'Microwave': PI, 'Toaster': PI / 2
}


def sample_pose(obj, aabb, obj_aabb=None, yaws=OBJ_YAWS):
    ## sample a pose in aabb that can fit an object in
    if obj_aabb != None:
        lower, upper = obj_aabb
        diff = [(upper[i] - lower[i]) / 2 for i in range(3)]
        lower = [aabb[0][i] + diff[i] for i in range(3)]
        upper = [aabb[1][i] - diff[i] for i in range(3)]
        aabb = AABB(lower=lower, upper=upper)
    x, y, z = sample_aabb(aabb)

    ## use pre-defined yaws for appliances like microwave
    if obj in yaws:
        yaw = yaws[obj]
    else:
        yaw = np.random.uniform(0, PI)

    return x, y, z, yaw


def sample_obj_on_body_link_surface(obj, body, link, scales=OBJ_SCALES, PLACEMENT_ONLY=False, max_trial=8):
    aabb = get_aabb(body, link)
    # x, y, z, yaw = sample_pose(obj, aabb)
    # maybe = load_asset(obj, x=round(x, 1), y=round(y, 1), yaw=yaw, floor=(body, link), scale=scales[obj], maybe=True)
    # sample_placement(maybe, body, bottom_link=link)

    x, y, z, yaw = sample_pose(obj, aabb)
    if isinstance(obj, str):
        obj = obj.lower()
        maybe = load_asset(obj, x=round(x, 1), y=round(y, 1), yaw=yaw, floor=(body, link), scale=scales[obj],
                           maybe=True)
    else:
        maybe = obj
    trial = 0
    while not aabb_contains_aabb(aabb2d_from_aabb(get_aabb(maybe)), aabb2d_from_aabb(aabb)):
        x, y, z, yaw = sample_pose(obj, aabb, get_aabb(maybe))
        if isinstance(obj, str):
            remove_body(maybe)
            maybe = load_asset(obj, x=round(x, 1), y=round(y, 1), yaw=yaw, floor=(body, link), scale=scales[obj],
                               maybe=True)
        else:
            pose = Pose(point=Point(x=x, y=y, z=z), euler=Euler(yaw=yaw))
            set_pose(maybe, pose)
        # print(f'sampling surface for {body}-{link}', nice(aabb2d_from_aabb(aabb)))
        trial += 1
        if trial > max_trial: break

    if isinstance(obj, str):
        remove_body(maybe)
        maybe = load_asset(obj, x=round(x, 1), y=round(y, 1), yaw=yaw, floor=(body, link), scale=scales[obj],
                           moveable=True)
    if PLACEMENT_ONLY: return x, y, z, yaw

    # print(nice(aabb2d_from_aabb(aabb)))
    # print(nice(aabb2d_from_aabb(get_aabb(maybe))))
    return maybe


def sample_obj_in_body_link_space(obj, body, link=None, scales=OBJ_SCALES,
                                  PLACEMENT_ONLY=False, XY_ONLY=False, verbose=False):
    set_renderer(verbose)
    if verbose: print()

    aabb = get_aabb(body, link)
    x, y, z, yaw = sample_pose(obj, aabb)
    if isinstance(obj, str):
        obj = obj.lower()
        maybe = load_asset(obj, x=x, y=y, yaw=yaw, z=z, scale=scales[obj], maybe=True)
    else:
        maybe = obj
    handles = draw_fitted_box(maybe)[-1]

    # def contained(maybe):
    #     if not XY_ONLY:
    #         return aabb_contains_aabb(get_aabb(maybe), aabb)
    #     return aabb_contains_aabb(aabb2d_from_aabb(get_aabb(maybe)), aabb2d_from_aabb(aabb))

    def sample_one(maybe, handles):
        x, y, z, yaw = sample_pose(obj, aabb, get_aabb(maybe))
        z += 0.01
        if isinstance(obj, str):
            remove_body(maybe)
            maybe = load_asset(obj, x=x, y=y, yaw=yaw, z=z, scale=scales[obj], maybe=True)
        else:
            pose = Pose(point=Point(x=x, y=y, z=z), euler=Euler(yaw=yaw))
            set_pose(maybe, pose)

        remove_handles(handles)
        handles = draw_fitted_box(maybe)[-1]
        return maybe, (x, y, z, yaw), handles

    def sample_maybe(body, maybe, pose, handles):
        (x, y, z, yaw) = pose

        while not aabb_contains_aabb(get_aabb(maybe), aabb) or body_collision(body, maybe, link1=link):
            maybe, (x, y, z, yaw), handles = sample_one(maybe, handles)

        if verbose:
            print(f'sampling space for {body}-{link} {nice(aabb)} : {obj} {nice(get_aabb(maybe))}', )
            print(f'   collision between {body}-{link} and {maybe}: {body_collision(body, maybe, link1=link)}')
            print(f'   aabb of {body}-{link} contains that of {maybe}: {aabb_contains_aabb(get_aabb(maybe), aabb)}')
            set_camera_target_body(maybe, dx=1.5, dy=0, dz=0.7)

        return maybe, (x, y, z, yaw), handles

    def adjust_z(body, maybe, pose, handles):
        (x, y, z, yaw) = pose
        just_added = False
        ## lower the object until collision
        for interval in [0.1, 0.05, 0.01]:
            while aabb_contains_aabb(get_aabb(maybe), aabb) and not body_collision(body, maybe, link1=link):
                z -= interval
                just_added = False
                pose = Pose(point=Point(x=x, y=y, z=z), euler=Euler(yaw=yaw))
                set_pose(maybe, pose)
                remove_handles(handles)
                handles = draw_fitted_box(maybe)[-1]
                if verbose:
                    print(f'trying pose for {obj}: z - interval = {nice(z + interval)} - {interval}) = {nice(z)}')
            if just_added:
                return None
            reason = f'b.c. collision = {body_collision(body, maybe, link1=link)}, containment = {aabb_contains_aabb(get_aabb(maybe), aabb)}'
            z += interval
            just_added = True
            pose = Pose(point=Point(x=x, y=y, z=z), euler=Euler(yaw=yaw))
            set_pose(maybe, pose)
            remove_handles(handles)
            handles = draw_fitted_box(maybe)[-1]
            if verbose:
                print(f'reset pose for {obj}: z + interval = {nice(z - interval)} + {interval}) = {nice(z)} | {reason}')
        z -= interval

        if verbose:
            print(f'finalize pose for {obj}: z - interval = {nice(z + interval)} - {interval}) = {nice(z)}')
            print(f'   collision between {body}-{link} and {maybe}: {body_collision(body, maybe, link1=link)}')
            print(f'   aabb of {body}-{link} contains that of {maybe}: {aabb_contains_aabb(get_aabb(maybe), aabb)}')

        return maybe, (x, y, z, yaw), handles

    pose = (x, y, z, yaw)
    maybe, pose, handles = sample_maybe(body, maybe, pose, handles)
    result = adjust_z(body, maybe, pose, handles)
    while result == None:
        maybe, pose, handles = sample_one(maybe, handles)
        maybe, pose, handles = sample_maybe(body, maybe, pose, handles)
        result = adjust_z(body, maybe, pose, handles)
    maybe, (x, y, z, yaw), handles = result

    if isinstance(obj, str):
        remove_body(maybe)
        maybe = load_asset(obj, x=x, y=y, yaw=yaw, z=z, scale=scales[obj], moveable=True)
        # maybe = load_asset(obj, x=round(x, 1), y=round(y, 1), yaw=yaw, z=round(z, 1), scale=scales[obj], moveable=True)

    remove_handles(handles)
    set_renderer(True)
    if PLACEMENT_ONLY: return x, y, z, yaw
    # print(nice(aabb2d_from_aabb(aabb)))
    # print(nice(aabb2d_from_aabb(get_aabb(maybe))))
    return maybe


def add_attachment(state=None, obj=None, parent=-1, parent_link=None, attach_distance=0.1):
    new_attachments = {}
    if state != None:
        new_attachments = dict(state.attachments)

    if parent == -1:  ## use robot as parent
        parent = state.robot
        link1 = None
        parent_link = state.robot.base_link
    else:
        link1 = parent_link

    joint = None
    if isinstance(obj, tuple):
        BODY_TO_OBJECT = state.world.BODY_TO_OBJECT
        link1 = BODY_TO_OBJECT[obj].handle_link
        obj, joint = obj

    collision_infos = get_closest_points(parent, obj, link1=link1, max_distance=INF)
    min_distance = min([INF] + [info.contactDistance for info in collision_infos])
    if attach_distance == None or (min_distance < attach_distance):  ## (obj not in new_attachments) and
        if joint != None:
            attachment = create_attachment(parent, parent_link, obj,
                                           child_link=link1, child_joint=joint)
        else:
            attachment = create_attachment(parent, parent_link, obj)
        new_attachments[obj] = attachment  ## may overwrite older attachment
    return new_attachments


def create_attachment(parent, parent_link, child, child_link=None, child_joint=None, OBJ=False):
    parent_link_pose = get_link_pose(parent, parent_link)
    child_pose = get_pose(child)
    grasp_pose = multiply(invert(parent_link_pose), child_pose)
    if OBJ:  ## attachment between objects
        return ObjAttachment(parent, parent_link, grasp_pose, child)
    return Attachment(parent, parent_link, grasp_pose, child,
                      child_link=child_link, child_joint=child_joint)


class Attachment(object):
    def __init__(self, parent, parent_link, grasp_pose, child,
                 child_joint=None, child_link=None):
        self.parent = parent  # TODO: support no parent
        self.parent_link = parent_link
        self.grasp_pose = grasp_pose
        self.child = child
        self.child_joint = child_joint
        self.child_link = child_link

    @property
    def bodies(self):
        return flatten_links(self.child) | flatten_links(self.parent, get_link_subtree(
            self.parent, self.parent_link))

    def assign(self):
        from .pr2_streams import LINK_POSE_TO_JOINT_POSITION
        # robot_base_pose = self.parent.get_positions(roundto=3)
        # robot_arm_pose = self.parent.get_positions(joint_group='left', roundto=3)  ## only left arm for now
        parent_link_pose = get_link_pose(self.parent, self.parent_link)
        child_pose = body_from_end_effector(parent_link_pose, self.grasp_pose)
        if self.child_link == None:
            set_pose(self.child, child_pose)
        elif self.child in LINK_POSE_TO_JOINT_POSITION:  ## pull drawer handle
            # for key in [robot_base_pose, robot_arm_pose]:
            for group in self.parent.joint_groups: ## ['base', 'left', 'hand']:
                key = self.parent.get_positions(joint_group=group, roundto=3)
                if key in LINK_POSE_TO_JOINT_POSITION[self.child][self.child_joint]:
                    position = LINK_POSE_TO_JOINT_POSITION[self.child][self.child_joint][key]
                    set_joint_position(self.child, self.child_joint, position)
                    # print(f'bullet.utils | Attachment | robot {key} @ {key} -> position @ {position}')
        return child_pose

    def apply_mapping(self, mapping):
        self.parent = mapping.get(self.parent, self.parent)
        self.child = mapping.get(self.child, self.child)

    def __repr__(self):
        name = self.__class__.__name__
        if self.child_link == None:
            return '{}({},{})'.format(name, self.parent, self.child)
        else:
            return '{}({},{}-{})'.format(name, self.parent, self.child, self.child_link)


def remove_attachment(state, obj=None):
    # print('bullet.utils | remove_attachment | old', state.attachments)
    if isinstance(obj, tuple): obj = obj[0]
    new_attachments = dict(state.attachments)
    if obj in new_attachments:
        new_attachments.pop(obj)
    # print('bullet.utils | remove_attachment | new', new_attachments)
    return new_attachments


class ObjAttachment(Attachment):
    def assign(self):
        parent_link_pose = get_link_pose(self.parent, self.parent_link)
        child_pose = body_from_end_effector(parent_link_pose, self.grasp_pose)
        set_pose(self.child, child_pose)

    # def __init__(self, parent, parent_link, child, rel_pose=None):
    #     super(ObjAttachment, self).__init__(parent, parent_link, None, child)
    #     if rel_pose == None:
    #         p_parent = get_link_pose(parent, parent_link)
    #         p_child= get_pose(child)
    #         rel_pose = (p_child[0][i] - p_parent[0][i] for i in range(len(p_child[0])))
    #     self.rel_pose = rel_pose
    # def assign(self):
    #     p_parent = get_link_pose(self.parent, self.parent_link)
    #     _, r_child = get_pose(self.child)
    #     p_child = (p_parent[0][i] + self.rel_pose[i] for i in range(len(self.rel_pose)))
    #     set_pose(self.child, (p_child, r_child))


#######################################################

def set_camera_target_body(body, link=None, dx=3.8, dy=0, dz=1):
    # if isinstance(body, tuple):
    #     link = BODY_TO_OBJECT[body].handle_link
    #     body = body[0]
    aabb = get_aabb(body, link)
    x = (aabb.upper[0] + aabb.lower[0]) / 2
    y = (aabb.upper[1] + aabb.lower[1]) / 2
    z = (aabb.upper[2] + aabb.lower[2]) / 2
    set_camera_pose(camera_point=[x + dx, y + dy, z + dz], target_point=[x, y, z])


def set_default_camera_pose():
    ## the whole kitchen & living room area
    # set_camera_pose(camera_point=[9, 8, 9], target_point=[6, 8, 0])

    ## just the kitchen
    set_camera_pose(camera_point=[4, 7, 4], target_point=[3, 7, 2])


def set_camera_target_robot(robot, distance=5, FRONT=False):
    x, y, yaw = get_pose2d(robot)
    target_point = (x, y, 2)
    yaw -= math.pi / 2
    pitch = - math.pi / 3
    if FRONT:
        yaw += math.pi
        pitch = -math.pi / 4  ## 0
        target_point = (x, y, 1)
    CLIENT = get_client()
    p.resetDebugVisualizerCamera(distance, math.degrees(yaw), math.degrees(pitch),
                                 target_point, physicsClientId=CLIENT)


#######################################################

# def summarize_links(body):
#     joints = get_joints(body)
#     for joint in joints:
#         check_joint_state(body, joint)
def get_point_distance(p1, p2):
    if isinstance(p1, tuple): p1 = np.asarray(p1)
    if isinstance(p2, tuple): p2 = np.asarray(p2)
    return np.linalg.norm(p1 - p2)


def get_pose2d(robot):
    if isinstance(robot, int):
        return get_group_conf(robot, 'base')
    point, quat = robot.get_pose()
    x, y, _ = point
    _, _, yaw = euler_from_quat(quat)
    return x, y, yaw


def summarize_joints(body):
    joints = get_joints(body)
    for joint in joints:
        check_joint_state(body, joint, verbose=True)


def check_joint_state(body, joint, verbose=False):
    name = get_joint_name(body, joint)
    pose = get_joint_position(body, joint)
    min_limit = get_min_limit(body, joint)
    max_limit = get_max_limit(body, joint)
    moveable = joint in get_movable_joints(body)
    joint_type = JOINT_TYPES[get_joint_type(body, joint)]

    category = 'fixed'
    state = None
    if min_limit < max_limit:

        if joint_type == 'revolute' and min_limit == 0:
            category = 'door-max'
            if pose == max_limit:
                state = 'door OPENED fully'
            elif pose == min_limit:
                state = 'door CLOSED'
            else:
                state = 'door OPENED partially'

        elif joint_type == 'revolute' and max_limit == 0:
            category = 'door-min'
            if pose == min_limit:
                state = 'door OPENED fully'
            elif pose == max_limit:
                state = 'door CLOSED'
            else:
                state = 'door OPENED partially'

        ## switch on faucet, machines
        elif joint_type == 'revolute' and min_limit + max_limit == 0:
            category = 'switch'
            if pose == min_limit:
                state = 'switch TURNED OFF'
            elif pose == max_limit:
                state = 'switch TURNED ON'

        elif joint_type == 'prismatic':  ## drawers
            category = 'drawer'
            if pose == max_limit:
                state = 'drawer OPENED fully'
            elif pose == min_limit:
                state = 'drawer CLOSED'
            else:
                state = 'drawer OPENED partially'

    else:
        state = 'fixed joint'

    if verbose:
        print(
            f'   joint {name}, pose = {pose}, limit = {nice((min_limit, max_limit))}, state = {state}, moveable = {moveable}')
    return category, state


def toggle_joint(body, joint):
    category, state = check_joint_state(body, joint)
    if 'OPENED' in state:
        close_joint(body, joint)
    elif 'CLOSED' in state:
        open_joint(body, joint)


def open_joint(body, joint, extent=0.95, pstn=None):
    if pstn == None:
        if isinstance(joint, str):
            joint = joint_from_name(body, joint)
        min_limit = get_min_limit(body, joint)
        max_limit = get_max_limit(body, joint)
        category, state = check_joint_state(body, joint)
        if category == 'door-max':
            pstn = max_limit * extent
        elif category == 'door-min':
            pstn = min_limit * extent
        elif category == 'drawer':
            pstn = max_limit
    set_joint_position(body, joint, pstn)


def close_joint(body, joint):
    min_limit = get_min_limit(body, joint)
    max_limit = get_max_limit(body, joint)
    category, state = check_joint_state(body, joint)
    if category == 'door-max':
        set_joint_position(body, joint, min_limit)
    elif category == 'door-min':
        set_joint_position(body, joint, max_limit)
    elif category == 'drawer':
        set_joint_position(body, joint, min_limit)


#######################################################

def get_readable_list(lst, world=None, NAME_ONLY=False):
    to_print = []
    for word in lst:
        if world != None:
            name = world.get_name(word)
            last_is_tuple = (len(to_print) != 0) and isinstance(to_print[-1], tuple)
            if name != None and not last_is_tuple: ## ['=', ('PickCost',), 'pr2|1']
                if not NAME_ONLY:
                    name = world.get_debug_name(word)
                to_print.append(name)
            else:
                to_print.append(word)
        else:
            to_print.append(word)
    return to_print


def summarize_facts(facts, world=None, name='Initial facts'):
    from pybullet_tools.logging import myprint as print
    print('----------------')
    print(f'{name} ({len(facts)})')
    predicates = {}
    for fact in facts:
        pred = fact[0].lower()
        if pred not in predicates:
            predicates[pred] = []
        predicates[pred].append(fact)
    predicates = {k: v for k, v in sorted(predicates.items())}
    # predicates = {k: v for k, v in sorted(predicates.items(), key=lambda item: len(item[1][0]))}
    for pred in predicates:
        to_print_line = [get_readable_list(fa, world) for fa in predicates[pred]]
        print('  ', pred, to_print_line)
    print('----------------')

def print_plan(plan, world=None):
    from pddlstream.language.constants import Equal, AND, PDDLProblem, is_plan
    from pybullet_tools.logging import myprint as print

    if not is_plan(plan):
        return
    step = 1
    print('Plan:')
    for action in plan:
        name, args = action
        args2 = [str(a) for a in get_readable_list(args, world)]
        print('{:2}) {} {}'.format(step, name, ' '.join(args2)))
        step += 1
    print()


def print_goal(goal, world=None):
    from pybullet_tools.logging import myprint as print

    print(f'Goal ({len(goal) - 1}): ({goal[0]}')
    for each in get_readable_list(goal[1:], world):
        print(f'   {each},')
    print(')')


#######################################################

def is_placement(body, surface, link=None, **kwargs):
    if isinstance(surface, tuple):
        surface, _, link = surface
    return is_placed_on_aabb(body, get_aabb(surface, link), **kwargs)


def is_contained(body, space):
    if isinstance(space, tuple):
        return aabb_contains_aabb(get_aabb(body), get_aabb(space[0], link=space[-1]))
    return aabb_contains_aabb(get_aabb(body), get_aabb(space))


#######################################################

def save_pickle(pddlstream_problem, plan, preimage):
    ## ------------------- save the plan for debugging ----------------------
    # doesn't work because the reconstructed plan and preimage by pickle have different variable index
    import pickle
    import os
    from os.path import join, dirname, abspath
    ROOT_DIR = abspath(join(dirname(__file__), os.pardir))
    file = join(ROOT_DIR, '..', 'leap', 'pddlstream_plan.pkl')
    if isfile(file): os.remove(file)
    with open(file, 'wb') as outp:
        pickle.dump(pddlstream_problem.init, outp, pickle.HIGHEST_PROTOCOL)
        pickle.dump(plan, outp, pickle.HIGHEST_PROTOCOL)
        pickle.dump(preimage, outp, pickle.HIGHEST_PROTOCOL)
    # ------------------- save the plan for debugging ----------------------


def pose_to_xyzyaw(pose):
    xyzyaw = list(nice_tuple(pose[0]))
    xyzyaw.append(nice_float(euler_from_quat(pose[1])[2]))
    return tuple(xyzyaw)


def xyzyaw_to_pose(xyzyaw):
    return tuple((tuple(xyzyaw[:3]), quat_from_euler(Euler(0, 0, xyzyaw[-1]))))

def draw_collision_shapes(body, links=[]):
    """ not working """
    if isinstance(body, tuple):
        body, link = body
        links.append(link)
    if len(links) == 0:
        links = get_links(body)
    body_from_world = get_pose(body)
    handles = []
    for link in links:
        collision_data = get_collision_data(body, link)
        for i in range(len(collision_data)):
            shape = collision_data[i]
            shape_from_body = (shape.local_frame_pos, shape.local_frame_orn)
            shape_from_world = multiply(shape_from_body, body_from_world)
            draw_bounding_lines(shape_from_world, shape.dimensions)
            print(f'link = {link}, colldion_body = {i} | dims = {nice(shape.dimensions)} | shape_from_world = {nice(shape_from_world)}')


def fit_dimensions(body, body_pose=unit_pose()):
    vertices = []
    for link in get_links(body):
        new_vertices = apply_affine(body_pose, vertices_from_rigid(body, link))
        for p in new_vertices[::10]:
            draw_point(p, size=0.01, color=YELLOW)
        vertices.extend(new_vertices)
    aabb = aabb_from_points(vertices)
    return aabb, get_aabb_center(aabb), get_aabb_extent(aabb)

def draw_fitted_box(body, link=None, draw_centroid=False, verbose=False):

    if link == None:
        body_pose = multiply(get_pose(body), Pose(euler=Euler(math.pi / 2, 0, -math.pi / 2)))  ##
        links = get_links(body)
    else:
        body_pose = get_link_pose(body, link)  ##
        if verbose:
            print(f'bullet_utils.draw_fitted_box | body_pose = get_link_pose({body}, {link}) = {nice(body_pose)}')
        links = [link]

    vertices = []
    for link in links:
        new_vertices = apply_affine(unit_pose(), vertices_from_rigid(body, link))
        vertices.extend(new_vertices)
    aabb = aabb_from_points(vertices)
    handles = draw_bounding_box(aabb, body_pose)
    if draw_centroid:
        handles.extend(draw_face_points(aabb, body_pose, dist=0.04))
    return body_pose, aabb, handles

def draw_bounding_box(aabb, body_pose):
    handles = []
    for a, b in get_aabb_edges(aabb):
        p1, p2 = apply_affine(body_pose, [a, b])
        handles.append(add_line(p1, p2))
    return handles

def draw_face_points(aabb, body_pose, dist=0.08):
    c = get_aabb_center(aabb)
    w, l, h = get_aabb_extent(aabb)
    faces = [(w/2+dist, 0, 0), (0, l/2+dist, 0), (0, 0, h/2+dist)]
    faces += [minus(0, f) for f in faces]
    faces = [add(f, c) for f in faces]
    faces = apply_affine(body_pose, faces)
    handles = []
    for f in faces:
        handles.append(draw_point(f, size=0.02, color=RED))
    return handles

def get_hand_grasps(state, body, link=None, grasp_length=0.1,
                    HANDLE_FILTER=False, LENGTH_VARIANTS=False,
                    visualize=False, RETAIN_ALL=False, verbose=False):
    from pybullet_tools.flying_gripper_utils import set_se3_conf, create_fe_gripper, se3_from_pose
    title = 'bullet_utils.get_hand_grasps | '
    dist = grasp_length
    robot = state.robot
    obstacles = state.fixed
    if body not in obstacles:
        obstacles += [body]
    body_pose, aabb, handles = draw_fitted_box(body, link=link, verbose=verbose)
    if link == None:
        body_pose = get_pose(body)
    else: ## for handle grasps, use the original pose of handle_link
        body_pose = multiply(body_pose, invert(robot.tool_from_hand))
        if verbose:
            print(f'{title}hand_link = {link} | body_pose = multiply(body_pose, invert(robot.tool_from_hand)) = {nice(body_pose)}')

    ## get the points in hand frame to be transformed to the origin of object frame in different directions
    c = get_aabb_center(aabb)
    w, l, h = dimensions = get_aabb_extent(aabb)
    faces = [(w/2+dist, 0, 0), (0, l/2+dist, 0), (0, 0, h/2+dist)]
    faces += [minus(0, f) for f in faces]

    ## for finding the longest dimension
    max_value = max(dimensions)
    filter = [int(x != max_value) for x in dimensions]

    P = math.pi
    rots = {
        (1, 0, 0): [(P/2, 0, -P/2), (P/2, P, -P/2), (P/2, -P/2, -P/2), (P/2, P/2, -P/2)],
        (-1, 0, 0): [(P/2, 0, P/2), (P/2, P, P/2), (P/2, -P/2, P/2), (P/2, P/2, P/2), (-P, -P/2, 0), (-P, -P/2, -P)],
        (0, 1, 0): [(0, P/2, -P/2), (0, -P/2, P/2), (P/2, P, 0), (P/2, 0, 0)],
        (0, -1, 0): [(0, P/2, P/2), (0, -P/2, -P/2), (-P/2, P, 0), (-P/2, 0, 0)],
        (0, 0, 1): [(P, 0, P/2), (P, 0, -P/2), (P, 0, 0), (P, 0, P)],
        (0, 0, -1): [(0, 0, -P/2), (0, 0, P/2), (0, 0, 0), (0, 0, P)],
    }
    set_renderer(visualize)
    grasps = []
    for f in faces:
        p = np.array(f)
        p = p / np.linalg.norm(p)

        ## only attempt the bigger surfaces
        on_longest = sum([filter[i]*p[i] for i in range(3)]) != 0
        if HANDLE_FILTER and not on_longest:
            continue

        ang = tuple(p)
        f = add(f, c)
        # r = rots[ang][0] ## random.choice(rots[tuple(p)]) ##
        for r in rots[ang]:
            grasp = multiply(Pose(point=f), Pose(euler=r))
            if check_cfree_gripper(grasp, state.world, body_pose, obstacles, verbose=verbose,
                                   visualize=visualize, RETAIN_ALL=RETAIN_ALL):
                grasps += [grasp]

                # # debug
                # if verbose:
                #     set_renderer(True)
                #     return grasps

                ## slide along the longest dimension
                if LENGTH_VARIANTS and on_longest:
                    dl_max = max_value / 3
                    dl_candidates = [random.uniform(-dl_max, dl_max) for k in range(3)]
                    dl_candidates = [dl_max, -dl_max]
                    for dl in dl_candidates:
                        grasp_dl = multiply(grasp, Pose(point=(dl,0,0)))
                        if check_cfree_gripper(grasp_dl, state.world, body_pose, obstacles, verbose=verbose,
                                               visualize=visualize, RETAIN_ALL=RETAIN_ALL, color=BROWN):
                            grasps += [grasp_dl]

    # set_renderer(True)
    print(title, [nice(g) for g in grasps])
    if len(grasps) == 0:
        print(title, 'no grasps found')
    return grasps

def check_cfree_gripper(grasp, world, object_pose, obstacles, visualize=True,
                        color=GREEN, min_num_pts=40, RETAIN_ALL=False, verbose=False):
    from pybullet_tools.flying_gripper_utils import get_cloned_se3_conf
    robot = world.robot
    # print(f'bullet_utils.check_cfree_gripper(object_pose={nice(object_pose)}) before robot.visualize_grasp')
    gripper_grasp = robot.visualize_grasp(object_pose, grasp, color=color, verbose=verbose)
    if gripper_grasp == None:
        return False

    if verbose:
        print(f'bullet_utils.check_cfree_gripper | gripper_grasp {gripper_grasp} | object_pose {nice(object_pose)}'
            f' | se_conf {nice(get_cloned_se3_conf(robot, gripper_grasp))} | grasp = {nice(grasp)}')

    if visualize: ## and not firstly: ## somtimes cameras blocked by robot, need to change dx, dy
        ## also helps slow down visualization of the sampling the testing process
        set_camera_target_body(gripper_grasp, dx=0.3, dy=0.5, dz=0.2) ## oven
        # set_camera_target_body(gripper_grasp, dx=1, dy=0.5, dz=0.8) ## faucet

    ## when gripper isn't closed, it shouldn't collide
    firstly = collided(gripper_grasp, obstacles, min_num_pts=min_num_pts,
                       world=world, verbose=False, tag='firstly')

    secondly = False
    if not firstly:
        ## when gripper is closed, it should collide with object
        robot.close_cloned_gripper(gripper_grasp)
        secondly = collided(gripper_grasp, obstacles, min_num_pts=0,
                            world=world, verbose=False, tag='secondly')

    result = not firstly and secondly
    if not result or not RETAIN_ALL:
        remove_body(gripper_grasp)
    elif RETAIN_ALL:
        robot.open_cloned_gripper(gripper_grasp)

    return result


def add(elem1, elem2):
    return tuple(np.asarray(elem1)+np.asarray(elem2))

def minus(elem1, elem2):
    return tuple(np.asarray(elem1)-np.asarray(elem2))

def dist(elem1, elem2):
    return np.linalg.norm(np.asarray(elem1)-np.asarray(elem2))

def draw_bounding_lines(pose, dimensions):
    w, l, h = dimensions  ## it's meshscale instead of wlh
    # tmp = create_box(w, l, h)
    # set_pose(tmp, pose)

    ## first get the points using local transforms
    def draw_given_transforms(transforms, color=RED):
        # if len(handles) > 0: remove_handles(handles)
        transforms.extend([(-t[0], t[1], t[2]) for t in transforms])
        transforms.extend([(t[0], -t[1], t[2]) for t in transforms])
        transforms.extend([(t[0], -t[1], -t[2]) for t in transforms])
        transforms = [Pose(t, Euler()) for t in transforms]

        def one_diff(t1, t2):
            return len([t1[k] != t2[k] for k in range(len(t1))]) == 1

        lines = []
        handles = []
        for t1 in transforms:
            pt1 = multiply(pose, t1)[0]
            for t2 in transforms:
                pt2 = multiply(pose, t2)[0]
                if pt1 != pt2 and one_diff(pt1, pt2):
                    if (pt1, pt2) not in lines:
                        handles.append(add_line(pt1, pt2, width=0.5, color=color))
                        lines.extend([(pt1, pt2), (pt2, pt1)])
            handles.extend(draw_point(pt1, size=0.02, color=color))
        return handles

    transforms = [(w/2, h/2, l/2)] ## [(h/2, l/2, w/2)]
    handles = draw_given_transforms(transforms, color=RED)
    # remove_body(tmp)
    return handles


def visualize_point(point, world=None):
    z = 0
    if len(point) == 3:
        x, y, z = point
    else:
        x, y = point
    body = create_box(.05, .05, .05, mass=1, color=(1, 0, 0, 1))
    set_pose(body, Pose(point=Point(x, y, z)))
    return body

def get_file_short_name(path):
    return path[path.rfind('/')+1:]

def equal_float(a, b, epsilon=0):
    return abs(a - b) <= epsilon

def equal(tup_a, tup_b, epsilon=0):
    if isinstance(tup_a, float) or isinstance(tup_a, int):
        return equal_float(tup_a, tup_b, epsilon)

    elif isinstance(tup_a, tuple):
        a = list(tup_a)
        b = list(tup_b)
        return all([equal_float(a[i], b[i], epsilon) for i in range(len(a))])

    return None

# def equal(tup1, tup2, epsilon=0.001):
#     if isinstance(tup1, float):
#         return abs(tup1 - tup2) < epsilon
#     if len(tup1) == 2:
#         return equal(tup1[0], tup2[0]) and equal(tup1[1], tup2[1])
#     return all([abs(tup1[i] - tup2[i]) < epsilon for i in range(len(tup1))])



def get_gripper_directions():
    """ for faces, 'sideways' = 'sagittal' , 'frontal' = 'frontback' """
    PI = math.pi
    label = "point {}, face {}"
    labels = {
        (PI/2, 0, 0): label.format('front', 'sideways'),
        (PI, 0, 0): label.format('front', 'horizontal'),

        (PI/2, 0, -PI): label.format('back', 'sideways'),
        (PI, 0, -PI): label.format('back', 'horizontal'),

        (PI/2, 0, -PI/2): label.format('left', 'frontal'),
        (PI, 0, -PI/2): label.format('left', 'horizontal'),

        (PI/2, 0, PI/2): label.format('right', 'frontal'),
        (PI, 0, PI/2): label.format('right', 'horizontal'),

        (PI, -PI/2, PI/2): label.format('up', 'sideways'),
        (PI, -PI/2, -PI/2): label.format('up', 'sideways'),
        (PI, -PI/2, 0): label.format('up', 'frontal'),

        (PI, PI/2, -PI/2): label.format('down', 'sideways'),
        (PI, PI/2, -3*PI/2): label.format('down', 'sideways'),
    }
    labels.update({(k[0]-PI, k[1], k[2]): v for k, v in labels.items()})
    return labels

GRIPPER_DIRECTIONS = get_gripper_directions()

def get_gripper_direction(pose, epsilon=0.01):
    """ fuzzy match of euler values to gripper direction label """
    euler = euler_from_quat(pose[1])
    for key in get_gripper_directions():
        if equal(euler, key, epsilon):
            return GRIPPER_DIRECTIONS[key]
    return None

def find_grasp_in_db(db_file_name, full_name=None):
    db_file = dirname(abspath(__file__))
    db_file = join(db_file, db_file_name)
    db = json.load(open(db_file, 'r'))

    found = None
    if full_name != None and full_name in db and len(db[full_name]) > 0:
        if len(db[full_name][0][1]) == 4:
            found = [(tuple(e[0]), tuple(e[1])) for e in db[full_name]]
        else:
            found = [(tuple(e[0]), quat_from_euler(e[1])) for e in db[full_name]]
    return found, db, db_file

def add_grasp_in_db(db, db_file, full_name, grasps):
    db[full_name] = []
    for g in grasps:
        g = nice(g, 4)
        db[full_name].append([list(g[0]), list(g[1])])
    os.remove(db_file)
    dump_json(db, db_file)

def visualize_camera_image(image, index=0, img_dir='.'):
    import seaborn as sns
    sns.set()
    import matplotlib.pyplot as plt

    name = join(img_dir, f"depth_image_feg_{index}.png")

    ax = sns.heatmap(image.depthPixels, annot=False, fmt="d")

    plt.title("Depth Image", fontsize=12)

    plt.savefig(name, bbox_inches='tight', dpi=100)
    plt.close()


    # plt.show()