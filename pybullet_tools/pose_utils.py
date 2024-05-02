from __future__ import print_function

import random
from itertools import product
from os.path import isfile, dirname, abspath, join, isdir
import sys

import numpy as np
import math
import pybullet as p
from pprint import pprint

import os
import json
from pybullet_tools.logging_utils import dump_json

from pybullet_tools.utils import unit_pose, get_collision_data, get_links, pairwise_collision, get_link_name, \
    is_movable, get_movable_joints, draw_pose, pose_from_pose2d, set_velocity, set_joint_states, get_bodies, \
    flatten, INF, inf_generator, get_time_step, get_all_links, get_visual_data, pose2d_from_pose, multiply, invert, \
    get_sample_fn, pairwise_collisions, sample_placement, aabb_contains_point, point_from_pose, \
    aabb2d_from_aabb, is_center_stable, aabb_contains_aabb, get_pose, get_aabb, GREEN, AABB, remove_body, stable_z, \
    get_joints, set_joint_position, Euler, PI, LockRenderer, HideOutput, load_model, \
    set_camera_pose, sample_aabb, get_min_limit, get_max_limit, get_joint_position, get_joint_name, \
    get_client, JOINT_TYPES, get_joint_type, get_link_pose, get_closest_points, \
    body_collision, is_placed_on_aabb, joint_from_name, body_from_end_effector, flatten_links, get_aabb_volume, \
    get_link_subtree, quat_from_euler, euler_from_quat, create_box, set_pose, Pose, Point, get_camera_matrix, \
    YELLOW, add_line, draw_point, RED, remove_handles, apply_affine, vertices_from_rigid, \
    aabb_from_points, get_aabb_extent, get_aabb_center, get_aabb_edges, set_renderer, draw_aabb, set_point, has_gui, get_rigid_clusters, \
    link_pairs_collision, wait_unlocked, apply_alpha, set_color, BASE_LINK as ROOT_LINK, \
    dimensions_from_camera_matrix, get_field_of_view, get_image, timeout, unit_point, get_joint_limits, ConfSaver, \
    BROWN, BLUE, WHITE, TAN, GREY, YELLOW, GREEN, BLACK, RED, tform_point, create_shape, STATIC_MASS, \
    get_box_geometry, create_body, get_link_parent, NULL_ID, get_joint_info, get_dynamics_info, \
    clone_collision_shape, clone_visual_shape, get_local_link_pose, get_joint_positions, \
    sample_placement_on_aabb, visual_shape_from_data, is_unknown_file, create_collision_shape
from pybullet_tools.bullet_utils import draw_fitted_box, draw_points, nice, nice_tuple, nice_float, \
    in_list
from pybullet_tools.pr2_primitives import Conf

MIN_DISTANCE = 1e-2


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

    def assign(self, verbose=False):
        # robot_base_pose = self.parent.get_positions(roundto=3)
        # robot_arm_pose = self.parent.get_positions(joint_group='left', roundto=3)  ## only left arm for now
        parent_link_pose = get_link_pose(self.parent.body, self.parent_link)
        child_pose = body_from_end_effector(parent_link_pose, self.grasp_pose)
        if self.child_link is None:
            set_pose(self.child, child_pose)
        else:
            LP2JP = self.parent.LINK_POSE_TO_JOINT_POSITION
            if verbose:
                print('\nbullet.Attachment.assign() | LINK_POSE_TO_JOINT_POSITION')
                pprint(LP2JP)
            if self.child in LP2JP:  ## pull drawer handle
                if self.child in LP2JP and self.child_joint in LP2JP[self.child]:
                    conf = self.parent.get_all_arm_conf()
                    if conf in LP2JP[self.child][self.child_joint]:
                        ls = LP2JP[self.child][self.child_joint][conf]
                        for group in self.parent.joint_groups: ## ['base', 'left', 'hand']:
                            key = self.parent.get_positions(joint_group=group, roundto=3)
                            result = in_list(key, ls)
                            if result is not None:
                                position = ls[result]
                                set_joint_position(self.child, self.child_joint, position)
                                # print(f'bullet.utils | Attachment | robot {key} @ {key} -> position @ {position}')
                            # elif len(key) == 4:
                            #     print('key', key)
                            #     print(ls)
        return child_pose

    def apply_mapping(self, mapping):
        self.parent = mapping.get(self.parent, self.parent)
        self.child = mapping.get(self.child, self.child)

    def __repr__(self):
        name = self.__class__.__name__
        if self.child_link is None:
            return '{}({}, {})'.format(name, self.parent, self.child)
        else:
            return '{}({}, {}-{})'.format(name, self.parent, self.child, self.child_link)


class ObjAttachment(Attachment):

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

    def assign(self):
        parent_link_pose = get_link_pose(self.parent, self.parent_link)
        child_pose = body_from_end_effector(parent_link_pose, self.grasp_pose)
        set_pose(self.child, child_pose)


def add_attachment_in_world(state=None, obj=None, parent=-1, parent_link=None, attach_distance=0.1, **kwargs):

    from robot_builder.robots import RobotAPI

    ## can attach without contact
    new_attachments = add_attachment(state=state, obj=obj, parent=parent, parent_link=parent_link,
                                     attach_distance=attach_distance, **kwargs)

    ## update object info
    world = state.world
    if hasattr(world, 'BODY_TO_OBJECT'):
        for body, attachment in new_attachments.items():
            obj = world.BODY_TO_OBJECT[body]
            if hasattr(obj, 'supporting_surface'):
                if isinstance(parent, RobotAPI) and obj.supporting_surface is not None:
                    obj.remove_supporting_surface()
                else:
                    obj.change_supporting_surface(parent)
    # for k in new_attachments:
    #     if k in state.world.ATTACHMENTS:
    #         state.world.ATTACHMENTS.pop(k)

    return new_attachments


def add_attachment(state=None, obj=None, parent=-1, parent_link=None, attach_distance=0.1,
                   OBJ=True, verbose=False, debug=False):
    """ can attach without contact """
    new_attachments = {}

    if parent == -1:  ## use robot as parent
        parent = state.robot
        link1 = None
        parent_link = state.robot.base_link
        OBJ = False
    else:
        link1 = parent_link

    joint = None
    if isinstance(obj, tuple):
        from pybullet_tools.general_streams import get_handle_link
        link1 = get_handle_link(obj)
        obj, joint = obj

    # collision_infos = get_closest_points(parent, obj, link1=link1, max_distance=INF)
    # min_distance = min([INF] + [info.contactDistance for info in collision_infos])
    # if True or attach_distance is None or (min_distance < attach_distance):  ## (obj not in new_attachments) and
    if True:
        if joint is not None:
            attachment = create_attachment(parent, parent_link, obj, child_link=link1, child_joint=joint, OBJ=OBJ)
        else:
            attachment = create_attachment(parent, parent_link, obj, OBJ=OBJ)
        new_attachments[obj] = attachment  ## may overwrite older attachment
        if verbose:
            print(f'\nbullet_utils.add_attachment | {attachment}\n')
        if debug:
            attachment.assign()
    return new_attachments


def create_attachment(parent, parent_link, child, child_link=None, child_joint=None, OBJ=False):
    parent_link_pose = get_link_pose(parent, parent_link)
    child_pose = get_pose(child)
    grasp_pose = multiply(invert(parent_link_pose), child_pose)
    if OBJ:  ## attachment between objects
        return ObjAttachment(parent, parent_link, grasp_pose, child)
    return Attachment(parent, parent_link, grasp_pose, child,
                      child_link=child_link, child_joint=child_joint)


def remove_attachment(state, obj=None, verbose=False):
    if isinstance(obj, tuple): obj = obj[0]
    new_attachments = dict(state.attachments)
    if obj in new_attachments:
        if verbose:
            print(f'\nbullet_utils.remove_attachment | {new_attachments[obj]}\n')
        new_attachments.pop(obj)
    return new_attachments


#######################################################


def sample_random_pose(aabb):
    ## sample point
    x = np.random.uniform(aabb[0][0], aabb[1][0])
    y = np.random.uniform(aabb[0][1], aabb[1][1])
    z = np.random.uniform(aabb[0][2], aabb[1][2])

    ## sample rotation
    case = np.random.randint(0, 4)
    roll, pitch, yaw = 0, 0, 0
    if case == 0:
        roll = np.random.uniform(0, 2 * np.pi)
    elif case == 1:
        pitch = np.random.uniform(0, 2 * np.pi)
    elif case == 2:
        yaw = np.random.uniform(0, 2 * np.pi)
    return Pose(Point(x, y, z), Euler(roll, pitch, yaw))


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


def has_much_larger_aabb(body_larger, body_smaller):
    extent_larger = get_aabb_extent(get_aabb(body_larger))[:2]
    extent_smaller = get_aabb_extent(get_aabb(body_smaller))[:2]
    area_larger = extent_larger[0] * extent_larger[1]
    area_smaller = extent_smaller[0] * extent_smaller[1]
    return area_larger > area_smaller * 4


def get_center_top_surface(body):
    aabb = get_aabb(body)
    x, y = get_aabb_center(aabb)[:2]
    z = aabb.upper[2]
    return x, y, z


def sample_center_top_surface(body, k=None):
    point = get_center_top_surface(body)
    yaw = random.uniform(0, 2 * np.pi)
    result = point, quat_from_euler(Euler(yaw=yaw))
    if k is not None:
        result = [result]
        for _ in range(k-1):
            result.append((point, quat_from_euler(Euler(yaw=random.uniform(0, 2 * np.pi)))))
    return result


def check_placement(obj, region):
    return is_center_stable(obj, region, above_epsilon=INF, below_epsilon=INF)  # is_center_stable | is_placement


def is_on(obj_aabb, region_aabb):
    return aabb_contains_aabb(aabb2d_from_aabb(obj_aabb), aabb2d_from_aabb(region_aabb))


def is_above(robot, aabb):
    # return is_center_stable(robot, self.button)
    return aabb_contains_point(point_from_pose(robot.get_pose())[:2], aabb2d_from_aabb(aabb))


def is_placement(body, surface, link=None, **kwargs):
    if isinstance(surface, tuple):
        surface, _, link = surface
    return is_placed_on_aabb(body, get_aabb(surface, link), **kwargs)


def is_contained(body, space):
    if isinstance(space, tuple):
        return aabb_contains_aabb(get_aabb(body), get_aabb(space[0], link=space[-1]))
    return aabb_contains_aabb(get_aabb(body), get_aabb(space))


def pose_to_xyzyaw(pose):
    xyzyaw = list(nice_tuple(pose[0]))
    xyzyaw.append(nice_float(euler_from_quat(pose[1])[2]))
    return tuple(xyzyaw)


def xyzyaw_to_pose(xyzyaw):
    return tuple((tuple(xyzyaw[:3]), quat_from_euler(Euler(0, 0, xyzyaw[-1]))))


def pose_from_xyzyaw(q):
    x, y, z, yaw = q
    return Pose(point=Point(x=x, y=y, z=z), euler=Euler(yaw=yaw))


def bconf_to_pose(bq):
    from pybullet_tools.utils import Pose
    if len(bq.values) == 3:
        x, y, yaw = bq.values
        z = 0
    elif len(bq.values) == 4:
        x, y, z, yaw = bq.values
    return Pose(point=Point(x, y, z), euler=Euler(yaw=yaw))


def pose_to_bconf(rpose, robot):
    (x, y, z), quant = rpose
    yaw = euler_from_quat(quant)[-1]
    if robot.use_torso:
        return Conf(robot, robot.get_group_joints('base-torso'), (x, y, z, yaw))
    return Conf(robot, robot.get_group_joints('base'), (x, y, yaw))


def add_pose(p1, p2):
    point = np.asarray(p1[0]) + np.asarray(p2[0])
    euler = np.asarray(euler_from_quat(p1[1])) + np.asarray(euler_from_quat(p2[1]))
    return (tuple(point.tolist()), quat_from_euler(tuple(euler.tolist())))


def sample_new_bconf(bq1):
    limits = [0.05] * 3
    def rand(limit):
        return np.random.uniform(-limit, limit)
    values = (bq1.values[i] + rand(limits[i]) for i in range(len(limits)))
    return Conf(bq1.body, bq1.joints, values)


def draw_pose2d(pose2d, z=0., **kwargs):
    return draw_pose(pose_from_pose2d(pose2d, z=z), **kwargs)


def draw_pose2d_path(path, z=0., **kwargs):
    # TODO: unify with open-world-tamp, namo, etc.
    # return list(flatten(draw_point(np.append(pose2d[:2], [z]), **kwargs) for pose2d in path))
    return list(flatten(draw_pose2d(pose2d, z=z, **kwargs) for pose2d in path))


def draw_pose3d_path(path, **kwargs):
    ## flying gripper
    if len(path[0]) == 6:
        from .flying_gripper_utils import pose_from_se3
        return list(flatten(draw_pose(pose_from_se3(q), **kwargs) for q in path))

    ## pr2
    elif len(path[0]) == 4:
        return list(flatten(draw_pose(pose_from_xyzyaw(q), **kwargs) for q in path))

    ## object
    elif len(path[0]) == 2:
        return list(flatten(draw_pose(q, **kwargs) for q in path))

    else:
        assert "What's this path" + path


def draw_colored_pose(pose, length=0.1, color=None, **kwargs):
    if color is None:
        handles = draw_pose(pose, length=length, **kwargs)
    else:
        origin_world = tform_point(pose, np.zeros(3))
        handles = []
        for k in range(3):
            axis = np.zeros(3)
            axis[k] = 1
            axis_world = tform_point(pose, length*axis)
            handles.append(add_line(origin_world, axis_world, color=color, **kwargs))
    return handles


## ----------------------------------------------------------------


OBJ_YAWS = {
    'Microwave': PI, 'Toaster': PI / 2
}


def sample_pose(obj, aabb, obj_aabb=None, yaws=OBJ_YAWS):
    ## sample a pose in aabb that can fit an object in
    if obj_aabb is not None:
        ori = aabb
        lower, upper = obj_aabb
        diff = [(upper[i] - lower[i]) / 2 for i in range(3)]

        # ## if the surface is large enough, give more space
        # dx1, dy1 = get_aabb_extent(aabb2d_from_aabb(aabb))
        # dx2, dy2 = get_aabb_extent(aabb2d_from_aabb(obj_aabb))
        # if dx1 > 2*dx2 and dy1 > 2*dy2:
        #     diff = [(upper[i] - lower[i]) / 2 * 3 for i in range(3)]

        lower = [aabb[0][i] + diff[i] for i in range(3)]
        upper = [aabb[1][i] - diff[i] for i in range(3)]
        aabb = AABB(lower=lower, upper=upper)
        # print('bullet_utils.sample_pose\tadjusted aabb for obj', nice(ori), '->', nice(aabb))

    ## adjust z to be lower
    height = get_aabb_extent(aabb)[2]
    if (obj_aabb is not None and height > 5 * get_aabb_extent(obj_aabb)[2]) or height > 1:
        x, y, _ = aabb.upper
        z = aabb.lower[2] + height / 3
        xl, yl, zl = aabb.lower
        aabb = AABB(lower=[xl, yl, zl+height / 4], upper=[x, y, z])
        # print('bullet_utils.sample_pose\t!adjusted z to be lower')

    x, y, z = sample_aabb(aabb)

    ## use pre-defined yaws for appliances like microwave
    if obj in yaws:
        yaw = yaws[obj]
    else:
        yaw = np.random.uniform(0, PI)

    return x, y, z, yaw


def sample_obj_on_body_link_surface(obj, body, link, PLACEMENT_ONLY=False, max_trial=3, verbose=False):
    aabb = get_aabb(body, link)
    # x, y, z, yaw = sample_pose(obj, aabb)
    # maybe = load_asset(obj, x=round(x, 1), y=round(y, 1), yaw=yaw, floor=(body, link), scale=scales[obj], maybe=True)
    # sample_placement(maybe, body, bottom_link=link)

    x, y, z, yaw = sample_pose(obj, aabb)
    maybe = obj
    trial = 0

    ## if surface smaller than object, just put in center
    if get_aabb_volume(aabb2d_from_aabb(get_aabb(maybe))) > get_aabb_volume(aabb2d_from_aabb(aabb)):
        x, y, z, yaw = sample_pose(obj, aabb, get_aabb(maybe))
        x, y = get_aabb_center(aabb2d_from_aabb(aabb))
        pose = Pose(point=Point(x=x, y=y, z=z), euler=Euler(yaw=yaw))
        set_pose(maybe, pose)

    else:
        while not aabb_contains_aabb(aabb2d_from_aabb(get_aabb(maybe)), aabb2d_from_aabb(aabb)):
            x, y, z, yaw = sample_pose(obj, aabb, get_aabb(maybe))
            pose = Pose(point=Point(x=x, y=y, z=z), euler=Euler(yaw=yaw))
            set_pose(maybe, pose)
            # print(f'sampling surface for {body}-{link}', nice(aabb2d_from_aabb(aabb)))
            trial += 1
            if trial > max_trial:
                if max_trial > 1 and verbose:
                    print(f'sample_obj_on_body_link_surface\t sample {obj} on {body}-{link} | exceed max trial {max_trial}')
                break

    if PLACEMENT_ONLY:
        z = stable_z(obj, body, link)
        return x, y, z, yaw

    # print(nice(aabb2d_from_aabb(aabb)))
    # print(nice(aabb2d_from_aabb(get_aabb(maybe))))
    return maybe


def sample_obj_in_body_link_space(obj, body, link=None, PLACEMENT_ONLY=False,
                                  draw=False, verbose=False, visualize=False, max_trial=3):
    if visualize:
        set_renderer(True)
    draw &= has_gui()
    if verbose:
        print(f'sample_obj_in_body_link_space(obj={obj}, body={body}, link={link})')
        # wait_for_user()

    aabb = get_aabb(body, link)
    # draw_aabb(aabb)

    x, y, z, yaw = sample_pose(obj, aabb, obj_aabb=get_aabb(obj))
    maybe = obj
    handles = draw_fitted_box(maybe)[-1] if draw else []

    def sample_one(maybe, handles):
        x, y, z, yaw = sample_pose(obj, aabb, get_aabb(maybe))
        z += 0.01
        pose = Pose(point=Point(x=x, y=y, z=z), euler=Euler(yaw=yaw))
        set_pose(maybe, pose)

        remove_handles(handles)
        handles = draw_fitted_box(maybe)[-1] if draw else []
        return maybe, (x, y, z, yaw), handles

    def sample_maybe(body, maybe, pose, handles):
        (x, y, z, yaw) = pose

        remaining_trials = 10
        while not aabb_contains_aabb(get_aabb(maybe), aabb) or body_collision(body, maybe, link1=link):
            remaining_trials -= 1
            if remaining_trials < 0:
                break
            maybe, (x, y, z, yaw), handles = sample_one(maybe, handles)
            if verbose:
                draw_points(body, link)
                print('\n ---- remaining_trials =', remaining_trials)
                print(f'sampling space for {body}-{link} {nice(aabb)} : {obj} {nice(get_aabb(maybe))}', )
                print(f'   collision between {body}-{link} and {maybe}: {body_collision(body, maybe, link1=link)}')
                print(f'   aabb of {body}-{link} contains that of {maybe}: {aabb_contains_aabb(get_aabb(maybe), aabb)}')
                print()
                # set_camera_target_body(maybe, dx=1.5, dy=0, dz=0.7)

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
                handles = draw_fitted_box(maybe)[-1] if draw else []
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
            handles = draw_fitted_box(maybe)[-1] if draw else []
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
    with timeout(duration=1, desc=f"({obj}, {body}, {link})"):
        while result is None:
            maybe, pose, handles = sample_one(maybe, handles)
            maybe, pose, handles = sample_maybe(body, maybe, pose, handles)
            result = adjust_z(body, maybe, pose, handles)
    if result is None:
        return None
    maybe, (x, y, z, yaw), handles = result

    remove_handles(handles)
    #set_renderer(True)
    if PLACEMENT_ONLY: return x, y, z, yaw
    # print(nice(aabb2d_from_aabb(aabb)))
    # print(nice(aabb2d_from_aabb(get_aabb(maybe))))
    return maybe


def get_noisy_pose(pose):
    noisy_point = np.random.normal(loc=pose[0], scale=0.05, size=3)
    noisy_point[-1] = pose[0][-1]
    yaw = np.random.uniform(low=-PI, high=PI)
    return tuple(noisy_point), quat_from_euler(Euler(yaw=yaw))


######################################################################################


def change_pose_interactive(obj):
    from pynput import keyboard

    exit_note = "(Press esc in terminal to exit)"

    pose = obj.get_pose()
    xyz, quat = pose
    euler = euler_from_quat(quat)
    pose = np.asarray([xyz, euler])
    adjustments = {
        # 'w': ((0, 0, 0.01), (0, 0, 0)),
        's': ((0, 0, -0.01), (0, 0, 0)),
        keyboard.Key.up: ((0, 0, 0.01), (0, 0, 0)),
        keyboard.Key.down: ((0, 0, -0.01), (0, 0, 0)),
        'a': ((0, -0.01, 0), (0, 0, 0)),
        'd': ((0, 0.01, 0), (0, 0, 0)),
        keyboard.Key.left: ((0, -0.01, 0), (0, 0, 0)),
        keyboard.Key.right: ((0, 0.01, 0), (0, 0, 0)),
        'f': ((0.01, 0, 0), (0, 0, 0)),
        'r': ((-0.01, 0, 0), (0, 0, 0)),
        'q': ((0, 0, 0), (0, 0, -0.1)),
        'e': ((0, 0, 0), (0, 0, 0.1)),
    }
    adjustments = {k: np.asarray(v) for k, v in adjustments.items()}

    def on_press(key, pose=pose):
        try:
            pressed = key.char.lower()
            print(f'alphanumeric key {key.char} pressed {exit_note}')
        except AttributeError:
            pressed = key
            print(f'special key {key} pressed {exit_note}')

        if pressed in adjustments:
            pose += adjustments[pressed]
            point = tuple(pose[0])
            euler = tuple(pose[1])
            pose = nice((point, quat_from_euler(euler)), keep_quat=True)
            print(f'\tnew pose of {obj.shorter_name}\t{pose}')
            set_pose(obj.body, pose)

    def on_release(key):
        if key == keyboard.Key.esc:
            return False

    print('-' * 10 + f' Enter WASDRF for poses and QE for yaw {exit_note}' + '-' * 10)
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


def has_getch():
    try:
        import getch
    except ImportError:
        print('Please install has_getch in order to use `step_by_step`: ```pip install getch```\n')
        sys.exit()
        return False
    return True

