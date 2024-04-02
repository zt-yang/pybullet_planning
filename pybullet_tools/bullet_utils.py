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
from pybullet_tools.logging import dump_json

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
    BROWN, BLUE, WHITE, TAN, GREY, YELLOW, GREEN, BLACK, RED, tform_point


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
CAMERA_MATRIX = get_camera_matrix(width=640, height=480, fx=525., fy=525.) # 319.5, 239.5 | 772.55, 772.5S

colors = [GREEN, BROWN, BLUE, WHITE, TAN, GREY, YELLOW, BLACK, RED]
color_names = ['GREEN', 'BROWN', 'BLUE', 'WHITE', 'TAN', 'GREY', 'YELLOW', 'BLACK', 'RED']


def load_robot_urdf(urdf_path):
    with LockRenderer():
        with HideOutput():
            robot = load_model(urdf_path, fixed_base=False)
    return robot



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


def pose_from_xyzyaw(q):
    x, y, z, yaw = q
    return Pose(point=Point(x=x, y=y, z=z), euler=Euler(yaw=yaw))


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
    if isinstance(ele, int) and '.' not in str(ele):
        return int(ele)
    else:
        return round(ele, round_to)


def nice_tuple(tup, round_to=3):
    new_tup = []
    for ele in tup:
        new_tup.append(nice_float(ele, round_to))
    return tuple(new_tup)


def nice(tuple_of_tuples, round_to=3, one_tuple=True, keep_quat=False):
    if tuple_of_tuples is None:
        return tuple_of_tuples

    ## float, int
    if isinstance(tuple_of_tuples, float) or isinstance(tuple_of_tuples, int):
        return nice_float(tuple_of_tuples, round_to)

    elif len(tuple_of_tuples) == 0:
        return []

    ## position, pose
    elif isinstance(tuple_of_tuples[0], tuple) or isinstance(tuple_of_tuples[0], list) \
            or isinstance(tuple_of_tuples[0], np.ndarray):

        ## pose = (point, quat) -> (point, euler)
        if len(tuple_of_tuples[0]) == 3 and len(tuple_of_tuples[1]) == 4:
            second_tuple = tuple_of_tuples[1]
            if keep_quat:
                one_tuple = False
            else:
                second_tuple = euler_from_quat(second_tuple)
            if one_tuple:
                one_list = list(tuple_of_tuples[0]) + list(second_tuple)
                return nice(tuple(one_list) , round_to)
            return nice( (tuple_of_tuples[0], second_tuple), round_to)
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

def get_root_links(body):
    [fixed_links] = get_rigid_clusters(body, links=[ROOT_LINK])
    return fixed_links

def articulated_collisions(obj, obstacles, verbose=False, **kwargs): # TODO: articulated_collision?
    # TODO: cache & compare aabbs
    for obstacle in obstacles:
        # dump_body(obstacle)
        # joints = get_movable_joints(obstacle)
        root_links = get_root_links(obstacle)
        if link_pairs_collision(body1=obstacle, links1=root_links, body2=obj, **kwargs):
            if verbose:
                print(f'articulated_collisions | obj = {obj}\tobstacle = {obstacle}\troot_links = {root_links}')
            # dump_body(obj)
            # dump_body(obstacle)
            # for link in root_links:
            #     collision_infos = get_closest_points(body1=obj, body2=obstacle, link2=link, **kwargs)
            #     for i, collision_info in enumerate(collision_infos):
            #         print(i, len(collision_infos), collision_info)
            #         draw_collision_info(collision_info)
            return True
    return False


def collided_around(obj, obstacles, padding=0.05, **kwargs):
    """ if obj is collided within 0.2 away from its area """
    (x, y, z), quat = get_pose(obj)
    lx = ly = padding
    # lx, ly, _ = get_aabb_extent(get_aabb(obj))

    # aabb = get_aabb(obj)
    # draw_aabb(aabb)
    # x_lower, y_lower, _ = aabb.lower
    # x_upper, y_upper, z_upper = aabb.upper
    # aabb = AABB(lower=(x_lower-lx, y_lower-ly, z_upper),
    #             upper=(x_upper+lx, y_upper+ly, z_upper+0.1))
    # set_camera_target_body(obj, dx=0.5, dy=0.5, dz=0.5)
    # draw_aabb(aabb)
    # wait_unlocked()

    for i, j in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_x = x + i * lx
        new_y = y + j * ly
        set_pose(obj, ((new_x, new_y, z), quat))
        if collided(obj, obstacles, **kwargs):
            return True
    return False


COLLISION_FILE = join(dirname(__file__), '..', '..', 'collisions.json')


def initialize_logs():
    from pybullet_tools.logging import TXT_FILE
    if isfile(TXT_FILE): os.remove(TXT_FILE)


def initialize_collision_logs():
    if isfile(COLLISION_FILE): os.remove(COLLISION_FILE)


def log_collided(obj, obs, visualize=False):
    # from world_builder.robots import RobotAPI

    collisions = {}
    if isfile(COLLISION_FILE):
        collisions = json.load(open(COLLISION_FILE, 'r'))
    key = f"({', '.join([str(obj), str(obs)])})"
    if key not in collisions:
        collisions[key] = 0
    collisions[key] += 1

    collisions = {k: v for k, v in sorted(collisions.items(), key=lambda item: item[1], reverse=True)}
    dump_json(collisions, COLLISION_FILE, indent=3, width=40, sort_dicts=False)

    if visualize: ## and (not isinstance(obj, str) or 'pr2' not in obj):
        set_renderer(True)
        wait_unlocked()


def collided(obj, obstacles=[], world=None, tag='', articulated=False, verbose=False,
             visualize=False, min_num_pts=3, use_aabb=True, ignored_pairs=[],
             log_collisions=False, **kwargs):

    prefix = '\t\tbullet_utils.collided '
    if len(tag) > 0: prefix += f'( {tag} )'

    ## first get answer
    if articulated:
        result = articulated_collisions(obj, obstacles, use_aabb=use_aabb, verbose=verbose, **kwargs)
        if verbose and result:
            print(prefix, '| articulated', obj, obstacles)
        return result
    # else:
    #     result = any(pairwise_collision(obj, b, use_aabb=use_aabb, **kwargs) for b in obstacles)
    # if not verbose:
    #     return result

    result = False
    ## first find the bodies that collides with obj
    bodies = []
    to_print = ''
    for b in obstacles:
        if pairwise_collision(obj, b) and (obj, b) not in ignored_pairs:
            if world is None:
                import traceback
                print('bullet_utils.collided | world is None')
                print(traceback.format_exc())
            obj_print = world.get_name(obj) if world is not None else obj
            b_print = world.get_name(b) if world is not None else b
            if verbose:
                # if b_print == 'floor1':
                #     print(obstacles)
                to_print += f'{prefix} {obj_print} collides with {b_print}'
            result = True
            bodies.append(b)
            if log_collisions:
                log_collided(obj_print, b_print)

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
        if verbose:
            if world is not None:
                obj = world.get_name(obj)
            line = f'{obj} with {body_links}'
            print(f"{prefix} | {line}")
            if not log_collisions:
                return line.replace("'", "")
    return result


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


#######################################################


ROTATIONAL_MATRICES = {}


def get_rotation_matrix(body, verbose=True):
    import untangle
    r = unit_pose()
    if is_mesh_entity(body):
        return r
    collision_data = get_collision_data(body, 0)
    # if set(get_all_links(body)) == {0, -1}:
    if len(collision_data) > 0:
        urdf_file = dirname(collision_data[0].filename.decode())
        count = 0
        while len(urdf_file.strip()) == 0:
            count += 1
            urdf_file = dirname(collision_data[count].filename.decode())
        urdf_file = urdf_file.replace('/textured_objs', '').replace('/base_objs', '').replace('/vhacd', '')
        mobility_urdf_file = join(urdf_file, 'mobility.urdf')
        if isfile(mobility_urdf_file):
            if urdf_file not in ROTATIONAL_MATRICES:
                # if verbose:
                #     print('get_rotation_matrix | urdf_file = ', abspath(urdf_file))
                joints = untangle.parse(mobility_urdf_file).robot.joint
                if isinstance(joints, list):
                    for j in joints:
                        if j.parent['link'] == 'base':
                            joint = j
                            break
                else:
                    joint = joints
                rpy = joint.origin['rpy'].split(' ')
                rpy = tuple([eval(e) for e in rpy])
                if equal(rpy, (1.57, 1.57, -1.57), epsilon=0.1):
                    r = Pose(euler=Euler(math.pi / 2, 0, -math.pi / 2))
                elif equal(rpy, (3.14, 3.14, -1.57), epsilon=0.1):
                    r = Pose(euler=Euler(0, 0, -math.pi / 2))
                elif equal(rpy, (1.57, 0, -1.57), epsilon=0.1):
                    r = Pose(euler=Euler(math.pi/2, 0, -math.pi / 2))
                elif equal(rpy, (0, 0, -1.57), epsilon=0.1):
                    r = Pose(euler=Euler(math.pi/2, 0, math.pi / 2))
                ROTATIONAL_MATRICES[urdf_file] = r
            r = ROTATIONAL_MATRICES[urdf_file]
    return r


def draw_points(body, link=None, size=0.2, **kwargs):
    body_pose = get_model_pose(body, link=link, **kwargs)
    vertices = get_model_points(body, link=link)
    vertices = apply_affine(body_pose, vertices)
    handles = []
    num_vertices = 40
    if len(vertices) > num_vertices:
        gap = int(len(vertices)/num_vertices)
        vertices = vertices[::gap]
    for v in vertices:
        handles.append(draw_point(v, size=size, color=RED))
    return handles


def get_model_pose(body, link=None, **kwargs):
    if link is None:
        body_pose = multiply(get_pose(body), get_rotation_matrix(body, **kwargs))
    else:
        body_pose = get_link_pose(body, link)
    return body_pose


def get_model_points(body, link=None):
    if link is None:
        links = get_links(body)
    else:
        links = [link]

    vertices = []
    for link in links:
        vv = vertices_from_rigid(body, link)
        if len(vv) > 0:
            new_vertices = apply_affine(unit_pose(), vv)
            vertices.extend(new_vertices)
    return vertices


def draw_fitted_box(body, link=None, draw_box=False, draw_centroid=False, verbose=False, **kwargs):
    body_pose = get_model_pose(body, link=link, verbose=verbose)
    vertices = get_model_points(body, link=link)
    if link is None:  link = -1
    data = get_collision_data(body, link)
    if len(data) == 0 or data[0].geometry_type == p.GEOM_MESH:
        aabb = aabb_from_points(vertices)
    else: ## if data.geometry_typep == p.GEOM_BOX:
        aabb = get_aabb(body)
    # TODO(caelan): global DRAW variable that disables
    handles = []
    if draw_box:
        handles += draw_bounding_box(aabb, body_pose, **kwargs)
    if draw_centroid:
        handles += draw_face_points(aabb, body_pose, dist=0.04)
    return aabb, handles


def draw_bounding_box(aabb, body_pose, **kwargs):
    handles = []
    for a, b in get_aabb_edges(aabb):
        p1, p2 = apply_affine(body_pose, [a, b])
        handles.append(add_line(p1, p2, **kwargs))
    return handles


def draw_face_points(aabb, body_pose, dist=0.08):
    c = get_aabb_center(aabb)
    w, l, h = get_aabb_extent(aabb)
    faces = [(w/2+dist, 0, 0), (0, l/2+dist, 0), (0, 0, h/2+dist)]
    faces += [minus(0, f) for f in faces]
    faces = [(w/2+dist, 0, 0)]
    faces = [add(f, c) for f in faces]
    faces = apply_affine(body_pose, faces)
    handles = []
    for f in faces:
        handles.extend(draw_point(f, size=0.02, color=RED))
    return handles


def is_type_of_entity(body, link=-1, geom_type=p.GEOM_BOX):
    if isinstance(body, tuple):
        return False
    if link is None:
        link = -1
    data = get_collision_data(body, link)
    return len(data) != 0 and data[0].geometry_type == geom_type


def is_box_entity(body, link=-1):
    return is_type_of_entity(body, link=link, geom_type=p.GEOM_BOX)


def is_mesh_entity(body, link=-1):
    return is_type_of_entity(body, link=link, geom_type=p.GEOM_MESH)


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


#################################################################


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
        from .pr2_streams import LINK_POSE_TO_JOINT_POSITION as LP2JP

        if verbose:
            print('\nbullet.Attachment.assign() | LINK_POSE_TO_JOINT_POSITION')
            pprint(LP2JP)

        # robot_base_pose = self.parent.get_positions(roundto=3)
        # robot_arm_pose = self.parent.get_positions(joint_group='left', roundto=3)  ## only left arm for now
        parent_link_pose = get_link_pose(self.parent, self.parent_link)
        child_pose = body_from_end_effector(parent_link_pose, self.grasp_pose)
        if self.child_link is None:
            set_pose(self.child, child_pose)
        elif self.child in LP2JP:  ## pull drawer handle
            if self.child in LP2JP and self.child_joint in LP2JP[self.child]:
                ls = LP2JP[self.child][self.child_joint]
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
            return '{}({},{})'.format(name, self.parent, self.child)
        else:
            return '{}({},{}-{})'.format(name, self.parent, self.child, self.child_link)


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


def add_attachment_in_world(state=None, obj=None, parent=-1, parent_link=None, attach_distance=0.1,
                            OBJ=True, verbose=False):

    from robot_builder.robots import RobotAPI

    ## can attach without contact
    new_attachments = add_attachment(state=state, obj=obj, parent=parent, parent_link=parent_link,
                                     attach_distance=attach_distance, OBJ=OBJ, verbose=verbose)

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


def add_attachment(state=None, obj=None, parent=-1, parent_link=None, attach_distance=0.1, OBJ=True, verbose=False):
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
            print(f'\nbullet_utils.add_attachment | {new_attachments[obj]}\n')
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


def get_camera_point_target():
    cam = p.getDebugVisualizerCamera()
    camForward = cam[5]
    dist, camTarget = cam[-2:]
    camPos = np.array(camTarget) - dist * np.array(camForward)
    return camPos, camTarget


def get_camera_image_at_pose(camera_point, target_point, camera_matrix, far=5.0, **kwargs):
    # far is the maximum depth value
    width, height = map(int, dimensions_from_camera_matrix(camera_matrix))
    _, vertical_fov = get_field_of_view(camera_matrix)
    return get_image(camera_point, target_point, width=width, height=height,
                     vertical_fov=vertical_fov, far=far, **kwargs)


def set_camera_target_body(body, link=None, dx=None, dy=None, dz=None, distance=1):
    # if isinstance(body, tuple):
    #     link = BODY_TO_OBJECT[body].handle_link
    #     body = body[0]
    aabb = get_aabb(body, link)
    x = (aabb.upper[0] + aabb.lower[0]) / 2
    y = (aabb.upper[1] + aabb.lower[1]) / 2
    z = (aabb.upper[2] + aabb.lower[2]) / 2
    if dx is None and dy is None and dz is None:
        dx = min(get_aabb_extent(aabb)[0] * 2 * distance, 2)
        dy = min(get_aabb_extent(aabb)[1] * 2 * distance, 2)
        dz = min(get_aabb_extent(aabb)[2] * 2 * distance, 2)
    camera_point = [x + dx, y + dy, z + dz]
    target_point = [x, y, z]
    set_camera_pose(camera_point=camera_point, target_point=target_point)
    return camera_point, target_point


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
        return BASE_JOINTS
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
    movable = joint in get_movable_joints(body)
    joint_type = JOINT_TYPES[get_joint_type(body, joint)]

    category = 'fixed'
    state = None
    if min_limit < max_limit:

        ## knob on faucet, oven
        if joint_type == 'revolute' and (min_limit + max_limit == 0 or 'knob' in name):
            category = 'knob'
            if pose == min_limit:
                state = 'knob TURNED OFF'
            elif pose == max_limit:
                state = 'knob TURNED ON'
            else:
                state = 'knob TURNED ON partially'

        elif joint_type == 'revolute' and min_limit == 0:
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
        print(f'   joint {name}, pose = {pose}, limit = {nice((min_limit, max_limit))}, \
            state = {state}, movable = {movable}')
    return category, state


def toggle_joint(body, joint):
    category, state = check_joint_state(body, joint)
    if 'OPENED' in state:
        close_joint(body, joint)
    elif 'CLOSED' in state:
        open_joint(body, joint)


def open_joint(body, joint, extent=0.95, pstn=None,
               return_pstn=False, hide_door=False):
    if pstn is None:
        if isinstance(joint, str):
            joint = joint_from_name(body, joint)
        min_limit = get_min_limit(body, joint)
        max_limit = get_max_limit(body, joint)
        category, state = check_joint_state(body, joint)
        if category == 'door-max':
            pstn = max_limit * extent
            if hide_door:
                pstn = 3/2 * PI
        elif category == 'door-min':
            pstn = min_limit * extent
            if hide_door:
                pstn = -3/2 * PI
        elif category == 'drawer':
            pstn = max_limit
    if return_pstn:
        return pstn
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

def get_readable_list(lst, world=None, NAME_ONLY=False, TO_LISDF=False):
    to_print = [lst[0]]
    for word in lst[1:]:
        if world is not None:
            name = world.get_name(word)
            last_is_tuple = (len(to_print) != 0) and isinstance(to_print[-1], tuple)
            if name is not None and not last_is_tuple: ## ['=', ('PickCost',), 'pr2|1']
                if TO_LISDF:
                    name = world.get_lisdf_name(word)
                elif not NAME_ONLY:
                    name = world.get_debug_name(word)
                to_print.append(name)
            else:
                to_print.append(word)
        else:
            to_print.append(word)
    return to_print


def summarize_facts(facts, world=None, name='Initial facts', print_fn=None):
    if print_fn is None:
        from pybullet_tools.logging import myprint as print_fn
    print_fn('----------------')
    print_fn(f'{name} ({len(facts)})')
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
        to_print_line = sorted([str(l).lower() for l in to_print_line])
        to_print = ', '.join(to_print_line)
        print_fn(f'  {pred} [{len(to_print_line)}] : {to_print}')
    print_fn('----------------')


def print_plan(plan, world=None, print_fn=None):
    from pddlstream.language.constants import is_plan
    if print_fn is None:
        from pybullet_tools.logging import myprint as print_fn

    if not is_plan(plan):
        return
    step = 1
    print_fn('Plan:')
    for action in plan:
        name, args = action
        if name.startswith('_'):
            print_fn(f' ) {name}')
            continue
        args2 = [str(a) for a in get_readable_list(args, world)]
        print_fn('{:2}) {} {}'.format(step, name, ' '.join(args2)))
        step += 1
    print_fn()


def print_goal(goal, world=None, print_fn=None):
    if print_fn is None:
        from pybullet_tools.logging import myprint as print_fn

    print_fn(f'Goal ({len(goal) - 1}): ({goal[0]}')
    for each in get_readable_list(goal[1:], world):
        print_fn(f'   {tuple(each)},')
    print_fn(')')


def summarize_poses(preimage):
    atposes = [f[-1] for f in preimage if f[0].lower() == 'atpose']
    poses = [f[-1] for f in preimage if f[0].lower() == 'pose' if f[-1] not in atposes]

    print('\n' + '=' * 25 + ' poses that can be cached to loaders_{domain}.py ' + '=' * 25)
    for pose in poses:
        print(nice(pose.value, keep_quat=True))
    print('-'*50+'\n')


def summarize_bconfs(preimage, plan):
    bconfs = [f[1] for f in preimage if f[0].lower() == 'bconf' and f[1].joint_state is not None]
    bconfs_ordered = []
    for action in plan:
        for arg in action.args:
            if arg in bconfs and arg not in bconfs_ordered:
                bconfs_ordered.append(arg)

    print('\n' + '=' * 25 + ' bconfs that can be cached to loaders_{domain}.py ' + '=' * 25)
    for bconf in bconfs_ordered:
        joint_state = {k: nice(v) for k, v in bconf.joint_state.items()}
        print(f"({nice(bconf.values)}, {joint_state}), ")
    print('-'*50+'\n')


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


def get_color_by_index(i):
    index = i % len(colors)
    return colors[index], color_names[index]


########################################################################


def add(elem1, elem2):
    return tuple(np.asarray(elem1)+np.asarray(elem2))


def minus(elem1, elem2):
    return tuple(np.asarray(elem1)-np.asarray(elem2))


def dist(elem1, elem2):
    return np.linalg.norm(np.asarray(elem1)-np.asarray(elem2))


def get_file_short_name(path):
    return path[path.rfind('/')+1:]


def in_list(elem, ls, epsilon=3e-3):
    for e in ls:
        if isinstance(elem, list) or isinstance(elem, tuple):
            if len(e) != len(elem):
                return None
        if equal(e, elem, epsilon=epsilon):
            return e
    return None


def equal_float(a, b, epsilon=0.0):
    return abs(a - b) <= epsilon


def equal(tup_a, tup_b, epsilon=0.001):
    if isinstance(tup_a, float) or isinstance(tup_a, int):
        return equal_float(tup_a, tup_b, epsilon)

    elif isinstance(tup_a, tuple):
        a = list(tup_a)
        b = list(tup_b)
        return all([equal(a[i], b[i], epsilon) for i in range(len(a))])

    return None


# def equal(tup1, tup2, epsilon=0.001):
#     if isinstance(tup1, float):
#         return abs(tup1 - tup2) < epsilon
#     if len(tup1) == 2:
#         return equal(tup1[0], tup2[0]) and equal(tup1[1], tup2[1])
#     return all([abs(tup1[i] - tup2[i]) < epsilon for i in range(len(tup1))])


def process_depth_pixels(pixels):
    """ scale float values 2-5 to int values 225-0 """
    n = (5 - pixels) / (5 - 2) * 225
    return n.astype('uint8')


def visualize_camera_image(image, index=0, img_dir='.', rgb=False, rgbd=False):
    import matplotlib.pyplot as plt

    if not isdir(img_dir):
        os.makedirs(img_dir, exist_ok=True)

    if rgbd:
        from PIL import Image

        depth_pixels = process_depth_pixels(image.depthPixels)

        for key, pixels in [('depth', depth_pixels),
                            ('rgb', image.rgbPixels[:,:,:3])]:
            sub_dir = join(img_dir, f"{key}s")
            if not isdir(sub_dir):
                os.makedirs(sub_dir, exist_ok=True)
            im = Image.fromarray(pixels)  ##
            im.save(join(sub_dir, f"{key}_{index}.png"))

    elif rgb:
        name = join(img_dir, f"rgb_image_{index}.png")
        plt.imshow(image.rgbPixels)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(name, bbox_inches='tight', dpi=100)
        plt.close()

    else:
        import seaborn as sns
        sns.set()
        name = join(img_dir, f"depth_map_{index}.png")
        ax = sns.heatmap(image.depthPixels, annot=False, fmt="d", vmin=2, vmax=5)
        plt.title(f"Depth Image ({index})", fontsize=12)
        plt.savefig(name, bbox_inches='tight', dpi=100)
        plt.close()

    # plt.show()


def get_obj_keys_for_segmentation(indices, unique=None):
    """ indices: {'(15, 1)': 'minifridge::joint_0', '0': 'pr20', '15': 'minifridge'}
        unique: {(23, 3): [(243, 0), ...], (23, 7): [...], ...}
        return obj_keys: {'minifridge::joint_0': [(15, 1), (15, 2)], 'pr20': [(0, 0)], 'minifridge': [(15, 0), (15, 1)]}
    """
    obj_keys = {}
    for k, v in indices.items():
        keys = []
        if isinstance(k, str):
            k = eval(k)
        if isinstance(k, int):  ##  and (k, 0) in unique
            if unique is not None:
                keys = [u for u in unique if u[0] == k]
                if len(keys) == 0:
                    keys = [(k, 0)]
            else:
                if k in get_bodies():
                    keys = [(k, l) for l in get_links(k)]
        elif isinstance(k, tuple) and len(k) == 3:
            keys = [(k[0], k[2])]
        elif isinstance(k, tuple) and len(k) == 2:
            if k[0] in get_bodies():
                keys = [(k[0], l) for l in get_door_links(k[0], k[1])]
        obj_keys[v] = keys
    return obj_keys


def get_segmask(seg, use_np=True):
    if use_np:
        # Mask to ignore pixels with value -1
        mask = seg != -1

        # Extract obUid and linkIndex using vectorized operations
        obUid = seg[mask] & ((1 << 24) - 1)
        linkIndex = (seg[mask] >> 24) - 1

        # Get coordinates
        coordinates = np.argwhere(mask)

        # Create a structured array with the necessary information
        structured_array = np.zeros(coordinates.shape[0],
                                    dtype=[('obUid', 'i4'), ('linkIndex', 'i4'), ('coordinates', 'i4', 2)])
        structured_array['obUid'] = obUid
        structured_array['linkIndex'] = linkIndex
        structured_array['coordinates'] = coordinates

        # Group by obUid and linkIndex
        unique = {}
        for uid_link in np.unique(structured_array[['obUid', 'linkIndex']], axis=0):
            mask = (structured_array['obUid'] == uid_link[0]) & (structured_array['linkIndex'] == uid_link[1])
            unique[tuple(uid_link)] = structured_array['coordinates'][mask].tolist()

    else:
        unique = {}
        for row in range(seg.shape[0]):
            for col in range(seg.shape[1]):
                pixel = seg[row, col]
                if pixel == -1: continue
                obUid = pixel & ((1 << 24) - 1)
                linkIndex = (pixel >> 24) - 1
                if (obUid, linkIndex) not in unique:
                    unique[(obUid, linkIndex)] = []
                    # print("obUid=", obUid, "linkIndex=", linkIndex)
                unique[(obUid, linkIndex)].append((row, col))
    return unique


def adjust_segmask(unique, world):
    """ looks ugly if we just remove some links, so create a doorless lisdf """
    # links_to_show = {}
    # for b, l in world.colored_links:
    #     if b not in links_to_show:
    #         links_to_show[b] = get_links(b)
    #     if l in links_to_show[b]:
    #         links_to_show[b].remove(l)
    # for b, ll in links_to_show.items():
    #     if len(ll) == len(get_links(b)):
    #         continue
    #     for l in links_to_show[b]:
    #         clone_body_link(b, l, visual=True, collision=True)
    #     remove_body(b)

    imgs = world.camera.get_image(segment=True, segment_links=True)
    seg = imgs.segmentationMaskBuffer
    movable_unique = get_segmask(seg)
    for k, v in movable_unique.items():
        if k not in unique:
            unique[k] = v
            # print(k, 'new', len(v))
        else:
            ## slow
            # new_v = [p for p in v if p not in unique[k]]
            # unique[k] += new_v
            old_count = len(unique[k])
            unique[k] += v
            unique[k] = list(set(unique[k]))
            # print(k, 'added', len(unique[k]) - old_count)
    return unique


def take_selected_seg_images(world, img_dir, body, indices, width=1280, height=960, **kwargs):
    from lisdf_tools.image_utils import save_seg_image_given_obj_keys

    ## take images from in front
    camera_point, target_point = set_camera_target_body(body, **kwargs)
    camera_kwargs = {'camera_point': camera_point, 'target_point': target_point}
    common = dict(img_dir=img_dir, width=width, height=height, fx=800)
    world.add_camera(**common, **camera_kwargs)

    ## take seg images
    imgs = world.camera.get_image(segment=True, segment_links=True)
    rgb = imgs.rgbPixels[:, :, :3]
    seg = imgs.segmentationMaskBuffer
    # seg = imgs.segmentationMaskBuffer[:, :, 0].astype('int32')
    unique = get_segmask(seg)
    obj_keys = get_obj_keys_for_segmentation(indices, unique)

    for k, v in indices.items():
        keys = obj_keys[v]
        mobility_id = world.get_mobility_id(body)
        file_name = join(img_dir, f"{mobility_id}_{v}.png")

        save_seg_image_given_obj_keys(rgb, keys, unique, file_name, crop=False)


def get_joint_range(body, joint):
    lower, upper = get_joint_limits(body, joint)
    return upper - lower


def is_joint_open(body, joint=-1, threshold=0.25, is_closed=False, verbose=True):
    if isinstance(body, tuple):
        body, joint = body
    lower, upper = get_joint_limits(body, joint)
    pstn_range = upper - lower
    limit_reversed = lower < 0 and upper == 0

    pstn = get_joint_position(body, joint)
    diff = (pstn - lower) if not limit_reversed else (upper - pstn)

    if is_closed:
        result = (diff == 0)
    else:
        result = (diff >= pstn_range * threshold)

    if verbose:
        status = 'open' if not is_closed else 'closed'
        joint_name = get_joint_name(body, joint)
        title = f'\tis_joint_{status}({(body, joint)}|{joint_name}, threshold={threshold})\t'
        print(title + f'at {pstn} in {nice((lower, upper))}\t', result)
    return result


def get_door_links(body, joint):
    with ConfSaver(body):
        min_pstn, max_pstn = get_joint_limits(body, joint)
        set_joint_position(body, joint, min_pstn)
        lps = [get_link_pose(body, l) for l in get_links(body)]
        set_joint_position(body, joint, max_pstn)
        links = [i for i in range(len(lps)) if lps[i] != get_link_pose(body, i)]
    return links


def sort_body_parts(bodies, existing=[]):
    indices = []
    links = {}
    joints = {}
    for body in bodies:
        if isinstance(body, tuple) and len(body) == 2:
            if body[0] not in joints:
                joints[body[0]] = []
            joints[body[0]].append(body)
            b = body[0]
        elif isinstance(body, tuple) and len(body) == 3:
            if body[0] not in links:
                links[body[0]] = []
            links[body[0]].append(body)
            b = body[0]
        else:
            b = body
        if b not in indices:
            indices.append(b)

    sorted_bodies = []
    for body in indices:
        if body not in existing:
            sorted_bodies.append(body)
        if body in joints:
            bodies = joints[body]
            bodies.sort()
            sorted_bodies.extend(bodies)
        if body in links:
            bodies = links[body]
            bodies.sort()
            sorted_bodies.extend(bodies)
    return sorted_bodies


def get_datetime(seconds=False, year=True):
    from datetime import datetime
    form = "%m%d_%H:%M"
    if seconds:
        form = "%m%d_%H%M%S"
    if year:
        form = "%y" + form
    return datetime.now().strftime(form)

# def remove_all_bodies():
#     for body in get_bodies():
#         remove_body(body)
#     remove_all_debug()


from pybullet_tools.utils import get_link_parent, NULL_ID, get_joint_info, get_dynamics_info, \
    clone_collision_shape, clone_visual_shape, get_local_link_pose, get_joint_positions, \
    collision_shape_from_data, visual_shape_from_data, is_unknown_file


def clone_visual_collision_shapes(body, link, client=None):
    client = get_client(client)
    visual_data = get_visual_data(body, link)
    collision_data = get_collision_data(body, link)
    if not visual_data and not collision_data:
        return [NULL_ID], [NULL_ID]
    collisions = [collision_shape_from_data(c, body, link, client) \
            for c in collision_data \
            if len(c.filename) > 0 and not is_unknown_file(c.filename)]
    visuals = [visual_shape_from_data(v, client) \
            for v in visual_data \
            if len(v.filename) > 0 and not is_unknown_file(v.filename)]
    if len(collisions) > len(visuals):
        visuals.extend([NULL_ID] * (len(collisions) - len(visuals)))
    return visuals, collisions


def clone_body_link(body, link, collision=True, visual=True, client=None):
    ## modified from pybullet_tools.utils.clone_body()
    ## problem is unable to handle multiple collision shapes in one link

    client = get_client(client)  # client is the new client for the body
    v, c = clone_visual_collision_shapes(body, link, client)

    for j in range(len(c)):
        new_from_original = {}
        base_link = get_link_parent(body, link)
        new_from_original[base_link] = NULL_ID

        masses = []
        collision_shapes = []
        visual_shapes = []
        positions = [] # list of local link positions, with respect to parent
        orientations = [] # list of local link orientations, w.r.t. parent
        inertial_positions = [] # list of local inertial frame pos. in link frame
        inertial_orientations = [] # list of local inertial frame orn. in link frame
        parent_indices = []
        joint_types = []
        joint_axes = []

        new_from_original[link] = 0
        joint_info = get_joint_info(body, link)
        dynamics_info = get_dynamics_info(body, link)
        masses.append(dynamics_info.mass)

        # collision_shapes.append(clone_collision_shape(body, link, client) if collision else NULL_ID)
        # visual_shapes.append(clone_visual_shape(body, link, client) if visual else NULL_ID)

        collision_shapes.append(c[j])
        visual_shapes.append(v[j])

        point, quat = get_local_link_pose(body, link)
        positions.append(point)
        orientations.append(quat)
        inertial_positions.append(dynamics_info.local_inertial_pos)
        inertial_orientations.append(dynamics_info.local_inertial_orn)
        parent_indices.append(new_from_original[joint_info.parentIndex] + 1) # TODO: need the increment to work
        joint_types.append(joint_info.jointType)
        joint_axes.append(joint_info.jointAxis)

        base_dynamics_info = get_dynamics_info(body, base_link)
        base_point, base_quat = get_link_pose(body, base_link)

        ## added by Yang
        baseCollisionShapeIndex = clone_collision_shape(body, base_link, client) if collision else NULL_ID
        baseVisualShapeIndex = clone_visual_shape(body, base_link, client) if visual else NULL_ID
        if len(collision_shapes) > 0: baseCollisionShapeIndex = NULL_ID  ## collision_shapes[0]
        if len(visual_shapes) > 0: baseVisualShapeIndex = visual_shapes[0]

        new_body = p.createMultiBody(baseMass=base_dynamics_info.mass,
                                     baseCollisionShapeIndex=baseCollisionShapeIndex,
                                     baseVisualShapeIndex=baseVisualShapeIndex,
                                     basePosition=base_point,
                                     baseOrientation=base_quat,
                                     baseInertialFramePosition=base_dynamics_info.local_inertial_pos,
                                     baseInertialFrameOrientation=base_dynamics_info.local_inertial_orn,
                                     linkMasses=masses,
                                     linkCollisionShapeIndices=collision_shapes,
                                     linkVisualShapeIndices=visual_shapes,
                                     linkPositions=positions,
                                     linkOrientations=orientations,
                                     linkInertialFramePositions=inertial_positions,
                                     linkInertialFrameOrientations=inertial_orientations,
                                     linkParentIndices=parent_indices,
                                     linkJointTypes=joint_types,
                                     linkJointAxis=joint_axes,
                                     physicsClientId=client)

        links = [link]
        for joint, value in zip(range(len(links)), get_joint_positions(body, links)):
            # TODO: check if movable?
            p.resetJointState(new_body, joint, value, targetVelocity=0, physicsClientId=client)
    return new_body


def colorize_link(body, link=0, transparency=0.5):
    if isinstance(body, tuple):
        body, link = body
    body_color = apply_alpha(0.9 * np.ones(3))
    link_color = np.array(body_color) + np.random.normal(0, 1e-2, 4)  # TODO: clip
    link_color = apply_alpha(link_color, alpha=transparency)
    set_color(body, link=link, color=link_color)


def colorize_world(fixed, transparency=0.5):
    # named_colors = get_named_colors(kind='xkcd')
    # colors = [color for name, color in named_colors.items()
    #           if any(color_type in name for color_type in color_types)]  # TODO: convex combination
    colored = []
    for body in fixed:
        joints = get_movable_joints(body)
        if not joints:
            continue
        # dump_body(body)
        # body_color = apply_alpha(WHITE, alpha=0.5)
        # body_color = random.choice(colors)
        body_color = apply_alpha(0.9 * np.ones(3))
        links = get_all_links(body)
        rigid = get_root_links(body)

        # links = set(links) - set(rigid)
        for link in links:
            # print('Body: {} | Link: {} | Joints: {}'.format(body, link, joints))
            # print(get_color(body, link=link))
            # print(get_texture(body, link=link))
            # clear_texture(body, link=link)
            # link_color = body_color
            link_color = np.array(body_color) + np.random.normal(0, 1e-2, 4)  # TODO: clip
            link_color = apply_alpha(link_color, alpha=1.0 if link in rigid else transparency)
            set_color(body, link=link, color=link_color)
            if link not in rigid:
                colored.append((body, link))
    return colored


def draw_base_limits(custom_limits, z=1e-2, **kwargs):
    if isinstance(custom_limits, dict):
        x_min, x_max = custom_limits[0]
        y_min, y_max = custom_limits[1]
        custom_limits = ((x_min, y_min), (x_max, y_max))
    if len(custom_limits[0]) == 3:
        draw_aabb(AABB(custom_limits[0], custom_limits[1]), **kwargs)
    elif len(custom_limits[0]) == 2:
        (x1, y1), (x2, y2) = custom_limits
        p1, p2, p3, p4 = [(x1, y1, z), (x1, y2, z), (x2, y2, z), (x2, y1, z)]
        points = [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]
        [add_line(p1, p2, **kwargs) for p1, p2 in points]


def has_tracik():
    try:
        import tracikpy
    except ImportError:
        return False
    return True

#############################################


def visualize_bconf(bconf):
    samp = create_box(.1, .1, .1, mass=1, color=(1, 0, 1, 1))
    if len(bconf) == 3:
        x, y, _ = bconf
        z = 0.2
    elif len(bconf) == 4:
        x, y, z, _ = bconf
    set_point(samp, (x, y, z))
    return samp


def visualize_point(point):
    z = 0
    if len(point) == 3:
        x, y, z = point
    else:
        x, y = point
    body = create_box(.05, .05, .05, mass=1, color=(1, 0, 0, 1))
    set_pose(body, Pose(point=Point(x, y, z)))
    return body


def clean_preimage(preimage):
    new_preimage = []
    for p in preimage:
        if 'UniqueOptValue' not in str(p):
            new_preimage.append(p)
    return new_preimage


def aabb_larger(one, two):
    if isinstance(one, tuple):
        lx, ly, _ = get_aabb_extent(get_aabb(one[0], one[1]))
    else:
        lx, ly, _ = get_aabb_extent(get_aabb(one))
    smaller = min(lx, ly)
    lx, ly, _ = get_aabb_extent(get_aabb(two))
    larger = max(lx, ly)
    return smaller > larger


def get_class_path(var):
    import inspect
    return inspect.getfile(var.__class__)


def find_closest_match(possible, all_possible=False):
    if len(possible) == 0:
        return None
    counts = {b: len(o) for b, o in possible.items()}
    counts = dict(sorted(counts.items(), key=lambda item: item[1]))
    if all_possible:
        return list(counts.keys())
    return list(counts.keys())[0]


def query_yes_no(question, default="no"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def query_right_left():
    """ return 1 for forward and 0 for backward, only works when run in terminal, not PyCharm """
    import getch

    while True:
        print('Please press `d` or `enter` to execute the next command; `a` to go back to the previous frame.')
        key = getch.getche()
        if ord(key) == 10:  # Enter
            return 1
        elif key == 'd':  # Right arrow
            return 1
        elif key == 'a':  # Left arrow
            return 0
        # elif key in [b'\xe0', b'\x00']:  # thing i do not understand
        #     key = ord(getch.getch())
        #     print('keyboard', key)
        #     if key == 77:  # Right arrow
        #         return 1
        #     elif key == 75:  # Left arrow
        #         return 0


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


def print_action_plan(action_plan, stream_plan, world=None):
    from pddlstream.language.object import Object, UniqueOptValue
    placements = {str(s.output_objects[0]): world.get_debug_name(s.input_objects[1].value)
                  for s in stream_plan if s.name == 'sample-pose'}
    action_plan_str = ''
    for action in action_plan:
        if action.name == 'move_base':
            action_plan_str += f"\n\t_"
            continue
        elems = []
        for elem in action.args:
            if elem.value == 'left':
                continue
            if isinstance(elem.value, int) or \
                    (isinstance(elem.value, tuple) and not isinstance(elem.value, UniqueOptValue)):
                elem = world.get_debug_name(elem.value)
            else:
                elem = str(elem.value) if isinstance(elem, Object) else elem.repr_name
                if '=' in elem:
                    elem = elem.split('=')[0]
                elif '#' in elem and elem in placements:
                    elem = f"{elem}_[{placements[elem]}]_"
            elems.append(elem)
        action_plan_str += f"\n\t{action.name} ( {', '.join(elems)} )"
    return action_plan_str


def get_fine_rainbow_colors(steps=2):
    from lisdf_tools.image_utils import RAINBOW_COLORS as rc
    def lerp(color1, color2, frac):
        return color1 * (1 - frac) + color2 * frac
    colors = []
    for i in range(len(rc) - 1):
        colors.extend([lerp(rc[i], rc[i+1], frac) * 255 for frac in np.linspace(0, 1, steps)])
    return colors


def multiply_quat(quat1, quat2):
    """ multiply two quaternions """
    return multiply((unit_point(), quat1), (unit_point(), quat2))[1]


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
            pose = (nice(pose[0]), quat_from_euler(pose[1]))
            print(f'\tnew pose of {obj.shorter_name}\t{pose}')
            set_pose(obj.body, pose)

    def on_release(key):
        if key == keyboard.Key.esc:
            return False

    print('-' * 10 + f' Enter WASDRF for poses and QE for yaw {exit_note}' + '-' * 10)
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


########################################################################


def has_srl_stream():
    try:
        import srl_stream
    except ImportError:
        print('Unfortunately, you cant use the library unless you are part of NVIDIA Seattle Robotics lab')
        return False
    return True


def has_getch():
    try:
        import getch
    except ImportError:
        print('Please install has_getch in order to use `step_by_step`: ```pip install getch```\n')
        return False
    return True