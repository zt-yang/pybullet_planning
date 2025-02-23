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
    collision_shape_from_data, visual_shape_from_data, is_unknown_file, create_collision_shape, \
    aabb_from_extent_center


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


def get_merged_aabb(aabbs):
    x_min, y_min, z_min = np.inf, np.inf, np.inf
    x_max, y_max, z_max = -np.inf, -np.inf, -np.inf
    for aabb in aabbs:
        x_min = min(x_min, aabb.lower[0])
        y_min = min(y_min, aabb.lower[1])
        z_min = min(z_min, aabb.lower[2])
        x_max = max(x_max, aabb.upper[0])
        y_max = max(y_max, aabb.upper[1])
        z_max = max(z_max, aabb.upper[2])
    return AABB(lower=[x_min, y_min, z_min], upper=[x_max, y_max, z_max])


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
    elif is_tuple(tuple_of_tuples[0]):

        ## pose = (point, quat) -> (point, euler)
        if len(tuple_of_tuples[0]) == 3 and len(tuple_of_tuples[1]) == 4:
            second_tuple = tuple_of_tuples[1]
            if keep_quat:
                one_tuple = False
            else:
                second_tuple = euler_from_quat(second_tuple)
            if one_tuple:
                one_list = list(tuple_of_tuples[0]) + list(second_tuple)
                return nice(tuple(one_list), round_to)
            return nice(tuple_of_tuples[0], round_to), nice(second_tuple, round_to)
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


def get_nice_pose(body):
    return nice(get_pose(body))


def get_nice_joint_positions(body, joints):
    return nice(get_joint_positions(body, joints))


def tupify_arr(arr):
    if isinstance(arr, np.ndarray) or isinstance(arr, list):
        arr = tuple(arr)
    return arr


def tupify(arr_of_arrs):
    if isinstance(arr_of_arrs, tuple):
        result = tuple([tupify_arr(arr) for arr in arr_of_arrs])
    else:
        result = tupify_arr(arr_of_arrs)
    return result


#######################################################

def get_root_links(body):
    [fixed_links] = get_rigid_clusters(body, links=[ROOT_LINK])
    return fixed_links


def articulated_collisions(obj, obstacles, verbose=False, world=None, **kwargs): # TODO: articulated_collision?
    # TODO: cache & compare aabbs
    for obstacle in obstacles:
        # dump_body(obstacle)
        # joints = get_movable_joints(obstacle)
        root_links = get_root_links(obstacle)
        if link_pairs_collision(body1=obstacle, links1=root_links, body2=obj, **kwargs):
            if verbose:
                to_print = f'\t\tarticulated_collisions | obj = {obj}\tobstacle = {obstacle}'
                if world is not None:
                    name = world.get_name_from_body(obstacle)
                    to_print += f' ({name})'
                print(to_print)

            if hasattr(obj, 'log_collisions'):
                obj.log_collisions(obstacle, source='collided.articulated_collisions')

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
    from pybullet_tools.logging_utils import TXT_FILE
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


def collided(obj, obstacles=[], world=None, articulated=False, verbose=False, tag='',
             visualize=False, min_num_pts=3, use_aabb=True, ignored_pairs=[],
             log_collisions=True, **kwargs):

    if world is None and hasattr(obj, 'world'):
        world = obj.world
    obj_print = world.get_name(obj) if world is not None else obj
    prefix = f'\t\tbullet_utils.collided({obj_print}) '
    if len(tag) > 0:
        prefix += f'( {tag} )'
    # verbose = True

    ## first get answer
    if articulated:
        body = obj if isinstance(obj, int) else obj.body
        obstacles_here = [o for o in obstacles if (o, body) not in ignored_pairs]
        result = articulated_collisions(body, obstacles_here, use_aabb=use_aabb, verbose=verbose,
                                        world=world, **kwargs)
        # if result:
        #     if verbose:
        #         print(prefix, '| articulated, obstacles =', obstacles)
        return result
    # else:
    #     result = any(pairwise_collision(obj, b, use_aabb=use_aabb, **kwargs) for b in obstacles)
    # if not verbose:
    #     return result

    result = False
    ## first find the bodies that collides with obj
    bodies = []
    verbose_bodies = []
    to_print = ''
    for b in obstacles:
        if pairwise_collision(obj, b) and (obj, b) not in ignored_pairs:
            if world is None:
                print('bullet_utils.collided | world is None')
                import traceback
                print(traceback.print_exc())

            b_print = world.get_name(b) if world is not None else b
            if verbose:
                # if b_print == 'floor1':
                #     print(obstacles)
                to_print += f'{prefix} collides with {b_print}'
            result = True

            if log_collisions:
                ## the robot keeps track of objects collided
                if hasattr(obj, 'log_collisions'):
                    obj.log_collisions(b, source='collided.pairwise_collision(robot)')
                elif world is not None:
                    log_kwargs = dict(source=f'collided.pairwise_collision({tag})')
                    if b not in bodies:
                        log_kwargs['verbose'] = False
                    world.robot.log_collisions(b, robot_body=obj, **log_kwargs)

            if b not in bodies:
                bodies.append(b)

    ## then find the exact links
    body_links = {}
    total = 0
    for b in bodies:
        key = world.get_debug_name(b) if (world is not None) else b
        d = get_links_collided(obj, b, visualize=False)
        total += sum(list(d.values()))
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


def get_links_collided(body, obstacle, names_as_keys=True, visualize=False):
    d = {}
    for l in get_links(obstacle):
        pts = get_closest_points(obstacle, body, link1=l, link2=None)
        if len(pts) > 0:
            if names_as_keys:
                l = get_link_name(obstacle, l)
            d[l] = len(pts)

            if visualize:  ## visualize collision points for debugging
                points = []
                for point in pts:
                    points.append(visualize_point(point.positionOnA))
                print(f'visualized {len(pts)} collision points')
                for point in points:
                    remove_body(point)
    d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
    return d


#######################################################


ROTATIONAL_MATRICES = {}


def get_rotation_matrix(body, verbose=True):
    import untangle
    r = unit_pose()
    if is_mesh_entity(body) or is_box_entity(body):
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
    """ return aabb when body pose is set to unit_pose() """
    body_pose = get_model_pose(body, link=link, verbose=verbose)
    vertices = get_model_points(body, link=link)
    if link is None:  link = -1
    data = get_collision_data(body, link)
    if len(data) == 0 or data[0].geometry_type == p.GEOM_MESH:
        aabb = aabb_from_points(vertices)
    else: ## if data.geometry_typep == p.GEOM_BOX:
        aabb = aabb_from_extent_center(get_aabb_extent(get_aabb(body)))  ## get_aabb(body)
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

#######################################################

# def summarize_links(body):
#     joints = get_joints(body)
#     for joint in joints:
#         check_joint_state(body, joint)


def get_point_distance(p1, p2):
    if isinstance(p1, tuple): p1 = np.asarray(p1)
    if isinstance(p2, tuple): p2 = np.asarray(p2)
    return np.linalg.norm(p1 - p2)


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
    return pstn


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


def get_joint_range(body, joint):
    lower, upper = get_joint_limits(body, joint)
    return upper - lower


def is_joint_open(body, joint=-1, threshold=0.25, is_closed=False, verbose=False):
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


def create_collision_box(w, l, h, mass=STATIC_MASS, pose=unit_pose()):
    geometry = get_box_geometry(w, l, h)
    collision_id = create_collision_shape(geometry, pose=pose)
    visual_id = NULL_ID
    return create_body(collision_id, visual_id, mass=mass)


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
        print('Please press `d` or `enter` to execute the next command; `a` to go back to the previous frame (only works in shell).')
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


def is_tuple(elems):
    return isinstance(elems, list) or isinstance(elems, tuple) or isinstance(elems, np.ndarray)


########################################################################


def has_srl_stream():
    return True
    # import ipdb; ipdb.set_trace()
    try:
        import srl_stream
    except ImportError:
        print('Unfortunately, you cant use the library unless you are part of NVIDIA Seattle Robotics lab')
        return False
    return True


def running_in_pycharm():
    return "PYCHARM_HOSTED" in os.environ
