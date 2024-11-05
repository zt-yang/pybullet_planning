import random
from os.path import isfile, dirname, abspath, join, isdir
import sys

import numpy as np
import math
import pybullet as p

import os
import json

from pybullet_tools.logging_utils import dump_json
from pybullet_tools.utils import unit_pose, get_collision_data, get_links, multiply, invert, \
    aabb_contains_aabb, get_pose, get_aabb, GREEN, AABB, remove_body, set_renderer, draw_aabb, \
    pose_from_tform, wait_for_user, Euler, PI, unit_point, LockRenderer, HideOutput, load_model, \
    get_client, JOINT_TYPES, get_joint_type, get_link_pose, get_closest_points, \
    get_link_subtree, quat_from_euler, euler_from_quat, create_box, set_pose, Pose, \
    YELLOW, add_line, draw_point, RED, remove_handles, apply_affine, vertices_from_rigid, \
    aabb_from_points, get_aabb_extent, get_aabb_center, get_aabb_edges, has_gui, get_rigid_clusters, \
    link_pairs_collision, wait_unlocked, apply_alpha, set_color, BASE_LINK as ROOT_LINK, \
    BROWN, BLUE, WHITE, TAN, GREY, YELLOW, GREEN, BLACK, RED, quat_from_pose, Point, tform_point, angle_between
from pybullet_tools.bullet_utils import nice, collided, equal, minus, add, get_color_by_index, \
    get_datetime, is_box_entity, draw_fitted_box, get_model_pose, colors
from pybullet_tools.camera_utils import set_camera_target_body
from pybullet_tools.pose_utils import draw_colored_pose


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


def find_grasp_in_db(db_file, instance_name, length_variants=False, scale=None,
                     use_all_grasps=False, verbose=False, world=None):
    """ find saved json files, prioritize databases/ subdir """
    db = json.load(open(db_file, 'r')) if isfile(db_file) else {}

    def rewrite_grasps(data):
        ## the newest format has poses written as (x, y, z, roll, pitch, row)
        if len(data[0]) == 6:
            found = [(tuple(e[:3]), quat_from_euler(e[3:])) for e in data]
        elif len(data[0][1]) == 3:
            found = [(tuple(e[0]), quat_from_euler(e[1])) for e in data]
        elif len(data[0][1]) == 4:
            found = [(tuple(e[0]), tuple(e[1])) for e in data]
        if verbose:
            print(f'    bullet_utils.find_grasp_in_db returned {len(found)}'
                  f' grasps for {instance_name} | scale = {scale}')
        return found

    found = None
    if instance_name in db:
        all_data = db[instance_name]
        grasp_key = 'grasps_all' if (use_all_grasps and 'grasps_all' in all_data) else 'grasps'
        if length_variants:
            grasp_key = 'grasps_l'

        ## the newest format has attr including 'name', 'grasps', 'grasps_length_variants'
        if '::' not in instance_name or ('scale' in all_data and scale == all_data['scale']) \
                and grasp_key in all_data:
            data = all_data[grasp_key]
            if len(data) > 0:
                found = rewrite_grasps(data)
                ## scale the grasps for object grasps but not handle grasps
                if scale is not None and 'scale' in all_data and scale != all_data['scale']:
                    found = [(tuple(scale/all_data['scale'] * np.array(p)), q) for p, q in found]
                    # new_found = []
                    # for p, q in found:
                    #     p = np.array(p)
                    #     p[:2] *= scale / all_data['scale']
                    #     p[2] *= scale * 1.4 / all_data['scale']
                    #     new_found.append((tuple(p), q))
                    # found = new_found
        elif 'other_scales' in all_data and str(scale) in all_data['other_scales']:
            data = all_data['other_scales'][str(scale)]
            found = rewrite_grasps(data)

    return found, db


def add_grasp_in_db(db, db_file, instance_name, grasps, name=None, length_variants=False, scale=None):
    if instance_name is None: return

    key = 'grasps' if not length_variants else 'grasps_l'
    add_grasps = []
    for g in grasps:
        add_grasps.append(list(nice(g, 4)))
    if len(add_grasps) == 0:
        return

    ## -------- save to json
    if name is None:
        name = 'None'

    if instance_name in db:
        if 'other_scales' not in db[instance_name]:
            db[instance_name]['other_scales'] = {}
        db[instance_name]['other_scales'][str(scale)] = add_grasps
    else:
        db[instance_name] = {
            'name': name,
            key: add_grasps,
            'datetime': get_datetime(),
            'scale': scale,
        }
    keys = {k: v['datetime'] for k, v in db.items()}
    keys = sorted(keys.items(), key=lambda x: x[1])
    db = {k: db[k] for k, v in keys}
    if isfile(db_file): os.remove(db_file)
    dump_json(db, db_file, sort_dicts=False)
    print(f'\n    bullet_utils.add_grasp_in_db saved {len(grasps)} grasps for {instance_name}\n')


def get_loaded_scale(body):
    data = get_collision_data(body, 0)
    scale = 1
    if len(data) > 0:
        scale = data[0].dimensions[0]
    else:
        print('get_scale | no collision data for body', body)
        # wait_unlocked()
    return scale


def get_grasp_db_file(robot, nudge=False, nudge_back=False):
    if isinstance(robot, str):
        robot_name = {'pr2': 'PR2Robot', 'feg': 'FEGripper'}[robot]
    else:
        robot_name = robot.__class__.__name__
    key = 'hand'
    if nudge:
        if nudge_back:
            key = 'back'
        else:
            key = 'nudge'
    db_file_name = f'{key}_grasps_{robot_name}.json'
    db_file = abspath(join(dirname(__file__), '..', 'databases', db_file_name))
    return db_file


def check_grasp_link(world, body, link):
    """ some movables, like pot and pot lid, have specific links for grasp """
    from world_builder.world import World
    from world_builder.world_utils import get_grasp_link
    using_grasp_link = False
    if isinstance(world, World) and link is None:
        path = world.BODY_TO_OBJECT[body].path
        grasp_link = get_grasp_link(path, body)
        if grasp_link is not None:
            link = grasp_link
            using_grasp_link = True
    return using_grasp_link, link


def enumerate_translation_matrices(x=0.02, negations=False):
    translations = [(x, 0, 0), (0, x, 0), (0, 0, x)]
    if negations:
        translations += [(-x, 0, 0), (0, -x, 0), (0, 0, -x)]
    return [(0, 0, 0)] + translations


def enumerate_rotational_matrices(return_list=False):
    P = math.pi
    rots = {
        (1, 0, 0): [(P / 2, 0, -P / 2), (P / 2, P, -P / 2), (P / 2, -P / 2, -P / 2), (P / 2, P / 2, -P / 2)],
        (-1, 0, 0): [(P / 2, 0, P / 2), (P / 2, P, P / 2), (P / 2, -P / 2, P / 2), (P / 2, P / 2, P / 2),
                     (-P, -P / 2, 0), (-P, -P / 2, -P)],
        (0, 1, 0): [(0, P / 2, -P / 2), (0, -P / 2, P / 2), (P / 2, P, 0), (P / 2, 0, 0)],
        (0, -1, 0): [(0, P / 2, P / 2), (0, -P / 2, -P / 2), (-P / 2, P, 0), (-P / 2, 0, 0)],
        (0, 0, 1): [(P, 0, P / 2), (P, 0, -P / 2), (P, 0, 0), (P, 0, P)],
        (0, 0, -1): [(0, 0, -P / 2), (0, 0, P / 2), (0, 0, 0), (0, 0, P)],
    }
    if return_list:
        lst = []
        for v in rots.values():
            lst += v
        return lst
    return rots


def visualize_found_grasps(found, robot, body, link, body_pose, retain_all=False, verbose=True):
    bodies = []
    bb = body if link is None else None
    for i, g in enumerate(found):
        color, color_name = get_color_by_index(i)
        print(f'\t{nice(g)}\t{color_name}')
        bodies.append(
            robot.visualize_grasp(body_pose, g, body=bb, verbose=verbose, new_gripper=retain_all, color=color)
        )
    # set_renderer(True)
    if retain_all:
        set_camera_target_body(body, dx=0.5, dy=0.5, dz=1)
        wait_unlocked()
    for b in bodies:
        remove_body(b)


def make_nudge_grasps_from_handle_grasps(world, found_hand_grasps, body, body_pose, nudge_back=False,
                                         verbose=False, interactive=False, debug=False):
    if nudge_back:
        x = ((-0.13, -0.25, 0.13), quat_from_euler(Euler(roll=-PI/2)))
    else:
        x = ((0, 0, 0.32), quat_from_euler(Euler(roll=PI)))

    if debug or interactive:
        set_renderer(True)
        g = found_hand_grasps[0]
        gripper = world.robot.visualize_grasp(body_pose, multiply(g, x), body=None, verbose=verbose, width=0)
        collided(gripper, [body])

        if interactive:
            from world_builder.entities import Object
            world.add_object(Object(gripper), name='gripper')

    return [multiply(g, x) for g in found_hand_grasps]


def get_hand_grasps(world, body, link=None, grasp_length=0.1, visualize=False,
                    handle_filter=False, length_variants=False, use_all_grasps=True,
                    retain_all=False, verbose=True, collisions=False, debug_del=False,
                    test_offset=False, skip_grasp_index_until=None, nudge=False, nudge_back=False):
    body_name = (body, None, link) if link is not None else body
    title = f'grasp_utils.get_hand_grasps({body_name}) | '
    dist = grasp_length
    robot = world.robot
    scale = 1 if is_box_entity(body) else get_loaded_scale(body)
    # print("get hand grasps scale", scale)
    # using_grasp_link, link = check_grasp_link(world, body, link)

    body_pose = get_model_pose(body, link=link, verbose=verbose)
    aabb, handles = draw_fitted_box(body, link=link, verbose=verbose, draw_box=False, draw_centroid=False)

    ## grasp the whole body
    if link is None:
        r = Pose(euler=Euler(math.pi / 2, 0, -math.pi / 2))
        body_pose = multiply(body_pose, invert(r))

    # ## grasp handle links
    # else:
    #     tool_from_hand = robot.get_tool_from_hand(body_name)
    #     body_pose = multiply(body_pose, invert(tool_from_hand))

    instance_name = world.get_instance_name(body_name)
    name_in_db = None if instance_name is None else world.get_name(body_name, use_default_link_name=True)
    if instance_name is not None:
        grasp_db_file = get_grasp_db_file(robot, nudge=nudge, nudge_back=nudge_back)
        found, db = find_grasp_in_db(grasp_db_file, instance_name, verbose=verbose, scale=scale,
                                     length_variants=length_variants, use_all_grasps=use_all_grasps, world=world)

        ## TODO: hack to hand adjust the found hand grasps to make nudge grasps
        if found is None and nudge:
            found_hand_grasps, _ = find_grasp_in_db(get_grasp_db_file(robot), instance_name, length_variants=length_variants,
                                                    scale=scale, use_all_grasps=use_all_grasps, verbose=verbose)
            if found_hand_grasps is not None:
                found = make_nudge_grasps_from_handle_grasps(world, found_hand_grasps, body, body_pose, nudge_back=nudge_back)
                # add_grasp_in_db(db, grasp_db_file, instance_name, found, name=name_in_db, scale=scale)

        if found is not None:
            if verbose:
                print(f'get_hand_grasps({instance_name}) | found {len(found)} grasp poses')
            if visualize:
                visualize_found_grasps(found, robot, body, link, body_pose, retain_all, verbose)
            remove_handles(handles)
            return found

    # obstacles = world.fixed
    # if body not in obstacles:
    #     obstacles += [body]
    obstacles = [body]

    ## only one in each direction
    def check_new(aabbs, aabb):
        yzs = [AABB(m.lower[1:], m.upper[1:]) for m in aabbs]
        if AABB(aabb.lower[1:], aabb.upper[1:]) in yzs: return False
        xys = [AABB(m.lower[:-1], m.upper[:-1]) for m in aabbs]
        if AABB(aabb.lower[:-1], aabb.upper[:-1]) in xys: return False
        xzs = [AABB(m.lower[::2], m.upper[::2]) for m in aabbs]
        if AABB(aabb.lower[::2], aabb.upper[::2]) in xzs: return False
        if debug_del and len(aabbs) > 0:
            print('aabbs\n  ', '\n  '.join([str(nice(n)) for n in aabbs]))
            print('\naabb\n  ', nice(aabb))
            print('check_new(aabbs, aabb)')
        return True

    def check_grasp(f, r):
        grasps = []
        grasp = multiply(Pose(point=f), Pose(euler=r))

        bb = body if link is None else (body, link)
        result, aabb, gripper = check_cfree_gripper(grasp, world, body_pose, obstacles, verbose=False, body=bb,
                                                    visualize=visualize, retain_all=retain_all, collisions=collisions,
                                                    test_offset=test_offset)
        if result:  ##  and check_new(aabbs, aabb):
            grasps += [grasp]
            # aabbs += [aabb]
            # these += [grasp]

            # # debug
            # if verbose:
            #     set_renderer(True)
            #     return grasps

            ## slide along the longest dimension
            if length_variants and on_longest:
                dl_max = max_value / 3
                dl_candidates = [random.uniform(-dl_max, dl_max) for k in range(3)]
                dl_candidates = [dl_max, -dl_max]
                for dl in dl_candidates:
                    grasp_dl = robot.mod_grasp_along_handle(grasp, dl)
                    result, aabb, gripper_dl = check_cfree_gripper(grasp, world, body_pose, obstacles, body=bb,
                                                                   verbose=verbose, collisions=collisions,
                                                                   visualize=visualize, retain_all=retain_all)
                    if result:  ## and check_new(aabbs, aabb):
                        grasps += [grasp_dl]
                        # aabbs += [aabb]
                        # these += [grasp_dl]
                    elif gripper_dl is not None:
                        remove_body(gripper_dl)

        return grasps

    ## get the points in hand frame to be transformed to the origin of object frame in different directions
    c = get_aabb_center(aabb)
    w, l, h = dimensions = get_aabb_extent(aabb)
    faces = [(w/2+dist, 0, 0), (0, l/2+dist, 0), (0, 0, h/2+dist)]
    faces += [minus(0, f) for f in faces]

    ## for finding the longest dimension
    max_value = max(dimensions)
    filter = [int(x == max_value) for x in dimensions]

    rots = enumerate_rotational_matrices()
    # set_renderer(visualize)
    grasps = []
    # aabbs = []
    index = 0
    for f in faces:
        p = np.array(f)
        p = p / np.linalg.norm(p)

        ## only attempt the bigger surfaces
        on_longest = sum([filter[i]*p[i] for i in range(3)]) == 0
        # if handle_filter and not on_longest:
        #     continue

        ## reduce num of grasps
        # these = []  ## debug

        ang = tuple(p)
        f = add(f, c)
        for r in rots[ang]:

            ## for debugging different grasps
            index += 1
            if skip_grasp_index_until is not None:
                if index - 1 < skip_grasp_index_until:
                    continue
                print(f'grasp_index\t{index}')

            grasps.extend(check_grasp(f, r))
            if test_offset:
                return grasps

        # ## just to look at the orientation
        # if debug_del:
        #     set_camera_target_body(body, dx=0.3, dy=0.3, dz=0.3)
        #     print(f'bullet_utils.get_hand_grasps | rots[{ang}]', len(rots[ang]), [nice(n) for n in rots[ang]])
        #     print(f'bullet_utils.get_hand_grasps -> ({len(these)})', [nice(n[1]) for n in these])
        #     print('bullet_utils.get_hand_grasps')

    # set_renderer(True)
    if verbose:
        print(f"{title} ({len(grasps)}) {[nice(g) for g in grasps]}")
        if len(grasps) == 0:
            print(title, 'no grasps found')

    ## lastly store the newly sampled grasps
    if instance_name is not None:
        add_grasp_in_db(db, grasp_db_file, instance_name, grasps, name=name_in_db,
                        length_variants=length_variants, scale=scale)
    remove_handles(handles)
    robot.hide_cloned_grippers()
    # if len(grasps) > num_samples:
    #     random.shuffle(grasps)
    #     return grasps[:num_samples]
    return grasps  ##[:1]


def check_cfree_gripper(grasp, world, body_pose, obstacles, verbose=False, visualize=False, body=None,
                        min_num_pts=40, retain_all=False, collisions=False, test_offset=False, **kwargs):
    robot = world.robot

    if verbose:
        print(f'\nbullet.check_cfree_gripper | '
              f'\trobot.visualize_grasp({nice(body_pose)}, ({nice(grasp)}):'
              f'\t{nice(robot.tool_from_hand)}\tkwargs = {kwargs}')

    gripper_grasp = robot.visualize_grasp(body_pose, grasp, verbose=verbose, body=body,
                                          new_gripper=test_offset, **kwargs)
    if test_offset:
        remove_body(gripper_grasp)
        return True, None, gripper_grasp

    if gripper_grasp is None:
        return False, None, None

    if visualize: ## and not firstly: ## somtimes cameras blocked by robot, need to change dx, dy
        ## also helps slow down visualization of the sampling the testing process
        set_camera_target_body(gripper_grasp, dx=0.3, dy=0.5, dz=0.2) ## oven
        # set_camera_target_body(gripper_grasp, dx=1, dy=0.5, dz=0.8) ## faucet
        # set_camera_target_body(gripper_grasp, dx=0.5, dy=-0.5, dz=0.5)  ## fridge shelf
        # set_camera_target_body(gripper_grasp, dx=0.05, dy=-0.05, dz=0.5)  ## above dishwasher
        # set_camera_target_body(gripper_grasp, dx=0.05, dy=-0.05, dz=0.15)  ## inside dishwasher
        # set_camera_target_body(gripper_grasp, dx=0.15, dy=-0.15, dz=0)  ## bottles on floor
        # set_camera_target_body(gripper_grasp, dx=2, dy=1, dz=1)  ## minifridges on the floor
        # set_camera_target_body(gripper_grasp, dx=0.5, dy=0.5, dz=1)  ## fork inside indigo drawer top

    ## criteria 1: when gripper isn't closed, it shouldn't collide
    firstly = collided(gripper_grasp, obstacles, min_num_pts=0,
                       world=world, verbose=False, tag='firstly')
    finger_link = robot.cloned_finger_link

    ## criteria 2: when gripper is closed, it should collide with object
    secondly = False
    if not firstly or not collisions:
        robot.close_cloned_gripper(gripper_grasp)
        secondly = collided(gripper_grasp, obstacles, min_num_pts=0, world=world, articulated=False,
                            verbose=False, tag='secondly')

        ## boxes don't contain vertices for collision checking
        if body is not None and isinstance(body, int) and len(get_collision_data(body)) > 0 \
                and get_collision_data(body)[0].geometry_type == p.GEOM_BOX:
            secondly = secondly or aabb_contains_aabb(get_aabb(gripper_grasp, finger_link), get_aabb(body))

    ## criteria 3: the gripper shouldn't be pointing upwards, heuristically
    aabb = get_aabb(gripper_grasp)
    upwards = robot.check_if_pointing_upwards(gripper_grasp)

    ## combining all criteria
    result = not firstly and secondly and not upwards

    if not result or not retain_all:
        # remove_body(gripper_grasp)  ## TODO
        gripper_grasp = None
        # # weiyu: also remove the gripper from the robot
        robot.remove_grippers()

    elif retain_all:
        robot.open_cloned_gripper(gripper_grasp)

    return result, aabb, gripper_grasp


def is_top_grasp(robot, arm, body, grasp, pose=None, top_grasp_tolerance=PI/4,
                 side_grasp_tolerance=None):  # None | PI/4 | INF
    result = True
    if top_grasp_tolerance is None and side_grasp_tolerance is None:
        return result
    if not isinstance(grasp, tuple):
        grasp = grasp.value
    if pose is None:
        pose = get_pose(body)
    grasp_pose = robot.get_grasp_pose(pose, grasp, arm, body=body)
    grasp_orientation = (Point(), quat_from_pose(grasp_pose))
    grasp_direction = tform_point(grasp_orientation, robot.grasp_direction)
    if top_grasp_tolerance is not None:
        result = result and angle_between(grasp_direction, Point(z=-1)) <= top_grasp_tolerance
    if side_grasp_tolerance is not None:
        tolerance = max(PI/2 - side_grasp_tolerance, 0)
        result = result and angle_between(grasp_direction, Point(z=-1)) >= tolerance
    return result

## --------------------------------------------------


def sample_from_pickled_grasps(grasps, world, body, pose=None, offset=None, k=15, debug=False,
                               verbose=False, visualize=False, retain_all=False):
    obstacles = [body]
    all_handles = []
    all_grasps = []
    filtered_grasps = []
    while len(all_grasps) < k:
        some_grasps = random.choices(grasps, k=k)

        ## for debugging grasp poses
        if pose is not None and debug:
            some_grasps = some_grasps[:len(colors)]
            for i, grasp in enumerate(some_grasps):
                g = multiply(pose, pose_from_tform(grasp))
                all_handles += draw_colored_pose(g, length=0.1, color=colors[i])

        ## the tool_link in the saved grasps are different from that here
        if offset is None:
            offset = Pose(point=(-0.01, 0, -0.03), euler=Euler(0, 0, -math.pi/2))
            offset = Pose(point=(0, -0.01, -0.03), euler=Euler(0, 0, 0))

        some_grasps = [multiply(pose_from_tform(g), offset) for g in some_grasps]
        if pose is not None and debug:
            break

        ## filter grasps to check if closing gripper contacts the object
        for grasp in some_grasps:
            result, aabb, gripper = check_cfree_gripper(
                grasp, world, pose, obstacles, verbose=verbose, body=body,
                visualize=visualize, retain_all=retain_all,
                collisions=True, test_offset=False)
            if result:
                filtered_grasps.append(grasp)
        k = k - len(filtered_grasps)
    return filtered_grasps, all_handles


## --------------------------------------------------


def test_transformations_template(rotations, translations, funk, title, skip_until=None):
    total = len(rotations) * len(translations)
    print(f'\n{title} ({total})')

    results = None
    for k1, r in enumerate(rotations):
        for k2, t in enumerate(translations):
            k = k1 * len(translations) + k2
            if skip_until is not None and k < skip_until:
                continue
            wait_for_user('\t\t next?')
            print(f'\n[{k}/{total}]',
                  f'\t{k1}/{len(rotations)} r = {nice(r)}',
                  f'\t{k2}/{len(translations)} t = {nice(t)}')
            results = funk(t, r)
    wait_for_user('finished')
    return results


def add_to_rc2oc(robot, group, a, o, mapping):
    body, joint = o
    conf = robot.get_all_base_conf()
    if group == 'base-torso':
        # group = (group, conf)
        conf = (a, robot.get_one_arm_conf(a))
    if body not in robot.ROBOT_CONF_TO_OBJECT_CONF:
        robot.ROBOT_CONF_TO_OBJECT_CONF[body] = {}
    if joint not in robot.ROBOT_CONF_TO_OBJECT_CONF[body]:
        robot.ROBOT_CONF_TO_OBJECT_CONF[body][joint] = {}
    if conf not in robot.ROBOT_CONF_TO_OBJECT_CONF[body][joint]:
        robot.ROBOT_CONF_TO_OBJECT_CONF[body][joint][conf] = {}
    robot.ROBOT_CONF_TO_OBJECT_CONF[body][joint][conf][group] = mapping


def needs_special_grasp(body, world):
    return world.get_category(body) in ['braiserbody']
