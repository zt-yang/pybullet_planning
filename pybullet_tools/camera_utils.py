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
    collision_shape_from_data, visual_shape_from_data, is_unknown_file, create_collision_shape
from pybullet_tools.bullet_utils import BASE_JOINTS, get_door_links


def adjust_camera_pose(camera_pose):
    """ used for set_camera_pose2() """
    if len(camera_pose) == 6:
        point = camera_pose[:3]
        euler = camera_pose[3:]
        camera_pose = (point, quat_from_euler(euler))
    (x, y, z), quat = camera_pose
    (r, p, w) = euler_from_quat(quat)
    if x < 6.5:
        x = np.random.normal(7, 0.2)
        # redo = True
    return [(x, y, z + 1), quat_from_euler((r - 0.3, p, w))]


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
    if isinstance(body, tuple) and len(body) == 3:
        body, _, link = body
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


def get_pose2d(robot):
    if isinstance(robot, int):
        return BASE_JOINTS
    point, quat = robot.get_pose()
    x, y, _ = point
    _, _, yaw = euler_from_quat(quat)
    return x, y, yaw


## ----------------------------------------------------------


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
    camera_image = world.camera.get_image(segment=True, segment_links=True)
    rgb = camera_image.rgbPixels[:, :, :3]
    seg = camera_image.segmentationMaskBuffer
    # seg = imgs.segmentationMaskBuffer[:, :, 0].astype('int32')
    unique = get_segmask(seg)
    obj_keys = get_obj_keys_for_segmentation(indices, unique)

    for k, v in indices.items():
        keys = obj_keys[v]
        mobility_id = world.get_mobility_id(body)
        file_name = join(img_dir, f"{mobility_id}_{v}.png")

        save_seg_image_given_obj_keys(rgb, keys, unique, file_name, crop=False)


def process_depth_pixels(pixels):
    """ scale float values 2-5 to int values 225-0 """
    n = (5 - pixels) / (5 - 2) * 225
    return n.astype('uint8')


def visualize_camera_image(image, index=0, img_dir='.', rgb=True, rgbd=False):
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
    for k, name in indices.items():
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
        obj_keys[name] = keys
    return obj_keys


def make_observation_collages(images_by_camera, one_col=False, one_row=False):
    collages = []
    camera_keys = list(images_by_camera.keys())
    duration = len(images_by_camera[camera_keys[0]])
    for t in range(duration):
        images = [lst[t] for lst in list(images_by_camera.values())]
        if one_col:
            collage = np.vstack(images)
        elif one_row:
            collage = np.hstack(images)
        else:
            obs_11, obs_12, obs_21, obs_22 = images
            collage = np.vstack([
                np.hstack([obs_11, obs_12]),
                np.hstack([obs_21, obs_22])
            ])
        collages += [collage]
    return collages
