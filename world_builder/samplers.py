from os.path import join, isdir, abspath
from os import listdir
import numpy as np
import random

from pybullet_tools.utils import quat_from_euler, euler_from_quat, get_aabb_center, get_aabb, get_pose, \
    set_pose, wait_for_user
from pybullet_tools.bullet_utils import xyzyaw_to_pose, set_camera_target_body
from world_builder.utils import get_instances

DATABASE = None

DATABASE_DIR = abspath(join(__file__, '..', '..', 'databases'))


def get_asset_to_poses(title='place', yaw_only=False, full_pose=True):
    import pandas as pd
    asset_to_pose = {}
    df = pd.read_csv(join(DATABASE_DIR, f'{title}.csv'))
    movables = df['Movable'].unique()
    surfaces = df['Surface'].unique()
    # categories = ['Bottle', 'Medicine']
    # categories = {k.lower(): k for k in categories}
    for m in movables:
        for s in surfaces:
            new_df = df.loc[(df['Movable'] == m) & (df['Surface'] == s)]
            if full_pose:
                poses = list(zip(new_df['x'], new_df['y'], new_df['z'], new_df['yaw']))
                poses = [xyzyaw_to_pose(xyzyaw) for xyzyaw in poses]
            elif yaw_only:
                poses = new_df['Radian']
            else:
                poses = list(zip(new_df['Distance'], new_df['Radian']))
            # if '/' in s:
            #     s = s.split('/')[1]
            asset_to_pose[(m, s)] = poses
            asset_to_pose[(m.lower(), s)] = poses

            # key = (m, s)
            # if m in categories:
            #     instances = get_instances(categories[m])
            #     for instance in instances:
            #         asset_to_pose[(instance, s)] = poses
            # else:
            #     asset_to_pose[key] = poses
    return asset_to_pose


def get_learned_yaw(category, quat=None):
    global DATABASE
    if DATABASE is None:
        DATABASE = get_asset_to_poses(title='place', yaw_only=True)

    if category in DATABASE:
        possibilities = DATABASE[category]
        yaw = np.random.choice(possibilities)
        if quat is not None:
            r, p, _ = euler_from_quat(quat)
            quat = quat_from_euler((r, p, yaw))
            return quat
        return yaw
    return None


def get_learned_poses(movable, surface, body, surface_body, num_samples=10, surface_point=None,
                      verbose=True, visualize=False):
    global DATABASE
    if DATABASE is None:
        DATABASE = get_asset_to_poses(title='pickplace', full_pose=True)

    def get_global_pose(pose, nudge=False):
        if surface_point is None:
            return pose
        rel_point, quat = pose
        point = [rel_point[i] + surface_point[i] for i in range(3)]
        if surface == 'box':
            point[0] = get_aabb(surface_body).upper[0] - rel_point[0]
        if nudge:
            delta = np.random.normal(scale=0.01, size=3)
            delta[2] = 0
            point = np.array(point) + delta
        point[2] += 0.01
        return (tuple(point), quat)

    key = (movable.lower(), surface)
    # nudge = surface not in ['100015', '100017', '100021', '100023', '100038', '100693']
    nudge = surface not in get_instances('BraiserBody')
    title = f'         get_learned_poses{key} |'
    if key in DATABASE:
        possibilities = DATABASE[key]
        if len(possibilities) == 0:
            if verbose:
                print(f'{title} has no data in database')
            return []
        choices = random.choices(range(len(possibilities)), k=num_samples)
        choices = [possibilities[i] for i in choices]
        choices = [get_global_pose(choice, nudge=nudge) for choice in choices]
        if verbose:
            print(f'{title} found {len(choices)} saved poses')
        if visualize:
            original_pose = get_pose(body)
            for i in range(len(choices)):
                set_pose(body, choices[i])
                wait_for_user(f'next {i}/{len(choices)}')
                set_camera_target_body(body)
            set_pose(body, original_pose)
        return choices
    if verbose:
        print(f'{title} doesnt exist in database')
    return []