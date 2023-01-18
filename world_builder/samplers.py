from os.path import join, isdir, abspath
from os import listdir
import numpy as np
import random

from pybullet_tools.utils import quat_from_euler, euler_from_quat, get_aabb_center
from pybullet_tools.bullet_utils import xyzyaw_to_pose

DATABASE = None

DATABASE_DIR = abspath(join(__file__, '..', '..', 'databases'))


def get_asset_to_poses(title='place', yaw_only=False, full_pose=True):
    import pandas as pd
    asset_to_pose = {}
    df = pd.read_csv(join(DATABASE_DIR, f'{title}.csv'))
    movables = df['Movable'].unique()
    surfaces = df['Surface'].unique()
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
            if '/' in s:
                s = s.split('/')[1]
            key = (m, s)
            asset_to_pose[key] = poses
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


def get_learned_poses(movable, surface, num_samples=10, surface_aabb=None, verbose=True):
    global DATABASE
    if DATABASE is None:
        DATABASE = get_asset_to_poses(title='pickplace', full_pose=True)

    def get_global_pose(pose):
        if surface_aabb is None:
            return pose
        x_upper = surface_aabb.upper[0]
        _, y_center, z_center = get_aabb_center(surface_aabb)
        (x, y, z), quat = pose
        return ((x_upper - x, y + y_center, z + z_center), quat)

    key = (movable.lower(), surface)
    title = f'         get_learned_poses{key} |'
    if key in DATABASE:
        possibilities = DATABASE[key]
        if len(possibilities) == 0:
            if verbose:
                print(f'{title} has no data in database')
            return []
        choices = random.choices(range(len(possibilities)), k=num_samples)
        choices = [possibilities[i] for i in choices]
        choices = [get_global_pose(choice) for choice in choices]
        if verbose:
            print(f'{title} found {len(choices)} saved poses for {key}')
        return choices
    if verbose:
        print(f'{title} doesnt exist in database')
    return []