from os.path import join, isdir, abspath
from os import listdir
import numpy as np

from pybullet_tools.utils import quat_from_euler, euler_from_quat

DATABASE = None

DATABASE_DIR = abspath(join(__file__, '..', '..', 'databases'))


def get_asset_to_poses(title='place', yaw_only=True):
    import pandas as pd
    asset_to_pose = {}
    df = pd.read_csv(join(DATABASE_DIR, f'{title}.csv'))
    categories = df['Category'].unique()
    for c in categories:
        new_df = df.loc[df['Category'] == c]
        if yaw_only:
            poses = new_df['Radian']
        else:
            poses = list(zip(new_df['Distance'], new_df['Radian']))
        asset_to_pose[c] = poses
    return asset_to_pose


def get_learned_yaw(category, quat=None):
    return None
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