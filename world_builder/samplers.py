from os.path import join, isdir, abspath
from os import listdir
import numpy as np

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


def get_learned_yaw(category):
    global DATABASE
    if DATABASE is None:
        DATABASE = get_asset_to_poses(title='place', yaw_only=True)

    if category in DATABASE:
        possibilities = DATABASE[category]
        return np.random.choice(possibilities)
    return None