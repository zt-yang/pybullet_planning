from os.path import join, isdir, abspath
from os import listdir
import json
import shutil
from tqdm import tqdm

from pybullet_tools.utils import euler_from_quat
from mamao_tools.data_utils import get_successful_plan, get_indices, get_sink_counter_x, get_lisdf_aabbs, \
    get_from_to

DATABASE_DIR = abspath(join(__file__, '..', '..', 'databases'))


def get_obj_pose(action, indices):
    name = indices[action[2]]
    category = name.replace('#1', '').replace('#2', '')
    pose = action[3]
    pose = eval(pose[pose.index('('):])
    return name, category, (pose[:3], pose[-1])


def save_pose_samples(asset_to_pose, title='pose_samples'):
    import pandas as pd

    lst = []
    for name, surface_to_poses in asset_to_pose.items():
        for surface, poses in surface_to_poses.items():
            for pose in poses:
                lst.append([name, surface] + list(pose))
    df = pd.DataFrame(lst, columns=['Movable', 'Surface', 'x', 'y', 'yaw'])
    df.to_csv(join(DATABASE_DIR, f'{title}.csv'))


def plot_pose_samples(asset_to_pose, title='Pose Samples'):
    """ https://stackoverflow.com/questions/63887605/2d-polar-histogram-with-python """
    import numpy as np
    import matplotlib.pylab as plt

    # define binning
    max_dist = 0.5  ## max(rlst)
    rbins = np.linspace(0, max_dist, 20)
    abins = np.radians(np.linspace(-180, 180, 30))

    fig, axs = plt.subplots(3, 3, figsize=(16, 16))

    for i, (name, poses) in enumerate(asset_to_pose.items()):

        rlst, alst = zip(*poses)
        rlst = max_dist - np.array(rlst)

        # calculate histogram
        hist, _, _ = np.histogram2d(alst, rlst, bins=(abins, rbins), density=1)
        A, R = np.meshgrid(abins, rbins)

        # plot
        ax = plt.subplot(3, 3, i+1, projection="polar")
        ax.set_title(f"{name} ({len(poses)} samples)")

        pc = ax.pcolormesh(A, R, hist.T, cmap="magma_r")
        fig.colorbar(pc)
        ax.grid(True)
    # plt.show()
    plt.title(title)
    plt.tight_layout()

    plt.savefig(join(DATABASE_DIR, f'{title}.png'))


def test_generate_pose_samples():
    data_dir = '/home/yang/Documents/fastamp-data-rss'
    subdirs = [join(data_dir, s) for s in listdir(data_dir) if \
               isdir(join(data_dir, s)) and s.startswith('mm_')]
    asset_to_pose = {}
    for subdir in subdirs:
        run_dirs = [join(subdir, s) for s in listdir(subdir) if isdir(join(subdir, s))]
        run_dirs = sorted(run_dirs, key=lambda x: eval(x.split('/')[-1]))
        for run_dir in tqdm(run_dirs):
            indices = get_indices(run_dir)
            plan = get_successful_plan(run_dir)[0]
            # counter_x = get_sink_counter_x(run_dir)
            aabbs = get_lisdf_aabbs(run_dir)
            placement_plan = []
            for action in plan:
                if action[0].startswith('pick') or action[0].startswith('place'):
                    if action[0] not in asset_to_pose:
                        asset_to_pose[action[0]] = {}
                    name, category, (point, yaw) = get_obj_pose(action, indices)
                    result = get_from_to(name, aabbs, point, run_dir=run_dir)
                    if result is None or len(result) == 3:
                        print(run_dir, name, result)
                        placement_plan.append((action[0], name, None))
                        continue
                    (relation, surface_category, surface_name), (x_upper, y_center) = result
                    if category not in asset_to_pose[action[0]]:
                        asset_to_pose[action[0]][category] = {}
                    if surface_category == 'box':
                        y_center = point[1]
                    if surface_category not in asset_to_pose[action[0]][category]:
                        asset_to_pose[action[0]][category][surface_category] = []
                    asset_to_pose[action[0]][category][surface_category].append((x_upper - point[0], point[1] - y_center, yaw))
                    placement_plan.append((action[0], name, surface_name))
            config_file = join(run_dir, 'planning_config.json')
            planning_config = json.load(open(config_file, 'r'))
            if 'placement_plan' not in planning_config and False:
                new_config_file = join(run_dir, 'planning_config_new.json')
                with open(new_config_file, 'w') as f:
                    planning_config['placement_plan'] = placement_plan
                    json.dump(planning_config, f, indent=3)
                shutil.move(new_config_file, config_file)

    for action in asset_to_pose:
        # plot_pose_samples(asset_to_pose[action], title=action)
        save_pose_samples(asset_to_pose[action], title=action)


def test_plotting(title='place'):
    import pandas as pd

    asset_to_pose = {}
    df = pd.read_csv(join(DATABASE_DIR, f'{title}.csv'))
    categories = df['Category'].unique()
    for c in categories:
        new_df = df.loc[df['Category'] == c]
        poses = list(zip(new_df['Distance'], new_df['Radian']))
        asset_to_pose[c] = poses
    plot_pose_samples(asset_to_pose, title='Reconstructed_' + title)


if __name__ == '__main__':
    test_generate_pose_samples()
    # test_plotting()