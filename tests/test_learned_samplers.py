from os.path import join, isdir, abspath, isfile, basename
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


def save_pose_samples(pose_dict, title='pose_samples'):
    import pandas as pd

    lst = []
    for action, asset_to_pose in pose_dict.items():
        for name, surface_to_poses in asset_to_pose.items():
            for surface, poses in surface_to_poses.items():
                for pose in poses:
                    lst.append([action, name, surface] + list(pose))
    df = pd.DataFrame(lst, columns=['Action', 'Movable', 'Surface', 'x', 'y', 'z', 'yaw', 'dx', 'run_name'])
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
    subdirs = [join(data_dir, s) for s in listdir(data_dir) if isdir(join(data_dir, s)) \
               and (s.startswith('mm_') or s.startswith('tt_') or s.startswith('_kc') or s == '_gmm')
               # and (s.startswith('mm_storage_long'))  ## for debugging
               ]
    subdirs.sort()
    asset_to_pose = {}

    found_count = 0
    misplaced_count = 0
    deleted_count = 0
    missed_count = 0
    for subdir in subdirs:
        run_dirs = [join(subdir, s) for s in listdir(subdir) if isdir(join(subdir, s))]
        run_dirs = sorted(run_dirs, key=lambda x: eval(x.split('/')[-1]))
        for run_dir in tqdm(run_dirs, desc=basename(subdir)):
            indices = get_indices(run_dir)
            plan = get_successful_plan(run_dir)[0]
            counter_x = get_sink_counter_x(run_dir)
            aabbs = get_lisdf_aabbs(run_dir)
            placement_plan = []
            for action in plan:
                if action[0].startswith('pick') or action[0].startswith('place'):
                    ## that of movable object
                    name, category, (point, yaw) = get_obj_pose(action, indices)
                    if point[0] > counter_x:
                        misplaced_count += 1
                        if 'pick' in action[0] and ('/mm_' in run_dir or '/tt_' in run_dir):
                            deleted_count += 1
                            # print(run_dir, name, point[0], round(counter_x, 3), action[0])
                            shutil.rmtree(run_dir)
                            break
                        continue

                    verbose = False
                    # if 'mm_storage_long/0' in run_dir and name == 'veggiepotato':
                    #     verbose = True
                    result = get_from_to(name, aabbs, point, run_dir=run_dir, verbose=verbose)
                    if verbose:
                        verbose = True

                    if result is None or result[0] is None:
                        # print(run_dir, name, result)
                        placement_plan.append((action[0], name, None))
                        missed_count += 1
                        continue
                    found_count += 1

                    (relation, surface_category, surface_name), x_upper, surface_point = result
                    placement_plan.append((action[0], name, surface_name))

                    dx = x_upper - point[0]
                    run_name = run_dir.replace(data_dir, '')
                    pp = [point[i] - surface_point[i] for i in range(3)] + [yaw, dx, run_name]
                    if surface_category == 'box':
                        pp[0] = pp[-2]

                    if action[0] not in asset_to_pose:
                        asset_to_pose[action[0]] = {}
                    if category not in asset_to_pose[action[0]]:
                        asset_to_pose[action[0]][category] = {}
                    if surface_category not in asset_to_pose[action[0]][category]:
                        asset_to_pose[action[0]][category][surface_category] = []
                    asset_to_pose[action[0]][category][surface_category].append(pp)

            ## the run_dir might have been deleted
            config_file = join(run_dir, 'planning_config.json')
            if isfile(config_file):
                planning_config = json.load(open(config_file, 'r'))
                if 'placement_plan' not in planning_config:
                    new_config_file = join(run_dir, 'planning_config_new.json')
                    with open(new_config_file, 'w') as f:
                        planning_config['placement_plan'] = placement_plan
                        json.dump(planning_config, f, indent=3)
                    shutil.move(new_config_file, config_file)
    # return

    ## found 26776 (0.892)	missed 332 (0.011)  misplaced 2899 (0.097)
    total_count = found_count + missed_count + misplaced_count
    line = f'found {found_count} ({round(found_count/total_count, 3)})'
    line += f'\tmissed {missed_count} ({round(missed_count/total_count, 3)})'
    line += f'\tmisplaced {misplaced_count} ({round(misplaced_count/total_count, 3)})'
    if deleted_count > 0:
        line += f'\tdeleted misplaced {deleted_count} ({round(deleted_count/(misplaced_count+deleted_count), 3)})'
    print(line)

    save_pose_samples(asset_to_pose, title='pickplace')
    # for action in asset_to_pose:
    #     # plot_pose_samples(asset_to_pose[action], title=action)
    #     save_pose_samples(asset_to_pose[action], title=action)


def test_polar_2d_plotting(title='place'):
    """ discarded """
    import pandas as pd

    asset_to_pose = {}
    df = pd.read_csv(join(DATABASE_DIR, f'{title}.csv'))
    categories = df['Category'].unique()
    for c in categories:
        new_df = df.loc[df['Category'] == c]
        poses = list(zip(new_df['Distance'], new_df['Radian']))
        asset_to_pose[c] = poses
    plot_pose_samples(asset_to_pose, title='Reconstructed_' + title)


def test_pose_3d_plotting(title='pickplace'):
    """ jupyter notebook in world_builder/samplers.ipynb """
    import pandas as pd
    import plotly.express as px

    df = pd.read_csv(join(DATABASE_DIR, f'{title}.csv'), index_col=0)

    def get_sub_df(cat=None, surface=None, action=None):
        condition = True
        if action is not None:
            condition = condition & (df.Action == action)
        if cat is not None:
            condition = condition & (df.Movable == cat)
        if surface is not None:
            condition = condition & (df.Surface == surface)
        return df[condition]

    sub_df = get_sub_df(surface='MiniFridge/10849')
    fig = px.scatter_3d(sub_df, x='x', y='y', z='z', color='yaw',
                        opacity=0.3, size_max=2, symbol='Movable',
                        hover_data=["run_name", "Action"])
    fig.show()


if __name__ == '__main__':
    test_generate_pose_samples()
    # test_polar_2d_plotting()
    # test_pose_3d_plotting()