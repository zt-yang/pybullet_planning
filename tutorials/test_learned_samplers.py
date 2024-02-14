from os.path import join, isdir, abspath, isfile, basename
from os import listdir
import json
from pprint import pprint
import shutil
from tqdm import tqdm

from pybullet_tools.utils import euler_from_quat
from pigi_tools.data_utils import get_successful_plan, get_indices, get_sink_counter_x, get_lisdf_aabbs, \
    get_from_to, add_to_planning_config
from world_builder.world_utils import get_placement_z

DATABASE_DIR = abspath(join(__file__, '..', '..', 'databases'))


def get_obj_pose(action, indices):
    if action[2] not in indices:
        return None
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
               and (s.startswith('mm_') or s.startswith('tt_') or s.startswith('_kc')
                    or s.startswith('ww_') or s == '_gmm') \
               # and (s.startswith('mm_storage_long'))  ## for debugging
               and not s.endswith('tmp')
               ]
    subdirs.sort()
    asset_to_pose = {}

    found_count = 0
    misplaced_count = 0
    deleted_count = 0
    missed_count = 0

    z_correction = get_placement_z()

    run_dirs = []
    for subdir in subdirs:
        some_run_dirs = [join(subdir, s) for s in listdir(subdir) if isdir(join(subdir, s))]
        run_dirs += sorted(some_run_dirs, key=lambda x: eval(x.split('/')[-1]))

    ## debugging
    # run_dirs = ['/home/yang/Documents/fastamp-data-rss' + '/mm_storage/381']
    # run_dirs = ['/home/yang/Documents/fastamp-data-rss/_gmm/3610']

    for run_dir in tqdm(run_dirs):  ## , desc=basename(subdir)
        indices = get_indices(run_dir, larger=False)
        plan = get_successful_plan(run_dir)[0]
        counter_x = get_sink_counter_x(run_dir)
        aabbs = get_lisdf_aabbs(run_dir)
        placement_plan = []
        for action in plan:
            if action[0].startswith('pick') or action[0].startswith('place'):
                ## that of movable object
                result = get_obj_pose(action, indices)
                if result is None:
                    print(run_dir, action)
                    continue
                name, category, (point, yaw) = result

                verbose = False or len(run_dirs) == 1
                # if 'mm_storage_long/0' in run_dir and name == 'veggiepotato':
                #     verbose = True
                result = get_from_to(name, aabbs, point, run_dir=run_dir, verbose=verbose)
                if verbose:
                    verbose = True

                if result is None or result[0] is None:
                    # print(run_dir, name, result)
                    placement_plan.append((action[0], name, None, point))
                    missed_count += 1
                    continue

                (relation, surface_category, surface_name), x_upper, surface_point = result
                placement_plan.append((action[0], name, surface_name, point))
                pp = [point[i] - surface_point[i] for i in range(3)]

                ######################################################################
                ## objects outside of counters
                if surface_category == 'box' and point[0] > counter_x:
                    misplaced_count += 1
                    if 'pick' in action[0] and ('/mm_' in run_dir or '/tt_' in run_dir):
                        deleted_count += 1
                        # print(run_dir, name, point[0], round(counter_x, 3), action[0])
                        # shutil.rmtree(run_dir)
                        break
                    continue

                ## objects on top of containers instead of inside
                elif 'minifridge' in surface_category.lower() and pp[2] > 0:
                    continue

                ######################################################################
                found_count += 1
                dx = x_upper - point[0]
                run_name = run_dir.replace(data_dir, '')
                pp += [yaw, dx, run_name]
                if surface_category == 'box':
                    pp[0] = pp[-2]
                movable_name = aabbs['category'][name]

                if movable_name in z_correction and surface_category in z_correction[movable_name]:
                    pp[2] = z_correction[movable_name][surface_category][2]  ## just get the mean
                elif not ('BraiserLid' in movable_name and 'BraiserBody' in surface_category):
                    print(movable_name, surface_category, 'not in z_correction', run_dir)

                if action[0] not in asset_to_pose:
                    asset_to_pose[action[0]] = {}
                if movable_name not in asset_to_pose[action[0]]:
                    asset_to_pose[action[0]][movable_name] = {}
                if surface_category not in asset_to_pose[action[0]][movable_name]:
                    asset_to_pose[action[0]][movable_name][surface_category] = []
                asset_to_pose[action[0]][movable_name][surface_category].append(pp)

        ## the run_dir might have been deleted
        config_file = join(run_dir, 'planning_config.json')
        if isfile(config_file):
            add_to_planning_config(run_dir, {'placement_plan': placement_plan})

    ## found 32940 (0.978)	missed 725 (0.022)	misplaced 20 (0.001)
    total_count = found_count + missed_count + misplaced_count
    line = f'## found {found_count} ({round(found_count/total_count, 3)})'
    line += f'\tmissed {missed_count} ({round(missed_count/total_count, 3)})'
    line += f'\tmisplaced {misplaced_count} ({round(misplaced_count/total_count, 3)})'
    if deleted_count > 0:
        line += f'\tdeleted misplaced {deleted_count} ({round(deleted_count/(misplaced_count+deleted_count), 3)})'
    print(line)

    if len(run_dirs) == 1:
        return

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