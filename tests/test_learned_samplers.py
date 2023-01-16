from os.path import join, isdir, abspath
from os import listdir
import pickle
from tqdm import tqdm

from pybullet_tools.utils import euler_from_quat
from mamao_tools.data_utils import get_successful_plan, get_indices

DATABASE_DIR = abspath(join(__file__, '..', '..', 'databases'))


def get_obj_pose(action, indices):
    obj = indices[action[2]].replace('#1', '').replace('#2', '')
    pose = action[3]
    pose = eval(pose[pose.index('('):])
    return obj, (pose[0], pose[-1])


def get_sink_counter_x(rundir, keyw='sink_counter'):
    def get_numbers(line, keep_strings=False):
        nums = line[line.index('>')+1:line.index('</')].split(' ')
        if keep_strings:
            return nums
        return [eval(n) for n in nums]
    lines = open(join(rundir, 'scene.lisdf'), 'r').readlines()
    for i in range(len(lines)):
        if f'name="{keyw}' in lines[i]:
            pose = lines[i+2]
            x = eval(get_numbers(pose, keep_strings=True)[0])
            size = lines[i+7]
            lx = eval(get_numbers(size, keep_strings=True)[0])
            return x + lx/2


def save_pose_samples(asset_to_pose, title='pose_samples'):
    import pandas as pd

    lst = []
    for name, poses in asset_to_pose.items():
        for pose in poses:
            lst.append([name] + list(pose))
    df = pd.DataFrame(lst, columns=['Category', 'Distance', 'Radian'], dtype=float)
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
        rundirs = [join(subdir, s) for s in listdir(subdir) if isdir(join(subdir, s))]
        rundirs = sorted(rundirs, key=lambda x: eval(x.split('/')[-1]))
        for rundir in tqdm(rundirs):
            indices = get_indices(rundir)
            plan = get_successful_plan(rundir)[0]
            counter_x = get_sink_counter_x(rundir)
            for action in plan:
                if action[0].startswith('pick') or action[0].startswith('place'):
                    if action[0] not in asset_to_pose:
                        asset_to_pose[action[0]] = {}
                    obj, (x, yaw) = get_obj_pose(action, indices)
                    if obj not in asset_to_pose[action[0]]:
                        asset_to_pose[action[0]][obj] = []
                    asset_to_pose[action[0]][obj].append((counter_x - x, yaw))
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
    # test_generate_pose_samples()
    test_plotting()