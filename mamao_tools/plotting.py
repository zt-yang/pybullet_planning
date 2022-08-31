"""
Bar chart demo with pairs of bars grouped for easy comparison.
"""
import json
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, isfile, isdir
from os import listdir


DATASET_PATH = '/home/zhutiany/Documents/mamao-data'
GROUPS = ['tt_one_fridge_pick', 'tt_one_fridge_table_in', 'tt_two_fridge_in']  ##
METHODS = ['None', 'oracle']  ## , 'random' , 'piginet'

def get_rundirs(task_name):
    data_dir = join(DATASET_PATH, task_name)
    dirs = [join(data_dir, f) for f in listdir(data_dir) if isdir(join(data_dir, f))]
    dirs.sort()
    return dirs


def get_time_data():
    data = {}
    for group in GROUPS:
        data[group] = {}
        for run_dir in get_rundirs(group):
            for method in METHODS:
                file = join(run_dir, f"plan_rerun_fc={method}.json")
                if not isfile(file):
                    if method == 'None':
                        file = join(run_dir, f"plan.json")
                    else:
                        continue
                if method not in data[group]:
                    data[group][method] = []
                if 'run_dir' not in data[group]:
                    data[group]['run_dir'] = []
                with open(file, 'r') as f:
                    d = json.load(f)
                    if len(d) == 2:
                        t = d[0]["planning"]
                    else:
                        t = d["planning_time"]
                    data[group][method].append(t)
                    if run_dir not in data[group]['run_dir']:
                        data[group]['run_dir'].append(run_dir)
    return data


def plot_bar_chart(data, save_path=None):
    """
        {
            "tt_one_fridge_pick": {
                "none": [t1, t2, t3, t4, t5],
                "oracle": [t1, t2, t3, t4, t5],
                "random": [t1, t2, t3, t4, t5],
                "piginet": [t1, t2, t3, t4, t5],
            }
        }
    """

    groups = list(data.keys())
    n_groups = len(data)
    means = {}
    stds = {}
    points_x = {}
    points_y = {}
    for i in range(n_groups):
        group = groups[i]
        for j in range(len(METHODS)):
            method = METHODS[j]
            if method not in means:
                means[method] = []
                stds[method] = []
                points_x[method] = []
                points_y[method] = []

            if method in data[group]:
                means[method].append(np.mean(data[group][method]))
                stds[method].append(np.std(data[group][method]))
                points_x[method].extend([i] * len(data[group][method]))
                points_y[method].extend(data[group][method])
            else:
                means[method].append(0)
                stds[method].append(0)

    # means_men = (20, 35, 30, 35, 27)
    # std_men = (2, 3, 4, 1, 2)
    #
    # means_women = (25, 32, 34, 20, 25)
    # std_women = (3, 5, 2, 3, 3)

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.35

    error_config = {'ecolor': '0.3'}

    colors = ['b', 'r', 'g', 'y']

    for i in range(len(means)):
        method = METHODS[i]
        plt.bar(index + bar_width * i, means[method], bar_width,
                alpha=0.4,
                color=colors[i],
                yerr=stds[method],
                error_kw=error_config,
                label=METHODS[i])
        x = np.asarray(points_x[method]) + bar_width * i
        y = points_y[method]
        # xy = np.vstack([x, y])  # Calculate the point density
        # z = gaussian_kde(xy)(xy)
        plt.scatter(x, y, alpha=1, color=colors[i])  ## c=z,

    plt.xlabel('Tasks')
    plt.ylabel('Planning time')
    plt.title('Planning time with feasibility checkers')
    plt.xticks(index + bar_width/2, tuple([g.replace('tt_', '') for g in groups]))
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_bar_chart(get_time_data())