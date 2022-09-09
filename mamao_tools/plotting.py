"""
Bar chart demo with pairs of bars grouped for easy comparison.
"""
import json
import time
import os
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, isfile, isdir, abspath
from os import listdir

from utils import DATASET_PATH

GROUPS = ['tt_one_fridge_pick', \
    'tt_one_fridge_table_pick', 'tt_one_fridge_table_in', 'tt_two_fridge_in']  ## 'tt_one_fridge_table_on', \
METHODS = ['None', 'oracle', 'pvt', 'pvt-task'] ## ## , 'random' , 'piginet'
check_time = 1662647476.5


def get_rundirs(task_name):
    data_dir = join(DATASET_PATH, task_name)
    dirs = [join(data_dir, f) for f in listdir(data_dir) if isdir(join(data_dir, f))]
    dirs.sort()
    return dirs


def get_time_data():
    data = {}
    for group in GROUPS:
        data[group] = {}
        run_dirs = get_rundirs(group)

        data[group]['count'] = len([f for f in run_dirs if isdir(join(f, 'crop_images'))])
        data[group]['missing'] = {}
        for run_dir in run_dirs:
            for method in METHODS:
                file = join(run_dir, f"plan_rerun_fc={method}.json")
                if method not in data[group]:
                    data[group][method] = []
                if method not in data[group]['missing']:
                    data[group]['missing'][method] = []

                if not isfile(file):
                    print(f"File not found: {file}")
                    data[group]['missing'][method].append(run_dir)
                    continue
                    if method == 'None':
                        file = join(run_dir, f"plan.json")
                    else:
                        continue

                if 'run_dir' not in data[group]:
                    data[group]['run_dir'] = []
                with open(file, 'r') as f:
                    d = json.load(f)
                    if len(d) == 2:
                        t = d[0]["planning"]
                    else:
                        if d["plan"] is None:
                            continue
                        t = d["planning_time"]
                    if t > 500:
                        print(f"Planning time too long: {t} s", file)
                        # t = 500

                    if run_dir not in data[group]['run_dir']:
                        data[group]['run_dir'].append(run_dir)

                    last_modified = os.path.getmtime(file)
                    if last_modified < check_time:
                        print('skipping old result', file)
                        data[group]['missing'][method].append(run_dir)
                        continue

                    data[group][method].append(t)
    return data


def plot_bar_chart(data, save_path=None):
    """
        {
            "tt_one_fridge_pick": {
                "none": [t1, t2, t3, t4, t5],
                "oracle": [t1, t2, t3, t4, t5],
                "random": [t1, t2, t3, t4, t5],
                "piginet": [t1, t2, t3, t4, t5],
                "run_dir": [s1, s2, s3, s4, s5],
            }
        }
    """

    groups = list(data.keys())
    n_groups = len(data)
    means = {}
    stds = {}
    maxs = {}
    argmaxs = {}
    counts = {}
    missing = {}
    points_x = {}
    points_y = {}
    for i in range(n_groups):
        group = groups[i]
        for j in range(len(METHODS)):
            method = METHODS[j]
            if method not in means:
                means[method] = []
                stds[method] = []
                maxs[method] = []
                argmaxs[method] = []
                counts[method] = []
                missing[method] = []
                points_x[method] = []
                points_y[method] = []

            if method in data[group] and len(data[group][method]) > 0:
                means[method].append(np.mean(data[group][method]))
                stds[method].append(np.std(data[group][method]))
                maxs[method].append(np.max(data[group][method]))
                label = data[group]['run_dir'][np.argmax(data[group][method])]
                label = label.replace(abspath(DATASET_PATH), '').replace('/', '').replace(group, '')
                argmaxs[method].append(f"#{label}")
                counts[method].append(len(data[group][method]))
                points_x[method].extend([i] * len(data[group][method]))
                points_y[method].extend(data[group][method])
            else:
                means[method].append(0)
                stds[method].append(0)
                maxs[method].append(0)
                argmaxs[method].append("")
                counts[method].append(0)
            missing[method].append(len(data[group]['missing'][method]))

    fig, ax = plt.subplots(figsize=(9, 6))

    index = np.arange(n_groups)
    bar_width = 0.1
    if len(METHODS) == 2:
        bar_width = 0.35
    elif len(METHODS) == 3:
        bar_width = 0.25
    elif len(METHODS) == 4:
        bar_width = 0.2
    x_ticks_offset = bar_width * (len(METHODS)-1)/2

    error_config = {'ecolor': '0.3'}

    colors = ['b', 'r', 'g', 'y']

    for i in range(len(means)):
        method = METHODS[i]

        """ mean & average of all rerun planning time """
        x = index + bar_width * i
        plt.bar(x, means[method], bar_width,
                alpha=0.4,
                color=colors[i],
                yerr=stds[method],
                error_kw=error_config,
                label=METHODS[i])
        
        """ max & count of rerun planning time """
        for j in range(len(x)):
            bar_label = f"{counts[method][j]} \n"
            if missing[method][j] > 0:
                bar_label += f"{missing[method][j]}"
            plt.annotate(bar_label,  # text
                        (x[j], 0),  # points location to label
                        textcoords="offset points",
                        xytext=(0, -24),  # distance between the points and label
                        ha='center',
                        fontsize=10)
            plt.annotate(argmaxs[method][j],  # text
                        (x[j], maxs[method][j]),  # points location to label
                        textcoords="offset points",
                        xytext=(0, 6),  # distance between the points and label
                        ha='center',
                        fontsize=10)

        """ median value of rerun planning time """
        x = np.asarray(points_x[method]) + bar_width * i
        y = points_y[method]

        # xy = np.vstack([x, y])  # Calculate the point density
        # z = gaussian_kde(xy)(xy)
        plt.scatter(x, y, alpha=1, color=colors[i], s=20)  ## c=z,

    plt.xlabel('Tasks (run count)', fontsize=12)
    plt.ylabel('Planning time', fontsize=12)
    plt.title('Planning time with feasibility checkers', fontsize=16, pad=35)
    labels = tuple([f"{g.replace('tt_', '')}\n({data[g]['count']})" for g in groups])
    plt.xticks(index + x_ticks_offset, labels, fontsize=10)
    plt.ylim([0, 500])
    plt.legend(ncol=4, fontsize=11, loc='upper center', bbox_to_anchor=(0.5, 1.1))
    ax.tick_params(axis='x', which='major', pad=28)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_bar_chart(get_time_data())