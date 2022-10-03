"""
Bar chart demo with pairs of bars grouped for easy comparison.
"""
import json
import shutil
import time
from datetime import datetime
import os
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, isfile, isdir, abspath
from os import listdir
import seaborn as sns
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

from utils import DATASET_PATH

AUTO_REFRESH = False
VIOLIN = False
FPC = False
PAPER_VERSION = False ## no preview, just save pdf

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Times']})
# rc('text', usetex=True)

GROUPS = ['tt_one_fridge_table_pick', 'tt_one_fridge_table_in', 'tt_two_fridge_pick',
          'tt_two_fridge_in']  ## ## 'tt_one_fridge_table_on', 'tt_one_fridge_pick',

METHODS = ['None', 'shuffle', 'binary', 'pvt', 'oracle']
METHOD_NAMES = ['Baseline', 'Shuffle', 'PST-0/1', 'PST', 'Oracle']
## ## , 'random' , 'piginet', 'pvt-task', 'pvt-2', 'pvt|rel=all'

METHODS = ['None', 'pvt', 'pvt*', 'pvt-task', 'oracle'] ##
METHOD_NAMES = ['Baseline', 'PST', 'PST*', 'PST-task', 'Oracle']
## ## , 'random' , 'piginet', 'pvt-task', 'pvt-2', 'pvt|rel=all'

check_time = 1664255601 ## 1664255601 for baselines | 1664750094  ## for d4

color_dict = {
    'b': ('#3498db', '#2980b9'),
    'g': ('#2ecc71', '#27ae60'),
    'r': ('#e74c3c', '#c0392b'),
    'y': ('#f1c40f', '#f39c12'),
    'p': ('#9b59b6', '#8e44ad'),
    'gray': ('#95a5a6', '#7f8c8d'),
}
colors = ['b', 'r', 'g', 'y', 'gray'] ## , 'p'
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#95a5a6'] ## , '#9b59b6'
colors_darker = ['#2980b9', '#c0392b', '#27ae60', '#f39c12', '#7f8c8d'] ## , '#8e44ad'
colors = [color_dict[k][0] for k in ['b', 'r', 'g', 'y', 'gray']]
colors_darker = [color_dict[k][0] for k in ['b', 'r', 'g', 'y', 'gray']]

## see which files are missing
# METHODS = ['None', 'oracle'] ## , 'pvt'
SAME_Y_AXES = False
RERUN_SUBDIR = 'rerun_2'


def get_rundirs(task_name):
    data_dir = join(DATASET_PATH, task_name)
    # data_dir = join(DATASET_PATH, 'tt_0915', 'tt1', task_name)
    dirs = [join(data_dir, f, RERUN_SUBDIR) for f in listdir(data_dir) if isdir(join(data_dir, f))]
    # dirs = []
    if PAPER_VERSION and False:
        data_dir = join(DATASET_PATH, 'tt_0915', 'tt1', task_name)
        dirs.extend([join(data_dir, f, RERUN_SUBDIR) for f in listdir(data_dir) if isdir(join(data_dir, f))])
    dirs.sort()
    return dirs


def check_run_dirs():
    for group in GROUPS:
        run_dirs = get_rundirs(group)
        for run_dir in run_dirs:

            domain_file = join(run_dir, f"domain_full.pddl")
            correct_file = '/home/yang/Documents/cognitive-architectures/bullet/assets/pddl/domains/pr2_mamao.pddl'
            num_lines = len(open(domain_file, 'r').readlines())

            # last_modified = os.path.getmtime(domain_file)
            # if last_modified > time.time() - 60 * 60:
            #     print('just modified domain', run_dir)
            #     if not isdir(run_dir.replace('/tt_', '/ss_')):
            #         shutil.copytree(run_dir, run_dir.replace('/tt_', '/ss_'))
            #
            # if num_lines != len(open(correct_file, 'r').readlines()): ## in [378, 379]:
            #     print('wrong domain', run_dir)
            #     os.remove(domain_file)
            #     shutil.copyfile(correct_file, domain_file)
            #     if 'tt_two_fridge_in' in run_dir:
            #         records = [join(run_dir, f) for f in listdir(run_dir) if 'plan_rerun_fc=' in f]
            #         [os.remove(f) for f in records]
            #         # print('   copying to', run_dir.replace('/tt_', '/ss_'))
            #         # print('   to remove\n', records)


def get_time_data(diverse=False):
    prefix = "diverse_" if diverse else ""
    data = {}
    for group in GROUPS:
        data[group] = {}
        run_dirs = get_rundirs(group)

        data[group]['count'] = len([f for f in run_dirs if isdir(join(f, 'crop_images'))])
        data[group]['missing'] = {}
        data[group]['run_dir'] = {}
        for run_dir in run_dirs:

            for method in METHODS:
                if method not in data[group]:
                    data[group][method] = []
                if method not in data[group]['missing']:
                    data[group]['missing'][method] = []
                if method not in data[group]['run_dir']:
                    data[group]['run_dir'][method] = []
                if 'run_dir' not in data[group]:
                    data[group]['run_dir'] = []

                if FPC: ## False positive count
                    file = join(run_dir, f"{prefix}fc_log={method}.json")
                else: ## planning time
                    file = join(run_dir, f"{prefix}plan_rerun_fc={method}.json")

                if not isfile(file):
                    print(f"File not found: {file}")
                    data[group]['missing'][method].append(run_dir)
                    continue
                    # if method == 'None':
                    #     file = join(run_dir, f"plan.json")
                    # else:
                    #     continue

                with open(file, 'r') as f:
                    last_modified = os.path.getmtime(file)
                    if last_modified < check_time:
                        print('skipping old result', file)
                        data[group]['missing'][method].append(run_dir)
                        continue

                    d = json.load(f)

                    if FPC:
                        print(d['checks'])
                    else:
                        ## original planning time
                        if len(d) == 2:
                            t = d[0]["planning"]

                        ## replanning time
                        else:
                            if d["plan"] is None:
                                print('Failed old result', file)
                                if last_modified < check_time:
                                    data[group]['missing'][method].append(run_dir)
                                continue
                            t = d["planning_time"]
                        if t > 500:
                            print(f"Planning time too long: {t} s", file)
                            # t = 500

                    data[group][method].append(t)
                    if run_dir not in data[group]['run_dir']:
                        data[group]['run_dir'][method].append(run_dir)
    return data


def plot_bar_chart(data, update=False, save_path=None, diverse=False):
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
        if not SAME_Y_AXES:
            points_x[i] = []
            points_y[i] = []

        for j in range(len(METHODS)):
            method = METHODS[j]
            if method not in means:
                means[method] = []
                stds[method] = []
                maxs[method] = []
                argmaxs[method] = []
                counts[method] = []
                missing[method] = []
                if SAME_Y_AXES:
                    points_x[method] = []
                    points_y[method] = []

            if method in data[group] and len(data[group][method]) > 0:
                means[method].append(np.mean(data[group][method]))
                stds[method].append(np.std(data[group][method]))
                maxs[method].append(np.max(data[group][method]))
                label = data[group]['run_dir'][method][np.argmax(data[group][method])]
                label = label.replace(abspath(DATASET_PATH), '')
                label = label.replace('/', '').replace(group, '').replace('tt1', '-').replace(RERUN_SUBDIR, '')
                argmaxs[method].append(f"#{label}")
                counts[method].append(len(data[group][method]))
                if SAME_Y_AXES:
                    points_x[method].extend([i] * len(data[group][method]))
                    points_y[method].extend(data[group][method])
                else:
                    points_x[i].extend([j] * len(data[group][method]))
                    points_y[i].extend(data[group][method])
            else:
                means[method].append(0)
                stds[method].append(0)
                maxs[method].append(0)
                argmaxs[method].append("")
                counts[method].append(0)
            missing[method].append(len(data[group]['missing'][method]))

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

    dt = datetime.now().strftime("%m%d_%H%M%S")
    title = 'Planning time comparison '
    title += '(diverse planning mode) ' if diverse else ''
    labels = tuple([f"{g.replace('tt_', '')}\n({data[g]['count']})" for g in groups])
    labels = tuple([f"{g.replace('tt_', '')}" for g in groups])

    if SAME_Y_AXES:
        figsize = (9, 6)
        fig, ax = plt.subplots(figsize=figsize)
        max_y = -1
        for i in range(len(means)):
            method = METHODS[i]

            """ mean & average of all rerun planning time """
            x = index + bar_width * i
            plt.bar(x, means[method], bar_width,
                    alpha=0.4,
                    color=colors[i],
                    yerr=stds[method],
                    error_kw={'ecolor': colors_darker[i]},
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
                if maxs[method][j] > max_y:
                    max_y = maxs[method][j]

            """ actual value of rerun planning time """
            x = np.asarray(points_x[method]) + bar_width * i
            y = points_y[method]
            # xy = np.vstack([x, y])  # Calculate the point density
            # z = gaussian_kde(xy)(xy)
            plt.scatter(x, y, s=20, color=colors[i], alpha=0.7)  ## c=z,

        plt.title(title + dt, fontsize=16, pad=35)
        plt.ylim([0, max_y+100]) if diverse else plt.ylim([0, 500])
        plt.legend(ncol=len(METHODS), fontsize=11, loc='upper center', bbox_to_anchor=(0.5, 1.1), fontname='Times')
        plt.xlabel('Tasks (run count)', fontsize=12, fontname='Times')
        plt.ylabel('Planning time', fontsize=12, fontname='Times')
        plt.xticks(index + x_ticks_offset, labels, fontsize=10)
        ax.tick_params(axis='x', which='major', pad=28)
        plt.tight_layout()

    ## ------------- different y axis to amplify improvements ------------ ##
    else:
        figsize = (15, 6) if not PAPER_VERSION else (15, 4)
        figsize = (12, 4) if len(groups) == 3 else figsize
        bar_width = 0.3
        fig, axs = plt.subplots(1, len(groups), figsize=figsize)

        scale = 0.4
        ll = [x for x in range(len(METHODS))]
        for i in range(len(groups)):
            mean = [means[method][i] for method in METHODS]
            std = [stds[method][i] for method in METHODS]
            count = [counts[method][i] for method in METHODS]
            miss = [missing[method][i] for method in METHODS]
            agm = [argmaxs[method][i] for method in METHODS]
            mx = [maxs[method][i] for method in METHODS]
            xx = points_x[i]
            yy = points_y[i]

            if VIOLIN:
                import pandas as pd
                from matplotlib.collections import PolyCollection

                df = pd.DataFrame({'x': xx, 'y': yy})
                my_pal = {i: colors[i] for i in range(len(METHODS))}
                sns.violinplot(data=df, x="x", y="y", bw=.4, ax=axs[i], scale="width", palette=my_pal)
                axs[i].set(xlabel=None, ylabel=None)
                # for art in axs[i].get_children():
                #     if isinstance(art, PolyCollection):
                #         art.set_edgecolor((1, 1, 1, 1))

            else:
                ll = [x * scale for x in range(len(METHODS))]
                axs[i].bar(ll, mean, bar_width,
                        alpha=0.4,
                        color=colors,
                        yerr=std,
                        error_kw={'ecolor': colors_darker},
                        label=METHODS)

                for j in range(len(METHODS)):
                    cc = [colors[k] for k in xx]
                    xxx = [x*scale for x in xx]
                    axs[i].scatter(xxx, yy, s=20, color=cc, alpha=0.7)

                    if not PAPER_VERSION:
                        bar_label = f"{count[j]} \n"
                        if miss[j] > 0:
                            bar_label += str(miss[j])
                        axs[i].annotate(bar_label,  # text
                                    (j*scale, 0),  # points location to label
                                    textcoords="offset points",
                                    xytext=(0, -40),  # distance between the points and label
                                    ha='center', color='gray',
                                    fontsize=10)
                        axs[i].annotate(agm[j],  # text
                                    (j*scale, mx[j]),  # points location to label
                                    textcoords="offset points",
                                    xytext=(0, 6),  # distance between the points and label
                                    ha='center',
                                    fontsize=10)

                axs[i].set_ylim([0, max(mx)*1.1])
                axs[i].tick_params(axis='x', which='major') ## , pad=28
                axs[i].tick_params(axis='y', which='major', pad=-4, rotation=45)

            if PAPER_VERSION:
                axs[i].set_title(labels[i], fontsize=11, y=-0.2)
            else:
                axs[i].set_title(f"{labels[i]} ({len(get_rundirs(groups[i]))})", fontsize=11, y=-0.24)

            axs[i].set_xticks(ll)
            axs[i].set_xticklabels(METHOD_NAMES, fontsize=10) ## , y=-0.25 , rotation=45

        axs[0].set_ylabel('Planning time', fontsize=12)
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(METHODS))]

        if not PAPER_VERSION:
            plt.subplots_adjust(hspace=0.55, left=0.1, right=0.95, top=0.8, bottom=0.2)
            fig.suptitle(title + dt, fontsize=16, y=0.96) ##
            fig.legend(handles, METHOD_NAMES, ncol=len(METHODS), fontsize=11,
                       loc='upper center', bbox_to_anchor=(0.5, 0.9))
        else:
            plt.subplots_adjust(hspace=0.55, left=0.05, right=0.99, top=0.85, bottom=0.15)
            fig.legend(handles, METHOD_NAMES, ncol=len(METHODS), fontsize=11,
                       loc='upper center', bbox_to_anchor=(0.5, 0.97))

    if PAPER_VERSION: ##  and False
        plt.savefig('/home/yang/evaluation.pdf', bbox_inches='tight')
    else:
        if update:
            plt.draw()
        else:
            plt.show()


if __name__ == '__main__':

    if not AUTO_REFRESH:
        print('time.time()', int(time.time()))
        plot_bar_chart(get_time_data(diverse=True), diverse=True)
        # plot_bar_chart(get_time_data())
    else:
        while True:
            plot_bar_chart(get_time_data(diverse=True), diverse=True, update=True)
            print('waiting for new data...')
            plt.pause(30)
            plt.close('all')

        # duration = 4
        # while True:
        #     plot_bar_chart(get_time_data(diverse=True), update=True)
        #     print('waiting for new data...')
        #     plt.pause(duration)
        #     plt.close('all')
        #     plot_bar_chart(get_time_data(), update=True)
        #     print('waiting for new data...')
        #     plt.pause(duration)
        #     plt.close('all')
