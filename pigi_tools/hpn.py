import platform
import json
import platform
import shutil
import time
import math
from datetime import datetime
import os
import sys
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, isfile, isdir, abspath
from collections import defaultdict
from os import listdir
import seaborn as sns
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

from pigi_tools.plotting_utils import *


#############################################################################

SUMMARIZE_SAMPLERS = True  ## when len(GROUPS) == 1
GROUPS = ['kitchen_food']
GROUPNAMES = ['Serve 6 Plates']
METHODS = ['original_more_plates', 'original_more_movables', 'original', 'hpn', 'hpn_goal-related']
METHOD_NAMES = ['original \n(+ 2 plates)', 'original \n(+ 2 food)', 'original', 'hpn', 'hpn \n(min object)']
#############################################################################
SUMMARIZE_SAMPLERS = False
GROUPS = ['kitchen_food', 'kitchen_food_cleaned']
GROUPNAMES = ['Serve 6 Plates', 'Serve 2 Cleaned']
METHODS = ['original', 'hpn']
METHOD_NAMES = ['original', 'hpn']


#############################################################################

exp_dir = '/home/yang/Documents/cognitive-architectures/bullet/experiments/'
PAPER_VERSION = True
cc = ['b', 'r', 'g', 'p', 'y', 'gray'][:len(METHODS)]
colors = [color_dict[k][0] for k in cc]
colors_darker = [color_dict[k][0] for k in cc]


def plot_bar_chart(data):
    """
        {
            "kitchen_food": {
                "original": [t1, t2, t3, t4, t5],
                "hpn": [t1, t2, t3, t4, t5],
                "hpn_goal-related": [t1, t2, t3, t4, t5],
            }
        }
    """

    groups = list(data.keys())
    n_groups = len(data)
    means = {}
    means_overhead = {}
    means_fd_time = {}
    stds = {}
    maxs = {}
    argmaxs = {}
    counts = {}
    missing = {}
    points_x = {}
    points_y = {}
    lines = {}  ## Group: lines
    for i in range(n_groups):
        group = groups[i]
        points_x[i] = []
        points_y[i] = []

        for j in range(len(METHODS)):
            method = METHODS[j]
            if method not in means:
                means[method] = []
                means_overhead[method] = []
                means_fd_time[method] = []
                stds[method] = []
                maxs[method] = []
                argmaxs[method] = []
                counts[method] = []
                missing[method] = []

            if method in data[group] and len(data[group][method]) > 0:
                means_planning = np.asarray(data[group][method]) ## - np.asarray(data[group]['overhead'][method])
                means[method].append(np.mean(means_planning))
                stds[method].append(np.std(means_planning))
                means_overhead[method].append(np.mean(data[group]['overhead'][method]))
                # means_fd_time[method].append(np.mean(data[group]['fd_time'][method]))
                maxs[method].append(np.max(data[group][method]))
                label = data[group]['run_dir'][method][np.argmax(data[group][method])]
                argmaxs[method].append(f"#{label}")
                counts[method].append(len(data[group][method]))
                points_x[i].extend([j] * len(data[group][method]))
                points_y[i].extend(data[group][method])
            else:
                means[method].append(0)
                means_overhead[method].append(0)
                # means_fd_time[method].append(0)
                stds[method].append(0)
                maxs[method].append(0)
                argmaxs[method].append("")
                counts[method].append(0)
            missing[method].append(len(data[group]['missing'][method]))

        lines[group] = {}
        run_dir_counts = {}
        for method, run_dirs in data[group]['run_dir'].items():
            for run_dir in run_dirs:
                if run_dir not in run_dir_counts:
                    run_dir_counts[run_dir] = []
                run_dir_counts[run_dir].append(method)
        # use_run_dirs = [k for k, v in run_dir_counts.items() if v == len(METHODS)]
        for run_dir, mm in run_dir_counts.items():
            if run_dir not in lines[group]:
                lines[group][run_dir] = {n: [] for n in ['x', 'y']}
            for method, run_dirs in data[group]['run_dir'].items():
                if method not in mm:
                    continue
                lines[group][run_dir]['x'].append(METHODS.index(method))
                lines[group][run_dir]['y'].append(data[group][method][run_dirs.index(run_dir)])

    index = np.arange(n_groups)
    bar_width = 0.1
    if len(METHODS) == 2:
        bar_width = 0.35
    elif len(METHODS) == 3:
        bar_width = 0.2
    elif len(METHODS) == 4:
        bar_width = 0.2
    elif len(METHODS) == 5:
        bar_width = 0.15
    x_ticks_offset = bar_width * (len(METHODS)-1)/2

    error_config = {'ecolor': '0.3'}

    dt = datetime.now().strftime("%m%d_%H%M%S")
    title = "Planning time"
    labels = list(data.keys())
    # if GROUPNAMES is not None:
    #     labels = tuple(GROUPNAMES)

    ## RSS paper version
    bar_width = 0.2 ## 0.5 / math.ceil(math.sqrt(len(groups)))
    fig_h = 5
    fig_w = fig_h * len(groups) * 0.8
    n_cols = len(groups)
    n_plots = len(groups)

    if SUMMARIZE_SAMPLERS:
        n_cols = math.ceil(len(groups)/2)
        figsize = (fig_w / 2, fig_h * 2)
        fig, axs = plt.subplots(2, n_cols, figsize=figsize)
        n_plots = 2 * n_cols
    else:
        figsize = (fig_w, fig_h)
        fig, axs = plt.subplots(1, len(groups), figsize=figsize)
        if len(groups) == 1:
            axs = [axs]


    scale = 0.4
    ll = [x for x in range(len(METHODS))]
    for i in range(n_plots):

        if SUMMARIZE_SAMPLERS:
            ax = axs[math.floor(i / n_cols)][i % n_cols]
            if i == n_cols:
                ax.axis('off')
                continue
            elif i > n_cols:
                i = i - 1
        else:
            ax = axs[i]

        mean = [means[method][i] for method in METHODS]
        overhead = [means_overhead[method][i] for method in METHODS]
        # fd_time = [means_fd_time[method][i] for method in METHODS]
        std = [stds[method][i] for method in METHODS]
        count = [counts[method][i] for method in METHODS]
        miss = [missing[method][i] for method in METHODS]
        agm = [argmaxs[method][i] for method in METHODS]
        mx = [maxs[method][i] for method in METHODS]
        xx = points_x[i]
        yy = points_y[i]

        ll = [x * scale for x in range(len(METHODS))]

        bar_kwargs = dict(color=colors, alpha=0.4, label=METHODS)
        overhead_kwargs = dict(color='#000000', alpha=0.4, bottom=mean)

        """ fixed cost """
        # fd_time = sum(fd_time) / len(fd_time)
        # axs[i].bar(ll, fd_time, bar_width,
        #            color=color_dict['p'][1], alpha=0.4)
        # # mean = [mean[i] - fd_time[i] for i in range(len(mean))]
        # mean = [m - fd_time for m in mean]
        # bar_kwargs['bottom'] = fd_time

        if i == 0:
            ylabel = 'Planning time (sec)'
            ax.set_ylabel(ylabel, fontsize=12)

        """ refinement cost """
        ax.bar(ll, mean, bar_width, yerr=std, error_kw={'ecolor': colors_darker}, **bar_kwargs)

        """ overhead cost """
        ax.bar(ll, overhead, bar_width, **overhead_kwargs)
        # print(f"task: {groups[i]} overhead: {[round(oo, 5) for oo in overhead]}")

        ## data points
        cc = []
        for k in xx:
            cc.append(colors[k])
        # cc = [colors[k] for k in xx]
        xxx = [x * scale for x in xx]
        s = 20

        ax.scatter(xxx, yy, s=s, color=cc, alpha=0.7)

        for j in range(len(METHODS)):

            ## number of unfinished / timeout cases
            if not PAPER_VERSION:
                bar_label = f"{count[j]} \n"
                if miss[j] > 0:
                    bar_label += f"timeout: {str(miss[j])}"
                ax.annotate(bar_label,  # text
                            (j*scale, 0),  # points location to label
                            textcoords="offset points",
                            xytext=(0, -40),  # distance between the points and label
                            ha='center', color='gray',
                            fontsize=10)

            ## run_dir of the extreme results
            if not PAPER_VERSION:
                ax.annotate(agm[j],  # text
                            (j*scale, mx[j]),  # points location to label
                            textcoords="offset points",
                            xytext=(0, 6),  # distance between the points and label
                            ha='center',
                            fontsize=10)

        ax.set_ylim([0, max(mx)*1.1])
        ax.tick_params(axis='x', which='major') ## , pad=28
        ax.tick_params(axis='y', which='major', pad=-4, rotation=45)

        ax.set_title(labels[i], fontsize=11, y=-0.2)

        ax.set_xticks(ll)
        ax.set_xticklabels(METHOD_NAMES, fontsize=10) ## , y=-0.25 , rotation=45

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(METHODS))]

    fig.suptitle(title, fontsize=16, y=0.96)

    ncol = math.ceil(len(METHODS) / 3)
    legends = [n.replace('\n', '') for n in METHOD_NAMES]

    if not PAPER_VERSION:
        plt.subplots_adjust(hspace=0.55, left=0.1, right=0.95, top=0.8, bottom=0.2)
        # fig.suptitle(title, fontsize=16, y=0.96) ##  + dt
        # fig.suptitle('.', fontsize=16, y=0.96) ##  + dt
        fig.legend(handles, legends, ncol=ncol, fontsize=11,
                   loc='upper center', bbox_to_anchor=(0.5, 0.97))
        plt.tight_layout()
    else:
        if SUMMARIZE_SAMPLERS:
            top = 0.87
            hspace = 0.3
            y_legend = 0.93
        else:
            top = 0.8
            hspace = 0.2
            y_legend = 0.9
        plt.subplots_adjust(hspace=hspace, left=0.05, right=0.99, top=top, bottom=0.1)
        fig.legend(handles, legends, ncol=len(METHODS), fontsize=11,
                   loc='upper center', bbox_to_anchor=(0.53, y_legend))

    if PAPER_VERSION:
        plt.savefig(f'hpn.png', bbox_inches='tight', dpi=300)
    else:
        plt.show()


def load_data():
    import csv
    data = {}
    for group in GROUPS:
        data[group] = {k: {} for k in ['count', 'overhead', 'run_dir', 'missing']}
        for method in METHODS:
            csv_file = join(exp_dir, group, f'{method}.csv')
            times = []
            overhead = []
            run_dirs = []
            missing = []
            with open(csv_file, newline='\n') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                i = 0
                for row in reader:
                    i += 1
                    if i == 1:
                        continue
                    name = row[0]

                    ## add those sampling time as new groups
                    if SUMMARIZE_SAMPLERS:
                        log_file = join(exp_dir, group, name, 'log.txt')
                        data = get_sampler_summary(log_file, method, data)

                    t = float(row[-3])
                    if t == 99999:
                        missing.append(name)
                        continue
                    times.append(t)
                    overhead.append(float(row[-2]))
                    run_dirs.append(name)
            data[group][method] = times
            data[group]['overhead'][method] = overhead
            data[group]['run_dir'][method] = run_dirs
            data[group]['count'][method] = i-1
            data[group]['missing'][method] = missing

    for g in data:
        if g not in GROUPS:
            data[g].update({
                'count': data[GROUPS[0]]['count'],
                'overhead': {k: [0]*len(data[GROUPS[0]]['overhead'][k]) for k in METHODS},
                'run_dir': data[GROUPS[0]]['run_dir'],
                'missing': data[GROUPS[0]]['missing'],
            })
    return data


def get_sampler_summary(log_file, method, data):
    samplers = ['inverse-reachability', 'inverse-kinematics', 'plan-base-motion']
    names = ['IR', 'IK', 'BasePlanning']
    groups = [f"time({name})" for name in names] + [f"n_calls({name})" for name in names]
    ##  + [f"p({name})" for name in ['IK', 'BasePlanning']]
    names = {s: n for s, n in zip(samplers, names)}

    ## extract sampler statistics from log.txt
    found = []
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if 'Local External Statistics' in lines[i]:
                found.append([l.replace('External: ', '').split(' | ') for l in lines[i:i+9] if 'External: ' in l])

    ## create groups from the statistics
    if groups[0] not in data:
        data.update({g: defaultdict(list) for g in groups})

    all_n_calls = {s: 0 for s in samplers}
    all_overhead = {s: 0 for s in samplers}
    for ff in found:
        for sampler, n_calls, p_success, mean_overhead, overhead in ff:
            if sampler in samplers:
                all_n_calls[sampler] += int(n_calls[n_calls.index(':') + 1:].strip())
                all_overhead[sampler] += float(overhead[overhead.index(':') + 1:].strip())
                #     p_success = float(p_success[p_success.index(':') + 1:].strip())

    for sampler in samplers:
        data[f"n_calls({names[sampler]})"][method].append(all_n_calls[sampler])
        data[f"time({names[sampler]})"][method].append(all_overhead[sampler])
        # data[f"p({names[sampler]})"][method].append(p_success)

    return data


if __name__ == '__main__':
    data = load_data()
    plot_bar_chart(data)
