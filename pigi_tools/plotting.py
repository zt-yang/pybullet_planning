"""
Bar chart demo with pairs of bars grouped for easy comparison.
"""
import platform
import json
import platform
import shutil
import time
from datetime import datetime
import os
import sys
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, isfile, isdir, abspath
from os import listdir
import seaborn as sns
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

from data_utils import DATASET_PATH, get_fc_record, get_fastdownward_time
from pigi_tools.plotting_utils import *

sys.path.append('/home/yang/Documents/fastamp')
# from fastamp_utils import get_fc_record

AUTO_REFRESH = False
VIOLIN = False
FPC = False
EVALUATE_GEOMETRY = False
PAPER_VERSION = False ## no preview, just save pdf
DEBUG_LINES = False
ANIMATION = None

SAVE_DATA = False
USE_DATA = platform.system() == 'Darwin'

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Times']})
# rc('text', usetex=True)

GROUPNAMES = None
FIXED_COST = None

############################## fridges #######################################3


GROUPS = ['tt_one_fridge_table_pick', 'tt_two_fridge_pick', 'tt_one_fridge_table_in',
          'tt_two_fridge_in'] ## , 'tt_two_fridge_goals']

METHODS = ['None', 'shuffle', 'binary', 'pvt', 'oracle']
METHOD_NAMES = ['Baseline', 'Shuffle', 'PST-0/1', 'PST', 'Oracle']
## ## , 'random' , 'piginet', 'pvt-task', 'pvt-2', 'pvt|rel=all'

METHODS = ['None', 'pvt', 'pvt*', 'pvt-task', 'oracle'] ##
METHOD_NAMES = ['Baseline', 'PST', 'PST*', 'PST-task', 'Oracle']
## ## , 'random' , 'piginet', 'pvt-task', 'pvt-2', 'pvt|rel=all'

METHODS = ['None', 'pvt-task', 'binary', 'pvt-all', 'oracle']  ## 'pvt-trans',
METHOD_NAMES = ['Baseline', 'PIGI', 'PIGI-1/0', 'PIGI-all', 'Oracle']  ## 'PIGI-trans',

## ------------ generalization to geometry ------------
## GROUPS = ['ss_two_fridge_pick', 'ss_two_fridge_in']
## METHODS = ['None', 'pvt-all', 'oracle']
## METHOD_NAMES = ['Baseline', 'PIGI-all', 'Oracle']


################################ kitchens ######################################


GROUPS = ['tt_storage', 'tt_braiser', 'tt_braiser_to_storage' ] ## 'tt_sink', 'tt_sink_to_storage',

METHODS = ['None', 'pvt-task', 'oracle']
METHOD_NAMES = ['Baseline', 'PIGI', 'Oracle']

# METHODS = ['None', 'oracle']
# METHOD_NAMES = ['Baseline', 'Oracle']

################################ draft ######################################


GROUPS = ['tt_one_fridge_table_in', 'tt_two_fridge_in',
          'tt_storage', 'tt_braiser', 'tt_braiser_to_storage',
          'tt_sink', 'tt_sink_to_storage'] ##
GROUPNAMES = ['Table-to-fridge', 'Fridge-to-fridge',
              'Counter-to-storage', 'Counter-to-pot', 'Pot-to-storage',
              'Counter-to-sink', 'Sink-to-storage' ]  ##

GROUPS = ['tt_sink', 'tt_sink_to_storage'] ##
GROUPNAMES = ['Counter-to-sink', 'Sink-to-storage' ]  ##

METHODS = ['None', 'pvt-task', 'oracle']  ## 'pvt-6',
METHOD_NAMES = ['Baseline', 'PIGI', 'Oracle']  ## 'PIGI*',

# FIXED_COST = []
# for task_name in GROUPS:
#     if 'to_storage' in task_name:
#         FIXED_COST.append(60)
#     else:
#         FIXED_COST.append(3)

################################ rebuttal ######################################

GROUPS = ['tt_braiser_to_storage']  ## 'tt_braiser', 'tt_storage',
GROUPNAMES = ['Pot-to-storage']  ## 'Counter-to-pot', 'Counter-to-storage',

METHODS = ['None', 'pvt-br2-1', 'pvt-br2-0.75', 'oracle']
METHOD_NAMES = ['Baseline', 'PIGI-250', 'PIGI-180', 'Oracle']

############################ generalization ################################

if EVALUATE_GEOMETRY:
    GROUPS = ['tt_two_fridge_in', 'ss_two_fridge_in']
    GROUPNAMES = ['Fridge-to-fridge (Food)', 'Fridge-to-fridge (Staplers)']

    METHODS = ['None', 'pvt-all', 'oracle']
    METHOD_NAMES = ['Baseline', 'PIGI', 'Oracle']

############################################################################

check_time = 1664255601 ## 1664255601 for baselines | 1664750094  ## for d4 | 1665010453 for d3


if len(METHODS) == 5:
    cc = ['b', 'r', 'g', 'p', 'gray']
elif len(METHODS) == 6:
    cc = ['b', 'r', 'g', 'p', 'y', 'gray']
colors = [color_dict[k][0] for k in cc]
colors_darker = [color_dict[k][0] for k in cc]

## see which files are missing
# METHODS = ['None', 'oracle'] ## , 'pvt'
SAME_Y_AXES = False
RERUN_SUBDIR = 'rerun_4'


def get_rundirs(task_name):
    data_dir = join(DATASET_PATH, task_name)
    if not isdir(data_dir):
        data_dir = join(DATASET_PATH.replace('-rss', ''), task_name)
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
        if 'two_fridge' not in group:
            continue
        run_dirs = get_rundirs(group)
        for run_dir in run_dirs:

            files = [join(run_dir, f) for f in listdir(run_dir) if 'pvt-task*' in f]
            print(run_dir, len(files))
            # for f in files:
            #     os.remove(f)

            # domain_file = join(run_dir, f"domain_full.pddl")
            # correct_file = '/home/yang/Documents/cognitive-architectures/bullet/assets/pddl/domains/pr2_mamao.pddl'
            # num_lines = len(open(domain_file, 'r').readlines())

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
        data[group]['overhead'] = {}
        data[group]['fd_time'] = {}
        data[group]['num_FP'] = {}
        for run_dir in run_dirs:

            for method in METHODS:
                if method not in data[group]:
                    data[group][method] = []
                if 'run_dir' not in data[group]:
                    data[group]['run_dir'] = []
                for k in ['missing', 'run_dir', 'overhead', 'fd_time']:
                    if method not in data[group][k]:
                        data[group][k][method] = []

                """ get planning time """
                file = join(run_dir, f"{prefix}plan_rerun_fc={method}.json")
                # if 'tt_two_fridge_pick' in run_dir and method == 'pvt-task*':
                #     file = join(run_dir, f"{prefix}plan_rerun_fc=pvt-124.json")
                # elif 'tt_two_fridge_in' in run_dir and method == 'pvt-task*':
                #     file = join(run_dir, f"{prefix}plan_rerun_fc=pvt-123.json")
                # if '/tt_sink/' in run_dir and method == 'pvt-task':
                #     file = file.replace('rerun_2', 'rerun_3')

                if not isfile(file):
                    # print(f"File not found: {file}")
                    data[group]['missing'][method].append(run_dir)
                    continue
                    # if method == 'None':
                    #     file = join(run_dir, f"plan.json")
                    # else:
                    #     continue

                last_modified = os.path.getmtime(file)
                if last_modified < check_time:
                    print('skipping old result', file)
                    data[group]['missing'][method].append(run_dir)
                    continue

                d = json.load(open(file, 'r'))
                if FPC:
                    rr = run_dir[:-len(RERUN_SUBDIR)-1]
                    record = get_fc_record(rr, fc_classes=[method], rerun_subdir=RERUN_SUBDIR)
                    if method not in record:
                        continue
                    num_FP = record[method][-1]
                    if num_FP is not None:
                        data[group][method].append(num_FP)
                    else:
                        continue
                    # num_FP = get_fc_record(rr, fc_classes=[method], rerun_subdir=RERUN_SUBDIR)
                    # if method in num_FP:
                    #     data[group][method].append(len(num_FP[method][-2]))

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

                ## may want to visualize overhead
                log_file = file.replace('plan_rerun_fc', 'fc_log')
                inf_t = json.load(open(log_file, 'r'))['run_time'][0]
                if isinstance(inf_t, list):
                    inf_t = inf_t[0]
                data[group]['overhead'][method].append(inf_t)

                fd_time = get_fastdownward_time(run_dir, method)
                data[group]['fd_time'][method].append(fd_time)

                if run_dir not in data[group]['run_dir']:
                    data[group]['run_dir'][method].append(run_dir)
    return data


if EVALUATE_GEOMETRY:
    file_name = 'plotting_geometry.json'
elif FPC:
    file_name = 'plotting_fpc.json'
else:
    file_name = 'plotting.json'


def dump_data(data):
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)


def load_data():
    with open(file_name, 'r') as f:
        data = json.load(f)
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
        if not SAME_Y_AXES:
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
                if SAME_Y_AXES:
                    points_x[method] = []
                    points_y[method] = []

            if method in data[group] and len(data[group][method]) > 0:
                if FPC:
                    means[method].append(np.mean(data[group][method]))
                    stds[method].append(np.std(data[group][method]))
                else:
                    means_planning = np.asarray(data[group][method]) - np.asarray(data[group]['overhead'][method])
                    means[method].append(np.mean(means_planning))
                    stds[method].append(np.std(means_planning))
                means_overhead[method].append(np.mean(data[group]['overhead'][method]))
                means_fd_time[method].append(np.mean(data[group]['fd_time'][method]))
                maxs[method].append(np.max(data[group][method]))
                label = data[group]['run_dir'][method][np.argmax(data[group][method])]
                label = label.replace(abspath(DATASET_PATH), '').replace(abspath(DATASET_PATH.replace('-rss', '')), '')
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
                means_overhead[method].append(0)
                means_fd_time[method].append(0)
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
    title = 'Planning time comparison '
    title += '(diverse planning mode) ' if diverse else ''
    if FPC:
        title = "Number of false positive skeletons before finding a solution"
    else:
        title = "Planning time"
    labels = tuple([f"{g.replace('tt_', '')}\n({data[g]['count']})" for g in groups])
    labels = tuple([f"{g.replace('tt_', '')}" for g in groups])
    if 'ss_' in groups[-1]:
        labels = tuple([f"{g.replace('ss_', '')} (stapler)" for g in groups])
    if GROUPNAMES is not None:
        labels = tuple(GROUPNAMES)

    print(means)

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
        ylabel = 'Planning time (ms)' if not FPC else 'Number of FP skeletons'
        plt.ylabel(ylabel, fontsize=12, fontname='Times')
        plt.xticks(index + x_ticks_offset, labels, fontsize=10)
        ax.tick_params(axis='x', which='major', pad=28)
        plt.tight_layout()

    ## ------------- different y axis to amplify improvements ------------ ##
    else:
        if PAPER_VERSION:
            if len(GROUPS) == 2:
                figsize = (5, 4)
            else:
                figsize = (15, 4)
        else:
            figsize = (12, 4)
            if len(METHODS) == 4 or len(GROUPS) == 4:
                figsize = (15, 4)
            if len(METHODS) == 5 or len(GROUPS) == 5:
                figsize = (18, 6)
                if len(METHODS) == 5 and len(GROUPS) == 5:
                    figsize = (21, 6)
            if len(METHODS) == 6 or len(GROUPS) == 6:
                figsize = (21, 6)

        ## RSS paper version
        bar_width = 0.3
        if len(METHODS) == 3 and len(GROUPS) == 7:
            if platform.system() == 'Darwin':
                figsize = (14, 3.5)
            else:
                figsize = (16, 4.5)
            bar_width = 0.27
        if len(METHODS) == 3 and len(GROUPS) == 2:
            if platform.system() == 'Darwin':
                figsize = (3.5, 3.5)
            else:
                figsize = (5, 5)
        if len(GROUPS) == 1:
            figsize = (5, 5)

        fig, axs = plt.subplots(1, len(groups), figsize=figsize)
        if len(groups) == 1:
            axs = [axs]

        scale = 0.4
        ll = [x for x in range(len(METHODS))]
        for i in range(len(groups)):
            mean = [means[method][i] for method in METHODS]
            overhead = [means_overhead[method][i] for method in METHODS]
            fd_time = [means_fd_time[method][i] for method in METHODS]
            std = [stds[method][i] for method in METHODS]
            count = [counts[method][i] for method in METHODS]
            miss = [missing[method][i] for method in METHODS]
            agm = [argmaxs[method][i] for method in METHODS]
            mx = [maxs[method][i] for method in METHODS]
            xx = points_x[i]
            yy = points_y[i]

            ## for animation
            if ANIMATION != 0:

                if ANIMATION in [1]:
                    mean[2] = 0
                    std[2] = 0
                    overhead[2] = 0

                    mean[1] = 0
                    std[1] = 0
                    overhead[1] = 0

                    xx_new = []
                    yy_new = []
                    for mm in range(len(xx)):
                        if xx[mm] == 0:
                            xx_new.append(xx[mm])
                            yy_new.append(yy[mm])
                    xx = xx_new
                    yy = yy_new

                elif ANIMATION in [2]:
                    step = 12
                    total_steps = 12

                    mean[1] = 0
                    std[1] = 0
                    overhead[1] = 0

                    xx_new = []
                    yy_new = []
                    for mm in range(len(xx)):
                        if xx[mm] in [0, 2]:
                            xx_new.append(xx[mm])
                            yy_new.append(yy[mm])
                        else:
                            for run_dir, line in lines[groups[i]].items():
                                if len(line['y']) < 3:
                                    continue
                                if line['y'][1] == yy[mm]:
                                    xxx = step / total_steps * xx[mm]
                                    yyy = line['y'][0] + (line['y'][1]-line['y'][0]) * step / total_steps
                                    xx_new.append(xxx)
                                    yy_new.append(yyy)

                    xx = xx_new
                    yy = yy_new

                if VIOLIN:
                    import pandas as pd
                    from matplotlib.collections import PolyCollection

                    scale = 1
                    df = pd.DataFrame({'x': xx, 'y': yy})
                    my_pal = {i: colors[i] for i in range(len(METHODS))}
                    ax = sns.violinplot(data=df, x="x", y="y", bw=.4, ax=axs[i], scale="width", palette=my_pal)
                    axs[i].set(xlabel=None, ylabel=None)
                    for violin in ax.collections[::2]:
                        violin.set_alpha(0.4)

                    for art in axs[i].get_children():
                        if isinstance(art, PolyCollection):
                            # color = hex_to_rgb()
                            art.set_edgecolor(art.get_facecolor()[0])

                else:
                    ll = [x * scale for x in range(len(METHODS))]

                    bar_kwargs = dict(color=colors, alpha=0.4, label=METHODS)
                    overhead_kwargs = dict(color='#000000', alpha=0.4, bottom=mean)

                    """ fixed cost """
                    if not FPC:
                        # fd_time = FIXED_COST[i]
                        fd_time = sum(fd_time) / len(fd_time)
                        axs[i].bar(ll, fd_time, bar_width,
                                   color=color_dict['p'][1], alpha=0.4)
                        # mean = [mean[i] - fd_time[i] for i in range(len(mean))]
                        mean = [m - fd_time for m in mean]
                        bar_kwargs['bottom'] = fd_time

                    """ refinement cost """
                    axs[i].bar(ll, mean, bar_width, yerr=std,
                               error_kw={'ecolor': colors_darker}, **bar_kwargs)

                    """ overhead cost """
                    if not FPC:
                        axs[i].bar(ll, overhead, bar_width, **overhead_kwargs)
                        print(f"task: {groups[i]} overhead: {[round(oo, 5) for oo in overhead]}")
                    else:
                        from matplotlib.ticker import MaxNLocator
                        axs[i].yaxis.set_major_locator(MaxNLocator(integer=True))

                ## data points
                cc = []
                for k in xx:
                    if isinstance(k, int):
                        cc.append(colors[k])
                    else:
                        cc.append(gradients[step-1])
                # cc = [colors[k] for k in xx]
                xxx = [x * scale for x in xx]
                s = 20
                if FPC:
                    ss = []
                    for j in range(len(METHODS)):
                        yy_method = [yy[m] for m in range(len(yy)) if xx[m] == j]
                        unique, num = np.unique(yy_method, return_counts=True)
                        unique = list(unique)
                        sss = [num[unique.index(m)] for m in yy_method]
                        if len(sss) == 0:
                            ss.extend([s]*len(yy_method))
                            continue
                        sss_min = min(sss)
                        # sss = [np.log(ssss/sss_min + 1)*s for ssss in sss]
                        # sss = [np.sqrt(ssss/sss_min + 1)*s for ssss in sss]
                        sss = [(ssss/sss_min + 1)*s/5 for ssss in sss]
                        ss.extend(sss)
                    s = np.asarray(ss)

                axs[i].scatter(xxx, yy, s=s, color=cc, alpha=0.7)

                """ line connections """
                if DEBUG_LINES:
                    for run_dir, line in lines[groups[i]].items():
                        if len(line['y']) < 2:
                            continue
                        color = None ## color_dict['gray'][0]
                        alpha = 0.2
                        if line['y'][0] != 0 and line['y'][1] != 0:
                            if line['y'][0] / line['y'][1] > 1.1:
                                color = color_dict['g'][0]
                            elif line['y'][1] / line['y'][0] > 1.1:
                                color = color_dict['r'][0]
                                print(run_dir, round(line['y'][1], 2))

                        if color is not None:
                            axs[i].plot([n*scale for n in line['x']], line['y'], color=color, alpha=alpha)

                for j in range(len(METHODS)):

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
                    if not PAPER_VERSION:
                        axs[i].annotate(agm[j],  # text
                                    (j*scale, mx[j]),  # points location to label
                                    textcoords="offset points",
                                    xytext=(0, 6),  # distance between the points and label
                                    ha='center',
                                    fontsize=10)

            else:
                ll = [x * scale for x in range(len(METHODS))]
                axs[i].bar(ll, [0]*len(ll), bar_width, label=METHODS)

            if GROUPS[-1].startswith('ss'):
                axs[i].set_ylim([0, 800 if not FPC else 15])
            else:
                axs[i].set_ylim([0, max(mx)*1.1])
            axs[i].tick_params(axis='x', which='major') ## , pad=28
            axs[i].tick_params(axis='y', which='major', pad=-4, rotation=45)

            if PAPER_VERSION: ##  and False
                axs[i].set_title(labels[i], fontsize=11, y=-0.2)
            else:
                axs[i].set_title(f"{labels[i]} ({len(get_rundirs(groups[i]))})", fontsize=11, y=-0.24)

            axs[i].set_xticks(ll)
            axs[i].set_xticklabels(METHOD_NAMES, fontsize=10) ## , y=-0.25 , rotation=45

        ylabel = 'Number of FP Skeletons' if FPC else 'Planning time (sec)'
        axs[0].set_ylabel(ylabel, fontsize=12)
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(METHODS))]

        if not PAPER_VERSION:
            plt.subplots_adjust(hspace=0.55, left=0.1, right=0.95, top=0.8, bottom=0.2)
            # fig.suptitle(title, fontsize=16, y=0.96) ##  + dt
            fig.suptitle('.', fontsize=16, y=0.96) ##  + dt
            fig.legend(handles, METHOD_NAMES, ncol=len(METHODS), fontsize=11,
                       loc='upper center', bbox_to_anchor=(0.5, 0.97))
        else:
            plt.subplots_adjust(hspace=0.55, left=0.05, right=0.99, top=0.85, bottom=0.15)
            fig.legend(handles, METHOD_NAMES, ncol=len(METHODS), fontsize=11,
                       loc='upper center', bbox_to_anchor=(0.5, 0.97))

    # if 'pvt-task' in means and len(means['pvt-task']) > 0 and len(means['None']) > 0:
    #     print([round(means['pvt-task'][i] / means['None'][i], 2) for i in range(len(groups))])

    if PAPER_VERSION: ##  and False
        file_name = 'evaluation' if GROUPS[-1].startswith('tt') else 'evaluation_geometry'
        if ANIMATION == 2:
            file_name += f'_{step}'
        if FPC:
            file_name += '_fpc'
        if platform.system() == 'Darwin':
            plt.savefig(f'{file_name}.png', bbox_inches='tight', dpi=300)
            # plt.savefig(f'{file_name}.pdf', bbox_inches='tight')
        else:
            plt.savefig(f'/home/yang/{file_name}.pdf', bbox_inches='tight')
        # plt.savefig(f'/home/yang/{file_name}.png', bbox_inches='tight')
    else:
        plt.tight_layout()
        if update:
            plt.draw()
        else:
            plt.show()


def make_plot():
    # check_run_dirs()

    ## on Ubuntu
    if not platform.system() == 'Darwin':
        data = get_time_data(diverse=True)
        dump_data(data)

    ## on Mac
    if USE_DATA:
        data = load_data()
     ## on Ubuntu
    else:
        data = get_time_data(diverse=True)
        if SAVE_DATA:
            dump_data(data)
            print('saved data, exit')
            sys.exit()

    if not AUTO_REFRESH:
        print('time.time()', int(time.time()))
        plot_bar_chart(data, diverse=True)
        # plot_bar_chart(get_time_data())
    else:
        while True:
            plot_bar_chart(data, diverse=True, update=True)
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


if __name__ == '__main__':
    make_plot()
