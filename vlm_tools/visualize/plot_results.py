import copy
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import json
from os.path import join
from os import listdir
from os.path import basename

from world_builder.colors import RED, ORANGE, YELLOW, CYAN, BLUE, GREEN, PURPLE
from vlm_tools.vlm_utils import RESULTS_DATA_DIR, RESULTS_DIR
from vlm_tools import get_kwargs_from_cache_path

final_version = True

A0 = r'Action $(\mathregular{N_{reprompt} = 0})$'
A1 = r'Action $(\mathregular{N = 1})$'
A2 = r'Action $(\mathregular{N = 2})$'
S0 = r'Subgoal $(\mathregular{N_{reprompt} = 0})$'
S1 = r'Subgoal $(\mathregular{N = 1})$'
S2 = r'Subgoal $(\mathregular{N = 2})$'
all_methods = [A0, A1, A2, S0, S1, S2]
all_methods_legend = [A0, S0, A1, S1, A2, S2]
# all_methods = ['Action', 'Action+Re', 'Action+Re2', 'Subgoal', 'Subgoal+Re', 'Subgoal+Re2']
method_colors = [RED, ORANGE, YELLOW, PURPLE, BLUE, GREEN]
colors = {all_methods[i]: method_colors[i] for i in range(len(all_methods))}


def get_method_name(f):
    reprompt = 0
    if 'reprompt1' in f:
        reprompt = 1
    elif 'reprompt' in f:
        reprompt = 2

    if 'actions' in f:
        f = [A0, A1, A2][reprompt]
    else:
        f = [S0, S1, S2][reprompt]
    return f

# domains = ['Single Arm', 'Dual Arm']
# metrics = ['success_rate', 'num_proposed_subgoals', 'num_refined_actions', 'subgoal_planning_time']

metrics_performance = ['task_success', 'continuous_success_rate', 'num_proposed_subgoals']  ## , 'task_progress'
metrics_plan = ['num_problems_solved_continuously', 'continuous_plan_len', 'continuous_subgoal_plan_len']  ## 'num_completed_problems',
metrics_stats = ['num_reprompts', 'num_ungrounded_subgoals', 'subgoal_planning_time']
metrics_time = ['total_planning_time', 'continuous_planning_time', 'subgoal_planning_time']  ## , 'wasted_planning_time'
renaming = {
    'continuous_success_rate': 'Task Completion Percentage',
    'num_problems_solved_continuously': '# of Completed Subproblems',
    'num_proposed_subgoals': '# of Predicted Subgoals / Actions',
    'continuous_plan_len': 'Completed Plan Length',
    'continuous_subgoal_plan_len': 'Plan Length of Subproblems',
}
# renaming['task_success'] = 'VLM-TAMP (ours)'
# renaming['continuous_success_rate'] = 'Baseline'

metric_groups = [metrics_performance, metrics_plan]


def get_metrics(debug_mode=False, video_mode='performance'):
    metric_groups_shown = copy.deepcopy(metric_groups)
    if debug_mode:
        metric_groups_shown.append(metrics_stats)

    all_metrics = []
    for group in metric_groups_shown:
        all_metrics.extend(group)

    if video_mode == 'performance':
        all_metrics = all_metrics[:2]
        metric_groups_shown = [all_metrics]
    elif video_mode == 'plan':
        all_metrics = list(renaming.keys())[-2:]
        metric_groups_shown = [all_metrics]

    return metric_groups_shown, all_metrics


def read_data(prefix='run_kitchen_chicken_soup', debug_mode=False, video_mode=None,
              skip_keywords=set(), include_keywords=set()):
    json_files = [f for f in listdir(RESULTS_DATA_DIR) if f.startswith(prefix)]
    elems_by_file = [(f, set(f.replace('.json', '').split('_'))) for f in json_files]
    json_files = [f for f, elems in elems_by_file if len(elems.intersection(skip_keywords)) == 0 \
                  and len(elems.intersection(include_keywords)) == len(include_keywords)]
    all_metrics = get_metrics(debug_mode, video_mode)[1]
    all_data = {m: {method: defaultdict(list) for method in all_methods} for m in all_metrics}

    for f in json_files:
        data = json.load(open(join(RESULTS_DATA_DIR, f), 'r'))
        method_name = get_method_name(f)
        exp_name = f.replace('.json', '')
        args = get_kwargs_from_cache_path(exp_name)

        difficulty = args['difficulty']
        dual_arm = 'Dual Arm' if args['dual_arm'] else 'Single Arm'
        difficulty = ['Easy', 'More Obstacles'][difficulty]
        domain_name = f'{difficulty}\n{dual_arm}'

        for k, v in data.items():
            if k in all_metrics:
                all_data[k][method_name][domain_name] = data[k]
    return all_data


def generate_plot(data, show=False, one_plot_per_domain=False, debug_mode=True, video_mode=None,
                  file_name='result', file_type='png'):
    """
    debug mode shows three rows
    """
    methods = list(list(data.values())[0].keys())
    domains = sorted(list(list(list(data.values())[0].values())[0].keys()))
    bar_width = 0.04
    group_gap = 0.02
    if not one_plot_per_domain:
        bar_width /= len(domains)
        domains = [domains]

    metric_groups_shown, all_metrics_shown = get_metrics(debug_mode, video_mode)
    metrics_max_given = {k: 1 for k in metrics_performance[:2]}
    metrics_max_given['num_completed_problems'] = 'num_proposed_subgoals'
    metrics_max = {metric: -np.inf for metric in all_metrics_shown}
    # for metrics, name in [
    #     (metrics_performance, 'performance'), (metrics_plan, 'plan'), (metrics_time, 'time')
    # ]:
    for name in domains:
        # Set up the subplots
        # fig, axes = plt.subplots(len(domains), len(metrics), figsize=(18, 8))
        # fig, axes = plt.subplots(3, len(all_metrics) // 3, figsize=(18, 12))  ## 3 plots
        # fig, axes = plt.subplots(3, len(all_metrics) // 3, figsize=(26, 12))  ## 1 plot for all domains, 3 by 4
        # fig, axes = plt.subplots(3, len(all_metrics) // 3, figsize=(18, 12))  ## 1 plot for all domains, 3 by 3

        figsize = (20, 8) if not debug_mode else (22, 12)
        num_columns = len(metrics_performance)
        if video_mode is not None:
            figsize = (14, 6)
            num_columns = 2
        fig, axes = plt.subplots(len(all_metrics_shown) // num_columns, num_columns, figsize=figsize)  ## six methods

        # Helper function to create scatter and bar plots
        def create_subplot(ax, data, title, ylabel, metric, dx=0.0):
            def get_xticklabel(method, count):
                if 'Re' in method:
                    return f'{method} ({count})'.replace('+', '\n+')
                return f'{method}\n({count})'

            means = [np.mean(data[method]) for method in methods]
            counts = []
            y_all = []
            used_colors = []
            for i, method in enumerate(methods):
                y = data[method]
                y_all.extend(y)
                if one_plot_per_domain:
                    x = np.full(len(y), i)
                else:
                    x = (i * bar_width * 1.1) + dx
                    x = np.full(len(y), x)

                color = colors[method]
                used_colors.append(color)
                alpha = 0.2 if debug_mode else 0.5
                ax.scatter(x, y, label=f'{method} runs', color=color, alpha=alpha)
                counts.append(len(y))

            y_max = np.max(y_all)
            if y_max > metrics_max[metric]:
                metrics_max[metric] = y_max
            this_y_max = metrics_max[metric]
            if metric in metrics_max_given:
                this_y_max = metrics_max_given[metric]
                if isinstance(this_y_max, str):
                    this_y_max = metrics_max[this_y_max]

            # color = 'grey'
            if one_plot_per_domain:
                x = np.arange(len(methods)) + dx
                ax.set_xticks(np.arange(len(methods)))
                ax.set_xticklabels([get_xticklabel(methods[i], counts[i]) for i in range(len(methods))], fontsize=12)
            else:
                x = np.arange(len(methods))
                x = (x * bar_width * 1.1) + dx

            bars = ax.bar(x, means, alpha=0.25, color=used_colors, label='Mean', width=bar_width)
            if debug_mode:
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.annotate(f'{round(height, 2)}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                                textcoords="offset points", ha='center', va='bottom', fontsize=9, color=used_colors[i])

            ax.tick_params(axis='y', labelsize=14)
            # ax.set_ylabel(ylabel, fontsize=13)

            hline_kwargs = dict(lw=1, linestyle=':', color='#bdc3c7', alpha=0.5)
            if 'success' in metric:
                ax.axhline(y=1, **hline_kwargs)

            ## set y to log scale
            if 'planning time' in ylabel:
                ax.set_yscale('log')
                title += ' (log)'

            index = 'abcdefghi'[all_metrics_shown.index(metric)]
            if video_mode is None:
                title = f"{index}) " + title
            # ax.set_title(title, fontsize=16)
            ax.set_title(title, fontsize=22, pad=14)
            ax.set_ylim(0, this_y_max * 1.2)
            if metric in ['continuous_subgoal_plan_len', 'task_success', 'continuous_success_rate']:
                if metric == 'continuous_subgoal_plan_len':
                    yticks = np.arange(6) * 2
                else:
                    yticks = np.arange(6) * 0.2
                ax.set_yticks(yticks)
                for y in yticks:
                    ax.axhline(y=y, **hline_kwargs)

        # # Create plots for each domain and each metric
        # for col, metric in enumerate(metrics):
        #     for row, domain in enumerate(domains):
        #         y = {m: data[metric][m][domain] for m in methods}  ## data[metric][domain]
        #         y_label = metric.replace('_', ' ')  ## .capitalize()
        #         create_subplot(axes[row, col], y, f'{y_label} in {domain}', y_label, metric)

        # Create plots for each domain and each metric

        for row, metrics in enumerate(metric_groups_shown):
            for col, metric in enumerate(metrics):
                ax = axes[row, col] if video_mode is None else axes[col]
                if metric in renaming:
                    y_label = renaming[metric]
                else:
                    y_label = ' '.join(s.capitalize() for s in metric.split('_'))
                show_counts = metric not in ['continuous_subgoal_plan_len'] and debug_mode
                if one_plot_per_domain:
                    domain_name = name
                    y = {m: data[metric][m][domain_name] for m in methods}
                    create_subplot(ax, y, y_label, y_label, metric)
                else:
                    domains = name
                    x_positions = []
                    x_ticklabels = []
                    all_average_increases = []
                    for i, domain_name in enumerate(domains):
                        dx = (bar_width * len(methods) + group_gap) * 1.1 * i
                        y = {m: data[metric][m][domain_name] for m in methods}
                        x_ticklabel = domain_name.replace('_dual_', '\ndual_').replace('_single_', '\nsingle_')
                        if show_counts:
                            # counts = '(' + ', '.join([f"{len(y[method])}" for method in methods]) + ')'
                            counts = ', '.join([f"{len(y[method])}" for method in methods])
                            remaining = ', '.join([f"{30 - len(y[method])}" for method in methods])
                            x_ticklabel += '\n' + counts + '\n' + remaining
                        create_subplot(ax, y, y_label, y_label, metric, dx=dx)
                        x_positions.append(dx + (bar_width * 1.1) * 2.5)
                        x_ticklabels.append(x_ticklabel)

                        means = [np.mean(y[method]) for method in methods]
                        all_average_increases.append(round(means[-1] / means[-3] - 1, 2))
                    if row == 0 and col in [0, 1]:
                        print('\t', metric, '\t', sorted(all_average_increases))

                    ax.set_xticks(x_positions)
                    fontsize = 9 if show_counts else 12
                    ax.set_xticklabels(x_ticklabels, fontsize=fontsize)

        # Create a single legend

        # Adjust layout
        plt.tight_layout()
        if one_plot_per_domain:
            handles, labels = axes[0, 0].get_legend_handles_labels()
            # fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.00), fontsize=16)
            fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 0.94), fontsize=16)
            # plt.subplots_adjust(top=0.9)  # Adjust the top to make space for the legend
            plt.subplots_adjust(top=0.86, hspace=0.4)  # Adjust the top to make space for the legend
            plt.suptitle(f"task: {name}", fontsize=26)
            file_name = name
        else:
            legend_kwargs = dict(loc='upper center', bbox_to_anchor=(0.5, 1), fontsize=18, markerscale=2)
            if video_mode is None:
                top = 0.85 if not debug_mode else 0.9
                hspace = 0.5 if not debug_mode else 0.5
                ncol = 6
                leg = fig.legend(methods, ncol=ncol, **legend_kwargs)
            else:
                top = 0.68
                hspace = 0.5
                ncol = 3
                handles, labels = ax.get_legend_handles_labels()
                new_methods = copy.copy(all_methods_legend)
                new_handles = [handles[methods.index(new_methods[i])] for i in range(6)]
                leg = fig.legend(new_handles, new_methods, ncol=ncol, **legend_kwargs)

            for i in range(len(leg.legendHandles)):
                leg.legendHandles[i].set_alpha(0.8)
                # leg.legendHandles[i].set_color(method_colors[i])

            plt.subplots_adjust(top=top, hspace=hspace)  # Adjust the top to make space for the legend

        if debug_mode:
            file_name += '_debug'

        if show:
            plt.show()
        else:
            file_path = join(RESULTS_DIR, f'{file_name}.{file_type}')
            print(f'generated image {file_path}')
            plt.savefig(file_path, dpi=200)


def generate_plots():
    ## for paper figures and debugging
    output_kwargs = dict(file_name='result', video_mode=None)

    ## for video submission
    output_kwargs = dict(file_name='performance', video_mode='performance')
    output_kwargs = dict(file_name='performance_title', video_mode='performance')
    output_kwargs = dict(file_name='plan', video_mode='plan')

    prefix = 'run_kitchen_chicken_soup'
    skip_keywords = set()  ## set(['reprompt'])
    include_keywords = set()  ## set(['meraki'])
    kwargs = dict(skip_keywords=skip_keywords, include_keywords=include_keywords)
    data = read_data(prefix, debug_mode=True, video_mode=output_kwargs['video_mode'], **kwargs)

    file_types = ['png']
    debug_modes = [False]
    if output_kwargs['video_mode'] is None:
        if final_version:
            file_types.append('pdf')
        debug_modes.append(True)

    for debug_mode in debug_modes:
        for file_type in file_types:
            generate_plot(data, debug_mode=debug_mode, file_type=file_type, **output_kwargs)


if __name__ == '__main__':
    generate_plots()
