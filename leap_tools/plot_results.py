from os.path import join, abspath, dirname, isdir, basename
from os import listdir
import csv
import json
import numpy as np
from pprint import pprint
import copy
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

EXPERIMENT_DIR = abspath(join(dirname(__file__), '..', 'experiments'))
OUTPUT_DIR = abspath(join(dirname(__file__), '..', 'visuals', 'results'))

field_names = ['domain', 'hierarchy', 'trial', 'planning_time_first', 'planning_time',
               'sequencing_time', 'sampling_time', 'preimage_time', 'plan_len']
marker_colors = ['rgba(111,114,249,0.3)', 'rgba(236,96,80,0.3)', 'rgba(48,208,154,0.3)']

yellow = '\033[93m'
green = '\033[92m'
red = '\033[91m'
blue = '\033[94m'
pink = '\033[95m'
gray = '\x1b[0;37m'
black = "\x1b[0m"


def get_plan_len(json_name, verbose=False):
    plan_len = 0
    for call in json.load(open(json_name, 'r'))[:-1]:
        for line in call['plan']:
            action = line[line.index("Action(name='") + 13:line.index("', args=(")]
            if '--no-' not in action:
                if verbose: print('count... \t', action)
                plan_len += 1
            elif verbose:
                print('not count... \t', action)
    return plan_len


def get_data(exp_dir, hierarchies, timeout=500, timeout_first=500, timeout_plan_len=70):
    all_lines = []
    stats = {}
    ## custom csv adapter because the row length aren't the same
    for hierarchy in hierarchies:
        csv_name = join(exp_dir, f"{hierarchy}.csv")
        with open(csv_name, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            lines = []
            line_count = 0
            for row in csv_reader:

                data = [v for v in row.values() if isinstance(v, str) and len(v) != 0]
                if 'average' in data[0]:
                    continue
                data = data[:2] + data[-3:]
                data[-1] = int(data[-1]) ## plan_len

                data[1:-1] = [round(float(d), 3) for d in data[1:-1]]
                if data[-3] == 99999:
                    continue
                    data[-3] = timeout
                    data[-4] = timeout_first
                    data[-1] = timeout_plan_len
                    print(data)

                if f'_{hierarchy}' in data[0]:
                    run_id = data[0].replace(f'_{hierarchy}', f'/{hierarchy}')
                else:
                    run_id = join(data[0], hierarchy)
                time_sequencing, time_sampling = extract_log_txt_statistics(join(exp_dir, run_id, 'log.txt'))[1]
                data = data[:-2] + [time_sequencing, time_sampling] + data[-2:]

                ## -- add in the problem name and exp_name,
                ##     i.e. name = 'make_omelette', exp_name = 'hpn'
                name = exp_dir
                if '/' in name:
                    name = name[name.rfind('/') + 1:]
                name = name.replace('test_kitchen_', '')

                data = [name, hierarchy] + data
                lines.append(data)
                line_count += 1
            all_lines += lines

            ## by hierarchical
            key = lines[0][1]
            if key not in stats:
                stats[key] = {}
            domain = lines[0][0]
            domain = domain.replace('test_kitchen_', '')
            stats[key][domain] = {}

            for i in range(len(field_names)):
                if '_' in field_names[i]:
                    data = [r[i] for r in lines if r[-4] != timeout and r[-3] != timeout]
                    print(field_names[i], data)
                    mean = round(np.mean(np.asarray(data)), 3)
                    std = round(np.std(np.asarray(data)), 3)
                    count = len(data)
                    stats[key][domain][field_names[i]] = {'mean': mean, 'std': std, 'count': count}
            print('\n', key, domain, 'count =', count, '\n')

    return all_lines, stats


def plot_dots_bar(df, stats, y, fig_name=None, fig_size=(600, 600)):
    # --- Create Scatter plot
    kwargs = dict(jitter=1, opacity=0.8, marker_size=10, marker_line_width=1,
                  marker_line_color='rgba(0,0,0,0.8)', showlegend=False) # marker_color='rgba(0,0,0,0.8)',
    fig = px.strip(df, x='domain', y=y, color="hierarchy", hover_data=["trial", y]).update_traces(**kwargs)

    # --- Create bar graphs with error bars
    index = 0
    for hierarchy, data in stats.items():
        domains = list(data.keys())
        means = [data[d][y]['mean'] for d in domains]
        stds = [data[d][y]['std'] for d in domains]
        fig.add_bar(
            name=hierarchy, x=domains, y=means, showlegend=True,
            marker_color=marker_colors[index], marker_line_color='rgba(0,0,0,1)', marker_line_width=1, opacity=0.8,
            error_y=dict(type='data', array=stds, color='rgba(0,0,0,1)', thickness=1.5, width=10)
        )
        index += 1

    # Customization of layout and traces
    fig.update_layout(template='simple_white', title=y, yaxis_title=y, barmode='group',
                      dragmode='drawrect', font_size=15, hoverlabel_namelength=-1, showlegend=True,

                      ## https://plotly.com/python/legend/#legend-positioning
                      legend=dict(  ## x=0.6, y=1,
                          title_text='',
                          traceorder="normal",
                          bordercolor="Black",
                          borderwidth=0,
                          font=dict(size=15, color="black"),
                          orientation="h",
                          yanchor="bottom",
                          y=1,
                          xanchor="right",
                          x=1
                      ),
                      height=fig_size[0], width=fig_size[1],  # 960,
                      )
    # fig.update_traces(marker_line_color='rgba(0,0,0,0.8)', marker_line_width=1.5, textfont_size=12, opacity=0.8)

    #  Customization of x-axis
    fig.update_xaxes(title='')

    if fig_name is not None:
        output_file = join(OUTPUT_DIR, fig_name)
        fig.write_image(output_file, scale=2)
        fig.write_html(output_file.replace('.png', '.html'))


def test_compare_flat_hpn(group_names):
    hierarchies = ['original', 'hpn']
    timeout = 200
    timeout_first = 200
    timeout_plan_len = 20
    fig_size = (500, 400)

    stats = {k: {} for k in hierarchies}
    all_data = []
    for group_name in group_names:
        exp_dir = join(EXPERIMENT_DIR, group_name)
        data = get_data(exp_dir, hierarchies, timeout=timeout, timeout_first=timeout_first,
                        timeout_plan_len=timeout_plan_len)
        all_data += data[0]
        for hierarchy in hierarchies:
            stats[hierarchy].update(data[1][hierarchy])
        # pprint(stats)

    df = pd.DataFrame(all_data, columns=field_names)
    for y in field_names:
        if '_' not in y: continue
        plot_dots_bar(df, stats, y=y, fig_name=join('plots', f"{y}.png"), fig_size=fig_size)

    print('\n\n/home/yang/Documents/cognitive-architectures/bullet/visuals/results/results_hpn.html')


def compare_multiple_hierarchies():
    stats = {}
    all_data = []

    exp_dir = join('../..', '0404-egg-levels')
    exp_names = ['flat.csv', 'one_level.csv', 'two_levels.csv']
    timeout = 500
    timeout_plan_len = 100
    fig_size = (500, 400)

    data = get_data(exp_dir, exp_names, timeout=timeout, timeout_plan_len=timeout_plan_len)
    all_data += data[0]
    stats.update(data[1])
    # pprint(stats)

    df = pd.DataFrame(all_data, columns=field_names)
    for y in field_names:
        if 'plan' not in y:
            continue
        plot_dots_bar(df, stats, y=y, fig_name=join('plots_multiple', f"plot_{y}.png"), fig_size=fig_size)


###################################################################################

HTML = """<!DOCTYPE html>
<html>
<head>
<link rel="stylesheet" href="view_plan.css">
</head>
<body>
{body}
</body>
</html>
"""


def save_html(text, file_name):
    with open(file_name, 'w') as f:
        f.write(HTML.format(body=text))
    print('\nSaved plan html', file_name)


def abstract(indent, action, goals, color):
    indent = ''.join(['&nbsp;'] * indent)
    return f"<details><summary><text style=color:{color}>{indent}{action}</text></summary>{indent}{goals}</details>"


def html_string(indent, action, preimage=None, color='black', fontsize=14):
    indent_action = ''.join(['&nbsp;'] * indent)
    indent_preimage_title = ''.join(['&nbsp;'] * (indent + 3))
    indent_preimage = ''.join(['&nbsp;'] * (indent + 6))
    action = action.replace("'", '')
    if preimage is not None:
        preimage = (indent_preimage_title + 'preimage: <br>' + indent_preimage +
                    f"<br>{indent_preimage}".join([g.replace("'", '') for g in preimage]))
        return (f'\n<details>\n\t<summary>\n\t\t<text style="color:{color}; font-size: {fontsize}px">'
                f'\n\t\t\t{indent_action}{action}'
                f'\n\t\t</text>\n\t</summary>{preimage}\n</details>')
    return f"\n<text style=color:{color}>\n\t{indent_action}{action}</text><br>"


def print_hierarchical_plan(json_name, verbose=False):
    curr_indent = 0
    indents = {}  ## {0: [], 1: []}
    plan_len = 0
    index = 0
    indent_str = '   '
    text = ''
    summaries = json.load(open(json_name, 'r'))[:-1]
    for j in range(len(summaries)):
        call = summaries[j]
        actions = [line[line.index("Action(name='") + 13:-1] for line in call['plan']]  ## line.index("', args=(")

        indent = ''.join([indent_str] * curr_indent)
        indents[curr_indent] = actions
        found_abstract = False
        while not found_abstract:
            print_actions = copy.deepcopy(indents[curr_indent])
            for i in range(len(print_actions)):
                action = print_actions[i]

                if '--no-' in action:
                    #                     print(gray, indent+action)
                    preimage = summaries[j + 1]['goal']
                    text += html_string(len(indent), action, preimage=preimage, color='red', fontsize=18)
                    found_abstract = True
                    indents[curr_indent] = print_actions[i + 1:]
                    curr_indent += 1
                    break
                else:
                    plan_len += 1
                    #                     print(red, indent+f"[{plan_len}] "+action)
                    text += html_string(len(indent), f"[{plan_len}] " + action, color='gray', fontsize=13)

            if not found_abstract:
                indents.pop(curr_indent)
                curr_indent -= 1
                curr_indent = max([0, curr_indent])
                indent = ''.join([indent_str] * curr_indent)
            if len(indents) == 0:
                break

        index += 1
    return text


def test_make_html_from_plan(exp_name='kitchen_food', run_name='230330_173112_hpn'):
    output_file = join(OUTPUT_DIR, f'view_plan_{exp_name}.html')
    json_name = join(EXPERIMENT_DIR, exp_name, run_name, 'time.json')
    text = print_hierarchical_plan(json_name)
    save_html(text, output_file)


def test_html_format():
    goals = ['atpose((4, p2039=(2.18, 0, 0.95, 0)))',
             'handempty((left,))',
             'cleaned((2,))',
             'not(((atrajobstacle, c456=t(7,52), 4),))']

    test = html_string(4, 'clean--no-on, args=(2, 5)', goals, 'gray')
    test += html_string(8, 'action1', goals, 'red')
    save_html(test, join(OUTPUT_DIR, 'view_plan_test.html'))


###################################################################################


def plotly_comparison_table_hierarchy(file_name="compute.xlsx"):
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd

    df = pd.read_excel(join(EXPERIMENT_DIR, file_name),
                       sheet_name='sheet_1', header=0, usecols="A,B,D", na_values=['NA'])

    # Group and calculate the mean and sem for planning time (s) during Original and HPN domain
    mean_domain = df.groupby(['experiment', 'domain']).mean()
    sem_domain = df.groupby(['experiment', 'domain']).sem()

    # Extract mean from the planning time (s) using the Original domain
    Original_mean_A = mean_domain['planning time (s)'].Tables['Original']
    Original_mean_C = mean_domain['planning time (s)'].Kitchen['Original']

    # Extract mean from the planning time (s) using the HPN domain
    HPN_mean_A = mean_domain['planning time (s)'].Tables['HPN']
    HPN_mean_C = mean_domain['planning time (s)'].Kitchen['HPN']

    # Extract mean from the planning time (s) using the HPN domain before 1st execution
    HPN1_mean_A = mean_domain['planning time (s)'].Tables['HPN (1st act)']
    HPN1_mean_C = mean_domain['planning time (s)'].Kitchen['HPN (1st act)']

    # Extract sem from the planning time (s) using the Original domain
    Original_sem_A = sem_domain['planning time (s)'].Tables['Original']
    Original_sem_C = sem_domain['planning time (s)'].Kitchen['Original']

    # Extract sem from the planning time (s) using the HPN domain
    HPN_sem_A = sem_domain['planning time (s)'].Tables['HPN']
    HPN_sem_C = sem_domain['planning time (s)'].Kitchen['HPN']

    # Extract sem from the planning time (s) using the HPN domain before 1st execution
    HPN1_sem_A = sem_domain['planning time (s)'].Tables['HPN (1st act)']
    HPN1_sem_C = sem_domain['planning time (s)'].Kitchen['HPN (1st act)']

    # Create Scatter plot
    fig = px.strip(df, x='experiment', y='planning time (s)', color="domain").update_traces(jitter=1,
                                                                                            opacity=0.8,
                                                                                            marker_size=10,
                                                                                            marker_line_width=1,
                                                                                            marker_line_color='rgba(0,0,0,0.8)',
                                                                                            # marker_color='rgba(0,0,0,0.8)',
                                                                                            showlegend=False)

    # Create bar graphs with error bars
    fig.add_bar(
        name='Flat',
        marker_color='rgba(0,0,0,0.5)', marker_line_color='rgba(0,0,0,1)', marker_line_width=1, opacity=0.8,
        x=['Tables', 'Kitchen'], y=[Original_mean_A, Original_mean_C], showlegend=True,
        error_y=dict(type='data', array=[Original_sem_A, Original_sem_C],
                     color='rgba(0,0,0,1)', thickness=1.5, width=10)
    )

    fig.add_bar(
        name='HPN',
        marker_color='rgba(255,255,0,0.5)', marker_line_color='rgba(0,0,0,1)', marker_line_width=1, opacity=0.8,
        x=['Tables', 'Kitchen'], y=[HPN_mean_A, HPN_mean_C], showlegend=True,
        error_y=dict(type='data', array=[HPN_sem_A, HPN_sem_C],
                     color='rgba(0,0,0,1)', thickness=1.5, width=10)
    )

    fig.add_bar(
        name='HPN (1st act)',
        marker_color='rgba(255,255,0,0.5)', marker_line_color='rgba(0,0,0,1)', marker_line_width=1, opacity=0.8,
        x=['Tables', 'Kitchen'], y=[HPN1_mean_A, HPN1_mean_C], showlegend=True,
        error_y=dict(type='data', array=[HPN1_sem_A, HPN1_sem_C],
                     color='rgba(0,0,0,1)', thickness=1.5, width=10)
    )

    # Customization of layout and traces
    fig.update_layout(template='simple_white', title='', yaxis_title='planning time (s)', barmode='group',
                      dragmode='drawrect', font_size=15, hoverlabel_namelength=-1, showlegend=True,
                      legend=dict(x=0.6, y=1,
                                  title_text='',
                                  traceorder="normal",
                                  bordercolor="Black",
                                  borderwidth=0,
                                  font=dict(size=15, color="black"),
                                  ),
                      height=600, width=600,  # 960,
                      )
    # fig.update_traces(marker_line_color='rgba(0,0,0,0.8)', marker_line_width=1.5, textfont_size=12, opacity=0.8)

    #  Customization of x-axis
    fig.update_xaxes(title='')

    print(df.groupby(['experiment', 'domain']).mean())
    print(df.groupby(['experiment', 'domain']).sem())

    fig.show()
    # fig.write_image(join(OUTPUT_DIR, "time_two.png"), scale=2)


def plotly_comparison_table_object(file_name="objects.xlsx"):
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd

    df = pd.read_excel(join(EXPERIMENT_DIR, file_name),
                       sheet_name='sheet_1', header=0, usecols="A,B,D", na_values=['NA']
                       )

    # Group and calculate the mean and sem for planning time (s) during Original and HPN domain
    mean_domain = df.groupby(['experiment', 'domain']).mean()
    sem_domain = df.groupby(['experiment', 'domain']).sem()

    # Extract mean from the planning time (s) using the Original domain
    Original_mean_A = mean_domain['planning time (s)'].ShortMinimum['Original']
    Original_mean_B = mean_domain['planning time (s)'].ShortExtra['Original']
    Original_mean_C = mean_domain['planning time (s)'].LongMinimum['Original']

    # Extract sem from the planning time (s) using the Original domain
    Original_sem_A = sem_domain['planning time (s)'].ShortMinimum['Original']
    Original_sem_B = sem_domain['planning time (s)'].ShortExtra['Original']
    Original_sem_C = sem_domain['planning time (s)'].LongMinimum['Original']

    # Create Scatter plot
    fig = px.strip(df, x='experiment', y='planning time (s)', color="domain").update_traces(jitter=1,
                                                                                            opacity=0.8,
                                                                                            marker_size=10,
                                                                                            marker_line_width=1,
                                                                                            marker_line_color='rgba(0,0,0,0.8)',
                                                                                            # marker_color='rgba(0,0,0,0.8)',
                                                                                            showlegend=False)

    # Create bar graphs with error bars
    fig.add_bar(
        name='Flat',
        marker_color='rgba(0,0,0,0.5)', marker_line_color='rgba(0,0,0,1)', marker_line_width=1, opacity=0.8,
        x=['ShortMinimum', 'ShortExtra', 'LongMinimum'],
        y=[Original_mean_A, Original_mean_B, Original_mean_C], showlegend=True,
        error_y=dict(type='data', array=[Original_sem_A, Original_sem_B, Original_sem_C],
                     color='rgba(0,0,0,1)', thickness=1.5, width=10)
    )

    # Customization of layout and traces
    fig.update_layout(template='simple_white', title='', yaxis_title='planning time (s)', barmode='group',
                      dragmode='drawrect', font_size=15, hoverlabel_namelength=-1, showlegend=False,
                      height=600, width=600,  # 960,
                      )
    # fig.update_traces(marker_line_color='rgba(0,0,0,0.8)', marker_line_width=1.5, textfont_size=12, opacity=0.8)

    #  Customization of x-axis
    fig.update_xaxes(title='')

    print(df.groupby(['experiment', 'domain']).mean())
    print(df.groupby(['experiment', 'domain']).sem())

    fig.show()
    # fig.write_image(join(OUTPUT_DIR, "objects_three.png"), scale=2)


def summarize_stream_statistics(problem_names):
    """ add cvs of stream statistics to the results folder, on the same level as originasl.csv
    """
    exp_names = ['original', 'hpn']
    column_names = ['hierarchy', 'group name', 'run id', 'stream name', 'total count', 'failed count']

    rows = [column_names]
    for problem_name in problem_names:
        exp_dir = join(EXPERIMENT_DIR, problem_name)
        runs = [join(exp_dir, f) for f in listdir(exp_dir) if isdir(join(exp_dir, f))]
        for run in runs:
            group_name = basename(dirname(run)).replace('test_kitchen_', '')
            for exp_name in exp_names:
                exp_run_dir = join(run, exp_name)

                ## get all_streams
                log_txt_file = join(exp_run_dir, 'log.txt')
                all_streams, _ = extract_log_txt_statistics(log_txt_file)

                ## get failed_streams
                failed_streams = {k: 0 for k in all_streams.keys()}
                log_json_files = [f for f in listdir(exp_run_dir) if f.endswith('.json') and f.startswith('log')]
                for log_json_file in log_json_files:
                    with open(join(exp_run_dir, log_json_file), 'r') as f:
                        log_data = json.load(f)
                        for i, log_arr in enumerate(log_data):
                            if i % 2 == 0 or len(log_arr) == 0:
                                continue
                            if isinstance(log_arr, dict):
                                log_arr = log_arr['summary']
                            else:  ## if isinstance(log_arr, list):
                                log_arr = log_arr[0]
                            for k, v in log_arr.items():
                                key = k.replace('(', '').replace(')', '').split(', ')[0]
                                if key not in failed_streams:
                                    failed_streams[key] = 0
                                    all_streams[key] = 0
                                failed_streams[key] += v

                for k, v in all_streams.items():
                    rows.append([exp_name, group_name, basename(run), k, v, failed_streams[k]])

    with open(join(EXPERIMENT_DIR, 'stream_statistics.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def extract_log_txt_statistics(log_txt_file):
    all_streams = {}
    time_sequencing = 0
    time_sampling = 0
    with open(log_txt_file, 'r') as f:
        for line in f.readlines():
            if 'Optimistic streams:' in line:
                line = line.strip().replace('Optimistic streams: Counter(', '').replace(')', '')
                all_streams = eval(line)
            if 'sequencing time:' in line:
                line = line.strip().split('|')
                time_sequencing += float(line[1].replace('sequencing time: ', '').strip())
                time_sampling += float(line[2].replace('sampling time: ', '').strip())
    return all_streams, [time_sequencing, time_sampling]


def plotly_comparison_hierarchy_streams():
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from world_builder.colors import BLUE, DARK_BLUE, ORANGE, DARK_ORANGE

    csv_file = join(EXPERIMENT_DIR, 'stream_statistics.csv')
    df = pd.read_csv(csv_file)

    stream_names = df['stream name'].unique()
    group_names = df['group name'].unique()

    # Define a color map for hierarchy
    bar_colors = {'original': BLUE, 'hpn': ORANGE}
    scatter_colors = {'original': DARK_BLUE, 'hpn': DARK_ORANGE}

    fig = make_subplots(rows=len(stream_names), cols=len(group_names),
                        subplot_titles=[f'{stream}<br>{group}' for stream in stream_names for group in group_names])

    for r, stream in enumerate(stream_names, start=1):
        for c, group in enumerate(group_names, start=1):
            filtered_df = df[(df['stream name'] == stream) & (df['group name'] == group)]
            hierarchies = filtered_df['hierarchy'].unique()

            # Calculate mean and standard deviation of failed count for each hierarchy
            means_failed = filtered_df.groupby('hierarchy')['failed count'].mean()
            std_devs_failed = filtered_df.groupby('hierarchy')['failed count'].std()

            # Calculate average total count for each hierarchy
            avg_total_counts = filtered_df.groupby('hierarchy')['total count'].mean()

            # Plot bar chart for each hierarchy
            for i, hierarchy in enumerate(hierarchies):
                color = bar_colors[hierarchy]
                dark_color = scatter_colors[hierarchy]

                fig.add_trace(go.Bar(
                    x=[hierarchy],
                    y=[means_failed[hierarchy]],
                    marker_color=color,  # Set color based on hierarchy
                    opacity=0.5,
                    error_y=dict(type='data', array=[std_devs_failed[hierarchy]], visible=True)
                ), row=r, col=c)

                fig.add_trace(go.Scatter(
                    x=filtered_df['hierarchy'],
                    y=filtered_df['failed count'],
                    text=filtered_df['run id'],
                    hoverinfo="text+y",
                    mode='markers',
                    marker_color=dark_color,
                    opacity=0.5,
                ), row=r, col=c)

                # # Add horizontal line for average total count for the specific hierarchy
                # fig.add_shape(
                #     type="line",
                #     x0=i-0.4,  # start x-coordinate
                #     x1=i+0.4,  # end x-coordinate
                #     y0=avg_total_counts[hierarchy],
                #     y1=avg_total_counts[hierarchy],
                #     line=dict(color=dark_color, dash="dash"),
                #     row=r, col=c
                # )

    fig.update_layout(title_text="Stream & Group Plots with Failed Counts", height=400 * len(stream_names),
                      showlegend=False)
    fig.show()
    # fig.write_html(csv_file.replace('.csv', '.html'))


if __name__ == '__main__':
    group = 'test_kitchen_dinner'
    groups = [group + suffix for suffix in ['', '_more_plates', '_more_movables']]

    """ make figures of results: total planning, first planning, planning length """
    test_compare_flat_hpn(groups)
    # compare_multiple_hierarchies()

    """ interactive plot to investigate which ones are causing problem_sets """
    summarize_stream_statistics(groups)
    # plotly_comparison_hierarchy_streams()
    # plotly_comparison_hierarchy()
    # plotly_comparison_object()

    """ print out the plans to see """
    # test_html_format()
    # test_make_html_from_plan(exp_name='test_kitchen_dinner', run_name=join('230914_220956', 'hpn'))
