from os.path import join, isdir, isfile
from os import listdir
import copy
import csv
from collections import defaultdict
import json

from vlm_tools.vlm_utils import EXP_DIR, plans_to_tree, export_tree_png, load_vlm_memory, idx_branch_from_subgoal

FAILURE_REASONS = ['translation', 'formulation', 'planning']
DISPLAY_LINK = "http://0.0.0.0:9000/{exp_name}/index.html"

results_html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{tab_name}</title>
    <!--link rel="stylesheet" href="log/style.css">
    <script src="log/scripts.js"></script--> 
    <style>
    {style}
    </style>
</head>
<body>
    <div>
      <div class="table-title">
        <h2>Summary of Experiment Results</h2>
        {summary_image}
      </div>
      <table style="table-layout: auto;">        
        <colgroup></colgroup>
        <thead>
          <tr>
            <th rowspan="2">Method</th>
            <th rowspan="2">Run Count</th>
            
            <th colspan="2">Success Rate</th>
            <th colspan="2">Plan</th>
            <th colspan="3">Time</th>
            <th colspan="1">Failure Analysis</th>
          </tr>
          <tr>
            <th class="th2">Task Success</th>
            <th class="th2">Task Completion Rate</th>
            
            <th class="th2"># Actions</th>
            <th class="th2">Effective # Actions</th>
            
            <th class="th2">Planning Time</th>
            <th class="th2">Query Time</th>
            <th class="th2">Total Run Time</th>
            
            <th class="th2">
                1. Parsing<br>
                2. Infeasibility<br>
                3. Planning
            </th>
          </tr>
        </thead>

        <tbody>{rows}
        </tbody>
      </table>
    </div>

</body>
</html>
"""


runs_row_template = """
          <tr>
            <td>{run_name}</td>
            <td class="tooltipTrigger">{num_subgoals}
                <span class="tooltip">{subgoals}</span>
            </td>
            
            <td class="tooltipTrigger">{num_actions}
                <span class="tooltip">{actions}</span>
            </td>
            
            <td>
              <div class="div-img">
                <img class='zoom_in' src='{planning_tree_path}'>
              </div>
            </td>
            <td>{failure}</td>
          </tr>
"""

runs_html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{tab_name}</title>
    <!--link rel="stylesheet" href="log/style.css">
    <script src="log/scripts.js"></script--> 
    <style>
    {style}
    </style>
</head>
<body>
    <div>
      <div class="table-title">
        <h2>Summary of Experiment Runs for "{exp_name}" ({num} Runs)</h2>
        {summary_image}
      </div>
      <table style="table-layout: auto;">        
        <colgroup></colgroup>
        <thead>
          <tr>
            <th rowspan="3">Run Name</th>
            <th colspan="3">Stats</th>
            <th colspan="1">Failure Analysis</th>
          </tr>
          <tr>
            <th class="th2">Number of Subgoals</th>
            <th class="th2">Number of Actions</th>
            <th class="th2">Planning Tree</th>
            <th class="th2">
                1. Parsing<br>
                2. Infeasibility<br>
                3. Planning
            </th>
          </tr>
        </thead>

        <tbody>{rows}
        </tbody>
      </table>
    </div>

</body>
</html>
"""

llm_response_wrapper = """
"""

## ------------------------------------------------------------------------------------------

included_style = """

.top_image {
    max-height: 1000px;
    width: auto;
    max-width: 100%;
}

.query_image {
    max-height: 200px;
    width: auto;
    max-width: 100%;
}

.table-title {
  border: 1px solid black;
  padding: 20px;
  background-color: #eceeef;
}

table {
  width: 100%;
}

table thead tr:first-child th:not(:first-child) {
  padding: 10px;
}

table thead tr th {
  padding: 7px;
}

th {
    position: sticky;
    top: 0;
    background-color: #fff; /* Prevents header from becoming transparent */
    z-index: 100; /* Ensures the header stays on top of other content */
}

.th2 {
    position: sticky;
    top: 44px;
    background-color: #fff; /* Prevents header from becoming transparent */
    z-index: 100; /* Ensures the header stays on top of other content */
}

table tbody tr:nth-child(even) {
  background-color: #cfd7e5;
}

table tbody tr:nth-child(even) td:first-child {
  background-color: #fff;
}

table tbody tr td {
  padding: 10px;
  max-width: 636px;
  word-wrap: break-word;
}

table th,
table td {
  border: 1px solid black;
}

.heading {
    font-size: 12pt; 
    color: #e74c3c;
    margin-bottom: -9pt;
    display: block;
}

.div-img{
    justify-content: center;
    align-items: center;
    display: flex;
}

.zoom_in {
    transition: transform .2s;
    width: auto;
    height: 50px;
    margin: 0 auto;
    background-color: rgb(173, 173, 237);
    border-radius: 10px;
    border: 1px solid black;
}
.zoom_in:hover{
    transform: scale(20) translate(0px, 25px);
    z-index: 101; /* Ensures the header stays on top of other content */
}

.tooltipTrigger .tooltip {
  display: none;
}
.tooltipTrigger:hover {
  cursor: pointer;
}
.tooltipTrigger:hover .tooltip {
  display: inline;
  position: relative; /* relative to .tooltipTrigger */
  left: 10px;
}

"""

vlm_response_cell = """<table style="border: 0">
                <tr style="line-height: 24pt;">
                  <th>
                  </th>
                  <th>
                  VLM Generated Task Plan
                  </th>
                  <th>
                  VLM Translated
                  </th>
                </tr>
                <tr>
                  <td>
                    {name}
                  </td>
                  <td style="font-size: 10pt">
                    {subgoals_english}
                  </td>
                  <td style="font-size: 10pt">
                    {subgoals_translated}
                  </td>
                </tr>
              </table>

"""

summary_image_line = """<img class="top_image" src="{summary_tree_path}"/>"""

query_image_line = """<img class="query_image" src="{img_path}"/>"""

link_to_page = """<a href='{link}' target='_blank'>{text}</a>"""

heading = """<b class="heading">{text}</b>"""

failed = """<p style="color: #e74c3c">{text}</p>"""

br = '-'*80

space = '&nbsp;&nbsp;'


## ----------------------------------------------------------------------------


def annotate_summary_tree(root, success_lookups):
    """ mark the count and success rate of each visited node """
    color_solved = 'green'
    color_somtimes = 'blue'
    color_failed = 'red'

    current_option = 'a'
    next_root = None
    for success_lookup in success_lookups:
        node = root
        for i, (k, v) in enumerate(success_lookup.items()):
            k = k.split('[')[1]
            for child in node.children:
                name = child.name
                if '_' not in name:
                    break
                option = name.split('_')[1]
                sub_name = name.split('[')[1]
                if sub_name.startswith(k) or soft_match(k, sub_name):
                    if '--' not in child.name:
                        child.name += f'--{v}/1'
                        child.color = color_solved if v == 1 else color_somtimes
                    else:
                        success, total = name.split('--')[1].split('/')
                        total = eval(total) + 1
                        success = eval(success) + v
                        child.name = name.split('--')[0] + f'--{success}/{total}'
                        child.color = color_solved if success == total else color_somtimes
                    node = child
                    break


def generate_summary_tree(lists, exp_name, success_lookups=None, output_name='planning_tree.png'):
    """ generate a planing tree from """
    export_path = join(EXP_DIR, exp_name, output_name)
    root, node_counts = plans_to_tree(lists, export_path, no_end=True)
    if success_lookups is not None:
        annotate_summary_tree(root, success_lookups)

    export_tree_png(root, export_path, verbose=False)


def soft_match(k, node_name):
    if node_name == "'on', 'chicken leg', 'pot bottom']" and k == "'in', 'chicken leg', 'pot body']":
        return True
    return False


## ----------------------------------------------------------------------------


def make_vlm_response_cell(name, exp_path, return_responses=False):
    result = load_vlm_responses(exp_path, verbose=False)
    if result is None:
        return ''
    vlm_english, vlm_translated = result
    if vlm_translated.startswith('<br><br>'):
        vlm_translated = vlm_translated.replace('<br><br>', '').strip()
    text = vlm_response_cell.format(name=name, subgoals_english=vlm_english, subgoals_translated=vlm_translated)
    if return_responses:
        return vlm_english.split('<br>'), vlm_translated.split('<br>'), text
    return text


def load_vlm_responses(exp_path, **kwargs):
    vlm_memory = load_vlm_memory(join(exp_path, 'llm_memory.json'), **kwargs)
    if vlm_memory is None:
        return None
    data = [v for v in vlm_memory[0].values() if isinstance(v, dict)]
    return [d['response'][0].replace('`', '').replace('python', '').replace('[', '').replace(']', '').replace('\n', '<br>') for d in data]


def load_failed_action_plans(log_path):
    if not isfile(log_path):
        return ''

    def simplify_action(action):
        if ', #' not in action:
            print('simplify_action', action)
        return space + space + action[:action.index(', #')] + ' )'

    skeletons = []
    lines = open(log_path, 'r').readlines()
    for i, line in enumerate(lines):
        if line.startswith('Action plan ('):
            for j in range(10):
                if i + j >= len(lines):
                    break
                new_line = lines[i+j]
                if new_line.startswith('Plans: ') and 'Actual time: ' in new_line:
                    t = new_line.split('Actual time: ')[1]
                    skeleton = [f'<br>{space}Action Plan {len(skeletons) + 1} ({t})']
                    skeleton += [simplify_action(l) for l in lines[i+1: i+j-1] \
                                 if '_\n' not in l and len(l) > 1 and '[log_collisions' not in l and ')->[' not in l]
                    skeletons.append('<br>'.join(skeleton))
    return '<br>'.join(skeletons) + '<br>' + br


def load_all_subproblems(exp_path):
    subproblems = {}
    subproblems_paths = [f for f in listdir(exp_path) if f.startswith('subgoals_')]
    for p in subproblems_paths:
        for prob in json.load(open(join(exp_path, p), 'r')):
            index = prob['subgoal']
            if '_[' in index:
                index = index.split('_[')[0]
            else: ## _end
                index = index[:index.index('_end')]
            subproblems[index] = prob
    return subproblems


def load_actions_and_time(time_log, subproblems, indices, exp_path, load_state_path=None):
    all_actions = [br]
    num_actions = []
    success_lookup = {}
    last_index = None
    last_success = False

    total_length = 0
    subgoal_planning_time = []

    j = 1
    for k, dic in enumerate(time_log):

        ## sometimes the last run didn't log properly
        object_reducer = dic['object_reducer'] if 'object_reducer' in dic else ''
        index = dic['last_node'] if 'last_node' in dic else ''
        if index is None:
            print(k, len(time_log), 'index = None', exp_path, dic)
            continue

        if 'num_success' in dic:
            continue
        index = index.split('[')[0][:-1]
        if len(index) > 0:
            if index not in indices:
                g = dic['last_node'].replace('_[', ' [')
            else:
                g = indices[index]
            last_index = index
        else:
            if last_index is None:
                break
            last_num, option = last_index.split('_')
            if last_success:
                last_num = eval(last_num) + 1
            index = f"{last_num}_{option}"
            if index not in indices:
                ## at some point there were logging mistake
                index = f"{last_num-1}_{option}"
                if index not in indices:
                    print('index not in indices')
            g = indices[index]
        line = f"{g} ({object_reducer})"

        if 'total_planning' in dic or dic['plan'] == 'FAILED':
            success_lookup[g] = 0
            last_success = False
            line = failed.format(text=line)
            if index not in subproblems:
                print('index not in subproblems')
            if index in subproblems:
                log_name = join('log', subproblems[index]['log_path'])
                log_path = join(exp_path, log_name)
                if not isfile(log_path) and load_state_path is not None:
                    log_path = join(load_state_path, log_name)

                failed_action_plans = load_failed_action_plans(log_path)
                failed_action_plans = f'Subgoal {g}<br>' + failed_action_plans
                all_actions.append(failed.format(text=failed_action_plans))
        else:
            success_lookup[g] = 1
            last_success = True
            if 'plan_skeleton' in dic:
                skeleton = dic['plan_skeleton']
            else:
                skeleton = [a[a.index('name=')+6: a.index(', args=')-1] for a in dic['plan']]
            length = len(skeleton)
            time = dic['planning']
            actions = '<br>'.join([f"{space}{j + i} {str(lst)}" for i, lst in enumerate(skeleton) if 'move_base(' not in lst])
            j += len(skeleton)

            line += f' | len = {length} | {time} sec'
            line = f'<p>{line}</p>'

            actions = f'Subgoal {g}<br>' + actions + '<br>' + br
            all_actions.append(actions)

            total_length += length
            subgoal_planning_time.append(time)

        num_actions.append(line)

    return all_actions, num_actions, success_lookup, total_length, subgoal_planning_time


def load_subgoals(agent_memory, actions_mode=False):
    """ sometimes there are options, sometimes there are replanning """
    alphabets = 'abcde'
    key = 'lists_of_subgoals_to_print' if not actions_mode else 'lists_of_actions_to_print'

    lists_of_subgoals = agent_memory[key]
    if 'replan_memory' in agent_memory and len(agent_memory['replan_memory']) > 0:
        alphabets = 'axyz'
        # length = len(agent_memory[key.replace('_to_print', '')][0])
        # lists_of_subgoals[0] = lists_of_subgoals[0][:length]
        lists_of_subgoals.extend([m[key][0] for m in agent_memory['replan_memory']])

    num_subgoals = [len(lst) for lst in lists_of_subgoals]

    all_subgoals = []
    indices = {}
    for k, subgoals in enumerate(lists_of_subgoals):
        formatted = [f"{i+1}_{alphabets[k]} {str(lst)}" for i, lst in enumerate(subgoals)]
        all_subgoals.append(formatted)
        indices.update({f"{i+1}_{alphabets[k]}": formatted[i] for i in range(len(formatted))})

    return all_subgoals, num_subgoals, indices


def load_success_counts(subproblems):
    from vlm_tools.llamp_agent import SOLVED, ALREADY
    success_count = {}
    for index, subproblem in subproblems.items():
        ## the end
        if '_' not in index:
            continue
        step, option = index.split('_')
        if subproblem['result'] in [SOLVED, ALREADY]:
            success_count[option] = eval(step)
    return success_count


def summarize_num_subgoals(num_subgoals, subproblems=None, actions_mode=False):
    alphabets = 'abcde'
    success_count = None

    if subproblems is not None:
        success_count = load_success_counts(subproblems)
        alphabets_used = set([x.split('_')[1] for x in subproblems if '_' in x])
        if 'x' in alphabets_used:
            alphabets = 'axyz'

    options = []
    all_solved = []
    for i in range(len(num_subgoals)):
        option = alphabets[i]
        key = 'subgoals' if not actions_mode else 'actions'
        line = f'{num_subgoals[i]} {key}'  ## Option {option}:
        if subproblems is not None:
            solved = success_count[option] if option in success_count else 0
            line += f', solved {solved}'
            all_solved.append(solved)
        options.append(heading.format(text=line))

    total_solved = sum(all_solved)
    total_subgoals = num_subgoals[0]
    if alphabets == 'axyz' and len(all_solved) >= 2:
        total_subgoals += num_subgoals[-1]
        total_subgoals += sum(all_solved[1:-1])
    success_rate = total_solved / total_subgoals
    return options, success_count, total_subgoals, success_rate


def get_llm_memory_dir_name(exp_name):
    return exp_name.replace('run_', 'llm_only_').replace('_reprompt1', '').replace('_reprompt', '').replace('_v2', '')


def from_llm_memory_dir_name(exp_name):
    return exp_name.replace('llm_only_', 'run_')


def get_llm_memory_names(exp_name):
    llm_dir = join(EXP_DIR, get_llm_memory_dir_name(exp_name))
    if not isdir(llm_dir):
        return []
    names = [f for f in listdir(llm_dir)]
    names.sort()
    return names


def get_link_to_page_line(run_name, exp_name=None):
    html_path = join('log', 'index.html')

    ## in other experiments dir
    if '/' in run_name:
        link = join('../..', run_name, html_path)
        run_name = run_name.split('/')[-1]
    else:
        link = join(run_name, html_path)
        if exp_name is not None:
            link = join('../..', exp_name, link)
    return link_to_page.format(text=run_name, link=link)


def replace_in_file(html_file, old_key, new_key):
    if not isfile(html_file):
        return
    with open(html_file, 'r') as f:
        body = f.read()
        body = body.replace(old_key, new_key)
    with open(html_file, 'w') as f:
        f.write(body)


def update_summary_pages_with_result_table(summary, header, prompts_only=False):
    from vlm_tools.visualize.viz_run_utils import _make_summary_table_from_rows
    exp_names = set([x[0] for x in summary] + [x[2] for x in summary])
    html_files = [join(EXP_DIR, exp_name, 'index.html') for exp_name in exp_names if len(exp_names) > 1]

    new_header_for_html = header[:-1] if prompts_only else header[:-2]
    new_summary_for_html = []
    for x in summary:
        if prompts_only:
            x = x[:-1]
        else:
            x = x[:-2]
            x[2] = link_to_page.format(link=join('../..', x[2], 'index.html'), text=x[2])
        x[0] = link_to_page.format(link=join('../..', x[0], 'index.html'), text=x[0])
        new_summary_for_html.append(x)
    hover_template = """
        <div class="tooltipTrigger">
            <div style="background-color: white; text-align: center; padding: 10px">Hover to Show Result Table</div>
            <div class="tooltip">{table}
            </div>
        </div>
    """
    table = _make_summary_table_from_rows([new_header_for_html] + new_summary_for_html)
    table = hover_template.format(table='\n\n'+table+'\n\n')
    key = '<div class="table-title">'
    for html_file in html_files:
        replace_in_file(html_file, key, key+'\n\n'+table+'\n\n')


def replace_key_line_with_line_in_html_body(html_path, replacements):
    new_lines = []
    used_keys = []
    with open(html_path, 'r') as f:
        for line in f.readlines():
            found = False
            for k, v in replacements.items():
                if k in line:
                    if k not in used_keys:
                        new_lines.append(v+'\n')
                    found = True
                    used_keys.append(k)

            if not found:
                new_lines.append(line)

    with open(html_path, 'w') as f:
        f.writelines(new_lines)


def get_whole_goal_sequence(all_subgoals, solved_subgoals, verbose=False):
    title = '[summarize_utils.get_whole_goal_sequence]\t'
    if len(all_subgoals) == 1 or len(solved_subgoals) == 0 or 'end' in solved_subgoals[-1]:
        return all_subgoals[0]
    failed_idx, failed_at_branch = idx_branch_from_subgoal(solved_subgoals[-1])
    branch_to_sequence = {idx_branch_from_subgoal(lst[0])[1]: lst for lst in all_subgoals if len(lst) > 0}
    future_subgoals = branch_to_sequence[failed_at_branch]
    future_subgoals = future_subgoals[int(failed_idx):]

    solved_subgoal_names = [f[f.index('['):] for f in solved_subgoals]
    future_subgoals_all = copy.deepcopy(future_subgoals)
    future_subgoals = [f for f in future_subgoals if f[f.index('['):] not in solved_subgoal_names]

    if verbose:
        num_solved = len(solved_subgoals)
        print(f'{title} solved_subgoals')
        print('\t' + '\n\t'.join([f"{i}\t{g}" for i, g in enumerate(solved_subgoals)]))
        print(f'{title} unsolved_subgoals')
        print('\t' + '\n\t'.join([
            f"{i + num_solved}\t{g}" if g in future_subgoals else f"{i + num_solved}\t{g}\t (already solved)" \
            for i, g in enumerate(future_subgoals_all)
        ]))
        print()
    return solved_subgoals + future_subgoals


def generate_failure_decomp_sheet(prompt_exp_name, outputs):
    columns = ['run_id', 'english', 'translated', 'subgoal', 'mistranslation', 'ungrounded', 'misspell']
    table = [columns]
    for run in outputs['run_info']:
        run_id = run['planning_tree_path'].split('/')[0].replace('_vlm-tamp', '')
        for i, english in enumerate(run['vlm_english']):
            translated = run['vlm_translated'][i] if i < len(run['vlm_translated']) else ''
            subgoal = run['vlm_repaired'][i] if i < len(run['vlm_repaired']) else ''
            table.append([run_id, english, translated, subgoal, 0, 0, 0])
            run_id = ''
    csv_name = join(EXP_DIR, prompt_exp_name, 'failure_vlm.csv')
    with open(csv_name, mode='w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(table)


def add_output_runs(outputs_runs, all_outputs_runs):
    for k, v in outputs_runs.items():
        if isinstance(v, list):
            if k not in all_outputs_runs:
                all_outputs_runs[k] = []
            all_outputs_runs[k].extend(v)
        elif isinstance(v, dict):
            vv = list(v.values())[0]

            ## prompt_to_runs
            if isinstance(vv, dict):
                if k not in all_outputs_runs:
                    all_outputs_runs[k] = {}
                for kk, vv in v.items():
                    all_outputs_runs[k][kk] = vv
            else:
                if k not in all_outputs_runs:
                    all_outputs_runs[k] = defaultdict(list)
                for kk, vv in v.items():
                    all_outputs_runs[k][kk].extend(vv)
