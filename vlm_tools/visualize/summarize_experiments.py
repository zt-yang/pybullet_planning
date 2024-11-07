import os
import sys
from os.path import join, abspath, dirname, basename
from tqdm import tqdm

R = abspath(join(dirname(__file__), os.pardir))
sys.path.extend([R] + [join(R, name) for name in ['pddlstream', 'pybullet_planning', 'lisdf']])

from tabulate import tabulate

from vlm_tools.llamp_agent import get_progress_summary_from_time_log
from vlm_tools.vlm_utils import RESULTS_DATA_DIR, get_alphabet, fix_server_path, enumerate_exp_names, \
    load_agent_memory
from vlm_tools.visualize.summarize_utils import *

from pybullet_tools.logging_utils import print_pink, get_success_rate_string, clear_cache_text


def get_hostname(exp_path):
    with open(join(exp_path, 'planning_config.json')) as f:
        hostname = json.load(f)['host']
    return hostname


def summarize_experiments(exp_name: str, llm_mode=False, outputs_runs=None,
                          hostname=None, data_root=None, verbose=True):
    """
    Usually a run generates both a task planning using LLM and motion plans using TAMP
    args:
        llm_mode=True:  only show LLM task plan and skipped planning
        outputs_runs:   show the results of various experiment runs at the LLM task plan page

    failure reasons:
        'translation':  producing a goal that involves bad typing, e.g. "(openedjoint, pot lid)"
                        the error causes the run to stop abruptly so the last problem was logged success
        'sequencing':   the planner didn't find the right skeleton given the subgoal
                        from a history of runs we know the ground truth skeleton, check if it's there
        'sampling':     the planner didn't find the continuous variables to refine the skeleton
                        if a failure is logged and it isn't sequencing failure
    """

    actions_mode = 'actions' in exp_name
    run_info = []
    exp_dir = join(EXP_DIR, exp_name) if data_root is None else join(EXP_DIR, data_root, exp_name)
    if not isdir(exp_dir):
        os.makedirs(exp_dir)
    run_names = [f for f in listdir(exp_dir) if isdir(join(exp_dir, f)) and not f.startswith('_')]
    lists_of_subgoals = []
    lists_of_suceeded_subgoals = []

    prompt_to_runs = defaultdict(list)
    llm_memory_names = get_llm_memory_names(exp_name)
    llm_memory_summary = {n: [] for n in llm_memory_names}
    success_lookups = []

    failure_reasons = {f: [] for f in FAILURE_REASONS}

    data = {k: [] for k in ['success_rate', 'num_proposed_subgoals',
                            'num_refined_actions', 'subgoal_planning_time']}
    data = defaultdict(list)

    for k, run_name in enumerate(sorted(run_names)):
        exp_path = join(exp_dir, run_name)

        exp_id = join(exp_name, run_name)

        ## sometimes agent_memory misses replan memory, fixed
        subgoals_path = join(exp_path, 'agent_memory.json')
        agent_memory = load_agent_memory(subgoals_path)

        all_subgoals, num_subgoals, indices = load_subgoals(agent_memory, actions_mode=actions_mode)
        subgoals_line = ['<br>'.join(lst) for lst in all_subgoals]

        key = 'VLM' if llm_mode else 'Run'
        name = 'Branch Idx: ' + get_alphabet(k) + \
               f'<br>Goal: ' + agent_memory['goal'] + \
               f"<br>{key}: " + get_link_to_page_line(run_name)
        init_image = join(exp_path, 'log', 'media', 'query_0.png')
        if isfile(init_image):
            init_image = join(run_name, 'log', 'media', 'query_0.png')
            name += f'<br><br>' + query_image_line.format(img_path=init_image)

        ## summarize experiments
        if not llm_mode:
            if verbose:
                print(exp_path)
            time_log_path = join(exp_path, 'time.json')
            subproblems_path = join(exp_path, 'subgoals_0.json')
            if not isfile(time_log_path) or not isfile(subgoals_path) or not isfile(subproblems_path):
                if verbose:
                    print(f'skipping {exp_id} because isfile(time.json) = {isfile(time_log_path)}'
                      f' or isfile(agent_memory.json) = {isfile(subgoals_path)}'
                      f' or isfile(subgoals_0.json) = {isfile(subproblems_path)}')
                continue

            this_host_name = get_hostname(exp_path)
            if hostname is not None and this_host_name != hostname:
                continue

            ## goal + object reducer
            time_log = json.load(open(time_log_path, 'r'))
            load_memory_path = agent_memory['load_memory']
            load_state_path = agent_memory['load_agent_state'] if 'load_agent_state' in agent_memory else None

            ## actions and time
            subproblems = load_all_subproblems(exp_path)
            actions, num_actions, success_lookup, total_length, subgoal_planning_time = \
                load_actions_and_time(time_log, subproblems, indices, exp_path, load_state_path=load_state_path)

            subgoals = [k for k, v in success_lookup.items() if v == 1]
            succeeded_subgoals = [subgoals]
            subgoals_line = ['<br>'.join(subgoals)]

            if len(num_actions) == 0:
                print(f'skipping {exp_id} because len(num_actions) = {len(num_actions)}')
                continue

            data['num_refined_actions'].append(total_length)
            data['subgoal_planning_time'].extend(subgoal_planning_time)
            success_lookups.append(success_lookup)

            ## goal + object reducer
            num_subgoals, success_count, total_subgoals, success_rate = \
                summarize_num_subgoals(num_subgoals, subproblems, actions_mode=actions_mode)
            data['success_rate'].append(success_rate)
            # data['num_proposed_subgoals'].append(total_subgoals)

            ## load the saved stats, correct the wrong in json and html
            whole_goal_sequence = get_whole_goal_sequence(all_subgoals, subgoals)  ## list(success_lookup.items())  ## time_log[-1]['whole_goal_sequence']
            data['num_proposed_subgoals'].append(len(whole_goal_sequence))

            ## ------------------------------ stats & performance ---------------------------------

            _, summary = get_progress_summary_from_time_log(time_log[:-1], whole_goal_sequence)
            time_log[-1] = summary
            with open(time_log_path, 'w') as f:
                json.dump(time_log, f, indent=4)
            values = [summary[k+"_string"] for k in ['continuous_success', 'task_progress', 'total_problems_solved']]
            keys = ['Continuous Success', 'Task Progress', 'Planning Success']
            replacements = {keys[i]: f'<td><span style="color: black">{keys[i]}<br>{values[i]}</span></td>' for i in range(3)}
            html_path = join(exp_path, 'log', 'index.html')
            # replace_key_line_with_line_in_html_body(html_path, replacements)

            ## incorporate the statistics
            performance_report_raw = []
            performance_report = []
            for key, v in summary.items():
                if key in ['whole_goal_sequence']:
                    continue
                if '_string' in key or key in ['task_success']:
                    kk = key.replace('_string', '')
                    if kk not in ['task_progress', 'total_problems_solved']:
                        performance_report_raw.append(f"{kk}: {v}")
                    performance_report.append(f"{kk}:<br> {v}<br><br>")

                if isinstance(v, list):
                    data[key].extend(v)
                else:
                    data[key].append(v)
            data['performance_report'].append(performance_report)

            failure_stats = []
            for key in ['ungrounded_subgoals', 'failed_nodes', 'reprompt_after_subgoals']:
                failure_stats.append(summary[key])
                data[f'num_{key}'].append(len(summary[key]))

            ungrounded_subgoals = summary['ungrounded_subgoals']
            if len(ungrounded_subgoals) > 0:
                print(f'\n{exp_path}\t ' + str(performance_report_raw))
                print('\t'+'\n\t'.join([str(s) for s in ungrounded_subgoals]))

            # print(summary['num_problems_solved_continuously'], '\t', summary['total_num_problems'], '\t',
            #       performance_report_raw)
            # if summary['num_problems_solved_continuously'] == summary['total_num_problems'] - 1:
            #     print_debug(f'\n[Almost] {exp_path}')

            p2r_stats = [run_name, performance_report_raw] + failure_stats

            ## -------------------------------------------------------------------------------

            num_reprompts = summary['num_reprompts']
            task_success = summary['task_success']
            des_dir = None
            if 'reprompt1' in exp_path:
                if num_reprompts == 0 and task_success == 0:
                    print_pink(f'\n{exp_path}\t {performance_report_raw}\t num_reprompts = {num_reprompts} < 1')
                    des_dir = exp_path.replace('_reprompt1', '')
                # else:
                #     print_green(f'\n{exp_path}\t {performance_report_raw}')

            elif 'reprompt' in exp_path:
                if num_reprompts <= 1 and task_success == 0:
                    print_pink(f'\n{exp_path}\t {performance_report_raw}\t num_reprompts = {num_reprompts} < 2')
                    if num_reprompts == 1:
                        des_dir = exp_path.replace('_reprompt', '_reprompt1')
                    if num_reprompts == 0:
                        des_dir = exp_path.replace('_reprompt', '')
                # else:
                #     print_green(f'\n{exp_path}\t {performance_report_raw}')

            if des_dir is not None:
                print_pink(f'moving it to {des_dir}')
                # shutil.move(exp_path, des_dir)
                # continue

            ## -------------------------------------------------------------------------------

            ## link to execution video
            mp4_path = join(exp_path, 'replay.mp4')
            if isfile(mp4_path):
                name += f'<br><br> ----> ' + link_to_page.format(text='mp4', link=mp4_path)

            ## get statistics about the bank of answers
            if load_memory_path is not None:

                load_memory_path = fix_server_path(load_memory_path)
                load_memory_name = basename(load_memory_path)
                if load_memory_name not in llm_memory_summary:
                    if verbose:
                        print(f'skipping {exp_id} because {load_memory_name} not in {list(llm_memory_summary.keys())}')
                    continue

                if data_root is None:
                    prompt_to_runs[load_memory_name].append(p2r_stats)
                else:
                    prompt_to_runs[load_memory_name].append(join(data_root, exp_id))

                llm_memory_summary[load_memory_name].append(success_count['a'])
                link_to_memory = get_link_to_page_line(load_memory_name, get_llm_memory_dir_name(exp_name))
                name += '<br><br>Memory: ' + link_to_memory
                name = make_vlm_response_cell(name, load_memory_path)

            info = {
                'num_subgoals': '<br>'.join(performance_report),  ## '<br>'.join(num_subgoals),
                'subgoals': '<br><br>' + '<br><br>'.join(subgoals_line),
                'num_actions': '\n'.join(num_actions),
                'actions': '<br><br>'.join(actions),
                'failure': '',
                'hostname': this_host_name
            }

        else:
            vlm_english, vlm_translated, name = make_vlm_response_cell(name, exp_path, return_responses=True)

            num_subgoals, _, _, _ = summarize_num_subgoals(num_subgoals, actions_mode=actions_mode)
            options = '<br><br>'.join([num_subgoals[i]+f'<br>{subgoals_line[i]}' for i in range(len(num_subgoals))])

            # run_exp_name = from_llm_memory_dir_name(exp_name)

            ## prompts only, no run history
            runs_by_exp_group = []
            if 'prompt_to_runs' in outputs_runs:
                for run_exp_name, this_prompt_to_runs in outputs_runs['prompt_to_runs'].items():
                    if run_name not in this_prompt_to_runs:
                        continue
                    all_runs = this_prompt_to_runs[run_name]
                    if len(all_runs) == 0:
                        continue
                    runs = []
                    first_one = all_runs[0]
                    if '(' in first_one:
                        runs = [first_one]
                        all_runs = all_runs[1:]

                    ## debug string on prompt summary page
                    for n in all_runs:
                        line = f"{get_link_to_page_line(n[0], run_exp_name)}: {n[1]}"
                        for i, (key, color) in enumerate([('ungrounded', 'purple'), ('failed', 'red'), ('reprompted', 'blue')]):
                            value = n[i+2]
                            if len(value) > 0:
                                line += f'<br><span style="color: {color}; fontsize: 10">{space}{space}{key}: {value}</span>'
                        runs.append(line)
                    runs_by_exp_group.append(f"{run_exp_name}<br><br>"+'<br><br>'.join(runs))

            info = {
                'vlm_english': vlm_english,
                'vlm_translated': vlm_translated,
                'vlm_repaired': all_subgoals[0],

                'num_subgoals': options,
                'subgoals': '',
                'num_actions': '<br><br><br>'.join(runs_by_exp_group),
                'actions': '',
                'failure': '',
            }

        info.update({
            'run_name': name,
            'exp_name': exp_name,
            'planning_tree_path': join(run_name, 'log', 'media', 'planning_tree.png'),
        })
        run_info.append(info)

        lists_of_subgoals += [[s[s.index('['):] for s in subgoals] for subgoals in all_subgoals]

    os.makedirs(RESULTS_DATA_DIR, exist_ok=True)
    result_path = join(RESULTS_DATA_DIR, f'{exp_name}.json')
    with open(result_path, 'w') as f:
        json.dump(dict(data), f, indent=4)

    outputs = {
        'run_info': run_info,
        'lists_of_subgoals': lists_of_subgoals,
        'prompt_to_runs': {exp_name: prompt_to_runs},
        'success_lookups': success_lookups,
    }
    return outputs


## ----------------------------------------------------------------------------


def generate_page(exp_name: str, summary_tree_path='planning_tree.png', verbose=True, **kwargs):
    """ generate an html with a summary planning tree one top, each row of the table is one run """
    outputs = summarize_experiments(exp_name, verbose=verbose, **kwargs)
    rows = [runs_row_template.format(**info) for info in outputs['run_info']]

    summary_image = ''
    if isfile(join(EXP_DIR, exp_name, summary_tree_path)):
        summary_image = summary_image_line.format(summary_tree_path=summary_tree_path)
        # summary_image = link_to_page.format(text=summary_image, link=summary_tree_path)

    output_path = join(EXP_DIR, exp_name, 'index.html')
    with open(output_path, 'w') as f:
        f.write(runs_html_template.format(tab_name=exp_name, style=included_style, exp_name=exp_name, num=len(rows),
                                          summary_image=summary_image, rows=''.join(rows)).replace('\\n', '<br>'))
    if verbose:
        print(f'Generated {output_path}')
        if len(rows) > 0:
            print(f'Visit', DISPLAY_LINK.format(exp_name=exp_name), f'to see {len(rows)} runs \n')
    return outputs


def summarize_subgoals_run_dir():
    exp_name = 'test_llm_kitchen_chicken_soup'
    exp_name = 'llm_kitchen_soup_subgoals_single'
    generate_page(exp_name)


def summarize_actions_run_dir():
    exp_name = 'test_llm_kitchen_chicken_soup_actions'
    generate_page(exp_name, actions_mode=True)


def summarize_prompts(exp_name):
    outputs = generate_page(exp_name, llm_mode=True)
    generate_summary_tree(outputs['lists_of_subgoals'], exp_name)


def summarize_all_prompts(prompts_only=True, hostname=None, data_roots=[None], verbose=False):
    exp_names = enumerate_exp_names(mode='eval')
    all_exp_names = [exp_name for exp_name in exp_names]
    # all_exp_names = ['run_kitchen_chicken_soup_v1_subgoals_dual_arm']  ## ; verbose = True
    # all_exp_names = [exp_name for exp_name in all_exp_names if 'reprompt' in exp_name and 'reprompt1' not in exp_name \
    #                  and 'v1' in exp_name and 'subgoal' in exp_name]

    summary = []
    outputs_runs_by_llm_group = {}
    lists_of_subgoals_by_llm_group = {}
    for exp_name in tqdm(all_exp_names):
        num_runs = 0
        if prompts_only:
            all_outputs_runs = {'success_lookups': {}}

        else:
            all_outputs_runs = {}
            for data_root in data_roots:
                outputs_runs = summarize_runs(exp_name, hostname=hostname, data_root=data_root, verbose=verbose)  ## summarize_experiments(exp_name)
                num_runs += len(outputs_runs['run_info'])
                add_output_runs(outputs_runs, all_outputs_runs)

        prompt_exp_name = get_llm_memory_dir_name(exp_name)
        if prompt_exp_name not in outputs_runs_by_llm_group:
            outputs_runs_by_llm_group[prompt_exp_name] = {}
        outputs_runs_by_llm_group[prompt_exp_name][exp_name] = all_outputs_runs

        ## generate llm page just to get the number of prompts
        num_output_prompts = ''
        if 'reprompt' not in exp_name:
            outputs_prompts = generate_page(prompt_exp_name, llm_mode=True, outputs_runs=all_outputs_runs, verbose=verbose)
            num_output_prompts = len(outputs_prompts['run_info'])
            generate_failure_decomp_sheet(prompt_exp_name, outputs_prompts)
            lists_of_subgoals_by_llm_group[prompt_exp_name] = outputs_prompts['lists_of_subgoals']

        summary.append([prompt_exp_name, num_output_prompts, exp_name, num_runs])

        if verbose:
            print('-'*80)

    ## ---------------------------------- summarizing all runs in llm_only page ----------------------------

    for prompt_exp_name, outputs_runs_by_exp_names in outputs_runs_by_llm_group.items():
        if len(outputs_runs_by_exp_names) == 1:
            continue
        lists_of_subgoals = lists_of_subgoals_by_llm_group[prompt_exp_name]
        all_outputs_runs = {}
        run_counts = {}
        for exp_name, outputs_runs in outputs_runs_by_exp_names.items():
            add_output_runs(outputs_runs, all_outputs_runs)
            n = 0
            if 'reprompt1' in exp_name:
                n = 1
            elif 'reprompt' in exp_name:
                n = 2
            generate_summary_tree(lists_of_subgoals, prompt_exp_name, output_name=f'planning_tree_n={n}.png',
                                  success_lookups=outputs_runs['success_lookups'])
            run_counts[exp_name] = len(outputs_runs['run_info'])

        new_prompt_to_runs_by_exp_name = {}
        for exp_name, run_count in run_counts.items():
            new_prompt_to_runs = {}
            for llm_name, run_names in all_outputs_runs['prompt_to_runs'][exp_name].items():
                new_prompt_to_runs[llm_name] = [get_success_rate_string(len(run_names), run_count)] + run_names
            new_prompt_to_runs_by_exp_name[exp_name] = new_prompt_to_runs
        all_outputs_runs['prompt_to_runs'] = new_prompt_to_runs_by_exp_name

        generate_page(prompt_exp_name, llm_mode=True, outputs_runs=all_outputs_runs, verbose=verbose)
        generate_summary_tree(lists_of_subgoals, prompt_exp_name, all_outputs_runs['success_lookups'])
        if verbose:
            print('-'*80)

    ## -------------------------------------------------------------------------------------------------------

    if prompts_only:
        header = ['prompt group', 'prompt count', 'prompt link']
        summary = [x[:2] + [DISPLAY_LINK.format(exp_name=x[0])] for x in summary if isinstance(x[1], int)]

    else:
        header = ['prompt group', 'prompt count', 'run group', 'run count', 'run link', 'prompt link']
        summary = [x + [DISPLAY_LINK.format(exp_name=x[2]), DISPLAY_LINK.format(exp_name=x[0]) if isinstance(x[1], int) else '']
                   for x in summary]

    print(tabulate(summary, header, tablefmt="psql"))
    update_summary_pages_with_result_table(summary, header, prompts_only=prompts_only)

    clear_cache_text()


## ---------------------------------------------------------------------------------------


def summarize_runs(exp_name, **kwargs):
    outputs = generate_page(exp_name, **kwargs)
    generate_summary_tree(outputs['lists_of_subgoals'], exp_name, outputs['success_lookups'])
    return outputs


def summarize_all_runs():
    for exp_name in enumerate_exp_names(mode='eval'):
        summarize_runs(exp_name)


## ---------------------------------------------------------------------------------------


if __name__ == '__main__':
    """ make sure local lost is launched first
    (cd experiments/; python -m http.server 9000)
    """
    # summarize_subgoals_run_dir()
    # summarize_actions_run_dir()
    summarize_all_prompts(prompts_only=False, data_roots=[None])  ## , data_roots=['_on_240911_bad_v1'], hostname="meraki"
    # summarize_all_runs()

