import json
import zipfile
import os
from os.path import join, isfile, isdir, abspath, dirname, relpath
from os import mkdir
from datetime import datetime
import csv
import pprint

TXT_FILE = abspath('txt_file.txt')


def clear_cache_text():
    if isfile(TXT_FILE):
        os.remove(TXT_FILE)


class bcolors(object):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'

    mapping = {
        'red': FAIL,
        'pink': HEADER,
        'blue': OKBLUE,
        'cyan': OKCYAN,
        'green': OKGREEN,
        'yellow': WARNING,

        'fail': FAIL,
        'header': HEADER,  ## pink
        'bold': BOLD,  ## not very visible
        'warning': WARNING,  ## yellow
        'underline': UNDERLINE,
    }

    def __init__(self):
        for k in list(self.mapping.keys()):
            if k not in ['bold']:
                self.mapping[k[0]] = self.mapping[k]

    def get_color(self, style=None):
        code = bcolors.WARNING
        if style is not None and style in self.mapping:
            code = self.mapping[style]
        return code

    def print(self, text, style=None):
        code = self.get_color(style)
        print(code + text + self.ENDC)


def print_debug(text, style=None):
    print_in_file(text)
    bcolors().print(text, style)


def print_yellow(text):
    print_debug(text, 'yellow')


def print_pink(text):
    print_debug(text, 'pink')


def print_green(text):
    print_debug(text, 'green')


def print_red(text):
    print_debug(text, 'red')


def print_blue(text):
    print_debug(text, 'blue')


def print_cyan(text):
    print_debug(text, 'cyan')


## -----------------------------------------------------------


def parallel_print(text='', *kwargs):
    string = get_string(text, kwargs)
    print_in_file(string, txt_file='txt_file.txt')


def myprint(text='', *kwargs):  ## , **kwargs2
    string = get_string(text, kwargs)
    print_in_file(string)


def get_string(text, kwargs, verbose=True):
    string = str(text)
    if len(kwargs) > 0:
        print(text, kwargs)
        string += ' '.join([str(n) for n in kwargs])
    else:
        print(text)
    return string


def print_in_file(string, txt_file=TXT_FILE):
    string = string.replace('\t', '    ') + '\n'
    with open(txt_file, 'a+') as f:
        f.writelines(string)


def record_results(goal, plan, planning_time, exp_name='default'):

    fieldnames = ['run name', 'planning time', 'plan length']

    results_dir = abspath(join(dirname(__file__), 'experiments', exp_name))
    if not isdir(results_dir): mkdir(results_dir)

    now = datetime.now().strftime("%m%d-%H%M%S")
    name = f'{now}_{exp_name}'

    ## record the planning time
    csv_name = join(results_dir, f'summary.csv')
    if not isfile(csv_name):
        with open(csv_name, mode='w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
    with open(csv_name, mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        row = [name, planning_time, len(plan)]
        writer.writerow({fieldnames[i]: row[i] for i in range(len(fieldnames))})

    ## record the plan
    with open(join(results_dir, f'{name}.txt'), 'w') as f:
        f.writelines(f'Goal: {goal}\n\nPlan:\n')
        f.writelines('\n'.join([str(n) for n in plan]))


def summarize_csv(csv_name):
    from tabulate import tabulate

    data = []
    fieldnames = []
    sums = []
    AVERAGE = 'average of'
    write_average = False
    line_count = 0
    result_count = 0
    with open(csv_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                fieldnames = row
                sums = [0] * (len(row)-1)
            elif AVERAGE not in row[0]:
                FAILED = False
                for i in range(1, len(row)):
                    if float(row[i]) == 0 or float(row[i]) == 99999:
                        FAILED = True
                if not FAILED:
                    for i in range(1, len(row)):
                        sums[i - 1] += float(row[i])
                    result_count += 1
                    data.append(row)
                write_average = True
            else:
                data.append(row)
                write_average = False
            line_count += 1

    if write_average and result_count > 0:
        row = [f'{AVERAGE} {result_count} runs']
        row.extend([round(s / result_count, 4) for s in sums])
        data.append(row)
        with open(csv_name, mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow({fieldnames[i]: row[i] for i in range(len(fieldnames))})

    print()
    print(tabulate(data, headers=fieldnames, showindex='always')) ## , tablefmt='fancy_grid'


def write_stream_statistics(externals, verbose=True):
    from pddlstream.language.statistics import dump_total_statistics, \
        dump_online_statistics, merge_data, get_data_path
    from pddlstream.utils import ensure_dir, write_pickle

    if not externals:
        return
    if verbose:
        dump_online_statistics(externals)
        # dump_total_statistics(externals)
    pddl_name = externals[0].pddl_name # TODO: ensure the same
    data = {}
    for external in externals:
        if not hasattr(external, 'instances'):
            continue # TODO: SynthesizerStreams
        data[external.name] = merge_data(external, {})

    filename = get_data_path(pddl_name)
    ensure_dir(filename)
    write_pickle(filename, data)
    if verbose:
        print('Wrote:', filename)


def dump_json(db, db_file, indent=2, width=160, sort_dicts=True, **kwargs):
    """ don't break lines for list elements """
    with open(db_file, 'w') as f:
        # pprint(db, f, indent=2, width=120) ## single quote
        f.write(pprint.pformat(db, indent=indent, width=width, sort_dicts=sort_dicts,
                               **kwargs).replace("'", '"'))


def save_commands(commands, commands_path):
    if len(commands) > 0:
        import pickle
        with open(commands_path, 'wb') as file:
            pickle.dump(commands, file)


#######################################################


def get_readable_list(lst, world=None, NAME_ONLY=False, TO_LISDF=False):
    if len(lst) == 0:
        return []
    to_print = [lst[0]]
    for word in lst[1:]:
        if world is not None:
            name = world.get_name(word)
            last_is_tuple = (len(to_print) != 0) and isinstance(to_print[-1], tuple)
            if name is not None and not last_is_tuple: ## ['=', ('PickCost',), 'pr2|1']
                if TO_LISDF:
                    name = world.get_lisdf_name(word)
                elif not NAME_ONLY:
                    name = world.get_debug_name(word)
                to_print.append(name)
            else:
                to_print.append(word)
        else:
            to_print.append(word)
    return to_print


def summarize_facts(facts, world=None, name='Initial facts', print_fn=None):
    if print_fn is None:
        print_fn = myprint
    print_fn('----------------')
    print_fn(f'{name} ({len(facts)})')
    predicates = {}
    for fact in facts:
        pred = fact[0].lower()
        if pred not in predicates:
            predicates[pred] = []
        predicates[pred].append(fact)
    predicates = {k: v for k, v in sorted(predicates.items())}
    # predicates = {k: v for k, v in sorted(predicates.items(), key=lambda item: len(item[1][0]))}
    for pred in predicates:
        to_print_line = [get_readable_list(fa, world) for fa in predicates[pred]]
        to_print_line = sorted([str(l).lower() for l in to_print_line])
        to_print = ', '.join(to_print_line)
        print_fn(f'  {pred} [{len(to_print_line)}] : {to_print}')
    print_fn('----------------')


def print_plan(plan, world=None, print_fn=None):
    from pddlstream.language.constants import is_plan
    if print_fn is None:
        print_fn = myprint

    if not is_plan(plan):
        return
    step = 1
    print_fn('Plan:')
    for action in plan:
        name, args = action
        if name.startswith('_'):
            print_fn(f' ) {name}')
            continue
        args2 = [str(a) for a in get_readable_list(args, world)]
        print_fn('{:2}) {} {}'.format(step, name, ' '.join(args2)))
        step += 1
    print_fn()


def print_goal(goal, world=None, print_fn=None):
    if print_fn is None:
        print_fn = myprint

    print_fn(f'Goal ({len(goal) - 1}): ({goal[0]}')
    goals_to_print = [get_readable_list(g, world) for g in goal[1:]]
    for each in goals_to_print:
        print_fn(f'   {tuple(each)},')
    print_fn(')')


def print_domain(domain_pddl, stream_pddl, custom_limits):
    myprint(f'stream_agent.pddlstream_from_state_goal(\n'
            f'\tdomain = {domain_pddl}, \n'
            f'\tstream = {stream_pddl}, \n'
            f'\tcustom_limits = {custom_limits}')


def summarize_poses(preimage):
    from pybullet_tools.bullet_utils import nice
    atposes = [f[-1] for f in preimage if f[0].lower() == 'atpose']
    poses = [f[1:] for f in preimage if f[0].lower() == 'pose' if f[-1] not in atposes]

    print('\n' + '=' * 25 + ' poses that can be cached to loaders_{domain}.py ' + '=' * 25)
    for obj_pose in poses:
        body, pose = obj_pose
        print(f'placing {body}')
        print(nice(pose.value, keep_quat=True))
    print('-'*50+'\n')


def summarize_bconfs(preimage, plan):
    from pybullet_tools.bullet_utils import nice
    bconfs = [f[1] for f in preimage if f[0].lower() == 'bconf' and f[1].joint_state is not None]
    bconfs_ordered = []
    for action in plan:
        for arg in action.args:
            if arg in bconfs and arg not in bconfs_ordered:
                bconfs_ordered.append(arg)

    print('\n' + '=' * 25 + ' bconfs that can be cached to loaders_{domain}.py ' + '=' * 25)
    for bconf in bconfs_ordered:
        joint_state = {k: nice(v) for k, v in bconf.joint_state.items()}
        print(f"({nice(bconf.values)}, {joint_state}), ")
    print('-'*50+'\n')


def print_lists(tuples):
    print()
    for lst, title in tuples:
        print_list(lst, title)
    print('-'*70)


def print_list(lst, title):
    lst = sorted(lst, key=lambda x: x[0])
    print(f'\t{title}({len(lst)})')
    print('\t\t' + '\n\t\t'.join([str(f) for f in lst]))


def print_dict(dic, title, indent=3, width=80):
    from pprint import pformat
    title = title.upper()  ## .replace('_', ' ')
    myprint('-' * 25 + f' {title} ' + '-' * 25)
    myprint(pformat(dict(dic), indent=indent, width=width))
    myprint('-' * 60)


def summarize_state_changes(current_facts, old_facts, title='summarize_state_changes', verbose=True):
    added = list(set(current_facts) - set(old_facts))
    added = process_facts(added)
    deled = list(set(old_facts) - set(current_facts))
    deled = process_facts(deled)
    if verbose:
        print_lists([(added, f'{title}.added'), (deled, f'{title}.deled')])
    return added, deled


def process_facts(facts):
    facts = [f for f in facts if f[0] not in ['='] and \
            not (f[0] == 'not' and f[1][0] in ['=', 'identical'])]
    facts = sorted(facts, key=lambda x: x[0])
    return facts


## -------------------------------------------------------------------------


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(join(root, file), relpath(join(root, file), join(path, '..')))


def zipit(dir_list, zip_name):
    zipf = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
    for dir in dir_list:
        zipdir(dir, zipf)
    zipf.close()


## -------------------------------------------------------------------------


def print_heading(text):
    separator = 50 * '-'
    spacing = 50 * ' '
    myprint(f'{separator}{separator}')
    myprint(f'{spacing} {text} {spacing} ')
    myprint(f'{separator}{separator}')


def get_success_rate_string(num_success, num_problems, roundto=2):
    if num_problems == 0:
        return f"0 (0 / 0)"
    return f"{round(num_success / num_problems, roundto)} ({num_success} / {num_problems})"


if __name__ == '__main__':
    text = 'test'
    for style in bcolors.mapping:
        bcolors().print(style+': ' + text, style)
