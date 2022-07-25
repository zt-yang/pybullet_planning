import json
import os
from os.path import join, isfile, isdir, abspath, dirname
from os import mkdir
from datetime import datetime
import csv
import pprint

TXT_FILE = abspath('txt_file.txt')
COMMANDS_FILE = abspath('commands.json')


def myprint(text='', *kwargs):
    string = [str(text)]
    if len(kwargs) > 0:
        print(text, kwargs)
        string.extend([str(n) for n in kwargs])
    else:
        print(text)
    string = ' '.join(string)+'\n'
    string = string.replace('\t', '    ')
    with open(TXT_FILE, 'a+') as f:
        f.writelines(string)


def record_command(action):
    from pybullet_tools.pr2_primitives import Commands
    from world_builder.robots import RobotAPI

    # def print_value(value):
    #     values = {}
    #     for f, v in value.__dict__.items():
    #         if isinstance(v, Commands):
    #             values[f] = [print_value(vv) for vv in v.commands]
    #         else:
    #             values[f] = [str(vv) for vv in v]

    commands = []
    if isfile(COMMANDS_FILE):
        with open(COMMANDS_FILE, 'r') as f:
            commands = json.load(f)
        os.remove(COMMANDS_FILE)

    commands.append(action)

    with open('file.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(myvar, file)

    step = len(commands)

    # command = {'timestamp': step, 'name': action.__class__.__name__, 'args': {}}
    # for field, value in action.__dict__.items():
    #     values = {}
    #     if hasattr(value, '__dict__'):
    #         for k, v in value.__dict__.items():
    #             if isinstance(v, tuple):
    #                 values[k] = list(v)
    #             elif isinstance(v, RobotAPI):
    #                 values[k] = v.name
    #         command['args'] = values
    #     # elif value is None:
    #     #
    commands.append(command)

    with open(COMMANDS_FILE, 'w') as f:
        json.dump(commands, f, indent=2)


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

    if write_average:
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

def dump_json(db, db_file, indent=2, width=160, **kwargs):
    """ don't break lines for list elements """
    with open(db_file, 'w') as f:
        # pprint(db, f, indent=2, width=120) ## single quote
        f.write(pprint.pformat(db, indent=indent, width=width, **kwargs).replace("'", '"'))