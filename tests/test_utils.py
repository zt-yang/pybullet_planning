from __future__ import print_function
import os
import json
import csv
from os.path import join, abspath, dirname, isdir, isfile
from collections import defaultdict

from config import EXP_PATH, modify_file_by_project

from pddlstream.algorithms.meta import solve, create_parser
from pybullet_tools.bullet_utils import nice


def init_experiment(exp_dir):
    from pybullet_tools.logging import TXT_FILE
    if isfile(TXT_FILE):
        os.remove(TXT_FILE)


def get_parser(exp_name=None):
    parser = create_parser()
    parser.add_argument('-test', type=str, default=exp_name, help='Name of the test case')
    parser.add_argument('-cfree', action='store_true', help='Disables collisions during planning')
    parser.add_argument('-enable', action='store_true', help='Enables rendering during planning')
    parser.add_argument('-teleport', action='store_true', help='Teleports between configurations')
    parser.add_argument('-simulate', action='store_true', help='Simulates the system')
    return parser


def get_args(exp_name=None):
    parser = get_parser(exp_name=exp_name)
    args = parser.parse_args()
    print('Arguments:', args)
    return args


def save_csv(csv_file, data):
    csv_file = modify_file_by_project(csv_file)
    col_names = list(data.keys())
    col_data = list(data.values())

    file_exists = isfile(csv_file)
    with open(csv_file, 'a') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(col_names)
        for row in zip(*col_data):
            writer.writerow(row)


def read_csv(csv_file, summarize=True):
    from tabulate import tabulate
    csv_file = modify_file_by_project(csv_file)
    keys = None
    data = {}
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for j, row in enumerate(reader):
            if j == 0:
                keys = row
                data = defaultdict(list)
            else:
                for i, elem in enumerate(row):
                    data[keys[i]].append(elem if i == 0 else eval(elem))
                    if i == 1 and elem == 0:  ## failed run
                        break

    ## summarize the average, min, max, and count of each column
    if summarize:
        stats = [["name", "avg", "min", "max", "count"]]
        for key, value in data.items():
            if key in ["date"]: continue
            numbers = [sum(value) / len(value), min(value), max(value), len(value)]
            stats.append([key] + [nice(n) for n in numbers])
        print(tabulate(stats, headers="firstrow"))

    return data
