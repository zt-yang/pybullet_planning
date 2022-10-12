from __future__ import print_function
import os
import json
from os.path import join, abspath, dirname, isdir, isfile
from config import EXP_PATH

from pddlstream.algorithms.meta import solve, create_parser


def init_experiment(exp_dir):
    from pybullet_tools.logging import TXT_FILE
    if isfile(TXT_FILE):
        os.remove(TXT_FILE)


def get_args(exp_name=None):
    parser = create_parser()
    parser.add_argument('-test', type=str, default=exp_name, help='Name of the test case')
    parser.add_argument('-cfree', action='store_true', help='Disables collisions during planning')
    parser.add_argument('-enable', action='store_true', help='Enables rendering during planning')
    parser.add_argument('-teleport', action='store_true', help='Teleports between configurations')
    parser.add_argument('-simulate', action='store_true', help='Simulates the system')
    args = parser.parse_args()
    print('Arguments:', args)
    return args