#!/usr/bin/env python

from __future__ import print_function

import sys
from os.path import join, abspath, dirname, isdir, isfile
from os import listdir, pardir
RD = abspath(join(dirname(__file__), pardir, pardir))
sys.path.extend([join(RD), join(RD, 'pddlstream'), join(RD, 'pybullet_planning'), join(RD, 'lisdf')])

from world_builder.paths import PBP_PATH
from pigi_tools.replay_utils import run_replay

CONFIG_YAML_PATH = join(PBP_PATH, 'pigi_tools', 'config', 'replay_debug.yaml')


if __name__ == '__main__':
    from pigi_tools.replay_utils import load_pigi_data

    run_replay(CONFIG_YAML_PATH, load_pigi_data)
