#!/usr/bin/env python

from __future__ import print_function

import sys
from os.path import join, abspath, dirname, isdir, isfile
from os import listdir, pardir
RD = abspath(join(dirname(__file__), pardir, pardir))
sys.path.extend([join(RD), join(RD, 'pddlstream'), join(RD, 'pybullet_planning'), join(RD, 'lisdf')])

import platform

if platform.node() == 'meraki':
    sys.path.append('/home/yang/Documents/playground/srl_stream/src')
else:
    sys.path.append('/home/zhutiany/Documents/playground/srl_stream/src')


from pigi_tools.replay_utils import run_replay, load_pigi_data, REPLAY_CONFIG_DEBUG


def replay_pr2():
    run_replay(REPLAY_CONFIG_DEBUG, load_pigi_data)


def replay_rummy():
    from rummy_tools.rummy_utils import RUMMY_CONFIG_PATH, load_rummy_data
    replay_config_file = join(RUMMY_CONFIG_PATH, 'replay_online.yaml')
    run_replay(replay_config_file, load_rummy_data)


if __name__ == '__main__':
    replay_pr2()
    # replay_rummy()
