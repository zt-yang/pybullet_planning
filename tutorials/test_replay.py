#!/usr/bin/env python

from __future__ import print_function

import sys
from os.path import join, abspath, dirname, isdir, isfile
from os import listdir, pardir
RD = abspath(join(dirname(__file__), pardir, pardir))
sys.path.extend([join(RD), join(RD, 'pddlstream'), join(RD, 'pybullet_planning'), join(RD, 'lisdf')])

sys.path.append('/home/zhutiany/Documents/playground/srl_stream/src')


from pigi_tools.replay_utils import run_replay, load_pigi_data, REPLAY_CONFIG_DEBUG


if __name__ == '__main__':
    run_replay(REPLAY_CONFIG_DEBUG, load_pigi_data)
