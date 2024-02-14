#!/usr/bin/env python

from __future__ import print_function
from os.path import join
from world_builder.paths import pbp_path
from pigi_tools.replay_utils import run_replay

CONFIG_YAML_PATH = join(pbp_path, 'pigi_tools', 'config', 'replay_debug.yaml')


if __name__ == '__main__':
    from pigi_tools.replay_utils import load_pigi_data

    run_replay(CONFIG_YAML_PATH, load_pigi_data)
