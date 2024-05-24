from __future__ import print_function

import sys
from os.path import join, abspath, dirname, isdir, isfile
from os import listdir, pardir
RD = abspath(join(dirname(__file__), pardir, pardir))
sys.path.extend([join(RD), join(RD, 'pddlstream'), join(RD, 'pybullet_planning'), join(RD, 'lisdf')])

################################################################

from cogarch_tools.cogarch_run import run_agent
from cogarch_tools.processes.pddlstream_agent import PDDLStreamAgent

from pigi_tools.replay_utils import run_replay, load_pigi_data, REPLAY_CONFIG_DEBUG

from world_builder.paths import OUTPUT_PATH


def test_pigi_data():
    seed = 378277
    goal_variations = [3]
    run_agent(
        agent_class=PDDLStreamAgent, config='config_pigi.yaml',
        goal_variations=goal_variations, seed=seed
    )


def test_replay_pigi_data():
    run_name = 'piginet_data/240523_133446_default'
    run_name = 'piginet_data/240523_144627_default'
    run_replay(REPLAY_CONFIG_DEBUG, load_pigi_data, given_path=join(OUTPUT_PATH, run_name),
               time_step=0.015)  ## , step_by_step=True, save_mp4=False


if __name__ == '__main__':
    test_pigi_data()
    # test_replay_pigi_data()
