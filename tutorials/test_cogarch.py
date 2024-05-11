from __future__ import print_function

import sys
from os.path import join, abspath, dirname, isdir, isfile
from os import listdir, pardir
RD = abspath(join(dirname(__file__), pardir, pardir))
sys.path.extend([join(RD), join(RD, 'pddlstream'), join(RD, 'pybullet_planning'), join(RD, 'lisdf')])

################################################################

from cogarch_tools.cogarch_run import run_agent
from cogarch_tools.processes.pddlstream_agent import PDDLStreamAgent
from leap_tools.hierarchical_agent import HierarchicalAgent, hpn_kwargs


def test_pick_place_domain():
    simple_problem = ['test_pick', 'test_small_sink'][0]
    run_agent(
        agent_class=HierarchicalAgent, config='config_dev.yaml', problem=simple_problem,
        # **hpn_kwargs
    )


def test_nvidia_kitchen_domain():
    kitchen_problem = ['test_kitchen_chicken_soup', 'test_kitchen_braiser', None][1]
    run_agent(
        agent_class=HierarchicalAgent, config='config_dev.yaml', problem=kitchen_problem,
        dual_arm=False, top_grasp_tolerance=0.8, visualization=False,
        separate_base_planning=False, # use_skeleton_constraints=True,
        # observation_model='exposed'
    )


def test_pigi_data():
    run_agent(
        agent_class=PDDLStreamAgent, config='config_pigi.yaml',
    )


if __name__ == '__main__':
    test_pick_place_domain()
    # test_nvidia_kitchen_domain()
    # test_pigi_data()
