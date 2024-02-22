from __future__ import print_function

import sys
from os.path import join, abspath, dirname, isdir, isfile
from os import listdir, pardir
RD = abspath(join(dirname(__file__), pardir, pardir))
sys.path.extend([join(RD), join(RD, 'pddlstream'), join(RD, 'pybullet_planning'), join(RD, 'lisdf')])

################################################################

from cogarch_tools.cogarch_run import run_agent
from leap_tools.hierarchical_agent import HierarchicalAgent


problem = ['test_kitchen_chicken_soup', 'test_kitchen_braiser', None][0]


def test_domain():
    run_agent(
        agent_class=HierarchicalAgent, problem=problem, use_rel_pose=True,
        # observation_model='exposed'
    )


if __name__ == '__main__':
    test_domain()
