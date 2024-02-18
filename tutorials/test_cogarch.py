from __future__ import print_function
import sys
from os.path import join, abspath, dirname, isdir, isfile
from os import listdir, pardir

ROOT_DIR = abspath(join(dirname(__file__), pardir, pardir))
sys.path.extend([
    join(ROOT_DIR),
    join(ROOT_DIR, 'pddlstream'),
    join(ROOT_DIR, 'pybullet_planning'),
    join(ROOT_DIR, 'lisdf'),
])
# print('\n'.join(sys.path) + '\n')

################################################################

from cogarch_tools.cogarch_run import run_agent
from leap_tools.hierarchical_agent import HierarchicalAgent


def test_domain():
    run_agent(
        agent_class=HierarchicalAgent,
        problem='test_kitchen_chicken_soup',  ## 'test_kitchen_chicken_soup',
        exp_subdir='kitchen', use_rel_pose=True,
        # observation_model='exposed'
    )


if __name__ == '__main__':
    test_domain()
