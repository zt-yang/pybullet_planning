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

from cogarch_tools.cogarch_run import main


def test_vlm_tamp_domain():
    main(
        problem='test_kitchen_chicken_soup', exp_subdir='kitchen_gpt', use_rel_pose=True,
        # observation_model='exposed'
    )


if __name__ == '__main__':
    test_vlm_tamp_domain()
