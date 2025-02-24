import os
import sys
from os.path import join, abspath, dirname, isdir, isfile
R = abspath(join(dirname(__file__), os.pardir, os.pardir))
sys.path.extend([R] + [join(R, name) for name in ['pybullet_planning', 'lisdf', 'pddlstream']])
# print('\n'.join([p for p in sys.path if 'vlm-tamp' in p]))

from pybullet_tools.utils import read_pickle, set_renderer, wait_for_user, PI

from tutorials.test_grasps import run_test_grasps, run_test_handle_grasps_counter

from cogarch_tools.cogarch_run import run_agent
from cogarch_tools.processes.teleop_agent import TeleOpAgent

from world_builder.paths import EXP_PATH

from leap_tools.hierarchical_agent import HierarchicalAgent


namo_kwargs = dict(domain='pddl_domains/feg_namo_domain.pddl', stream='pddl_domains/feg_namo_stream.pddl')


def test_feg_grasps():
    kwargs = dict(categories=['veggiesweetpotato'], skip_grasps=False, base_q=(0, 0, 0, 0))
    kwargs['categories'] = ['Food']

    ## --- step 1: find tool_from_hand transformation
    debug_kwargs = dict(verbose=True, visualize=False, retain_all=False, top_grasp_tolerance=PI/4)

    run_test_grasps('feg', **debug_kwargs, **kwargs)


def test_block_domain_feg():
    problem = ['test_pick'][1]
    run_agent(config='config_feg.yaml', problem=problem)


def test_cart_domain_feg():
    from pddl_domains.pddl_utils import update_namo_pddl
    update_namo_pddl(domain_name='feg')

    problem = ['test_navigation', 'test_cart_pull'][0]
    run_agent(config='config_feg.yaml', problem=problem, **namo_kwargs)


if __name__ == '__main__':
    test_feg_grasps()
    # test_block_domain_feg()
    # test_cart_domain_feg()
