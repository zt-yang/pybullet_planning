import os
import sys
from os.path import join, abspath, dirname, isdir, isfile
R = abspath(join(dirname(__file__), os.pardir))
sys.path.extend([R] + [join(R, name) for name in ['pybullet_planning', 'lisdf', 'pddlstream']])

from pybullet_tools.utils import read_pickle, set_renderer, wait_for_user, PI

from tutorials.test_grasps import test_grasps, test_handle_grasps_counter

from cogarch_tools.cogarch_run import run_agent
from cogarch_tools.processes.teleop_agent import TeleOpAgent

from world_builder.paths import EXP_PATH

from leap_tools.hierarchical_agent import HierarchicalAgent


def test_spot_grasps():
    kwargs = dict(categories=['VeggieCabbage'], skip_grasps=False, base_q=(0, 0, 0, 0))
    kwargs['categories'] = ['Food']

    ## --- step 1: find tool_from_hand transformation
    debug_kwargs = dict(verbose=True, test_rotation_matrix=True, skip_grasp_index=1,
                        test_translation_matrix=False)

    ## --- step 2: find tool_from_root transformation (multiple rotations may look correct,
    #               but only one works after running IR - IK)
    ## (1.571, 3.142, -1.571) (1.571, 3.142, 1.571) (1.571, 3.142, 0) (-1.571, 0, 0)
    debug_kwargs = dict(verbose=True, test_attachment=True, visualize=False, retain_all=False)

    ## --- step 3: verify all grasps generated for one object
    # debug_kwargs = dict(verbose=True, test_attachment=False, visualize=True, retain_all=True)

    ## --- step 4: verify top_grasp_tolerance filtering
    debug_kwargs = dict(verbose=True, visualize=False, retain_all=False, top_grasp_tolerance=PI/4)

    test_grasps('spot', **debug_kwargs, **kwargs)
    # test_grasps('pr2', **debug_kwargs, **kwargs)
    # test_grasps('feg', **debug_kwargs, **kwargs)


def test_office_chair_domain_spot():
    problem = ['test_spot_pick', 'test_office_chairs'][1]
    run_agent(config='config_spot.yaml', problem=problem)


if __name__ == '__main__':
    # test_spot_grasps()
    test_office_chair_domain_spot()
