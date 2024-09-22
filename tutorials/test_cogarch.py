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

from pddl_domains.pddl_utils import *

domain_name = 'pddl_domains/mobile_v5'
domain_kwargs = dict(agent_class=HierarchicalAgent, config='config_dev.yaml',
                     domain_pddl=f'{domain_name}_domain.pddl', stream_pddl=f'{domain_name}_stream.pddl')


def test_pick_place_domain():
    simple_problem = ['test_pick', 'test_small_sink'][0]
    run_agent(
        problem=simple_problem, **domain_kwargs  #, **hpn_kwargs
    )


def test_nvidia_kitchen_domain():
    """ minimum examples to test basic motion and mechanism """
    kitchen_problem = ['test_kitchen_chicken_soup', 'test_kitchen_braiser',
                       'test_skill_knob_stove', 'test_kitchen_fridge',
                       'test_kitchen_drawers', 'test_skill_knob_faucet'][-2]
    # update_kitchen_action_pddl()
    update_kitchen_pull_pddl()
    run_agent(
        problem=kitchen_problem,
        dual_arm=False, visualization=False, top_grasp_tolerance=0.8,
        separate_base_planning=False, **domain_kwargs,
        # use_skeleton_constraints=True,
        # observation_model='exposed'
    )


def test_cooking_domain():
    """ testing skill in the full kitchen, with a single arm
    remember to include world.remove_bodies_from_planning(goals)
    """
    kitchen_problem = ['test_kitchen_sprinkle', 'test_kitchen_nudge_door',
                       'test_kitchen_faucet_braiser_and_stove'][-1]
    run_agent(
        problem=kitchen_problem,
        dual_arm=False, top_grasp_tolerance=None, visualization=False,
        separate_base_planning=False, **domain_kwargs, # use_skeleton_constraints=True,
        # observation_model='exposed'
    )


if __name__ == '__main__':
    # test_pick_place_domain()
    test_nvidia_kitchen_domain()
    # test_cooking_domain()
