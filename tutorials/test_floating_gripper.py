#!/usr/bin/env python

from __future__ import print_function
import os
from os.path import join, abspath, dirname, isdir, isfile
from config import EXP_PATH

from pybullet_tools.utils import disconnect, LockRenderer, has_gui, WorldSaver, wait_if_gui, \
    SEPARATOR, get_aabb, wait_for_duration
from pybullet_tools.bullet_utils import summarize_facts, print_goal, nice
from pybullet_tools.stream_agent import get_stream_info, post_process, move_cost_fn
from pybullet_tools.logging_utils import TXT_FILE

from pybullet_tools.pr2_primitives import control_commands

from pddlstream.language.constants import Equal, AND, print_solution, PDDLProblem
from pddlstream.utils import read, INF, get_file_path, find_unique, Profiler, str_from_object
from pddlstream.algorithms.meta import solve, create_parser
from pddlstream.algorithms.focused import solve_focused

from lisdf_tools.lisdf_utils import pddlstream_from_dir
from lisdf_tools.lisdf_loader import load_lisdf_pybullet
from lisdf_tools.lisdf_planning import pddl_to_init_goal, Problem

from world_builder.actions import apply_commands


""" a scene with two food items, fridge, pot, and basin """
DEFAULT_TEST = 'test_feg_pick'

GOAL = None
GOAL = [('holding', 'hand', 'veggiecabbage#1')]
GOAL = [('holding', 'hand', 'braiserlid#1')]
GOAL = [('on', 'veggiecabbage#1', 'fridge#1::shelf_bottom')]
GOAL = [('on', 'veggiecabbage#1', 'counter#1::indigo_tmp')]
GOAL = [('on', 'veggiecabbage#1', 'braiserbody#1::braiser_bottom')]  ## remove lid first
GOAL = [('graspedhandle', 'faucet#1::joint_faucet_1')]
GOAL = [('cleaned', 'veggiecabbage#1')]

""" a scene with open cabinet doors """
# DEFAULT_TEST = 'test_feg_cabinets_rearrange'
# GOAL = None
# GOAL = [('in', 'oilbottle#1', 'counter#1::sektion')]
# GOAL = [('in', 'oilbottle#1', 'counter#1::sektion'),
#         ('in', 'vinegarbottle#1', 'counter#1::sektion')]
# GOAL = [('storedinspace', '@bottle', 'counter#1::sektion')]

""" a scene with doors that should be opened - very slow """
# DEFAULT_TEST = 'test_feg_closed_doors'
# GOAL = None
# GOAL = [('graspedhandle', 'fridge#1::fridge_door')]
# GOAL = [('graspedhandle', 'fridge#1::fridge_door'),
#         ('holding', 'hand', 'veggiecabbage#1')]
# GOAL = [('holding', 'hand', 'veggiecabbage#1')]
# GOAL = [('on', 'veggiecabbage#1', 'basin#1::basin_bottom')]
# GOAL = [('graspedhandle', 'faucet#1::joint_faucet_0')]
# GOAL = [('cleaned', 'veggiecabbage#1')]


def init_experiment(exp_dir):
    if isfile(TXT_FILE):
        os.remove(TXT_FILE)


def get_args(exp_name):
    parser = create_parser()
    parser.add_argument('-test', type=str, default=exp_name, help='Name of the test case')
    parser.add_argument('-cfree', action='store_true', help='Disables collisions during planning')
    parser.add_argument('-enable', action='store_true', help='Enables rendering during planning')
    parser.add_argument('-teleport', action='store_true', help='Teleports between configurations')
    parser.add_argument('-simulate', action='store_true', help='Simulates the system')
    args = parser.parse_args()
    print('Arguments:', args)
    return args

#####################################


def main(exp_name, verbose=True):

    args = get_args(exp_name)

    exp_dir = join(EXP_PATH, args.test)
    world = load_lisdf_pybullet(exp_dir, width=1280, height=960) ## , width=720, height=560)
    saver = WorldSaver()
    problem = Problem(world)

    pddlstream_problem = pddlstream_from_dir(problem, exp_dir=exp_dir, collisions=not args.cfree,
                                             teleport=args.teleport)  ## , goal=GOAL
    _, _, _, stream_map, init, goal = pddlstream_problem
    world.summarize_all_objects(init)
    stream_info = world.robot.get_stream_info()

    # stream_info = get_stream_info(partial=False, defer=False)  ## problem
    summarize_facts(init, world=world)
    print_goal(goal)
    print(SEPARATOR)
    init_experiment(exp_dir)

    with Profiler():
        with LockRenderer(lock=not args.enable):
            solution = solve_focused(pddlstream_problem, stream_info=stream_info,
                                     planner='ff-astar1', max_planner_time=10, debug=False,
                                     unit_costs=True, success_cost=INF,
                                     max_time=INF, verbose=True, visualize=False,
                                     unit_efforts=True, effort_weight=1,
                                     bind=True, max_skeletons=INF,
                                     search_sample_ratio=0, world=world)
            # solution = solve(pddlstream_problem, algorithm=args.algorithm, unit_costs=args.unit,
            #                  stream_info=stream_info, success_cost=INF, verbose=True, debug=False)
            saver.restore()

    print_solution(solution)
    plan, cost, evaluations = solution
    if (plan is None) or not has_gui():
        disconnect()
        return

    print(SEPARATOR)
    with LockRenderer(lock=not args.enable):
        commands = post_process(problem, plan)
        problem.remove_gripper()
        saver.restore()

    world.remove_redundant_bodies()
    saver.restore()
    wait_if_gui('Execute?')
    if args.simulate:  ## real physics
        control_commands(commands)
    else:
        # apply_commands(State(), commands, time_step=0.01)
        apply_commands(problem, commands, time_step=0.1)
    wait_if_gui('Finish?')
    disconnect()


if __name__ == '__main__':
    main(exp_name=DEFAULT_TEST)
