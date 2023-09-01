#!/usr/bin/env python

from __future__ import print_function
import os
import json
from os.path import join, abspath, dirname, isdir, isfile
from config import EXP_PATH

from pybullet_tools.utils import disconnect, LockRenderer, has_gui, WorldSaver, wait_if_gui, \
    SEPARATOR, get_aabb, wait_for_duration
from pybullet_tools.bullet_utils import summarize_facts, print_goal, nice
from pybullet_tools.pr2_agent import get_stream_info, post_process, move_cost_fn, get_stream_map

from pybullet_tools.pr2_primitives import control_commands
from pddlstream.language.constants import Equal, AND, print_solution, PDDLProblem
from pddlstream.utils import read, INF, get_file_path, find_unique, Profiler, str_from_object
from pddlstream.algorithms.meta import solve, create_parser

from lisdf_tools.lisdf_loader import load_lisdf_pybullet, pddlstream_from_dir
from lisdf_tools.lisdf_planning import pddl_to_init_goal, Problem

from world_builder.actions import apply_actions

from test_utils import get_args, init_experiment

DEFAULT_TEST = 'test_pr2_kitchen' ## 'test_pr2_kitchen' | 'test_blocks_kitchen' ##


#####################################


def main(exp_name, verbose=True):

    args = get_args(exp_name)

    exp_dir = join(EXP_PATH, args.test)
    world = load_lisdf_pybullet(exp_dir, width=720, height=560)
    saver = WorldSaver()
    problem = Problem(world)

    pddlstream_problem = pddlstream_from_dir(problem, exp_dir=exp_dir, collisions=not args.cfree,
                                             teleport=args.teleport, replace_pddl=True)
    world.summarize_all_objects()

    stream_info = world.robot.get_stream_info(partial=False, defer=False)
    _, _, _, stream_map, init, goal = pddlstream_problem
    summarize_facts(init, world=world)
    print_goal(goal)
    print(SEPARATOR)
    init_experiment(exp_dir)

    with Profiler():
        with LockRenderer(lock=not args.enable):
            solution = solve(pddlstream_problem, algorithm=args.algorithm, unit_costs=args.unit,
                             stream_info=stream_info, success_cost=INF, verbose=True, debug=False)
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

    saver.restore()
    wait_if_gui('Execute?')
    if args.simulate:  ## real physics
        control_commands(commands)
    else:
        # apply_commands(State(), commands, time_step=0.01)
        apply_actions(problem, commands, time_step=0.1)
    wait_if_gui('Finish?')
    disconnect()


if __name__ == '__main__':
    main(exp_name=DEFAULT_TEST)
