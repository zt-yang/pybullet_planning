#!/usr/bin/env python

from __future__ import print_function
import time
from os.path import join
from config import EXP_PATH

from pybullet_tools.utils import disconnect, LockRenderer, has_gui, WorldSaver, wait_if_gui, \
    SEPARATOR
from pybullet_tools.bullet_utils import summarize_facts, print_goal, get_datetime
from pybullet_tools.stream_agent import post_process

from pybullet_tools.pr2_primitives import control_commands
from pddlstream.language.constants import print_solution
from pddlstream.utils import INF, Profiler
from pddlstream.algorithms.meta import solve

from lisdf_tools.lisdf_utils import pddlstream_from_dir
from lisdf_tools.lisdf_loader import load_lisdf_pybullet
from lisdf_tools.lisdf_planning import Problem

from world_builder.actions import apply_commands

from examples.test_utils import init_experiment, get_parser, save_csv, read_csv

DEFAULT_TEST = 'test_pr2_kitchen'


#####################################


def main(args, execute=True):

    exp_dir = join(EXP_PATH, args.test)
    world = load_lisdf_pybullet(exp_dir, width=1440, height=1120)
    saver = WorldSaver()
    problem = Problem(world)

    pddlstream_problem = pddlstream_from_dir(problem, exp_dir=exp_dir, collisions=not args.cfree,
                                             teleport=args.teleport, replace_pddl=False)
    world.summarize_all_objects()

    stream_info = world.robot.get_stream_info(partial=False, defer=False)
    _, _, _, stream_map, init, goal = pddlstream_problem
    summarize_facts(init, world=world)
    print_goal(goal)
    print(SEPARATOR)
    init_experiment(exp_dir)

    planner_kwargs = dict(algorithm=args.algorithm, unit_costs=args.unit, stream_info=stream_info,
                          success_cost=INF, verbose=True, debug=False, world=world)
    # planner_kwargs, plan_dataset = get_diverse_kwargs(planner_kwargs)

    with Profiler():
        with LockRenderer(lock=not args.enable):
            solution = solve(pddlstream_problem, **planner_kwargs)
            saver.restore()

    print_solution(solution)
    plan, cost, evaluations = solution

    ## failed without a plan
    if plan is None:
        disconnect()
        return 0, None
    print(SEPARATOR)

    ## play out the plan
    if execute and has_gui():
        with LockRenderer(lock=not args.enable):
            commands = post_process(problem, plan)
            problem.remove_gripper()
            saver.restore()

        saver.restore()
        wait_if_gui('Execute?')
        if args.simulate:  ## real physics
            control_commands(commands)
        else:
            apply_commands(problem, commands, time_step=0.1)
        wait_if_gui('Finish?')

    disconnect()
    return 1, len(plan)


def run_multiple(args, n=10):
    """ run planner multiple times & print stats """
    csv_file = join(EXP_PATH, args.test, 'results.csv')

    success_count = []
    plan_len = []
    time_passed = []
    dates = []
    for i in range(n):
        start = time.time()
        result, length = main(args, execute=False)
        success_count.append(result)
        plan_len.append(length)
        time_passed.append(time.time() - start)
        dates.append(get_datetime())

    ## save plan_len and time_passed in cvs file
    save_csv(csv_file, {'date': dates, 'success': success_count,
                        'plan_length': plan_len, 'time_passed': time_passed})
    read_csv(csv_file, summarize=True)


if __name__ == '__main__':
    """
    python test_pddlstream.py -t test_pr2_kitchen -n 10
    """
    parser = get_parser(exp_name=DEFAULT_TEST)
    parser.add_argument('-n', type=int, default=1, help='Number of trials')
    args = parser.parse_args()
    if args.n == 1:
        main(args)
    else:
        run_multiple(args, n=args.n)
