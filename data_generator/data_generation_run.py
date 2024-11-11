from __future__ import print_function
import shutil
import pickle
import os
import time
import random
import copy
import json
import argparse
from os.path import join, abspath, dirname, isdir, isfile, basename
from os import listdir

from pddlstream.language.constants import Equal, AND, print_solution, PDDLProblem
from pddlstream.algorithms.meta import solve, create_parser

from pybullet_tools.utils import disconnect, LockRenderer, has_gui, WorldSaver, wait_if_gui, \
    SEPARATOR, get_aabb, wait_for_duration, has_gui, reset_simulation, set_random_seed, \
    set_numpy_seed, set_renderer
from pybullet_tools.bullet_utils import nice, get_datetime, initialize_logs
from pybullet_tools.stream_agent import solve_multiple, post_process, pddlstream_from_state_goal, \
    create_cwd_saver
from pybullet_tools.pr2_primitives import control_commands, apply_commands
from pybullet_tools.logging_utils import parallel_print, myprint, summarize_facts, print_goal

from world_builder.world import State
from world_builder.actions import apply_commands
from world_builder.builders import sample_world_and_goal, save_world_problem


def data_generation_process(config, world_only=False):
    """ exist a version in cognitive-architectures for generating mini-datasets (single process),
        run in kitchen-worlds for parallelization, but no reliable planning time data

        inside each data folder, to be generated:
        - before planning:
            [x] scene.lisdf
            [x] problem.pddl
            [x] planning_config.json
            [x] log.txt (generated before planning)
        - after planning:
            [x] plan.json
            [x] commands.pkl
            [x] log.json (updated by pddlstream)
    """

    new_config = copy.deepcopy(config)
    seed = config.seed
    if seed is None:
        seed = random.randint(0, 10 ** 6 - 1)
        new_config.seed = seed
    set_random_seed(seed)
    set_numpy_seed(seed)
    print('Seed:', seed)

    initialize_logs()
    if not new_config.parallel:
        clear_failed_out_dirs(dirname(new_config.data.out_dir))

    """ STEP 1 -- GENERATE SCENES """
    world, goal = sample_world_and_goal(new_config)
    if world_only:
        lisdf_file = save_world_problem(world, goal, config)
        reset_simulation()
        return lisdf_file

    domain_path = abspath(config.planner.domain_pddl)
    stream_path = abspath(config.planner.stream_pddl)

    saver = WorldSaver()
    cwd_saver = create_cwd_saver()  ## so all the log will be printed to tmp/
    print_fn = parallel_print  ## if args.parallel else myprint
    print_fn(config)

    state = State(world)
    pddlstream_problem = pddlstream_from_state_goal(
        state, goal, domain_pddl=domain_path, stream_pddl=stream_path,
        custom_limits=world.robot.custom_limits, print_fn=print_fn, **config.streams)
    stream_info = world.robot.get_stream_info()

    kwargs = {'visualize': config.planner.visualize}
    if config.planner.diverse:
        kwargs.update(dict(
            diverse=True,
            downward_time=config.planner.downward_time,  ## max time to get 100, 10 sec, 30 sec for 300
            evaluation_time=60,  ## on each skeleton
            max_plans=200,  ## number of skeletons
        ))
    start = time.time()
    solution, cwd_saver = solve_multiple(pddlstream_problem, stream_info, lock=config.sim.lock,
                                         cwd_saver=cwd_saver, world=world, **kwargs)

    print_solution(solution)
    plan, cost, evaluations = solution

    """ ============== save world configuration ==================== """
    tmp_dir = cwd_saver.tmp_cwd
    cwd_saver.restore()
    exp_dir = save_world_problem(world, goal, new_config)

    """ =============== log plan and planning time =============== """
    t = None if config.parallel else round(time.time() - start, 3)
    if plan is None:
        plan_log = None
        plan_len = None
        init = None
    else:
        plan_log = [str(a) for a in plan]
        plan_len = len(plan)
        init = [[str(a) for a in f] for f in evaluations.preimage_facts]
    time_log = [{
        'planning': t, 'plan': plan_log, 'plan_len': plan_len, 'init': init
    }, {'total_planning': t}]
    with open(join(exp_dir, f'plan.json'), 'w') as f:
        json.dump(time_log, f, indent=4)

    """ =============== save planing log =============== """
    txt_file = join(tmp_dir, 'txt_file.txt')
    if isfile(txt_file):
        shutil.move(txt_file, join(exp_dir, f"log.txt"))
    txt_file = join(tmp_dir, 'visualizations', 'log.json')
    if isfile(txt_file):
        shutil.move(txt_file, join(exp_dir, f"log.json"))
    cwd_saver.restore()

    """ =============== save commands for replay =============== """
    with LockRenderer(lock=config.sim.lock):
        commands = post_process(state, plan, simulate=config.sim.simulate)
        state.remove_gripper()
        saver.restore()
    with open(join(exp_dir, f"commands.pkl"), 'wb') as f:
        pickle.dump(commands, f)

    """ =============== save visuals =============== """
    # if save_rgb:
    #     world.visualize_image(img_dir=output_dir, rgb=True)
    # if save_depth:
    #     world.visualize_image(img_dir=output_dir)

    """ =============== visualize the plan =============== """
    if (plan is None) or not has_gui():
        reset_simulation()
        disconnect()
        return

    print(f'\nSAVED DATA in {abspath(exp_dir)}\n')

    print(SEPARATOR)
    visualize = not new_config.parallel and new_config.n_data == 1 and not new_config.sim.skip_prompt
    saver.restore()
    set_renderer(True)
    if config.sim.simulate:  ## real physics
        control_commands(commands)
    else:
        if visualize:
            wait_if_gui('Execute?')
        apply_commands(state, commands, time_step=config.sim.time_step, verbose=False)
        if visualize:
            wait_if_gui('Exit?')
    print(SEPARATOR)
    reset_simulation()
    disconnect()


def clear_failed_out_dirs(out_dir):
    if not isdir(out_dir):
        return
    exp_dirs = [join(out_dir, f) for f in listdir(out_dir) if isdir(join(out_dir, f))]
    for exp_dir in exp_dirs:
        solution_file = join(exp_dir, 'plan.json')
        if not isfile(solution_file):
            shutil.rmtree(exp_dir)
        else:
            solution = json.load(open(solution_file, 'r'))[0]['plan']
            if solution is None:
                shutil.rmtree(exp_dir)
