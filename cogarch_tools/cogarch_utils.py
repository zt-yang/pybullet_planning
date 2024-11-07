from __future__ import print_function

import shutil
import argparse
import numpy as np
import pybullet as p
import random

import os
from os import listdir
from os.path import join, abspath, dirname, isdir, isfile

from pybullet_tools.utils import connect, draw_pose, enable_preview, unit_pose, set_random_seed, set_camera_pose, \
    set_numpy_seed, add_parameter, add_button

from problem_sets import problem_fn_from_name

from world_builder.paths import PBP_PATH

from pigi_tools.data_utils import get_feasibility_checker

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


PROBLEM_CONFIG_PATH = abspath(join(dirname(__file__), 'configs'))


def parse_config(path):
    import yaml
    from pathlib import Path
    from argparse import Namespace
    conf = yaml.safe_load(Path(path).read_text())
    conf = Namespace(**conf)
    return conf


def parse_config_for_agent(path):
    from argparse import Namespace
    conf = parse_config(path)
    conf.sim = Namespace(**conf.sim)
    conf.data = Namespace(**conf.data)
    conf.problem = Namespace(**conf.problem)
    conf.planner = Namespace(**conf.planner)
    conf.robot = Namespace(**conf.robot)
    conf.streams = Namespace(**conf.streams)
    if hasattr(conf, 'agent'):
        conf.agent = Namespace(**conf.agent)
    if hasattr(conf, 'rummy_pipeline'):
        conf.rummy_pipeline = Namespace(**conf.rummy_pipeline)
    if isinstance(conf.seed, str) and conf.seed.lower() == 'none':
        conf.seed = None
    return conf


def get_default_agent_parser_given_config(conf):
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=conf.seed, help='')
    if hasattr(conf, 'debug'):
        parser.add_argument('--debug', action='store_true', default=conf.debug, help='')

    ## -------- simulation related
    parser.add_argument('-v', '--viewer', action='store_true', default=conf.sim.viewer,
                        help='When enabled, enables the PyBullet viewer.')
    parser.add_argument('--lock', action='store_true', default=conf.sim.lock,
                        help='When enabled, locks the viewer during planning.')
    parser.add_argument('-d', '--drive', action='store_true', default=conf.sim.drive, help='')
    parser.add_argument('-t', '--time_step', type=float, default=conf.sim.time_step, help='')
    parser.add_argument('--window_width', type=int, default=conf.sim.window_width, help='')
    parser.add_argument('--window_height', type=int, default=conf.sim.window_height, help='')
    parser.add_argument('-cam', '--camera', action='store_true', default=conf.sim.camera, help='')
    parser.add_argument('-seg', '--segment', action='store_true', default=conf.sim.segment, help='')
    parser.add_argument('-mon', '--monitoring', action='store_true', default=conf.sim.monitoring, help='')
    parser.add_argument('--show_object_names', action='store_true', default=conf.sim.show_object_names, help='')

    ## --------- streams related
    parser.add_argument('-c', '--cfree', action='store_true', default=conf.streams.cfree,
                        help='When enabled, DISABLES collision checking.')
    parser.add_argument('--movable', action='store_true', default=conf.streams.disable_movable_collision,
                        help='When enabled, DISABLES movable collisions.')
    parser.add_argument('--base', action='store_true', default=conf.streams.disable_base_collision,
                        help='When enabled, DISABLES base collisions.')
    parser.add_argument('--teleport', action='store_true', default=conf.streams.teleport,
                        help='When enabled, teleports the robot between base configurations.')

    ## -------- planning problem related
    parser.add_argument('-p', '--problem', type=str, default=conf.problem.problem,
                        help='name of the problem function that initiate both the world and goal')
    parser.add_argument('-domain', '--domain_pddl', type=str, default=conf.problem.domain_pddl,
                        help='name to the domain pddl file')
    parser.add_argument('-stream', '--stream_pddl', type=str, default=conf.problem.stream_pddl,
                        help='name to the stream pddl file')
    parser.add_argument('--use_skeleton_constraints', action='store_true',
                        default=conf.problem.use_skeleton_constraints,
                        help='When enabled, planner will use skeleton constraints predefined in problem loader, if any')
    parser.add_argument('--use_subgoal_constraints', action='store_true',
                        default=conf.problem.use_subgoal_constraints,
                        help='When enabled, planner will use subgoal constraints predefined in problem loader, if any')
    parser.add_argument('-rel', '--use_rel_pose', action='store_true', default=conf.problem.use_rel_pose,
                        help='When enabled, domain will use relative pose for objects in movable links, e.g. drawers')
    parser.add_argument('--preview_scene', action='store_true', default=conf.problem.preview_scene,
                        help='When enabled, previews the scene and press Enter before solving the problem.')
    parser.add_argument('--preview_plan', action='store_true', default=conf.problem.preview_plan,
                        help='When enabled, previews the plan before returning.')

    ## -------- output data related
    parser.add_argument('--exp_dir', type=str, default=conf.data.exp_dir,
                        help='path to `experiments` to save outputs')
    parser.add_argument('--exp_subdir', type=str, default=conf.data.exp_subdir,
                        help='name of the sub-directory in `../experiments` to save outputs')
    parser.add_argument('--exp_name', type=str, default=conf.data.exp_name,
                        help='name of the comparison group for which planning time and success rate will be compared')
    parser.add_argument('--record_problem', action='store_true', default=conf.data.record_problem,
                        help='When enabled, the world lisdf, problem pddl, and solution json will be saved')
    parser.add_argument('-mp4', '--record_mp4', action='store_true', default=conf.data.record_mp4,
                        help='When enabled, the solution mp4 will be saved')
    parser.add_argument('--save_testcase', action='store_true', default=conf.data.save_testcase,
                        help='When enabled, the problem and solution will be saved into test_cases')

    ## -------- PDDLStream planner related
    parser.add_argument('-viz', '--visualization', action='store_true', default=conf.planner.visualization,
                        help='When enabled, PDDLStream will generate stream plans and stream summary')
    parser.add_argument('--scene_only', action='store_true', default=conf.planner.scene_only,
                        help='When enabled, generate the scene and exit the problem')
    parser.add_argument('--use_heuristic_fc', action='store_true', default=conf.planner.use_heuristic_fc,
                        help='When enabled, use heuristic feasibility checker to speed up planning')
    parser.add_argument('--data', action='store_true', default=conf.planner.dataset_mode,
                        help='When enabled, collects plan data.')
    parser.add_argument('--evaluation_time', type=float, default=conf.planner.evaluation_time,
                        help='Amount of time for overall TAMP')
    parser.add_argument('--downward_time', type=float, default=conf.planner.downward_time,
                        help='Amount of time for task planning with Fast Downward')
    parser.add_argument('--max_plans', type=int, default=conf.planner.max_plans,
                        help='In diverse planning / dataset mode, number of skeletons to consider')
    parser.add_argument('--max_solutions', type=int, default=conf.planner.max_solutions,
                        help='In diverse planning / dataset mode, number of solutions to return')
    parser.add_argument('--log_failures', action='store_true', default=conf.planner.log_failures,
                        help='When enabled, log failed streams for analysis.')

    ## -------- real robot related
    parser.add_argument('--execute', action='store_true', default=False,
                        help='Ignore this, just for plumbing purposes')
    return parser


def parse_agent_args(config='config_dev.yaml', config_root=PROBLEM_CONFIG_PATH,
                     get_agent_parser_given_config=get_default_agent_parser_given_config,
                     modify_agent_args_fn=None, **kwargs):
    """ default values are given at yaml, custom values are provided by commandline flags, overwritten by kwargs """

    from pybullet_tools.logging_utils import myprint as print
    np.set_printoptions(precision=3, threshold=3, suppress=True)  #, edgeitems=1) #, linewidth=1000)

    conf = parse_config_for_agent(join(config_root, config))
    parser = get_agent_parser_given_config(conf)

    ## seed determines asset instances and object poses initiated for the problem
    args = parser.parse_args()

    ## replace the default values with values provided, when running in IDE
    for k, v in kwargs.items():
        args.__dict__[k] = v

    ## initialize random seed
    seed = args.seed
    if seed is None:
        seed = random.randint(0, 10**6-1)
    else:
        args.viewer = True
    set_random_seed(seed)
    set_numpy_seed(seed)
    args.seed = seed

    if args.exp_subdir is None and isinstance(args.problem, str):
        args.exp_subdir = args.problem

    ## other args, especially those related to problem and planner may be added directly in config files
    args.__dict__['robot_builder_args'] = conf.robot.__dict__
    for attr in ['problem', 'planner', 'agent', 'data', 'streams', 'sim']:
        if attr not in conf.__dict__:
            continue
        for k, v in conf.__dict__[attr].__dict__.items():
            if k not in args.__dict__:
                args.__dict__[k] = v

    ## update robot_builder_args
    if args.record_mp4:
        args.robot_builder_args['draw_base_limits'] = False
    if hasattr(args, 'separate_base_planning'):
        args.robot_builder_args['separate_base_planning'] = args.separate_base_planning
    if hasattr(args, 'dual_arm'):
        args.robot_builder_args['dual_arm'] = args.dual_arm

    ## other processing
    args.exp_dir = abspath(join(PBP_PATH, args.exp_dir))

    if modify_agent_args_fn is not None:
        args = modify_agent_args_fn(args)

    print(f'Seed: {args.seed}')
    print(f'Args: {args}')
    return args


def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    import sys
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None


def get_pddlstream_kwargs(args, skeleton, subgoals, initializer, pddlstream_debug=False,
                          soft_subgoals=False, max_evaluation_plans=30, max_complexity=5,
                          max_iterations=4):
    fc = None if not args.use_heuristic_fc else get_feasibility_checker(initializer, mode='heuristic')
    debug = args.pddlstream_debug if hasattr(args, 'pddlstream_debug') else pddlstream_debug
    solver_kwargs = dict(
        skeleton=skeleton,
        subgoals=subgoals,
        soft_subgoals=soft_subgoals,
        fc=fc,
        collect_dataset=args.data,
        lock=args.lock,
        preview=args.preview_plan,
        max_solutions=args.max_solutions,
        visualization=args.visualization,  ## to draw constraint networks and stream plans
        log_failures=args.log_failures,  ## to summarize failed streams
        evaluation_time=args.evaluation_time,
        downward_time=args.downward_time,
        stream_planning_timeout=args.stream_planning_timeout,
        total_planning_timeout=args.total_planning_timeout,
        max_plans=args.max_plans,  ## used by diverse planning
        max_evaluation_plans=max_evaluation_plans,  ## used by focused planning loop
        max_complexity=max_complexity,
        max_iterations=max_iterations,
        debug=debug
    )
    for k in ['soft_subgoals', 'max_evaluation_plans', 'max_complexity', 'max_iterations']:
        if hasattr(args, k):
            solver_kwargs[k] = getattr(args, k)
    solver_kwargs = update_timeout_for_debugging(solver_kwargs)
    return solver_kwargs


def update_timeout_for_debugging(solver_kwargs):
    if debugger_is_active():
        solver_kwargs['evaluation_time'] *= 2
        solver_kwargs['total_planning_timeout'] *= 2
        solver_kwargs['stream_planning_timeout'] *= 2
    return solver_kwargs


def init_pybullet_client(args):
    connect(use_gui=args.viewer, shadows=False, width=args.window_width, height=args.window_height)

    if args.camera:
        enable_preview()
    if not args.segment:
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, False)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, False)

    draw_pose(unit_pose(), length=1.)


def clear_planning_dir(run_dir=dirname(__file__)):
    for cwd in [run_dir, dirname(run_dir)]:
        for dir_name in ['temp', 'statistics', 'visualization', 'cache']:
            temp_dir = join(cwd, dir_name)
            if isdir(temp_dir):
                shutil.rmtree(temp_dir)

        log_file = join(cwd, 'txt_file.txt')
        if isfile(log_file):
            os.remove(log_file)

##################################################


def get_pddlstream_problem(args, **kwargs):

    problem_kwargs = dict(
        movable_collisions=not args.movable,
        motion_collisions=not args.movable,
        base_collisions=not args.base,
    )

    def set_kitchen_camera_pose():
        set_camera_pose(camera_point=[4, 7, 4], target_point=[3, 7, 2])

    def set_default_camera_pose():
        set_camera_pose(camera_point=[3, -3, 3], target_point=[0, 0, 0])

    set_default_camera_pose()

    ## ------------------- old PR2 problem_sets
    if callable(args.problem):
        problem_fn = args.problem
    # elif args.problem == 'test_pick':
    #     set_kitchen_camera_pose()
    #     from problem_sets.pr2_problems import test_pick as problem_fn
    # elif args.problem == 'test_plated_food':
    #     set_kitchen_camera_pose()
    #     from problem_sets.pr2_problems import test_plated_food as problem_fn
    # elif args.problem == 'test_small_sink':
    #     set_kitchen_camera_pose()
    #     from problem_sets.pr2_problems import test_small_sink as problem_fn
    # elif args.problem == 'test_five_tables':
    #     from problem_sets.pr2_problems import test_five_tables as problem_fn
    # elif args.problem == 'test_exist_omelette':
    #     from problem_sets.pr2_problems import test_exist_omelette as problem_fn
    #
    # elif args.problem == 'test_cart_obstacle':
    #     from problem_sets.pr2_problems import test_cart_obstacle as problem_fn
    # elif args.problem == 'test_moving_carts':
    #     set_camera_pose(camera_point=[4, -2, 4], target_point=[0, -2, 0])
    #     set_camera_pose(camera_point=[5, -2, 4], target_point=[1, -2, 0])  ## laundry area
    #     from problem_sets.pr2_problems import test_moving_carts as problem_fn
    # elif args.problem == 'test_three_moving_carts':
    #     set_camera_pose(camera_point=[5, 0, 4], target_point=[1, 0, 0])
    #     from problem_sets.pr2_problems import test_three_moving_carts as problem_fn
    # elif args.problem == 'test_fridge_pose':
    #     set_camera_pose(camera_point=[3, 6.5, 2], target_point=[1, 4, 1])
    #     from problem_sets.pr2_problems import test_fridge_pose as problem_fn
    # elif args.problem == 'test_kitchen_oven':
    #     set_camera_pose(camera_point=[3, 5, 3], target_point=[0, 6, 1])
    #     from problem_sets.pr2_problems import test_kitchen_oven as problem_fn
    # elif args.problem == 'test_oven_egg':
    #     set_camera_pose(camera_point=[3, 5, 3], target_point=[0, 6, 1])
    #     set_kitchen_camera_pose()
    #     from problem_sets.pr2_problems import test_oven_egg as problem_fn
    # elif args.problem == 'test_braiser_lid':
    #     set_kitchen_camera_pose()
    #     from problem_sets.pr2_problems import test_braiser_lid as problem_fn
    # elif args.problem == 'test_egg_movements':
    #     set_camera_pose(camera_point=[3, 5, 3], target_point=[0, 6, 1])
    #     from problem_sets.pr2_problems import test_egg_movements as problem_fn

    else:
        problem_fn = problem_fn_from_name(args.problem)

    ## problem_sets.problem_template()
    return problem_fn(args, **kwargs, **problem_kwargs)


def reorg_output_dirs(exp_name, output_dir, log_failures=False):
    if exp_name == 'original':
        new_output_dir = output_dir.replace(f'_{exp_name}', '')
        results_dir = join(new_output_dir, exp_name)
        os.makedirs(results_dir, exist_ok=True)

        ## move problem-related files
        for file in ['scene.lisdf', 'problem.pddl', 'planning_config.json']:
            shutil.move(join(output_dir, file), join(new_output_dir, file))

        ## move solution-related files
        for file in ['commands.pkl', 'log.txt', 'time.json']:
            if isfile(join(output_dir, file)):
                shutil.move(join(output_dir, file), join(results_dir, file))

        shutil.rmtree(output_dir)
    else:
        results_dir = join(dirname(output_dir), exp_name)
        if isdir(results_dir):
            shutil.rmtree(results_dir)
        shutil.move(output_dir, results_dir)
        # os.remove(join(dirname(output_dir), 'tmp'))

    ## move planning-related files
    visualization_dir = join(dirname(__file__), 'visualizations')
    if log_failures and isdir(visualization_dir):
        logs = [f for f in listdir(visualization_dir) if f.startswith('log') and f.endswith('.json')]
        for log_file in logs:
            shutil.move(join(visualization_dir, log_file), join(results_dir, log_file))
    print(f"given_path: '{results_dir}'")


def clear_empty_exp_dirs(exp_dir):
    if not isdir(exp_dir):
        return
    run_dirs = [join(exp_dir, f) for f in listdir(exp_dir) if isdir(join(exp_dir, f))]
    for run_dir in run_dirs:
        paths = listdir(run_dir)
        if len(paths) <= 1:
            shutil.rmtree(run_dir)
