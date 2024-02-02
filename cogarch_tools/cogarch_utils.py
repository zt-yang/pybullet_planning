from __future__ import print_function

import shutil
import argparse
import numpy as np
import pybullet as p
import random

import os
from os.path import join, abspath, dirname, isdir, isfile

from pybullet_tools.utils import connect, draw_pose, enable_preview, unit_pose, set_random_seed, set_camera_pose, \
    set_numpy_seed
from problem_sets import problem_fn_from_name

from mamao_tools.data_utils import get_feasibility_checker

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


PROBLEM_CONFIG_PATH = abspath(dirname(__file__))


def parse_config(path):
    import yaml
    from pathlib import Path
    from argparse import Namespace
    conf = yaml.safe_load(Path(path).read_text())
    conf = Namespace(**conf)
    conf.sim = Namespace(**conf.sim)
    conf.problem = Namespace(**conf.problem)
    conf.planner = Namespace(**conf.planner)
    conf.robot = Namespace(**conf.robot)
    return conf


def get_parser(config='config_dev.yaml', config_root=PROBLEM_CONFIG_PATH, **kwargs):

    conf = parse_config(join(config_root, config))

    np.set_printoptions(precision=3, threshold=3, suppress=True)  #, edgeitems=1) #, linewidth=1000)
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=conf.seed, help='')
    if hasattr(conf, 'debug'):
        parser.add_argument('--debug', action='store_true', default=conf.debug, help='')

    ## -------- simulation related
    parser.add_argument('-v', '--viewer', action='store_true', default=conf.sim.viewer,
                        help='When enabled, enables the PyBullet viewer.')
    parser.add_argument('--lock', action='store_true', default=conf.sim.lock,
                        help='When enabled, locks the viewer during planning.')
    parser.add_argument('-c', '--cfree', action='store_true', default=conf.sim.cfree,
                        help='When enabled, DISABLES collision checking.')
    parser.add_argument('--movable', action='store_true', default=conf.sim.disable_movable_collision,
                        help='When enabled, DISABLES movable collisions.')
    parser.add_argument('--base', action='store_true', default=conf.sim.disable_base_collision,
                        help='When enabled, DISABLES base collisions.')
    parser.add_argument('--teleport', action='store_true', default=conf.sim.teleport,
                        help='When enabled, teleports the robot between base configurations.')
    parser.add_argument('-d', '--drive', action='store_true', default=conf.sim.drive, help='')
    parser.add_argument('-t', '--time_step', type=float, default=conf.sim.time_step, help='')
    parser.add_argument('-cam', '--camera', action='store_true', default=conf.sim.camera, help='')
    parser.add_argument('-seg', '--segment', action='store_true', default=conf.sim.segment, help='')
    parser.add_argument('-mon', '--monitoring', action='store_true', default=conf.sim.monitoring, help='')

    ## -------- planning problem related
    parser.add_argument('-p', '--problem', type=str, default=conf.problem.problem,
                        help='name of the problem function that initiate both the world and goal')
    parser.add_argument('-g', '--goal', type=str, default=conf.problem.goal,
                        help='natual language goal or predicates to initiate the problem function')
    parser.add_argument('-exdir', '--exp_dir', type=str, default=conf.problem.exp_dir,
                        help='path to `experiments` to save outputs')
    parser.add_argument('-exsubdir', '--exp_subdir', type=str, default=conf.problem.exp_subdir,
                        help='name of the sub-directory in `../experiments` to save outputs')
    parser.add_argument('-exname', '--exp_name', type=str, default=conf.problem.exp_name,
                        help='name of the comparison group for which planning time and success rate will be compared')
    parser.add_argument('-domain', '--domain_pddl', type=str, default=conf.problem.domain_pddl,
                        help='name to the domain pddl file')
    parser.add_argument('-stream', '--stream_pddl', type=str, default=conf.problem.stream_pddl,
                        help='name to the stream pddl file')
    parser.add_argument('-rel', '--use_rel_pose', action='store_true', default=conf.problem.use_rel_pose,
                        help='When enabled, domain will use relative pose for objects in movable links, e.g. drawers')
    parser.add_argument('--preview_scene', action='store_true', default=conf.problem.preview_scene,
                        help='When enabled, previews the scene and press Enter before solving the problem.')
    parser.add_argument('--preview_plan', action='store_true', default=conf.problem.preview_plan,
                        help='When enabled, previews the plan before returning.')
    parser.add_argument('--record_problem', action='store_true', default=conf.problem.record_problem,
                        help='When enabled, the world lisdf, problem pddl, and solution json will be saved')
    parser.add_argument('-mp4', '--record_mp4', action='store_true', default=conf.problem.record_mp4,
                        help='When enabled, the solution mp4 will be saved')
    parser.add_argument('--save_testcase', action='store_true', default=conf.problem.save_testcase,
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

    ## seed determines asset instances and object poses initiated for the problem
    args = parser.parse_args()
    seed = args.seed
    if seed is None:
        seed = random.randint(0, 10**6-1)
    else:
        args.viewer = True
    set_random_seed(seed)
    set_numpy_seed(seed)
    args.seed = seed

    ## replace the default values with values provided
    for k, v in kwargs.items():
        args.__dict__[k] = v

    ## other args
    args.__dict__['robot_builder_args'] = conf.robot.__dict__
    for k, v in conf.problem.__dict__.items():
        if k not in args.__dict__:
            args.__dict__[k] = v

    print(f'Seed: {args.seed}')
    print(f'Args: {args}')
    return args


def get_pddlstream_kwargs(args, skeleton, initializer):
    fc = None if not args.use_heuristic_fc else get_feasibility_checker(initializer, mode='heuristic')
    solver_kwargs = dict(
        skeleton=skeleton,
        fc=fc,
        collect_dataset=args.data,
        lock=args.lock,
        preview=args.preview_plan,
        max_solutions=args.max_solutions,
        visualization=args.visualization,  ## to draw constraint networks and stream plans
        log_failures=args.log_failures,  ## to summarize failed streams
        evaluation_time=args.evaluation_time,
        downward_time=args.downward_time,
        max_plans=args.max_plans,
    )
    return solver_kwargs


def init_gui(args, width=1980, height=1238):
    connect(use_gui=args.viewer, shadows=False, width=width, height=height)

    if args.camera:
        enable_preview()
    if not args.segment:
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, False)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, False)

    # p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, True)
    # parameter = add_parameter(name='facts')
    # button = add_button(name='TBD')

    draw_pose(unit_pose(), length=1.)


def clear_planning_dir(run_dir=dirname(__file__)):
    temp_dir = join(run_dir, 'temp')
    if isdir(temp_dir):
        shutil.rmtree(temp_dir)

    log_file = join(run_dir, 'txt_file.txt')
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
    elif args.problem == 'test_pick':
        set_kitchen_camera_pose()
        from problem_sets.pr2_problems import test_pick as problem_fn
    elif args.problem == 'test_plated_food':
        set_kitchen_camera_pose()
        from problem_sets.pr2_problems import test_plated_food as problem_fn
    elif args.problem == 'test_small_sink':
        set_kitchen_camera_pose()
        from problem_sets.pr2_problems import test_small_sink as problem_fn
    elif args.problem == 'test_five_tables':
        from problem_sets.pr2_problems import test_five_tables as problem_fn
    elif args.problem == 'test_exist_omelette':
        from problem_sets.pr2_problems import test_exist_omelette as problem_fn
    # elif args.problem == 'test_three_omelettes':
    #     from bullet.examples.pr2_problems import test_three_omelettes as problem_fn
    # elif args.problem == 'test_bucket_lift':
    #     from bullet.examples.pr2_problems import test_bucket_lift as problem_fn
    elif args.problem == 'test_navigation':
        from problem_sets.pr2_problems import test_navigation as problem_fn
    elif args.problem == 'test_cart_pull':
        from problem_sets.pr2_problems import test_cart_pull as problem_fn
    elif args.problem == 'test_cart_obstacle_wconf': ## testing with other regions in room
        # set_camera_pose(camera_point=[4, -4, 4], target_point=[2, 0, 0])
        from problem_sets.pr2_problems import test_cart_obstacle_wconf as problem_fn
    elif args.problem == 'test_cart_obstacle':
        from problem_sets.pr2_problems import test_cart_obstacle as problem_fn
    elif args.problem == 'test_moving_carts':
        set_camera_pose(camera_point=[4, -2, 4], target_point=[0, -2, 0])
        set_camera_pose(camera_point=[5, -2, 4], target_point=[1, -2, 0])  ## laundry area
        from problem_sets.pr2_problems import test_moving_carts as problem_fn
    elif args.problem == 'test_three_moving_carts':
        set_camera_pose(camera_point=[5, 0, 4], target_point=[1, 0, 0])
        from problem_sets.pr2_problems import test_three_moving_carts as problem_fn
    elif args.problem == 'test_fridge_pose':
        set_camera_pose(camera_point=[3, 6.5, 2], target_point=[1, 4, 1])
        from problem_sets.pr2_problems import test_fridge_pose as problem_fn
    elif args.problem == 'test_kitchen_fridge':
        set_camera_pose(camera_point=[3, 5, 3], target_point=[0, 6, 1])
        from problem_sets.pr2_problems import test_kitchen_fridge as problem_fn
    elif args.problem == 'test_kitchen_oven':
        set_camera_pose(camera_point=[3, 5, 3], target_point=[0, 6, 1])
        from problem_sets.pr2_problems import test_kitchen_oven as problem_fn
    elif args.problem == 'test_oven_egg':
        set_camera_pose(camera_point=[3, 5, 3], target_point=[0, 6, 1])
        set_kitchen_camera_pose()
        from problem_sets.pr2_problems import test_oven_egg as problem_fn
    elif args.problem == 'test_braiser_lid':
        set_kitchen_camera_pose()
        from problem_sets.pr2_problems import test_braiser_lid as problem_fn
    elif args.problem == 'test_egg_movements':
        set_camera_pose(camera_point=[3, 5, 3], target_point=[0, 6, 1])
        from problem_sets.pr2_problems import test_egg_movements as problem_fn

    # ## ------------------- demo PR2 problem_sets
    # elif args.problem == 'test_skill_knob_faucet':
    #     set_camera_pose(camera_point=[3, 5, 3], target_point=[0, 6, 1])
    #     from bullet.examples.pr2_problems import test_skill_knob_faucet as problem_fn
    # elif args.problem == 'test_skill_knob_stove':
    #     set_camera_pose(camera_point=[3, 5, 3], target_point=[0, 6, 1])
    #     from bullet.examples.pr2_problems import test_skill_knob_stove as problem_fn
    # elif args.problem == 'test_kitchen_demo':
    #     set_camera_pose(camera_point=[3, 5, 3], target_point=[0, 6, 1])
    #     from bullet.examples.pr2_problems import test_kitchen_demo as problem_fn
    # elif args.problem == 'test_kitchen_demo_two':
    #     set_camera_pose(camera_point=[3, 5, 3], target_point=[0, 6, 1])
    #     from bullet.examples.pr2_problems import test_kitchen_demo_two as problem_fn
    # elif args.problem == 'test_kitchen_demo_objects':
    #     set_camera_pose(camera_point=[3, 5, 3], target_point=[0, 6, 1])
    #     from bullet.examples.pr2_problems import test_kitchen_demo_objects as problem_fn
    # elif args.problem == 'test_kitchen_joints':
    #     set_camera_pose(camera_point=[3, 5, 3], target_point=[0, 6, 1])
    #     from bullet.examples.pr2_problems import test_kitchen_joints as problem_fn

    else:
        problem_fn = problem_fn_from_name(args.problem)

    return problem_fn(args, **kwargs, **problem_kwargs)
