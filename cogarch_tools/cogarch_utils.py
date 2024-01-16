from __future__ import print_function

import copy
import shutil
import argparse
import numpy as np
import pybullet as p
import random

import os
import sys
from os.path import join, abspath, dirname, isdir, isfile
from os import listdir

from pybullet_tools.utils import connect, disconnect, draw_pose, enable_preview, unit_pose, set_random_seed, reset_simulation, \
    set_camera_pose, VideoSaver, wait_unlocked, set_numpy_seed, is_darwin, timeout
from pybullet_tools.bullet_utils import BASE_LIMITS, get_datetime, initialize_logs
from pybullet_tools.logging import summarize_csv
from problem_sets import problem_fn_from_name

from lisdf_tools.lisdf_loader import load_lisdf_pybullet, pddlstream_from_dir
from lisdf_tools.lisdf_planning import Problem

from world_builder.world import State, evolve_processes
from world_builder.world_generator import save_to_outputs_folder
from mamao_tools.data_utils import get_feasibility_checker

from cogarch_tools.agent import PDDLStreamAgent

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
    return conf


def get_parser(config='config_dev.yaml', **kwargs):

    conf = parse_config(join(PROBLEM_CONFIG_PATH, config))

    np.set_printoptions(precision=3, threshold=3, suppress=True)  #, edgeitems=1) #, linewidth=1000)
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=conf.seed, help='')

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
    parser.add_argument('-exdir', '--exp_dir', type=str, default=conf.problem.exp_dir,
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


def clear_planning_dir():
    temp_dir = join(dirname(__file__), 'temp')
    if isdir(temp_dir):
        shutil.rmtree(temp_dir)

    log_file = join(dirname(__file__), 'txt_file.txt')
    if isfile(log_file):
        os.remove(log_file)
