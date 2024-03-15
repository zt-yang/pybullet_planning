from __future__ import print_function
from os.path import join, abspath, basename, isdir, isfile, dirname
from os import listdir
import os
import sys
import time
import shutil
import argparse

from world_builder.world_utils import parse_yaml
from world_builder.paths import DATA_CONFIG_PATH, PBP_PATH, TEMP_PATH


def parallel_processing(process, inputs, parallel):
    start_time = time.time()
    num_cases = len(inputs)

    if parallel:
        import multiprocessing
        from multiprocessing import Pool

        max_cpus = 11
        num_cpus = min(multiprocessing.cpu_count(), max_cpus)
        print(f'using {num_cpus} cpus')
        with Pool(processes=num_cpus) as pool:
            pool.map(process, inputs)

    else:
        for i in range(num_cases):
            process(inputs[i])

    print(f'went through {num_cases} run_dirs (parallel={parallel}) in {round(time.time() - start_time, 3)} sec')


def get_config_file_from_argparse(default_config_name='kitchen_full_feg.yaml', default_config_path=None,
                                  default_config_dir=DATA_CONFIG_PATH):
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--config_name', type=str, default=default_config_name,
                        help='name of config file inside pybullet_planning/data_generator/configs/ directory.')
    parser.add_argument('-p', '--config_path', type=str, default=default_config_path,
                        help='absolute path to config file.')
    args = parser.parse_args()

    if args.config_path is not None:
        config_file = args.config_path
    else:
        config_file = join(default_config_dir, args.config_name)
    return config_file


def get_config_from_argparse(default_config_name='kitchen_full_feg.yaml', default_config_path=None,
                             default_config_dir=DATA_CONFIG_PATH):
    config_file = get_config_file_from_argparse(default_config_name, default_config_path,
                                                default_config_dir)
    return get_config(config_file)


def get_config(config_file):
    """ all relative paths in config file has pybullet_planning as the root dir """
    config = parse_yaml(config_file)
    if isinstance(config.seed, str) and config.seed.lower() == 'none':
        config.seed = None
    config.data.out_dir = join(PBP_PATH, config.data.out_dir)
    config.planner.domain_pddl = join(PBP_PATH, config.planner.domain_pddl)
    config.planner.stream_pddl = join(PBP_PATH, config.planner.stream_pddl)

    if not isfile(config.planner.domain_pddl):
        print('provided domain file doesnt exist: ' + config.planner.domain_pddl)
        sys.exit()
    return config


def get_data_processing_parser(task_name=None, parallel=False, use_viewer=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, default=task_name)
    parser.add_argument('-p', action='store_true', default=parallel)
    parser.add_argument('-v', '--viewer', action='store_true', default=use_viewer,
                        help='When enabled, enables the PyBullet viewer.')
    return parser


def clear_constraint_networks(viz_dir):
    constraint_dir = join(viz_dir, 'constraint_networks')
    stream_dir = join(viz_dir, 'stream_plans')
    if isdir(constraint_dir) and len(listdir(constraint_dir)) == 0:
        shutil.rmtree(constraint_dir)
    if isdir(stream_dir) and len(listdir(stream_dir)) == 0:
        shutil.rmtree(stream_dir)


def fix_robot_path(scene_path):
    new_lines = []
    with open(scene_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if '_description/' in line:
                new_lines.append(line.replace('../../assets/', '../../pybullet_planning/'))
            else:
                new_lines.append(line)
    with open(scene_path, 'w') as f:
        f.writelines(new_lines)


def copy_dir_for_process(viz_dir, tag=None, verbose=True, print_fn=None):
    """ for lisdf loader to run successfully, the run_dir need to reside on the same level as pybullet_planning """
    if not verbose:
        print_fn = print
    elif print_fn is None:
        from pybullet_tools.logging import myprint as print_fn

    clear_constraint_networks(viz_dir)

    subdir = basename(viz_dir)
    task_name = basename(viz_dir.replace(f"/{subdir}", ''))

    ## temporarily move the dir to the test_cases folder for asset paths to be found
    test_dir = join(TEMP_PATH, f"temp_{task_name}_{subdir}")
    if isdir(test_dir):
        if verbose:
            print_fn('copy_dir_for_process | removing', test_dir)
        shutil.rmtree(test_dir)
    if not isdir(test_dir):
        shutil.copytree(viz_dir, test_dir)

    ## fix tht robot path in assets/models/ to pybullet_planning/models/
    fix_robot_path(join(test_dir, 'scene.lisdf'))

    if not verbose:
        print_fn(viz_dir)
    else:
        if tag is None:
            print_fn(viz_dir, end='\r')
        elif verbose:
            print_fn(f'\n\n\n--------------------------\n    {tag} {viz_dir} \n------------------------\n\n\n')

    return test_dir
