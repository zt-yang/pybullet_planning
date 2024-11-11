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
    parser.add_argument('--skip_prompt', action='store_true',
                        help='if true, the console will prompt the user to press Enter to see the trajectory play out.')

    ## a hackish way to get arbitrary arguments
    parsed, unknown = parser.parse_known_args()

    for i, arg in enumerate(unknown):
        next_arg = unknown[i + 1] if i + 1 < len(unknown) else None
        if arg.startswith(('-', '--')):
            is_bool = next_arg is None or next_arg.startswith(('-', '--'))
            print('arg.split()[0]', arg.split('=')[0])
            arg = arg.split('=')[0]
            if is_bool:
                parser.add_argument(arg, action='store_true')
            else:
                parser.add_argument(arg, type=str)

    args_parsed = parser.parse_args()
    parsed_config = dict(args_parsed.__dict__)
    parsed_config.pop('config_name')
    parsed_config.pop('config_path')

    if args_parsed.config_path is not None:
        config_file = args_parsed.config_path
    else:
        config_file = join(default_config_dir, args_parsed.config_name)
    return config_file, parsed_config


def get_config_from_argparse(default_config_name='kitchen_full_feg.yaml', default_config_path=None,
                             default_config_dir=DATA_CONFIG_PATH):
    config_file, parsed_config = get_config_file_from_argparse(default_config_name, default_config_path,
                                                               default_config_dir)
    config = get_config(config_file)
    for key, value in config.__dict__.items():
        if key in parsed_config:
            setattr(config, key, parsed_config[key])
        elif isinstance(value, argparse.Namespace):
            for sub_key, sub_value in value.__dict__.items():
                if sub_key in parsed_config:
                    setattr(value, sub_key, parsed_config[sub_key])
                    setattr(config, key, value)
    return config


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


def get_data_processing_parser(task_name=None, given_path=None, parallel=False, viewer=False):
    if given_path is not None:
        parallel = False
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default=task_name)
    parser.add_argument('--path', type=str, default=given_path)
    parser.add_argument('--parallel', action='store_true', default=parallel)

    parser.add_argument('-v', '--viewer', action='store_true', default=viewer,
                        help='When enabled, enables the PyBullet viewer.')
    return parser


def parse_image_rendering_args(task_name='test_feg_kitchen_full',
                               given_path=None, parallel=False, viewer=True,
                               generate_seg=False, n_px=224, redo=False, modified_time=None,
                               new_key='default_key', accepted_keys=[]):
    parser = get_data_processing_parser(task_name=task_name, given_path=given_path, parallel=parallel,
                                        viewer=viewer)
    parser.add_argument('--seg', action='store_true', default=generate_seg)
    parser.add_argument('--redo', action='store_true', default=redo)
    parser.add_argument('--modified_time', type=int, default=modified_time)
    parser.add_argument('--crop_px', type=int, default=n_px)
    parser.add_argument('--new_key', type=str, default=new_key)
    parser.add_argument('--accepted_keys', type=list, default=accepted_keys)
    args = parser.parse_args()
    args.larger_world = 'mm_' in args.task or 'tt_' in args.task
    return args


##################################################################


def process_all_tasks(process, task_name=None, dataset_root=None, parallel=False, cases=None, path=None,
                      dir=None, case_filter=None, return_dirs=False, input_args=None):
    from cogarch_tools.cogarch_utils import clear_planning_dir
    clear_planning_dir()

    if path is not None:
        cases = [path]
    elif dir is not None:
        cases = [join(dir, c) for c in listdir(dir)]
        cases = [c for c in cases if isdir(c) and not isfile(join(c, 'gym_replay.gif'))]
        # cases = cases[:1]
    elif cases is not None and len(cases) > 0:
        cases = [join(dataset_root, task_name, case) for case in cases]
    else:
        cases = get_run_dirs(task_name, dataset_root)

    if len(cases) > 1:
        def get_digit(string):
            string = string.split('/')[-1]
            if '_' in string:
                string = ''.join(string.split('_')[:-1])
            return eval(string)
        cases = sorted(cases, key=lambda x: get_digit(x))

    if case_filter is not None:
        cases = [c for c in cases if case_filter(c)]

    if return_dirs:
        return cases

    if input_args is not None:
        cases = [(case, input_args) for case in cases]

    parallel_processing(process, cases, parallel)


def get_run_dirs(task_names, dataset_root):
    if isinstance(task_names, str):
        task_names = [task_names]
    all_subdirs = []
    for task_name in task_names:
        dataset_dir = join(dataset_root, task_name)
        # organize_dataset(task_name)
        if not isdir(dataset_dir):
            print('get_run_dirs | no directory', dataset_dir)
            continue
        subdirs = listdir(dataset_dir)
        subdirs.sort()
        subdirs = [join(dataset_dir, s) for s in subdirs if isdir(join(dataset_dir, s))]
        all_subdirs += subdirs
    return all_subdirs


#########################################################


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
        from pybullet_tools.logging_utils import myprint as print_fn

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
    lisdf_path = join(test_dir, 'scene.lisdf')
    if isfile(lisdf_path):
        fix_robot_path(lisdf_path)

    if not verbose:
        print_fn(viz_dir)
    else:
        # if tag is None:
        #     print_fn(viz_dir)
        if verbose:
            print_fn(f'\n\n\n--------------------------\n    {tag} {viz_dir} \n------------------------\n\n\n')

    return test_dir
