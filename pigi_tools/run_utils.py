from os import listdir
from os.path import join, abspath, dirname, basename, isdir, isfile
from tabnanny import verbose
import os
import math
import json
import numpy as np
import random
import time
import sys
import pickle
import shutil
import argparse

from world_builder.world_utils import parse_yaml
from cogarch_tools.cogarch_utils import clear_planning_dir


def get_task_names(task_name):
    # if task_name == 'mm':
    #     task_names = ['mm_one_fridge_pick',
    #                   'mm_one_fridge_table_pick', 'mm_one_fridge_table_in', 'mm_one_fridge_table_on',
    #                   'mm_two_fridge_in', 'mm_two_fridge_pick'] ## , 'mm_two_fridge_goals'
    # elif task_name == 'tt':
    #     task_names = ['tt_one_fridge_table_pick', 'tt_one_fridge_table_in',
    #                   'tt_two_fridge_pick', 'tt_two_fridge_in', 'tt_two_fridge_goals']  ##
    # elif task_name == 'ff':
    #     task_names = ['ff_one_fridge_table_pick', 'ff_one_fridge_table_in',
    #                   'ff_two_fridge_in', 'ff_two_fridge_pick']
    # elif task_name == 'ww':
    #     task_names = ['ww_one_fridge_table_pick', 'ww_one_fridge_table_in',
    #                   'ww_two_fridge_in', 'ww_two_fridge_pick']
    # elif task_name == 'bb':
    #     task_names = ['bb_one_fridge_pick',
    #                   'bb_one_fridge_table_pick', 'bb_one_fridge_table_in', 'bb_one_fridge_table_on',
    #                   'bb_two_fridge_in', 'bb_two_fridge_pick']
    # elif task_name == 'zz':
    #     task_names = ['zz_three_fridge', 'ss_two_fridge_pick', 'ss_two_fridge_in']

    mm_task_names = ['mm_storage', 'mm_sink', 'mm_braiser',
                     'mm_sink_to_storage', 'mm_braiser_to_storage']
    if task_name == 'mm':
        task_names = mm_task_names
    elif task_name == 'tt':
        task_names = [t.replace('mm_', 'tt_') for t in mm_task_names]
    else:
        task_names = [task_name]
    return task_names


def get_run_dirs(task_name):
    task_names = get_task_names(task_name)
    all_subdirs = []
    for task_name in task_names:
        dataset_dir = join('/home/yang/Documents/fastamp-data-rss/', task_name)
        # organize_dataset(task_name)
        if not isdir(dataset_dir):
            print('get_run_dirs | no directory', dataset_dir)
            continue
        subdirs = listdir(dataset_dir)
        subdirs.sort()
        subdirs = [join(dataset_dir, s) for s in subdirs if isdir(join(dataset_dir, s))]
        all_subdirs += subdirs
    return all_subdirs


def process_all_tasks(process, task_name, parallel=False, cases=None, path=None,
                      dir=None, case_filter=None, return_dirs=False):
    clear_planning_dir()

    if path is not None:
        cases = [path]
    elif dir is not None:
        cases = [join(dir, c) for c in listdir(dir)]
        cases = [c for c in cases if isdir(c) and not isfile(join(c, 'gym_replay.gif'))]
        # cases = cases[:1]
    elif cases is not None and len(cases) > 0:
        cases = [join(MAMAO_DATA_PATH, task_name, case) for case in cases]
    else:
        cases = get_run_dirs(task_name)

    if len(cases) > 1:
        cases = sorted(cases, key=lambda x: eval(x.split('/')[-1]))

    if case_filter is not None:
        cases = [c for c in cases if case_filter(c)]

    if return_dirs:
        return cases

    parallel_processing(process, cases, parallel)


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