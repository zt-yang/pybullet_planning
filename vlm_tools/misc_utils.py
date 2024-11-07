import os
import random
import sys
from os import listdir
from os.path import join, abspath, dirname, isdir, isfile
R = abspath(join(dirname(__file__), os.pardir))
sys.path.extend([R] + [join(R, name) for name in ['pddlstream', 'pybullet_planning', 'lisdf']])

import datetime
from pathlib import Path
import shutil

from pybullet_tools.logging_utils import zipit
from vlm_tools.vlm_utils import EXP_DIR

DOWNLOADS_DIR = join(str(Path.home()), 'Downloads')


def is_nvidia_machine():
    import platform
    return platform.node() == '223da4c-lcedt'


def get_date():
    return datetime.datetime.now().strftime('%y%m%d_%H')


def zip_data_for_sharing(lst, name=None):
    paths = []
    for path in lst:
        if not isdir(path) and path.endswith('.pkl'):
            path = dirname(dirname(path))
        paths.append(path)

    if name is None:
        name = f'_memory_{get_date()}.zip'
    outpath = join(DOWNLOADS_DIR, name)
    print(abspath(outpath))
    zipit(paths, outpath)
    sys.exit()


def zip_runs_without_states(keyword='run_kitchen_chicken_soup'):
    zip_root = join(DOWNLOADS_DIR, f'_zip_memory_{get_date()}')
    if isdir(zip_root):
        shutil.rmtree(zip_root)
    os.makedirs(zip_root, exist_ok=True)
    zip_dirs = []

    for exp_name in listdir(EXP_DIR):
        if exp_name.startswith(keyword):
            exp_dir = join(EXP_DIR, exp_name)
            zip_dir = join(zip_root, exp_name)
            os.makedirs(zip_dir, exist_ok=True)
            run_names = [f for f in listdir(exp_dir) if isdir(join(exp_dir, f))]
            for run_name in run_names:
                run_dir = join(exp_dir, run_name)
                zip_run_dir = join(zip_dir, run_name)
                os.makedirs(zip_run_dir, exist_ok=True)
                files = [f for f in listdir(run_dir) if f not in ['states']]
                for f in files:
                    old_file = join(run_dir, f)
                    if isfile(old_file):
                        shutil.copy(old_file, join(zip_run_dir, f))
                    elif isdir(old_file):
                        shutil.copytree(old_file, join(zip_run_dir, f))

            file_names = [f for f in listdir(exp_dir) if isfile(join(exp_dir, f))]
            for f in file_names:
                shutil.copy(join(exp_dir, f), join(zip_dir, f))

            zip_dirs.append(zip_dir)

    zip_data_for_sharing(zip_dirs, name=f"{zip_root}.zip")


if __name__ == '__main__':
    zip_runs_without_states()
