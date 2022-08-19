import json
import shutil
from os.path import join, isdir, abspath
from os import listdir
from .feasibility_checkers import Oracle


DATASET_PATH = '/home/zhutiany/Documents/mamao-data'


def get_oracle_answer(run_dir):
    return len(json.load(open(join(run_dir, 'plan.json'), 'r'))[0]['plan']) != 2


def get_feasibility_checker(run_dir, mode):
    if mode == 'oracle':
        return Oracle(get_oracle_answer(run_dir))
    return None


def organize_dataset(task_name):
    out_path = join(DATASET_PATH, task_name)
    names = [eval(l) for l in listdir(out_path) if isdir(join(out_path, l))]
    if len(names) == 0: return
    index = max(names)
    missing = list(set(range(index)) - set(names))
    if len(missing) == 0: return
    missing.sort()
    top = index
    moved = []
    for i in range(len(missing)):
        old_dir = join(out_path, str(top))
        while not isdir(old_dir) or str(top) in missing:
            top -= 1
            old_dir = join(out_path, str(top))
        if top in moved:
            break
        print(f'---- moving {top} to {missing[i]}')
        top -= 1
        shutil.move(old_dir, join(out_path, str(missing[i])))
        moved.append(missing[i])
