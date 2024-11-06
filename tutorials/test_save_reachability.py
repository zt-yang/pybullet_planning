import json
import pickle
from os.path import join, isdir, isfile
from os import listdir
from collections import defaultdict

from vlm_tools.vlm_utils import enumerate_exp_names, EXP_DIR
from pybullet_tools.learned_samplers import summarize_reachability


def get_exp_dirs(dual_arm=True):
    """ find history """
    exp_runs = []
    count = 0
    root_dirs = [EXP_DIR] + [join(EXP_DIR, s) for s in ['_', '_before_240905']]
    for root in root_dirs:
        for exp_name in listdir(root):
            exp_dir = join(root, exp_name)
            if exp_name.startswith('run_kitchen_') and isdir(exp_dir):
                if (dual_arm and 'dual_arm' not in exp_name) or (not dual_arm and 'dual_arm' in exp_name):
                    continue
                runs = [join(exp_dir, f, 'time.json') for f in listdir(exp_dir) if isdir(join(exp_dir, f)) and
                        isfile(join(exp_dir, f, 'time.json'))]
                exp_runs += runs
                count += len(runs)
    print(f'found {len(exp_runs)} experiments and in total {count} runs for dual_arm = {dual_arm}')
    return exp_runs


if __name__ == '__main__':
    for dual_arm in [True, False]:
        prefix = 'test_nvidia_kitchen_'
        if dual_arm:
            prefix += 'dual_arm_'
        summarize_reachability(get_exp_dirs(dual_arm=dual_arm), prefix=prefix)
