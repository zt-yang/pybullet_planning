import json
from os.path import join
from .feasibility_checkers import Oracle


def get_oracle_answer(run_dir):
    return len(json.load(open(join(run_dir, 'plan.json'), 'r'))[0]['plan']) != 2


def get_feasibility_checker(run_dir, mode):
    if mode == 'oracle':
        return Oracle(get_oracle_answer(run_dir))
    return None