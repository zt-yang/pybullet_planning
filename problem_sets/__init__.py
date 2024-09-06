from os.path import join, dirname
from os import listdir
import importlib
import sys


def problem_fn_from_name(problem_name):
    """ search through all problem functions from the parent folder """
    funk_names = get_all_modules_in_directory()
    return find_problem_fn_from_name(problem_name, funk_names)


def get_all_modules_in_directory():
    """ equivalent to
    import problem_sets.pr2_problems as pr2_problems
    import problem_sets.problems_block_world as pr2_problems_new
    funk_names = [pr2_problems, pr2_problems_new]
    """
    ignore_files = ['__init__.py', 'problem_utils.py']
    root = dirname(__file__)
    paths = [join(root, f) for f in listdir(root) if f.endswith('.py') and f not in ignore_files]
    return [get_functions_in_file(path) for path in paths]


def get_functions_in_file(path, name=None):
    if name is None:
        ## e.g. problem_sets.pr2_problems_pigi, vlm_tools.vlm_tamp_problems
        ## assumes the root dir has been added to PYTHONPATH
        name = '/'.join(path.split('/')[-2:]).replace('.py', '')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def find_problem_fn_from_name(name, funk_names):
    from inspect import getmembers, isfunction
    for problem_bank in funk_names:
        result = [a[1] for a in getmembers(problem_bank) if isfunction(a[1]) and a[0] == name]
        if len(result) > 0:
            break
    if len(result) == 0:
        raise ValueError('Problem {} not found'.format(name))
    return result[0]


"""
from robot_builder.robot_builders import build_table_domain_robot
from problem_sets.problem_utils import *

def problem_fn(args, **kwargs):

    def loader_fn(world, **world_builder_args):
        goals = []
        skeleton = []
        objects = []
        world.remove_bodies_from_planning(goals, exceptions=objects)
        return {'goals': goals, 'skeleton': skeleton}

    return problem_template(args, robot_builder_fn=build_table_domain_robot,
                            world_loader_fn=loader_fn, **kwargs)
"""
