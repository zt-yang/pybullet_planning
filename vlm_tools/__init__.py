import os
import random
import sys
from os import listdir
from os.path import join, abspath, dirname, isdir, isfile
R = abspath(join(dirname(__file__), os.pardir))
sys.path.extend([R] + [join(R, name) for name in ['pddlstream', 'pybullet_planning', 'lisdf']])
from pathlib import Path

from cogarch_tools.cogarch_run import run_agent

from pigi_tools.replay_utils import run_replay, load_pigi_data, REPLAY_CONFIG_DEBUG

from pddl_domains.pddl_utils import update_kitchen_nudge_pddl

from vlm_tools.llamp_agent import LLAMPAgent, VLM_AGENT_CONFIG_ROOT
from vlm_tools.problems_vlm_tamp import test_kitchen_chicken_soup, vlm_tamp_problem_fn_from_name
from vlm_tools.vlm_utils import EXP_REL_PATH, get_subdir_name, load_prompts, sample_prompt, EXP_DIR


def modify_agent_args_for_vlm_tamp(args):
    if hasattr(args, 'planning_mode'):
        args.llamp_planning_mode = args.planning_mode
    if hasattr(args, 'load_llm_memory') and args.load_llm_memory is not None:
        args.load_llm_memory = join(EXP_DIR, args.load_llm_memory)
    else:
        args.load_llm_memory = None
    args.load_agent_state = None
    # if args.vlm_type is not None:
    #     args.api_class_name = args.vlm_type
    return args


def modify_world_builder_args_for_vlm_tamp(args, world_builder_args):
    world_builder_args['difficulty'] = args.difficulty
    return world_builder_args


def run_vlm_tamp_with_argparse(problem_name='test_kitchen_chicken_soup', world_builder_args=dict(), **kwargs):
    problem = vlm_tamp_problem_fn_from_name(problem_name)
    domain_kwargs = dict(agent_class=LLAMPAgent, config='config_nvidia_kitchen.yaml', config_root=VLM_AGENT_CONFIG_ROOT,
                         use_learned_ir=True, serve_page=True)

    if 'exp_subdir' in kwargs and 'actions' in kwargs['exp_subdir']:
        kwargs.update({k: 4 for k in ['max_plans', 'max_evaluation_plans']})
    run_agent(problem=problem, world_builder_args=world_builder_args,
              modify_agent_args_fn=modify_agent_args_for_vlm_tamp,
              modify_world_builder_args_fn=modify_world_builder_args_for_vlm_tamp, **domain_kwargs, **kwargs)


## ---------------------------------------------------------------------------------------------------


def run_vlm_tamp_agent(problem_name='test_kitchen_chicken_soup', difficulty=0,
                       config='config_nvidia_kitchen.yaml', config_root=VLM_AGENT_CONFIG_ROOT,
                       serve_page=True, use_learned_ir=True, world_builder_args=dict(), **kwargs):
    problem = vlm_tamp_problem_fn_from_name(problem_name)
    domain_kwargs = dict(agent_class=LLAMPAgent, config=config, config_root=config_root,
                         use_learned_ir=use_learned_ir, serve_page=serve_page)

    if 'exp_subdir' in kwargs and 'actions' in kwargs['exp_subdir']:
        kwargs.update({k: 4 for k in ['max_plans', 'max_evaluation_plans']})
    world_builder_args['difficulty'] = difficulty
    run_agent(problem=problem, world_builder_args=world_builder_args,
              modify_agent_args_fn=modify_agent_args_for_vlm_tamp, **domain_kwargs, **kwargs)


def enumerate_exp_conditions(
        problem_names=['test_kitchen_chicken_soup'],
        planning_modes=['actions-reprompt', 'sequence-reprompt', 'sequence', 'actions'],
        if_dual_arm=[True, False],
        difficulties=[0, 1],
        num_runs=10,
        max_num_data=None,
        get_subdir_fn=None
    ):
    line = '-' * 20

    random.shuffle(problem_names)
    random.shuffle(difficulties)
    random.shuffle(if_dual_arm)
    random.shuffle(planning_modes)
    for k in range(num_runs):
        for problem_name in problem_names:
            for difficulty in difficulties:
                for dual_arm in if_dual_arm:
                    for planning_mode in planning_modes:
                        print(f'\n\n\n\n{line} {k}-th {problem_name} + difficulty={difficulty} '
                              f'+ dual_arm={dual_arm} + {planning_mode} {line}\n\n\n\n')
                        ## once exceeding target number of runs, stop generating this combination
                        if get_subdir_fn is not None and max_num_data is not None:
                            subdir_name = get_subdir_fn(problem_name, difficulty, dual_arm, planning_mode)
                            parent_dir = join(EXP_DIR, subdir_name)
                            if isdir(parent_dir):
                                num_subdirs = len([s for s in listdir(parent_dir) if isdir(join(parent_dir, s))])
                                if num_subdirs >= max_num_data:
                                    continue
                        yield problem_name, difficulty, dual_arm, planning_mode


def run_vlm_recipe_generation(viewer=False, **kwargs):
    """
    but somehow always fail at the third run
    save_testcase: save the world conf, high-level plan, without running pddlstream
    difficulty:
        if difficulty == 0: cabinet door open 80%, lid already on stove
        e.g. loaders_nvidia_kitchen.py
    """
    def get_subdir_fn(problem_name, difficulty, dual_arm, planning_mode):
        return get_subdir_name(planning_mode=planning_mode, dual_arm=dual_arm, mode='llm',
                               version=f'v{difficulty}', problem_name=problem_name)

    for problem_name, difficulty, dual_arm, planning_mode in enumerate_exp_conditions(get_subdir_fn=get_subdir_fn, **kwargs):
        exp_subdir = get_subdir_name(planning_mode, dual_arm, mode='llm',
                                     problem_name=problem_name, version=f'v{difficulty}')
        run_vlm_tamp_agent(save_testcase=True, load_llm_memory=None, load_agent_state=None, viewer=viewer,
                           problem_name=problem_name, difficulty=difficulty,
                           llamp_planning_mode=planning_mode, dual_arm=dual_arm, exp_subdir=exp_subdir)


def run_vlm_tamp_experiments(**kwargs):

    def get_subdir_fn(problem_name, difficulty, dual_arm, planning_mode):
        return get_subdir_name(planning_mode=planning_mode, dual_arm=dual_arm, mode='eval',
                               version=f'v{difficulty}', problem_name=problem_name)

    for problem_name, difficulty, dual_arm, planning_mode in enumerate_exp_conditions(get_subdir_fn=get_subdir_fn, **kwargs):
        run_vlm_tamp_experiment(planning_mode=planning_mode, dual_arm=dual_arm,
                                difficulty=difficulty, serve_page=False, **kwargs)


def run_vlm_tamp_experiment(problem_name='test_kitchen_chicken_soup', difficulty=0,
                            planning_mode='sequence', dual_arm=True, **args):
    exp_kwargs = dict(planning_mode=planning_mode, dual_arm=dual_arm, problem_name=problem_name, version=f'v{difficulty}')
    exp_subdir = get_subdir_name(mode='eval', **exp_kwargs)
    load_llm_memory = join(EXP_REL_PATH, sample_prompt(**exp_kwargs))
    kwargs = dict(llamp_planning_mode=planning_mode, dual_arm=dual_arm, exp_subdir=exp_subdir, difficulty=difficulty)
    run_vlm_tamp_agent(problem_name=problem_name, load_llm_memory=load_llm_memory, **kwargs, **args)


def get_kwargs_from_cache_path(run_path):
    """ either for load_memory or load_agent_state """
    exp_name = run_path.split('/')[0]  ## e.g. run_kitchen_chicken_soup_v0_subgoals_dual_arm
    dual_arm = 'dual_arm' in run_path
    difficulty = 0 if '_v0_' in run_path else 1
    problem_name, planning_mode = exp_name.split(f"_v{difficulty}_")
    problem_name = problem_name.replace('run_', 'test_').replace('llm_only_', 'test_')
    planning_mode = planning_mode.replace(f"subgoals", 'sequence').replace(f"_dual_arm", '').replace('_reprompt', '-reprompt')
    return make_kwargs(problem_name, difficulty, planning_mode, dual_arm)


def make_kwargs(problem_name='test_kitchen_chicken_soup', difficulty=0, planning_mode='sequence', dual_arm=True):
    planning_mode = planning_mode.replace(f"_dual_arm", '')
    exp_kwargs = dict(planning_mode=planning_mode, dual_arm=dual_arm, problem_name=problem_name, version=f'v{difficulty}')
    exp_subdir = get_subdir_name(mode='eval', **exp_kwargs)
    return dict(llamp_planning_mode=planning_mode, difficulty=difficulty, dual_arm=dual_arm, exp_subdir=exp_subdir)
