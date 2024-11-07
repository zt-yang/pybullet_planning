import json
from os.path import join, abspath, isfile, isdir

from pybullet_tools.utils import set_random_seed, set_numpy_seed
from problem_sets.problems_nvidia import *
from problem_sets import find_problem_fn_from_name, get_functions_in_file

from world_builder.world_utils import save_world_aabbs_to_json

from vlm_tools.vlm_utils import SUBGOALS_GROUP, ACTIONS_GROUP, DEFAULT_TEMP_DIR, load_vlm_memory, \
    load_agent_memory, fix_server_path, fix_experiment_path
from vlm_tools.vlm_planning_api import get_llamp_api


def vlm_tamp_problem_fn_from_name(problem_name):
    funk_names = [get_functions_in_file(abspath(__file__))]
    return find_problem_fn_from_name(problem_name, funk_names)


def get_problem_dict_from_open_goal(world, objects, args, load_llm_memory=None, load_agent_state=None,
                                    temp_dir=DEFAULT_TEMP_DIR):
    planning_mode = args.llamp_planning_mode
    open_goal = args.open_goal

    ## defining LLM Planning agent
    if load_agent_state is not None:
        agent_memory_path = load_agent_state.split('/states/')[0]
        agent_memory_path = fix_experiment_path(agent_memory_path)
        load_llm_memory = load_agent_memory(agent_memory_path)['load_memory']
        if load_llm_memory is None:
            load_llm_memory = agent_memory_path

    elif load_llm_memory is None:
        load_llm_memory = args.load_llm_memory

    if load_llm_memory is not None:
        load_llm_memory = fix_server_path(load_llm_memory)

    log_dir = join(temp_dir, 'log')
    img_dir = join(temp_dir, 'log', 'media')
    llamp_api_class = get_llamp_api(args.api_class_name)
    llamp_api = llamp_api_class(open_goal, planning_mode=planning_mode, load_memory=load_llm_memory, seed=args.seed,
                                log_dir=log_dir, vlm_kwargs=dict(image_dir=img_dir))
    problem_dict = {'english_goal': open_goal, 'llamp_api': llamp_api}

    subgoals = llamp_api.get_subgoals(world, objects=objects)

    ## , n=args.k_plans
    if planning_mode in SUBGOALS_GROUP + ACTIONS_GROUP:
        problem_dict['goal_sequence'] = subgoals
    elif planning_mode in ['constraints', 'soft_constraints']:
        problem_dict['subgoals'] = subgoals
    elif planning_mode is None:
        problem_dict['goals'] = []

    if load_agent_state is not None:
        llamp_api.agent_state_path = load_agent_state

    return problem_dict


def _get_vlm_agent_kwargs(args, world_builder_args):
    title = '[problem_vlm_tamp._get_vlm_agent_kwargs]\t'
    load_llm_memory = world_builder_args['load_llm_memory'] if 'load_llm_memory' in world_builder_args else None
    load_agent_state = world_builder_args['load_agent_state'] if 'load_agent_state' in world_builder_args else None
    randomize_joint_positions = False

    ## ------------- changing world generation seed to make sure actions can be replayed ---------
    memory = None
    if load_agent_state:
        path = abspath(load_agent_state.split('/states/')[0])
        path = fix_experiment_path(path)
        memory = load_agent_memory(path)
    elif load_llm_memory:
        memory = load_agent_memory(abspath(load_llm_memory))

    if memory is not None and 'seed' in memory:
        seed = memory['seed']
        print(f'{title} found seed saved in vlm memory: {seed}')
        set_random_seed(seed)
        set_numpy_seed(seed)
        args.seed = seed
    ## -------------------------------------------------------------------------------------------

    # for key in ['240907_']:
    #     if (load_llm_memory and key in load_llm_memory) or (load_agent_state and key in load_agent_state):
    #         randomize_joint_positions = False

    print(f'{title} randomize_joint_positions = {randomize_joint_positions}')
    return dict(load_llm_memory=load_llm_memory, load_agent_state=load_agent_state), randomize_joint_positions


def test_kitchen_chicken_soup(args, **kwargs):
    def loader_fn(world, temp_dir=DEFAULT_TEMP_DIR, **world_builder_args):
        difficulty = world_builder_args['difficulty'] if 'difficulty' in world_builder_args else 1
        vlm_agent_kwargs, randomize_joint_positions = _get_vlm_agent_kwargs(args, world_builder_args)

        objects, movables = load_open_problem_kitchen(world, reduce_objects=True, difficulty=difficulty,
                                                      randomize_joint_positions=randomize_joint_positions)
        # save_world_aabbs_to_json(world, json_name='world_shapes.json')
        return get_problem_dict_from_open_goal(world, objects, args, temp_dir=temp_dir, **vlm_agent_kwargs)

    return test_nvidia_kitchen_domain(args, loader_fn, initial_xy=(2, 6.25), **kwargs)
