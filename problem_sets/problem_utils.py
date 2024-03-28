from os.path import join, abspath, isfile

from pybullet_tools.utils import set_all_static

from world_builder.world import World, State
from world_builder.world import World
from world_builder.paths import KITCHEN_WORLD, PBP_PATH


PDDL_PATH = abspath(join(__file__, '..', '..', 'assets', 'pddl'))

pull_actions = ['grasp_handle', 'pull_handle', 'ungrasp_handle']
pull_with_link_actions = ['grasp_handle', 'pull_handle_with_link', 'ungrasp_handle']
pick_place_actions = ['pick', 'place']
pick_place_rel_actions = ['pick_from_supporter', 'place_to_supporter']


def create_world(args):
    return World(time_step=args.time_step, segment=args.segment, use_rel_pose=args.use_rel_pose)


def pddlstream_from_state_goal(state, goals, args=None, custom_limits=None, domain_name=None, stream_name=None,
                               cfree=False, teleport=False, use_all_grasps=False, **kwargs):
    from pybullet_tools.pr2_agent import pddlstream_from_state_goal as pddlstream_helper
    domain_name = args.domain_pddl if args is not None else domain_name
    domain_pddl = join(PDDL_PATH, 'domains', domain_name)
    stream_name = args.stream_pddl if args is not None else stream_name
    stream_pddl = join(PDDL_PATH, 'streams', stream_name)
    if not isfile(domain_pddl):
        domain_pddl = join(PBP_PATH, domain_name)
        stream_pddl = join(PBP_PATH, stream_name)
    if args is not None:
        cfree = args.cfree
        teleport = args.teleport
        if hasattr(args, 'use_all_grasps'):
            use_all_grasps = args.use_all_grasps
            print(f'\n\npddlstream_from_state_goal | using all grasps? {use_all_grasps} \n\n')
    if custom_limits is None:
        custom_limits = state.robot.custom_limits
    return pddlstream_helper(state, goals, custom_limits=custom_limits, domain_pddl=domain_pddl, stream_pddl=stream_pddl,
                             collisions=not cfree, teleport=teleport, use_all_grasps=use_all_grasps, **kwargs)


def save_to_kitchen_worlds(state, pddlstream_problem, EXIT=False, **kwargs):
    from world_builder.world_generator import save_to_kitchen_worlds as save_helper
    return save_helper(state, pddlstream_problem, EXIT=EXIT, root_path=KITCHEN_WORLD, **kwargs)


def test_template(args, robot_builder_fn, robot_builder_args, world_loader_fn,
                  observation_model=None, world_builder_args={}, **kwargs):
    """ the most general form of a test """
    world = create_world(args)
    robot = robot_builder_fn(world, **robot_builder_args)

    ## add skeleton or a sequence of goals as planning constraint
    problem_dict = {k: None for k in ['goals', 'skeleton', 'llamp_api', 'subgoals', 'goal_sequence']}
    loaded_problem_dict = world_loader_fn(world, **world_builder_args)

    if not isinstance(loaded_problem_dict, dict):
        goals = problem_dict['goals'] = loaded_problem_dict
    else:
        problem_dict.update(loaded_problem_dict)
        goals = problem_dict['goals']
        if goals is None:
            if problem_dict['subgoals'] is not None:
                goals = problem_dict['subgoals']
            elif problem_dict['goal_sequence'] is not None:
                goals = problem_dict['goal_sequence'][:1]
        problem_dict['goals'] = goals

    set_all_static()
    state = State(world, objects=world.objects, observation_model=observation_model)
    exogenous = []

    ## may change the goal if they are debugging goals
    problem_dict['pddlstream_problem'] = pddlstream_from_state_goal(state, goals, args, **kwargs)
    goals = problem_dict['pddlstream_problem'].goal[1:]
    # save_to_kitchen_worlds(state, pddlstream_problem, exp_name='blocks_pick', world_name='blocks_pick')

    return state, exogenous, goals, problem_dict
