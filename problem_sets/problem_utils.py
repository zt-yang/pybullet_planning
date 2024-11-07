import math
import pprint
from os.path import join, abspath, isfile

from pybullet_tools.stream_agent import pddlstream_from_state_goal
from pybullet_tools.pr2_streams import DEFAULT_RESOLUTION
from pybullet_tools.utils import set_all_static
from pybullet_tools.logging_utils import print_dict, print_green

from world_builder.world import World, State
from world_builder.world import World
from world_builder.paths import KITCHEN_WORLD, PBP_PATH

from pddlstream.language.constants import Action, AND, PDDLProblem


PDDL_PATH = abspath(join(__file__, '..', '..', 'assets', 'pddl'))


def create_world(args):
    return World(time_step=args.time_step, segment=args.segment, use_rel_pose=args.use_rel_pose)


def pddlstream_from_state_goal_args(state, goals, args=None, custom_limits=None, debug=False, verbose=False,
                                    domain_name=None, stream_name=None, cfree=False, teleport=False,
                                    use_all_grasps=False, top_grasp_tolerance=None, side_grasp_tolerance=None,
                                    resolution=DEFAULT_RESOLUTION, use_learned_ir=True, ir_max_attempts=60, **kwargs):
    if args is not None:
        domain_name = args.domain_pddl
        stream_name = args.stream_pddl
        debug = args.debug
        verbose = debug
        cfree = args.cfree
        teleport = args.teleport
        if hasattr(args, 'use_all_grasps'):
            use_all_grasps = args.use_all_grasps
        if hasattr(args, 'top_grasp_tolerance'):
            top_grasp_tolerance = args.top_grasp_tolerance
        if hasattr(args, 'side_grasp_tolerance'):
            side_grasp_tolerance = args.side_grasp_tolerance
        if hasattr(args, 'resolution_angular'):
            resolution = math.radians(args.resolution_angular)
        if hasattr(args, 'ir_max_attempts'):
            ir_max_attempts = args.ir_max_attempts
        if hasattr(args, 'use_learned_ir'):
            use_learned_ir = args.use_learned_ir
    stream_kwargs = dict(use_all_grasps=use_all_grasps, top_grasp_tolerance=top_grasp_tolerance,
                         side_grasp_tolerance=side_grasp_tolerance, resolution=resolution,
                         ir_max_attempts=ir_max_attempts, use_learned_ir=use_learned_ir)
    state.world.stream_kwargs = stream_kwargs
    print_dict(stream_kwargs, 'stream_kwargs')

    domain_pddl = join(PDDL_PATH, 'domains', domain_name)
    stream_pddl = join(PDDL_PATH, 'streams', stream_name)
    if not isfile(domain_pddl):
        domain_pddl = join(PBP_PATH, domain_name)
        stream_pddl = join(PBP_PATH, stream_name)
    args.domain_pddl = domain_pddl
    args.stream_pddl = stream_pddl

    if custom_limits is None:
        custom_limits = state.robot.custom_limits

    return pddlstream_from_state_goal(
        state, goals, custom_limits=custom_limits, debug=debug, verbose=verbose,
        domain_pddl=domain_pddl, stream_pddl=stream_pddl, collisions=not cfree, teleport=teleport,
        **stream_kwargs, **kwargs)


def save_to_kitchen_worlds(state, pddlstream_problem, exit=False, **kwargs):
    from world_builder.world_generator import save_to_kitchen_worlds as save_helper
    return save_helper(state, pddlstream_problem, exit=exit, root_path=KITCHEN_WORLD, **kwargs)


def problem_template(args, robot_builder_fn, robot_builder_args, world_loader_fn,
                     observation_model=None, world_builder_args={}, **kwargs):
    """ the most general form of a test """

    print_dict(robot_builder_args, 'robot_builder_args')
    print_dict(world_builder_args, 'world_builder_args')

    world = create_world(args)
    robot = robot_builder_fn(world, **robot_builder_args)

    ## add skeleton or a sequence of goals as planning constraint
    problem_dict = {k: None for k in ['goals', 'skeleton', 'llamp_api', 'subgoals', 'goal_sequence']}
    loaded_problem_dict = world_loader_fn(world, **world_builder_args)

    ## just process subgoals with llamp_agent, without actually plan
    if loaded_problem_dict is None:
        return None

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

    # print_green(f'[problem_template]\t using obstacles {state.fixed}')
    # world.print_ignored_pairs()

    ## may change the goal if they are debugging goals
    pddlstream_problem = pddlstream_from_state_goal_args(state, goals, args, problem_dict=problem_dict, **kwargs)
    goals = pddlstream_problem.goal[1:]
    problem_dict['pddlstream_problem'] = pddlstream_problem
    # save_to_kitchen_worlds(state, pddlstream_problem, exp_name='blocks_pick', world_name='blocks_pick')

    return state, exogenous, goals, problem_dict


def update_stream_map(pddlstream_problem, stream_map):
    a, b, c, _, e, f = pddlstream_problem
    return PDDLProblem(a, b, c, stream_map, e, f)
