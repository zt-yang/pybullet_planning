#!/usr/bin/env python

from __future__ import print_function

import copy
import shutil

import os
import sys
from os.path import join, abspath, dirname, isdir, isfile
from os import listdir

ROOT_DIR = abspath(join(dirname(__file__), os.pardir))
sys.path.extend([
    join(ROOT_DIR, '..'),
    join(ROOT_DIR, '..', 'pddlstream'),
    join(ROOT_DIR, '..', 'bullet'),
    join(ROOT_DIR, '..', 'bullet', 'pybullet_planning'),
    join(ROOT_DIR, '..', 'bullet', 'lisdf'),
])

from pybullet_tools.utils import disconnect, reset_simulation, \
    VideoSaver, wait_unlocked, timeout
from pybullet_tools.bullet_utils import get_datetime, initialize_logs

from lisdf_tools.lisdf_loader import load_lisdf_pybullet, pddlstream_from_dir
from lisdf_tools.lisdf_planning import Problem

from world_builder.world import State, evolve_processes
from world_builder.world_generator import save_to_outputs_folder

from cogarch_tools.processes.pddlstream_agent import PDDLStreamAgent
from cogarch_tools.cogarch_utils import get_parser, init_gui, get_pddlstream_kwargs, clear_planning_dir, \
    get_pddlstream_problem, PROBLEM_CONFIG_PATH


SAVE_COLLISIONS = False
EXPERIMENT_DIR = abspath(join(dirname(__file__), '..', '..', 'experiments'))


#######################################################


def main(config='config_dev.yaml', config_root=PROBLEM_CONFIG_PATH,
         problem='test_studio', domain='pr2_mamao.pddl', stream='pr2_stream_mamao.pddl',
         exp_dir=EXPERIMENT_DIR, exp_subdir='test', exp_name='default', reset=False,
         record_mp4=False, record_problem=True, save_testcase=False,
         record_plans=False, data_generation=False, use_rel_pose=False,
         domain_modifier=None, object_reducer=None, comparing=False, **kwargs):
    """
    problem:    name of the problem builder function to solve
    exp_dir:    sub-directory in `bullet/experiments` to save the planning data
    exp_name:   for comparison groups of different algorithms, e.g. ['original', 'hpn', 'hpn_goal-related']
    comparing:  put solutions inside exp_dir/exp_name instead of inside exp_dir
    """

    from pybullet_tools.logging import myprint as print
    initialize_logs()  ## so everything would be loaded to txt log file
    args = get_parser(config=config, config_root=config_root,
                      problem=problem, exp_dir=exp_dir, exp_subdir=exp_subdir, exp_name=exp_name,
                      domain=domain, stream=stream, use_rel_pose=use_rel_pose,
                      record_problem=record_problem, record_mp4=record_mp4, save_testcase=save_testcase)
    if 'robot_builder_args' not in kwargs:
        kwargs['robot_builder_args'] = args.robot_builder_args

    """ load problem """
    if '/' in args.exp_subdir:
        world = load_lisdf_pybullet(args.exp_subdir, width=1440, height=1120, time_step=args.time_step)
        problem = Problem(world)
        pddlstream_problem = pddlstream_from_dir(problem, exp_dir=abspath(args.exp_dir), collisions=not args.cfree,
                                                 teleport=args.teleport, replace_pddl=True,
                                                 domain_name=domain, stream_name=stream)
        state = State(world)
        exogenous = []
        goals = pddlstream_problem[-1]
        subgoals = None
        skeleton = None
        problem_dict = {'pddlstream_problem': pddlstream_problem}

        """ sample problem """
    else:
        init_gui(args, width=1440, height=1120)
        state, exogenous, goals, problem_dict = get_pddlstream_problem(args, **kwargs)
        pddlstream_problem = problem_dict['pddlstream_problem']
        subgoals = problem_dict['subgoals']
        skeleton = problem_dict['skeleton']

    init = pddlstream_problem.init

    ## load next test problem
    if args.save_testcase:
        disconnect()
        return

    ## for visualizing observation
    if hasattr(args, 'save_initial_observation') and args.save_initial_observation:
        state.world.initiate_observation_cameras()
        obs_dir = join('log', 'media')
        if not isdir(obs_dir):
            os.makedirs(obs_dir)
        state.save_default_observation(output_path=join(obs_dir, 'observation_0.png'))

    """ load planning agent """
    solver_kwargs = get_pddlstream_kwargs(args, skeleton, subgoals, [copy.deepcopy(state), goals, init])
    if SAVE_COLLISIONS:
        solver_kwargs['evaluation_time'] = 10
    # agent = TeleOpAgent(state.world)
    agent = PDDLStreamAgent(state.world, init, goals=goals, processes=exogenous, pddlstream_kwargs=solver_kwargs)
    agent.set_pddlstream_problem(problem_dict, state)

    # note = kwargs['world_builder_args'].get('note', None) if 'world_builder_args' in kwargs else None
    agent.init_experiment(args, domain_modifier=domain_modifier, object_reducer=object_reducer, comparing=comparing)

    """ before planning """
    if args.preview_scene and args.viewer:
        wait_unlocked()

    """ solving the problem """
    if not args.scene_only:
        agents = [agent]
        processes = exogenous + agents
        evolve_kwargs = dict(processes=processes, ONCE=not args.monitoring, verbose=False)

        if record_mp4:
            video_path = 'video_tmp.mp4'
            with VideoSaver(video_path):
                evolve_processes(state, **evolve_kwargs)
            shutil.move(video_path, agent.mp4_path)
            print(f'\n\nsaved mp4 to {agent.mp4_path}\n\n')
        else:
            max_time = 8 * 60
            with timeout(duration=max_time):
                evolve_processes(state, **evolve_kwargs)

        ## failed
        if agent.mp4_path is None:
            if not SAVE_COLLISIONS:
                print('failed to find any plans')
                disconnect()
                return
            agent.timestamped_name = f'{get_datetime(TO_LISDF=True)}'
            if not comparing:
                agent.timestamped_name += f'_{args.exp_name}'
            agent.mp4_path = join(EXPERIMENT_DIR, args.exp_dir, f"{agent.timestamped_name}.mp4")

        output_dir = agent.mp4_path.replace('.mp4', '')

    else:
        name = agent.timestamped_name = f'{get_datetime(TO_LISDF=True)}_{args.exp_name}'
        output_dir = join(EXPERIMENT_DIR, args.exp_dir, name)

    if record_problem and not isinstance(goals, tuple):
        state.restore()  ## go back to initial state
        state.world.save_test_case(output_dir, goal=goals, init=init, domain=domain, stream=stream,
                                   pddlstream_kwargs=solver_kwargs, problem=problem)

        ## putting solutions from all methods in the same directory as the problem
        if comparing:
            if args.exp_name == 'original':
                new_output_dir = output_dir.replace(f'_{args.exp_name}', '')
                results_dir = join(new_output_dir, args.exp_name)
                os.makedirs(results_dir, exist_ok=True)

                ## move problem-related files
                for file in ['scene.lisdf', 'problem.pddl', 'planning_config.json']:
                    shutil.move(join(output_dir, file), join(new_output_dir, file))

                ## move solution-related files
                for file in ['commands.pkl', 'log.txt', 'time.json']:
                    if isfile(join(output_dir, file)):
                        shutil.move(join(output_dir, file), join(results_dir, file))

                shutil.rmtree(output_dir)
            else:
                results_dir = join(dirname(output_dir), args.exp_name)
                if isdir(results_dir):
                    shutil.rmtree(results_dir)
                shutil.move(output_dir, results_dir)
                # os.remove(join(dirname(output_dir), 'tmp'))

            ## move planning-related files
            visualization_dir = join(dirname(__file__), 'visualizations')
            if solver_kwargs['log_failures'] and isdir(visualization_dir):
                logs = [f for f in listdir(visualization_dir) if f.startswith('log') and f.endswith('.json')]
                for log_file in logs:
                    shutil.move(join(visualization_dir, log_file), join(results_dir, log_file))

            print('saved planning data to', results_dir)
        else:
            print('saved planning data to', output_dir)

    if record_plans:
        from world_builder.paths import KITCHEN_WORLD
        KITCHEN_WORLD = abspath(KITCHEN_WORLD.replace('kitchen-worlds', '../kitchen-worlds'))
        dir_name = agent.timestamped_name
        if state.world.note is not None and not args.scene_only:
            dir_name += f'_{state.world.note}'
        data_path = join(KITCHEN_WORLD, 'outputs', problem, dir_name)
        save_to_outputs_folder(output_dir, data_path, data_generation=data_generation,
                               multiple_solutions=args.dataset_mode)
        if SAVE_COLLISIONS:
            file = 'collisions.pkl'
            root_dir = '/home/yang/Documents/cognitive-architectures/bullet/'
            paths = ['pybullet_planning/world_builder/', 'examples/']
            for pp in paths:
                ff = join(root_dir, pp, file)
                if isfile(ff):
                    shutil.move(ff, join(data_path, file))
                    print('moved', ff, 'to', data_path)
                    break
        if solver_kwargs['fc'] is not None:
            solver_kwargs['fc'].dump_log(join(data_path, f'diverse_plans.json'))
        if isdir(data_path) or args.scene_only:
            print('saved data to', data_path)
        else:
            print('failed to find any plans', data_path)

    clear_planning_dir(run_dir=dirname(__file__))

    if reset:
        reset_simulation()
    else:
        disconnect()
