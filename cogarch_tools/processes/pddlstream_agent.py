from __future__ import print_function

import copy
import os
import shutil
from os.path import isdir, isfile, dirname, abspath, basename
from pprint import pprint
import time
import json
import csv
import pickle
from collections import defaultdict
from os.path import join
import sys

from pybullet_tools.bullet_utils import summarize_facts, print_goal, get_datetime
from pybullet_tools.pr2_primitives import Trajectory
from pybullet_tools.pr2_agent import solve_pddlstream
from pybullet_tools.utils import SEPARATOR
from pybullet_tools.logging import save_commands, TXT_FILE

from world_builder.actions import get_primitive_actions
from world_builder.world_utils import get_camera_image

from cogarch_tools.processes.motion_agent import MotionAgent
from problem_sets.pr2_problems import pddlstream_from_state_goal

from leap_tools.domain_modifiers import initialize_domain_modifier
from leap_tools.object_reducers import initialize_object_reducer

from pddlstream.language.constants import Action, AND, PDDLProblem


ZOOM_IN_AT_OBJECT = False
SAVE_TIME = False

from world_builder.paths import pbp_path
PDDL_PATH = join(pbp_path, '..', 'assets', 'pddl')
VISUALIZATIONS_PATH = join(pbp_path, '..', 'examples', 'visualizations')


def get_traj(t):
    [t] = t.commands
    if SAVE_TIME:
        t = Trajectory([t.path[0]] + [t.path[-1]])
    return t


class PDDLStreamAgent(MotionAgent):

    requires_conf = requires_poses = requires_facts = requires_variables = True

    def __init__(self, world, init=[], goals=[], processes=[],
                 replan_frequency=1., pddlstream_kwargs={}, **kwargs):
        super(PDDLStreamAgent, self).__init__(world, **kwargs)
        self.goals = list(goals)
        self.replan_frequency = replan_frequency # TODO: include
        self.plan_step = None
        self.plan = None
        self.processes = processes  ## YANG-HPN
        self.solve_pddlstream = solve_pddlstream
        self.step_count = 0
        self.refinement_count = 0
        self.env = None  ## for preimage computation
        self.envs = []  ## for preimage computation
        self.env_execution = None  ## for saving states during execution
        self.preimages_after_op = {}  ## to be updated by each abstract planning call
        self.domains_for_action = {}
        self.time_log = []  ## for recording time

        self.exp_dir = None
        self.mp4_path = None
        self.timestamped_name = None

        self.pddlstream_problem = None
        self.initial_state = None
        self.problem_count = 0
        self.goal_sequence = None
        self.llamp_api = None
        self.last_removed_facts = []

        self.state = init
        self.static_facts = []
        self.failed_count = None
        self.last_action_name = None
        self.actions = []
        self.commands = []
        self.plan_len = 0
        self.pddlstream_kwargs = pddlstream_kwargs
        self.useful_variables = {}
        self.on_map = {}

    """ planning related """
    def set_pddlstream_problem(self, problem_dict, state):
        self.pddlstream_problem = problem_dict['pddlstream_problem']
        self.initial_state = state
        if 'llamp_api' in problem_dict and problem_dict['llamp_api'] is not None:
            self.llamp_api = problem_dict['llamp_api']
            self.llamp_api.output_html()
        if 'goal_sequence' in problem_dict and problem_dict['goal_sequence'] is not None:
            self.goal_sequence = problem_dict['goal_sequence']

    def init_experiment(self, args, domain_modifier=None, object_reducer=None, comparing=False):
        """ important for using the right files in replanning """
        if hasattr(args, 'object_reducer'):
            object_reducer = args.object_reducer
        if object_reducer is not None:
            args.exp_name += '_' + object_reducer

        ## related to saving data
        self.exp_dir = abspath(join(args.exp_dir, args.exp_subdir))
        if not isdir(self.exp_dir):
            os.makedirs(self.exp_dir, exist_ok=True)
        self.exp_name = args.exp_name
        if self.llamp_api is not None:
            self.timestamped_name = add_timestamp(self.exp_name)
            exp_path = join(self.exp_dir, self.timestamped_name)
            os.makedirs(exp_path)

        self.domain_pddl = args.domain_pddl
        self.stream_pddl = args.stream_pddl
        self.domain_modifier = initialize_domain_modifier(domain_modifier)
        self.object_reducer = initialize_object_reducer(object_reducer)
        self.custom_limits = self.robot.custom_limits

        ## HPN experiments
        self.comparing = comparing

        ## LLAMP debugging
        self.debug_step = args.debug_step if hasattr(args, 'debug_step') else None

    def process_plan(self, observation):
        """
        example self.plan
        00 = {MoveBaseAction} MoveBaseAction{conf: q552=(1.529, 5.989, 0.228, 3.173)}
        01 = {MoveBaseAction} MoveBaseAction{conf: q592=(1.607, 6.104, 0.371, 3.123)}
        10 = {MoveBaseAction} MoveBaseAction{conf: q744=(1.474, 7.326, 0.808, 9.192)}
        11 = {Action: 2} Action(name='pick', args=('left', 4, p1=(0.75, 7.3, 1.24, 0.0, -0.0, 0.0), g104=(-0.0, 0.027, -0.137, 0.0, -0.0, -3.142), q728=(1.474, 7.326, 0.808, 2.909), c544=t(7, 129)))
        """
        from pybullet_tools.logging import myprint as print

        if self.plan:
            self.world.remove_redundant_bodies()

        if observation.unobserved_objs is not None:
            newly_observed = observation.update_unobserved_objs()
            if len(newly_observed) > 0:
                self.replan(observation)

        while self.plan:
            action = self.plan.pop(0)
            name = get_action_name(action)
            if self.last_action_name != name:
                print(self.step_count, print_action(action))
            self.last_action_name = name

            ## already broken down to specific robot commands
            if not isinstance(action, Action):
                if isinstance(action, list):
                    rest = action[1:]
                    self.plan = rest + self.plan
                    action = action[0]
                self.actions.append(action)
                return action

            name, args = action
            self.step_count += 1

            if self.env_execution is not None and name in self.env_execution.domain.operators:
                facts_old = self.state
                self.state = self.env_execution.step(action) + self.static_facts
                self.state = list(set(self.state))
                added = list(set(self.state) - set(facts_old))
                deled = list(set(facts_old) - set(self.state))
                if len(added) > 0:
                    print(f'\tadded: {added}')
                    print(f'\tdeled: {deled}')

                # summarize_facts(self.state, self.world, name='Facts computed during execution')

            if '--no-' in name:
                self.refine_plan(action, observation)
            else:
                commands = get_primitive_actions(action, self.world, teleport=SAVE_TIME)
                self.plan = commands + self.plan
                self.plan_len += 1
                return self.process_plan(observation)

        return None

    def replan(self, observation, **kwargs):
        """ the first planning """

        if self.llamp_api is not None:
            obs_path = self._replan_preprocess(observation)

        self.plan_step = self.num_steps
        self.plan, env, knowledge, time_log, preimage = self.solve_pddlstream(
            self.pddlstream_problem, observation.state, domain_pddl=self.domain_pddl,
            domain_modifier=self.domain_modifier, **self.pddlstream_kwargs, **kwargs)  ## observation.objects
        self.record_time(time_log)
        self.initial_state.remove_gripper()  ## after the first planning

        self.evaluations, self.goal_exp, self.domain, self.externals = knowledge
        self.state = list(set(self.pddlstream_problem.init + preimage))

        ## save the failed streams
        failures_file = join(VISUALIZATIONS_PATH, 'log.json')
        if self.exp_name == 'hpn' and isdir(VISUALIZATIONS_PATH) and isfile(failures_file):
            shutil.move(failures_file, join(VISUALIZATIONS_PATH, f'log_0.json'))

        if 'hpn' in self.exp_name:
            self._replan_postprocess(env, preimage)

        ## move the txt_file.txt to log directory
        if self.llamp_api is not None:
            self._replan_postprocess(obs_path)

        return self.plan

    def _replan_preprocess(self, observation):
        assert NotImplemented

    def _replan_postprocess(self, **kwargs):
        assert NotImplemented

    """ state related """

    def state_changed(self, observation): # TODO: consider the continuous values
        step = self.plan_step
        return not self.observations or (self.observations[step].objects != observation.objects)

    def goal_achieved(self, observation):
        from pybullet_tools.logging import myprint as print

        ## hack for checking if the plan has been executed
        if self.plan is not None and len(self.plan) == 0: ## []
            print('\n\nfinished executing plan\n')
            # wait_if_gui('finish?')
            return True
        return False

    def check_goal_achieved(self, facts, next_goal):
        from world_builder.world_utils import check_goal_achieved
        return check_goal_achieved(facts, next_goal, self.world)

    def policy(self, observation):
        observation.assign()
        action = self.process_plan(observation)

        if action is not None:
            self.record_command(action)
            return action

        """ if no more action to execute, check success or replan """
        while not self.plan:
            facts = observation.facts
            seq_planning_mode = self.goal_sequence is not None and len(self.goal_sequence) > 1
            if seq_planning_mode:
                ## the first planning problem also need to be processed to reduce objects
                if self.problem_count > 0:
                    self.goal_sequence.pop(0)
                next_goal = self.goal_sequence[0]
                if str(next_goal) == 'on([10, (4, None, 1)])':
                    print('check_goal_achieved on([10, (4, None, 1)])')
                while self.check_goal_achieved(facts, next_goal):
                    print(f'\ncheck_goal_achieved({next_goal})\n')
                    self.goal_sequence.pop(0)
                    next_goal = self.goal_sequence[0]
                self.update_pddlstream_problem(facts, [next_goal])

            elif self.goal_achieved(observation):
                self.save_stats()  ## save the planning time statistics
                return None

            self.replan(observation)

            if not self.plan:
                ## backtrack planning tree to use other subgoals
                if seq_planning_mode:
                    status = self.llamp_api.backtrack_planing_tree()
                    if status in ['succeed', 'failed']:
                        self.save_stats(solved=(status == 'succeed'))
                    else:
                        self.goal_sequence = status
                else:
                    break

        # if (self.plan is None) or (len(self.plan) == 0):
        #     return None
        return self.process_plan(observation)
        #return self.process_commands(current_conf)

    ###############################################################################

    def record_time(self, time_log):
        self.time_log.append(time_log)
        print('\n[TIME LOG]\n' + '\n'.join([str(v) for v in self.time_log]))

    def record_command(self, action):
        self.commands.append(action)

    def save_commands(self, commands_path):
        save_commands(self.commands, commands_path)

    def save_stats(self, solved=True):
        print('\n\nsaving statistics\n\n')
        IS_HPN = self.comparing and self.exp_name != 'original'
        name = add_timestamp(self.exp_name)
        exp_name = name if not IS_HPN else basename(self.exp_dir)

        for i in range(len(self.time_log)):
            if 'planning' not in self.time_log[i]:
                print('save_stats', i, self.time_log[i])
        print(self.time_log[0]['planning']) #  --monitoring
        durations = {i: self.time_log[i]['planning'] for i in range(len(self.time_log))}
        durations2 = {i: self.time_log[i]['preimage'] for i in range(len(self.time_log))}
        total_planning = sum(list(durations.values()))
        total_preimage = sum(list(durations2.values()))
        if not solved:
            total_planning = 99999

        ## save a line in cvs of planning time
        csv_name = join(self.exp_dir, f'{self.exp_name}.csv')
        if IS_HPN:  ## put one directory up
            csv_name = join(dirname(self.exp_dir), f'{self.exp_name}.csv')
        fieldnames = ['exp_name']
        fieldnames.extend(list(durations.keys()))
        fieldnames.append('total_planning')
        fieldnames.append('preimage')
        fieldnames.append('plan_len')
        if not isfile(csv_name):
            with open(csv_name, mode='w') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
        with open(csv_name, mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            row = {'exp_name': exp_name}
            row.update(durations)
            row['total_planning'] = total_planning
            row['preimage'] = total_preimage
            row['plan_len'] = self.plan_len
            writer.writerow(row)

        ## save a txt of plan / time_log
        self.time_log.append({'total_planning': total_planning})
        with open(join(self.exp_dir, f'{name}_time.json'), 'w') as f:
            json.dump(self.time_log, f, indent=4)
            # f.write('\n'.join([str(t) for t in self.time_log]))

        if os.path.isfile(TXT_FILE):
            shutil.move(TXT_FILE, join(self.exp_dir, f"{name}_log.txt"))

        command_file = join(self.exp_dir, f"{name}_commands.pkl")
        self.save_commands(command_file)
        self.mp4_path = join(self.exp_dir, f"{name}.mp4")
        self.timestamped_name = name


#########################


def add_timestamp(exp_name):
    return f'{get_datetime(seconds=True)}_{exp_name}'


def get_action_name(action):
    if isinstance(action, Action):
        return action.name
    return action.__class__.__name__


def print_action(action):
    """ not the action in bullet.actions, but just a named tuple """
    if isinstance(action, tuple):
        name, args = action
        if 'move' in name:
            new_args = []
            for ele in args:
                if isinstance(ele, Trajectory):
                    new_args.append(f'{ele}-{ele.path[0]}')
                else:
                    new_args.append(ele)
            return Action(name, new_args)
    return action