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

from pybullet_tools.logging_utils import summarize_facts, print_goal
from pybullet_tools.bullet_utils import get_datetime
from pybullet_tools.pr2_primitives import Trajectory
from pybullet_tools.stream_agent import solve_pddlstream
from pybullet_tools.utils import SEPARATOR
from pybullet_tools.logging_utils import save_commands, TXT_FILE, summarize_state_changes

from world_builder.actions import get_primitive_actions
from world_builder.world_utils import get_camera_image

from cogarch_tools.processes.motion_agent import MotionAgent
from cogarch_tools.cogarch_utils import clear_empty_exp_dirs
from problem_sets.pr2_problems import pddlstream_from_state_goal

from leap_tools.domain_modifiers import initialize_domain_modifier
from leap_tools.object_reducers import initialize_object_reducer

from pddlstream.language.constants import Action, AND, PDDLProblem


ZOOM_IN_AT_OBJECT = False
SAVE_TIME = False

from world_builder.paths import PBP_PATH
PDDL_PATH = join(PBP_PATH, '..', 'assets', 'pddl')
VISUALIZATIONS_PATH = join(PBP_PATH, '..', 'examples', 'visualizations')


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
        self.replan_frequency = replan_frequency
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
        self.time_log = []  ## for recording time

        self.exp_dir = None  ## path to store runs
        self.exp_name = None  ## name to group runs
        self.timestamped_name = None  ## name to identify a run

        self.pddlstream_problem = None
        self.initial_state = None

        self.state = init
        self.static_facts = [('=', ('PlaceCost',), 1), ('=', ('PickCost',), 1)]
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
        pddlstream_problem = problem_dict['pddlstream_problem']
        self.pddlstream_problem = state.robot.modify_pddl(pddlstream_problem)
        self.initial_state = state

    def init_experiment(self, args, domain_modifier=None, object_reducer=None, comparing=False):
        """ important for using the right files in replaning """

        ## related to saving data
        exp_name = args.exp_name
        if comparing and (exp_name != 'original'):
            exp_name = args.exp_subdir
        exp_name = self._init_object_reducer(args, object_reducer, exp_name)

        self.exp_name = exp_name
        self.timestamped_name = add_timestamp(exp_name)
        self.exp_dir = abspath(join(args.exp_dir, args.exp_subdir, self.timestamped_name))
        clear_empty_exp_dirs(dirname(self.exp_dir))
        if not isdir(self.exp_dir):
            os.makedirs(self.exp_dir, exist_ok=True)

        self.domain_pddl = args.domain_pddl
        self.stream_pddl = args.stream_pddl
        self.domain_modifier = initialize_domain_modifier(domain_modifier)
        self.custom_limits = self.robot.custom_limits

        ## HPN experiments
        self.comparing = comparing

    def _init_object_reducer(self, args, object_reducer, exp_name):
        if hasattr(args, 'object_reducer'):
            object_reducer = args.object_reducer

        if object_reducer is not None:
            exp_name += '_' + object_reducer.replace(';', '_')
        self.object_reducer = initialize_object_reducer(object_reducer)
        return exp_name

    def goal_achieved(self, observation):
        from pybullet_tools.logging_utils import myprint as print

        ## hack for checking if the plan has been executed
        if self.plan is not None and len(self.plan) == 0:
            print('\n\nfinished executing plan\n')
            # wait_if_gui('finish?')
            return True
        return False

    def policy(self, observation):
        observation.assign()
        action = self.process_plan(observation)
        if action is not None:
            self.record_command(action)
            return action

        ## if no more action to execute, check success or replan
        if not self.plan:
            if self.goal_achieved(observation):
                self.save_stats()
                return None
            self.replan(observation)

        print('\n\npddlstream_agent.policy | start self.process_plan')
        return self.process_plan(observation)

    def process_plan(self, observation):
        """ get the next action if a plan has been made
            example self.plan
            00 = {MoveBaseAction} MoveBaseAction{conf: q552=(1.529, 5.989, 0.228, 3.173)}
            01 = {MoveBaseAction} MoveBaseAction{conf: q592=(1.607, 6.104, 0.371, 3.123)}
            10 = {MoveBaseAction} MoveBaseAction{conf: q744=(1.474, 7.326, 0.808, 9.192)}
            11 = {Action: 2} Action(name='pick', args=('left', 4, p1=(0.75, 7.3, 1.24, 0.0, -0.0, 0.0), g104=(-0.0, 0.027, -0.137, 0.0, -0.0, -3.142), q728=(1.474, 7.326, 0.808, 2.909), c544=t(7, 129)))
        """
        from pybullet_tools.logging_utils import myprint as print

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
                if len(self.plan) > 0 and get_action_name(self.plan[0]) != get_action_name(action):
                    print('pddlstream_agent.process_plan\tgetting new action', action)
                return action

            # if self.step_count in [7,  8]:
            #     print('self.step_count in [7,  8]')

            name, args = action
            incomplete_action = '?t' in args

            ## may be an abstract action or move_base action that hasn't been solved
            if '--no-' in name or incomplete_action:
                self.refine_plan(action, observation)
            else:
                if self.env_execution is not None and name in self.env_execution.domain.operators:
                    self._update_state(action)

                commands = get_primitive_actions(action, self.world, teleport=SAVE_TIME)
                self.plan = commands + self.plan
                self.plan_len += 1
                return self.process_plan(observation)

        return None

    def _update_state(self, action):
        facts_old = set(self.state)
        added, deled = self.env_execution.step(action)
        self.state += added + self.static_facts
        self.state = [f for f in set(self.state) if f not in deled]
        print(f'pddlstream_agent._update_state(step={self.step_count}, {action})')
        summarize_state_changes(self.state, facts_old, title='')
        # summarize_facts(self.state, self.world, name='Facts computed during execution')
        self.step_count += 1

    def replan(self, observation, **kwargs):
        """ make new plans given a pddlstream_problem """

        self.plan_step = self.num_steps
        self.plan, env, knowledge, time_log, preimage = self.solve_pddlstream(
            self.pddlstream_problem, observation.state, domain_pddl=self.domain_pddl,
            domain_modifier=self.domain_modifier, **self.pddlstream_kwargs, **kwargs)  ## observation.objects
        self.pddlstream_kwargs.update({'skeleton': None, 'subgoals': None})

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

        return self.plan

    def _replan_preprocess(self, observation):
        assert NotImplemented

    def _replan_postprocess(self, **kwargs):
        assert NotImplemented

    ###############################################################################

    def record_time(self, time_log):
        self.time_log.append(time_log)
        print('-'*50)
        print('\n[TIME LOG]\n' + '\n'.join([str(v) for v in self.time_log]))
        print('-'*50, '\n')

    def remove_unpickleble_attributes(self):
        self.world.remove_unpickleble_attributes()

    def recover_unpickleble_attributes(self):
        self.world.recover_unpickleble_attributes()

    def record_command(self, action):
        self.commands.append(action)

    def save_commands(self, commands_path):
        self.remove_unpickleble_attributes()
        save_commands(self.commands, commands_path)

    def save_time_log(self, csv_name, solved=True):
        """ compare the planning time and plan length across runs """
        from pybullet_tools.logging_utils import myprint as print
        from tabulate import tabulate

        for i in range(len(self.time_log)):
            if 'planning' not in self.time_log[i]:
                print('save_time_log', i, self.time_log[i])
        print('pddlstream_agent.save_time_log\n\ttotal planning time:',
              self.time_log[0]['planning'])  # --monitoring
        durations = {i: self.time_log[i]['planning'] for i in range(len(self.time_log))}
        durations2 = {i: self.time_log[i]['preimage'] for i in range(len(self.time_log))}
        total_planning = sum(list(durations.values()))
        total_preimage = sum(list(durations2.values()))
        if not solved:
            total_planning = 99999

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
            row = {'exp_name': self.exp_name}
            row.update(durations)
            row['total_planning'] = total_planning
            row['preimage'] = total_preimage
            row['plan_len'] = self.plan_len
            writer.writerow(row)

        print(tabulate([list(row.values())], headers=fieldnames, tablefmt='orgtbl'))

        return total_planning

    def save_stats(self, solved=True):
        print('\n\nsaving statistics\n\n')
        name = self.timestamped_name

        ## save one line in cvs of planning time and plan length
        csv_name = join(dirname(self.exp_dir), f'{self.exp_name}.csv')
        if self.comparing and ('original' not in self.exp_name):  ## put one directory up
            csv_name = join(dirname(dirname(self.exp_dir)), f'{self.exp_name}.csv')
        total_planning = self.save_time_log(csv_name, solved=solved)

        ## save the final plan
        plan_log_path = join(self.exp_dir, f'time.json')
        self.time_log.append({'total_planning': total_planning})
        with open(plan_log_path, 'w') as f:
            json.dump(self.time_log, f, indent=4)
            # f.write('\n'.join([str(t) for t in self.time_log]))

        ## save the log txt, commands, and video recording
        if os.path.isfile(TXT_FILE):
            shutil.move(TXT_FILE, join(self.exp_dir, f"log.txt"))

        self.save_commands(join(self.exp_dir, f"commands.pkl"))


###########################################################################


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
                    new_args.append(f'{ele}-{ele.path[0]}-{ele.path[-1]}')
                else:
                    new_args.append(ele)
            return Action(name, new_args)
    return action
