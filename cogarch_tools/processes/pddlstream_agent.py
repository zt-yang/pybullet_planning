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
from pybullet_tools.stream_agent import solve_pddlstream, make_init_lower_case
from pybullet_tools.utils import SEPARATOR, wait_if_gui, WorldSaver
from pybullet_tools.logging_utils import save_commands, TXT_FILE, summarize_state_changes, print_lists

from world_builder.actions import get_primitive_actions
from world_builder.world_utils import get_camera_image

from cogarch_tools.processes.motion_agent import MotionAgent
from cogarch_tools.cogarch_utils import clear_empty_exp_dirs

from leap_tools.domain_modifiers import initialize_domain_modifier
from leap_tools.object_reducers import initialize_object_reducer

from pigi_tools.replay_utils import apply_actions, load_basic_plan_commands
from lisdf_tools.lisdf_planning import Problem

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
        self.world_state = None

        self.state = init
        self.last_plan_state = None  ## updated after planning
        self.last_added_facts = []
        self.last_deled_facts = []
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

    ###################################################################

    def initialize(self, state):
        """ when the agent state is loaded from previous saved runs, the world state changes """
        if self.world_state is None:
            self.world_state = state
        return self.world_state

    def set_world_state(self, state):
        self.world_state = state

    ###################################################################

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
            wait_if_gui('finish?')
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

        # if self.plan:
        #     self.world.remove_redundant_bodies()

        if observation.unobserved_objs is not None:
            newly_observed = observation.update_unobserved_objs()
            if len(newly_observed) > 0:
                self.replan(observation)

        while self.plan:
            action = self.plan.pop(0)
            name = get_action_name(action)
            if self.last_action_name != name:
                print(f"{self.step_count}\t{print_action(action)}")
            self.last_action_name = name

            ## already broken down to specific robot commands
            if not isinstance(action, Action):
                if isinstance(action, list):
                    rest = action[1:]
                    self.plan = rest + self.plan
                    action = action[0]
                self.actions.append(action)
                if len(self.plan) > 0 and get_action_name(self.plan[0]) != get_action_name(action):
                    print(f'pddlstream_agent.process_plan\tgetting new action {action}')
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
        self.state = update_facts(self.state, added=added + self.static_facts, deled=deled)
        print(f'pddlstream_agent._update_state(step={self.step_count}, {action})')
        summarize_state_changes(self.state, facts_old, title='')
        # summarize_facts(self.state, self.world, name='Facts computed during execution')
        self.step_count += 1
        self.last_added_facts = added
        self.last_deled_facts = deled

    def replan(self, observation, **kwargs):
        """ make new plans given a pddlstream_problem """

        self.plan_step = self.num_steps
        self.plan, env, knowledge, time_log, preimage = self.solve_pddlstream(
            self.pddlstream_problem, observation.state, domain_pddl=self.domain_pddl,
            domain_modifier=self.domain_modifier, **self.pddlstream_kwargs, **kwargs)  ## observation.objects
        self.pddlstream_kwargs.update({'skeleton': None, 'subgoals': None})
        self.evaluations, self.goal_exp, self.domain, _ = knowledge
        self.record_time(time_log)
        is_HPN = 'hpn' in self.exp_name or env is not None

        if is_HPN:
            self.state = make_init_lower_case(set(self.pddlstream_problem.init + preimage))

            ## save the failed streams
            failures_file = join(VISUALIZATIONS_PATH, 'log.json')
            if isdir(VISUALIZATIONS_PATH) and isfile(failures_file):
                shutil.move(failures_file, join(VISUALIZATIONS_PATH, f'log_0.json'))
        else:
            print(f'pddlstream.replan\tstep_count = {self.step_count}')
            self.state = make_init_lower_case(self.pddlstream_problem.init)
            self.last_plan_state = copy.deepcopy(self.state)

        ## the first planning problem - only for
        if self.env_execution is None:  ## and not self.pddlstream_kwargs['visualization']:
            if self.plan is None:
                self.save_stats(solved=False)
            self._init_env_execution()

        ## hierarchical planning in the now
        if is_HPN:
            self._replan_postprocess(env, preimage)

        return self.plan

    def _init_env_execution(self):
        from leap_tools.hierarchical import PDDLStreamForwardEnv

        domain_pddl = self.domain_pddl
        domain_pddl = join(PDDL_PATH, 'domains', domain_pddl)
        init = self.state
        self.env_execution = PDDLStreamForwardEnv(domain_pddl, self.pddlstream_problem, init=init)
        self.env_execution.reset()

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
        if len(self.commands) > 0:
            self.remove_unpickleble_attributes()
            save_commands(self.commands, commands_path)
            shutil.copy(commands_path, join(self.exp_dir, 'states', f'commands_{self.problem_count}.pkl'))
            self.recover_unpickleble_attributes()

    def save_time_log(self, csv_name, solved=True, failed_time=False):
        """ compare the planning time and plan length across runs """
        from pybullet_tools.logging_utils import myprint as print
        from tabulate import tabulate

        for i in range(len(self.time_log)):
            if 'planning' not in self.time_log[i]:
                print(f'save_time_log, self.time_log[{i}] = {self.time_log[i]}')
        print('pddlstream_agent.save_time_log\n\ttotal planning time:'+str(self.time_log[0]['planning']))
        durations = {i: self.time_log[i]['planning'] for i in range(len(self.time_log))}
        durations2 = {i: self.time_log[i]['preimage'] for i in range(len(self.time_log))}
        total_planning = sum(list(durations.values()))
        total_preimage = sum(list(durations2.values()))
        if not solved and not failed_time:
            total_planning = 99999

        fieldnames = ['exp_name']
        fieldnames.extend(list(durations.keys()))
        fieldnames.append('total_planning')
        fieldnames.append('preimage')
        fieldnames.append('plan_len')

        row = {'exp_name': self.exp_name}
        row.update(durations)
        row['total_planning'] = total_planning
        row['preimage'] = total_preimage
        row['plan_len'] = self.plan_len

        if csv_name is not None:
            if not isfile(csv_name):
                with open(csv_name, mode='w') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()
            with open(csv_name, mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow(row)

        print(tabulate([list(row.values())], headers=fieldnames, tablefmt='orgtbl'))

        return total_planning

    def save_stats(self, solved=True, final=True, failed_time=False):
        print('\n\nsaving statistics\n\n')
        name = self.timestamped_name

        ## save one line in cvs of planning time and plan length
        if final:

            ## save the log txt, commands, and video recording
            if os.path.isfile(TXT_FILE):
                shutil.move(TXT_FILE, join(self.exp_dir, f"log.txt"))

            csv_name = join(dirname(self.exp_dir), f'{self.exp_name}.csv')
            if self.comparing and ('original' not in self.exp_name):  ## put one directory up
                csv_name = join(dirname(dirname(self.exp_dir)), f'{self.exp_name}.csv')
        else:
            csv_name = None

        total_planning = self.save_time_log(csv_name, solved=solved, failed_time=failed_time)

        if final:
            print('save_stats.total_planning.final')
            self.time_log.append({'total_planning': total_planning})

        ## save the final plan
        plan_log_path = join(self.exp_dir, f'time.json')
        with open(plan_log_path, 'w') as f:
            json.dump(self.time_log, f, indent=4)
            # f.write('\n'.join([str(t) for t in self.time_log]))

        self.save_commands(join(self.exp_dir, f"commands.pkl"))

    def save_agent_state(self):
        """ resume planning """
        agent_state_dir = join(self.exp_dir, 'states')
        if not isdir(agent_state_dir):
            os.makedirs(agent_state_dir)

        stream_map = self.env_execution._pddlstream_problem[3]
        static_literals = self.env_execution.static_literals
        env_externals = self.env_execution.externals
        _action_space = self.env_execution._action_space
        _observation_space = self.env_execution._observation_space
        variables = self.initial_state.variables

        self.sample_fn = None
        self.difference_fn = None
        self.distance_fn = None
        self.extend_fn = None
        self.observations = []

        a, b, c, _, e, f = self.env_execution._pddlstream_problem
        self.env_execution._pddlstream_problem = PDDLProblem(a, b, c, None, e, f)
        a, b, c, _, e, f = self.pddlstream_problem
        self.pddlstream_problem = PDDLProblem(a, b, c, None, e, f)
        self.env_execution.static_literals = None
        self.env_execution.externals = None
        self.env_execution._action_space = None
        self.env_execution._observation_space = None
        self.initial_state.variables = None

        agent_state_path = join(agent_state_dir, f'agent_state_{self.problem_count}.pkl')
        with open(agent_state_path, 'bw') as f:
            pickle.dump(self, f)

        # world_state_path = join(agent_state_dir, f'world_state_{self.problem_count}.pkl')
        # with open(world_state_path, 'bw') as f:
        #     pickle.dump(WorldSaver(), f)

        # for k, v in self.__dict__.items():
        #     print(k)
        #     with open(agent_state_path, 'bw') as f:
        #         pickle.dump(v, f)
        #
        # for k, v in self.env_execution.__dict__.items():
        #     print(k)
        #     with open(agent_state_path, 'bw') as f:
        #         pickle.dump(v, f)

        a, b, c, _, e, f = self.env_execution._pddlstream_problem
        self.env_execution._pddlstream_problem = PDDLProblem(a, b, c, stream_map, e, f)
        a, b, c, _, e, f = self.pddlstream_problem
        self.pddlstream_problem = PDDLProblem(a, b, c, stream_map, e, f)
        self.env_execution.static_literals = static_literals
        self.env_execution.externals = env_externals
        self.env_execution._action_space = _action_space
        self.env_execution._observation_space = _observation_space
        self.initial_state.variables = variables

    def load_agent_state(self, agent_state_path):
        """ resume planning """

        print('\n\n'+'-'*60+f'\n[load_agent_state] from {agent_state_path}\n')

        if self.env_execution is None:
            self._init_env_execution()

        exp_dir = self.exp_dir
        stream_map = self.pddlstream_problem[3]
        static_literals = self.env_execution.static_literals
        env_externals = self.env_execution.externals
        _action_space = self.env_execution._action_space
        _observation_space = self.env_execution._observation_space
        variables = self.initial_state.variables

        with open(agent_state_path, 'br') as f:
            self = pickle.load(f)

        commands_path = agent_state_path.replace('agent_state_', 'commands_')
        # commands_path = agent_state_path.replace('agent_state_', 'commands_')
        self.apply_commands(commands_path)

        self.exp_dir = exp_dir
        self.plan = []

        a, b, c, _, e, f = self.env_execution._pddlstream_problem
        self.env_execution._pddlstream_problem = PDDLProblem(a, b, c, stream_map, e, f)
        a, b, c, _, e, f = self.pddlstream_problem
        self.pddlstream_problem = PDDLProblem(a, b, c, stream_map, e, f)
        self.env_execution.static_literals = static_literals
        self.env_execution.externals = env_externals
        self.env_execution._action_space = _action_space
        self.env_execution._observation_space = _observation_space
        self.initial_state.variables = variables

        print('\n'+'-'*60+f'\n\n')

        return self

    def apply_commands(self, commands_path):

        with open(commands_path, 'br') as f:
            commands = pickle.load(f)

        problem, _, plan, body_map = load_basic_plan_commands(self.world, self.exp_dir, self.exp_dir,
                                                              load_attach=False, maybe_hpn=False)
        apply_actions(problem, commands, time_step=0.001, verbose=False, plan=plan)  ## , body_map=body_map


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


def update_facts(facts, added, deled):
    return [f for f in set(facts + added) if f not in deled]


def filter_dynamic_facts(facts):
    ## the following predicates are generated again by state thus ignored
    ignored_preds = ['aconf', 'ataconf', 'atbconf', 'atpose', 'atposition', 'atrelpose', 'basemotion',
         'bconf', 'contained', 'defaultaconf', 'isclosedposition',
         'kingrasphandle', 'kinpulldoorhandle', 'pose', 'position', 'relpose']
    facts = [f for f in facts if f[0] not in ignored_preds]

    ## some positive and negative effects cancel out
    to_remove = []
    for i in range(len(facts)):
        fact = facts[i]
        if fact[0] == 'not' and fact[1] in facts:
            to_remove.extend([fact, fact[1]])
    facts = [f for f in facts if f not in to_remove]
    return facts
