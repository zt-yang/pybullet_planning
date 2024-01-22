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

from world_builder.actions import get_primitive_actions
from world_builder.world_utils import get_camera_image

from cogarch_tools.processes.motion_agent import MotionAgent
from problem_sets.pr2_problems import pddlstream_from_state_goal

from leap_tools.domain_modifiers import initialize_domain_modifier
from leap_tools.object_reducers import initialize_object_reducer

from pddlstream.language.constants import Action

# ORIGINAL_DOMAIN = {
#     'pr2_kitchen_abstract.pddl': 'pr2_kitchen.pddl',
#     'pr2_eggs_no_atbconf.pddl': 'pr2_eggs.pddl',
#     'pr2_eggs_no_on.pddl': 'pr2_eggs.pddl',
#     'pr2_eggs_no_on_atbconf.pddl': 'pr2_eggs.pddl',
#     'pr2_eggs_no_stuff.pddl': 'pr2_eggs.pddl',
#     'pr2_rearrange_no_atbconf.pddl': 'pr2_rearrange.pddl',
#     'pr2_kitchen_demo_no_atbconf.pddl': 'pr2_kitchen_demo.pddl',
#     'pr2_kitchen_demo_no_on_atbconf.pddl': 'pr2_kitchen_demo.pddl',
#     'pr2_food_rearrange_open_no_atbconf.pddl': 'pr2_food_rearrange_open.pddl',
#     'pr2_food_rearrange_no_atbconf.pddl': 'pr2_food_rearrange.pddl',
#     'feg_kitchen_no_atseconf.pddl': 'feg_kitchen.pddl',
#     'pr2_mamao_no_atbconf.pddl': 'pr2_mamao.pddl',
# }
# REFINEMENT_DOMAIN = copy.deepcopy(ORIGINAL_DOMAIN)
# REFINEMENT_DOMAIN.update({
#     'pr2_eggs_no_on_atbconf.pddl': 'pr2_eggs_no_atbconf.pddl',
#     'pr2_kitchen_demo_no_on_atbconf.pddl': 'pr2_kitchen_demo_no_atbconf.pddl'
# })

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
        self.mp4_path = None
        self.timestamped_name = None

        self.pddlstream_problem = None
        self.initial_state = None
        self.goal_sequence = None
        self.llamp_agent = None

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

    def set_pddlstream_problem(self, pddlstream_problem, state):
        self.pddlstream_problem = pddlstream_problem
        self.initial_state = state

    def set_goal_sequence(self, goal_sequence):
        goal_sequence, llamp_agent = goal_sequence
        self.goal_sequence = goal_sequence
        self.llamp_agent = llamp_agent
        self.llamp_agent.output_html()

    def init_experiment(self, args, domain_modifier=None, object_reducer=None, comparing=False):
        """ important for using the right files in replanning """
        if object_reducer is not None:
            args.exp_name += '_' + object_reducer
        self.exp_name = args.exp_name
        self.exp_dir = abspath(join(args.exp_dir, args.exp_subdir))
        if not isdir(self.exp_dir):
            os.makedirs(self.exp_dir, exist_ok=True)
        self.domain_pddl = args.domain_pddl
        self.stream_pddl = args.stream_pddl
        self.domain_modifier = initialize_domain_modifier(domain_modifier)
        self.object_reducer = initialize_object_reducer(object_reducer)
        self.base_limits = self.robot.custom_limits
        self.comparing = comparing
        # print('\n\n\n agent.init_experiment | self.base_limits = ', self.base_limits)

    def dump_actions(self):
        import pickle
        with open(self.mp4_path.replace('.mp4', '_plan.pickle'), 'wb') as outp:
            for action in self.actions:
                pickle.dump(action, outp, pickle.HIGHEST_PROTOCOL)

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

        if self.llamp_agent is not None:
            obs_path = self.llamp_agent.log_obs_image(self.world.cameras)
            self.llamp_agent.log_subgoal(self.pddlstream_problem.goal[1], obs_path, 'started')
            self.llamp_agent.output_html()

        self.plan_step = self.num_steps
        self.plan, env, knowledge, time_log, preimage = self.solve_pddlstream(
            self.pddlstream_problem, observation.state, domain_pddl=self.domain_pddl,
            domain_modifier=self.domain_modifier, **self.pddlstream_kwargs, **kwargs) ## observation.objects
        self.record_time(time_log)
        self.initial_state.remove_gripper()  ## after the first planning

        ## save the failures
        failures_file = join(VISUALIZATIONS_PATH, 'log.json')
        if self.exp_name == 'hpn' and isdir(VISUALIZATIONS_PATH) and isfile(failures_file):
            shutil.move(failures_file, join(VISUALIZATIONS_PATH, f'log_0.json'))

        self.evaluations, self.goal_exp, self.domain, self.externals = knowledge
        self.state = list(set(self.pddlstream_problem.init + preimage))

        ## the first planning problem - only for HPN planning mode
        if self.env_execution is None and 'hpn' in self.exp_name: ## and not self.pddlstream_kwargs['visualization']:
            from leap_tools.hierarchical import PDDLStreamForwardEnv

            if self.plan is None:
                self.save_stats(FAILED=True)
            domain_pddl = self.domain_pddl
            domain_pddl = join(PDDL_PATH, 'domains', domain_pddl)
            init = self.state
            self.env_execution = PDDLStreamForwardEnv(domain_pddl, self.pddlstream_problem, init=init)
            self.env_execution.reset()

        if env is not None:
            self.env = env
            self.process_hierarchical(env, self.plan, self.domain_pddl)
            for f in preimage:
                if f[0] in ['grasp', 'pose'] and str(f[-1]) in env.useful_variables:
                    self.useful_variables[f[-1]] = f
            print('-' * 20, '\nuseful_variables:')
            pprint(list(self.useful_variables.keys()))

            self.on_map = {v.variables[1].name: [kk.name for kk in k.variables] for k, v in env.on_map.items()}
            self.on_map = {('on', eval(k[0]), eval(k[1].replace('none', 'None'))): v for v, k in self.on_map.items()}
            print('-' * 20, '\non_map:')
            pprint(self.on_map)
            print('-' * 20)

        ## move the txt_file.txt to log directory
        if self.llamp_agent is not None:
            status = 'failed' if self.plan is None else 'solved'
            self.llamp_agent.log_subgoal(self.pddlstream_problem.goal[1], obs_path, status)
            self.llamp_agent.output_html()

        # wait_if_gui('continue to execute?')
        return self.plan

    def get_refinement_goal_init(self, action):
        # facts = observation.get_facts(self.env.init_preimage)
        op = self.env_execution.to_literal(action)
        preimage = self.preimages_after_op[op]
        goals = [self.env_execution.from_literal(n) for n in preimage]

        goals_not = [g for g in goals if g[0] == 'not']
        goals = [g for g in goals if g[0] != 'not']
        goals.extend(goals_not)
        # goals.sort(key = len, reverse = True)
        if self.failed_count is not None:
            goals = goals[:self.failed_count]
        # goals = [n for n in goals if n[0] != 'not']  ## for debugging 2nd refinement
        # goals = [n for n in goals if n[0] != 'cfreetrajpose']  ## for debugging replacement tests
        # goals = [n for n in goals if n[0] == 'safeatraj'][:1]  ## for debugging replacement tests

        facts = []

        ## add in potentially useful facts
        all_goal_elements = goals + [g[1] for g in goals_not]
        for goal in all_goal_elements:
            for elem in goal:
                if elem in self.useful_variables:
                    useful = self.useful_variables[elem]
                    if useful not in facts and useful not in self.state:
                        facts.append(useful)
                        print('\tadded useful fact', useful)

        ## remove irrelevant facts
        ignore = ['basemotion', 'btraj', 'cfreeapproachpose', 'cfreeposepose', 'not']
        removed = []
        for f in self.state:
            if f[0] in self.env_execution.domain.predicates and f[0] not in ignore and \
                    not self.env_execution.domain.predicates[f[0]].is_derived:
                if f[0] == 'pose' and (
                        ('atpose', f[1], f[2]) not in self.state and ('atpose', f[1], f[2]) not in goals):
                    print('removed irrelevant pose', f)
                    removed.append(f)
                    continue
                if f[0] == 'bconf' and ('atbconf', f[1]) not in self.state:
                    print('removed irrelevant bconf', f)
                    removed.append(f)
                    continue
                if f[0] == 'reach' and ('bconf', f[3]) not in self.state:
                    print('removed irrelevant reach', f)
                    removed.append(f)
                    continue
                if f[0] in ['supported', 'kin', 'atraj']:
                    print('removed irrelevant', f)
                    removed.append(f)
                    continue
                facts.append(f)
                if f in goals:
                    goals.remove(f)

        ## hack to make sure the init is the minimal required to solve the sub problem
        goals_check = copy.deepcopy(goals)
        goal_on = [g for g in goals if g[0] == 'on']
        if len(goal_on) > 0:
            goal_on = goal_on[0]
            pose = self.on_map[goal_on]
            goals_check += [('pose', goal_on[1], pose)]

        add_relevant_facts_given_goals(facts, goals_check, removed)

        # ## hack to make sure the init is the minimal required to solve the sub problem
        # goal_on = [g for g in goals if g[0] == 'on']
        # if len(goal_on) > 0:
        #     add_facts = []
        #     goal_on = goal_on[0]
        #     pose = self.on_map[goal_on]
        #     reach = [f for f in removed if f[0] == 'reach' and f[3] == pose][0]
        #     kin = [f for f in removed if f[[0] == 'kin' and f[3] == pose]][0]
        #     add_facts += [('pose', goal_on[1], pose), ('bconf', reach[-1]), reach, kin]
        #     add_facts += [f for f in removed if f[0] == 'supported' and f[2] == pose]
        #     print(f'\n{goal_on}\t->\tadded facts\n\t'+'\n\t'.join([str(f) for f in add_facts]))
        #     facts += add_facts
        #
        # goal_at_grasp = [g for g in goals if g[0] == 'atgrasp']
        # if len(goal_at_grasp) > 0:
        #     add_facts = []
        #     goal_at_grasp = goal_at_grasp[0]
        #     pose = [f[2] for f in facts if f[0] == 'atpose'][0]
        #     reach = [f for f in removed if f[0] == 'reach' and f[3] == pose][0]
        #     add_facts += [('bconf', reach[-1]), reach]
        #     add_facts += [f for f in removed if f[0] == 'kin' and f[3] == pose]
        #     print(f'\n{goal_at_grasp}\t->\tadded facts\n\t'+'\n\t'.join([str(f) for f in add_facts]))
        #     facts += add_facts

        return goals, facts

    def refine_plan(self, action, observation, **kwargs):
        from pybullet_tools.logging import myprint
        from pddlstream.algorithms.algorithm import reset_globals

        self.refinement_count += 1
        myprint(f'\n## {self.refinement_count}th refinement problem')

        goals, facts = self.get_refinement_goal_init(action)

        facts = self.object_reducer(facts, goals)

        # self.last_goals = goals  ## used to find new goals
        if self.domain_modifier is not None:
            domain_pddl = self.domain_pddl
            predicates = action.name.split('--no-')[1:]
            if len(predicates) == 1 or True:  ## TODO: better way to schedule the postponing
                domain_modifier = None
            else:
                domain_modifier = initialize_domain_modifier(predicates[:-1])
        else:
            domain_pddl = self.domains_for_action[action]
            domain_modifier = None

        pddlstream_problem = pddlstream_from_state_goal(observation.state, goals, custom_limits=self.base_limits,
                                                        domain_pddl=domain_pddl, stream_pddl=self.stream_pddl,
                                                        facts=facts, PRINT=True)

        sub_problem = pddlstream_problem
        sub_state = observation.state
        # sub_problem = get_smaller_world(pddlstream_problem, observation.state.world)

        ## get new plan, by default it's using the original domain file
        reset_globals()
        plan, env, knowledge, time_log, preimage = self.solve_pddlstream(
            sub_problem, sub_state, domain_pddl=domain_pddl,
            domain_modifier=domain_modifier,
            **self.pddlstream_kwargs, **kwargs)  ## observation.objects
        observation.state.remove_gripper()

        ## save the failures
        failures_file = join(VISUALIZATIONS_PATH, 'log.json')
        if isdir(VISUALIZATIONS_PATH) and isfile(failures_file):
            shutil.move(failures_file, join(VISUALIZATIONS_PATH, f'log_{self.refinement_count}.json'))

        print('------------------------ \nRefined plan:', plan)
        if plan is not None:
            self.plan = plan + self.plan
            add_facts = [s for s in preimage if s not in self.state]
            self.static_facts += add_facts

            ## need to have here because it may have just been refining and no action yet
            self.state += self.static_facts
            self.state = list(set(self.state))
            self.record_time(time_log)

            print('\nnew plan:')
            [print('  ', p.name) for p in self.plan]
            print('\nadded facts:')
            [print('  ', p) for p in sorted([str(f) for f in add_facts])]
            print('\n')

            if env is not None:
                self.envs.append(env)
                self.process_hierarchical(env, plan, domain_pddl)

            if self.failed_count is not None:
                print('self.failed_count is not None')
                sys.exit()
        else:
            if plan is None:
                self.save_stats(FAILED=True)
                self.plan = None
            print('failed to refine plan! exiting...')
            sys.exit()
            # if self.failed_count == None:
            #     self.failed_count = len(goals) - 1
            # else:
            #     self.failed_count -= 1
            # self.refine_plan(action, observation)

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

    def record_time(self, time_log):
        self.time_log.append(time_log)
        print('\n[TIME LOG]\n' + '\n'.join([str(v) for v in self.time_log]))

    def process_hierarchical(self, env, plan, domain_pddl):

        ## add new continuous vars to env_exeution so from_literal can be used
        self.env_execution.add_objects(env)

        self.preimages_after_op.update(env.get_all_preimages_after_op())

        index = 0
        print('\nupdated library of preimages:')
        for action in self.plan:
            index += 1
            op = self.env_execution.to_literal(action)
            preimage = self.preimages_after_op[op]
            goals = [self.env_execution.from_literal(n) for n in preimage]
            print(f"\n{index}\t{action}")
            print('   eff:\n' + f'\n   '.join([str(g) for g in goals if g[0] != 'not']))
            g2 = [str(g) for g in goals if g[0] == 'not']
            if len(g2) > 0:
                print('   ' + f'\n   '.join(g2))

    def policy(self, observation):
        observation.assign()
        action = self.process_plan(observation)

        if action is not None:
            self.record_command(action)
            return action

        """ if no more action to execute, check success or replan """
        if not self.plan:
            if self.goal_achieved(observation):
                if self.goal_sequence is not None and len(self.goal_sequence) > 1:
                    self.goal_sequence.pop(0)
                    self.update_pddlstream_problem(observation.facts, [self.goal_sequence[0]])
                else:
                    self.save_stats() ## save the planning time statistics
                    return None
            self.replan(observation)
        # if (self.plan is None) or (len(self.plan) == 0):
        #     return None
        return self.process_plan(observation)
        #return self.process_commands(current_conf)

    def update_pddlstream_problem(self, init, goals):
        from pddlstream.language.constants import AND, PDDLProblem
        from pybullet_tools.logging import myprint as print_fn

        goal = [AND] + goals

        world = self.world
        print_fn(SEPARATOR)
        summarize_facts(init, self.world, name='Facts extracted from observation', print_fn=print_fn)
        print_goal(goal, world=world, print_fn=print_fn)
        print_fn(f'Robot: {world.robot} | Objects: {world.objects}\n'
                 f'Movable: {world.movable} | Fixed: {world.fixed} | Floor: {world.floors}')
        print_fn(SEPARATOR)

        domain_pddl, constant_map, stream_pddl, stream_map, _, _ = self.pddlstream_problem
        self.pddlstream_problem = PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

    def record_command(self, action):
        self.commands.append(action)

    def save_stats(self, FAILED=False):
        print('\n\nsaving statistics\n\n')
        IS_HPN = self.comparing and self.exp_name != 'original'
        name = f'{get_datetime(TO_LISDF=True)}_{self.exp_name}'
        exp_name = name if not IS_HPN else basename(self.exp_dir)

        for i in range(len(self.time_log)):
            if 'planning' not in self.time_log[i]:
                print('save_stats', i, self.time_log[i])
        print(self.time_log[0]['planning']) #  --monitoring
        durations = {i: self.time_log[i]['planning'] for i in range(len(self.time_log))}
        durations2 = {i: self.time_log[i]['preimage'] for i in range(len(self.time_log))}
        total_planning = sum(list(durations.values()))
        total_preimage = sum(list(durations2.values()))
        if FAILED:
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

        from pybullet_tools.logging import TXT_FILE
        import shutil
        if os.path.isfile(TXT_FILE):
            shutil.move(TXT_FILE, join(self.exp_dir, f"{name}_log.txt"))

        command_file = join(self.exp_dir, f"{name}_commands.pkl")
        with open(command_file, 'wb') as file:
            pickle.dump(self.commands, file)
        self.mp4_path = join(self.exp_dir, f"{name}.mp4")
        self.timestamped_name = name


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


def get_smaller_world(p, world):
    from pddlstream.language.constants import PDDLProblem
    hack = {
        (2, None, 17): [(2, 19), (2, 23)],  ## dagger
        (2, None, 0): [(2, 10), (2, 14)]  ## chewie
    }

    init = p.init
    goal = p.goal
    # world = state.world

    ## find joints to keep because on things in the goal
    movable = []
    joints = []
    for lit in goal:
        if lit[0] == 'atgrasp':
            o = lit[2]
            movable.append(o)
            pose = [f[2] for f in init if f[0] == 'atpose' and f[1] == o][0].value
            containers = [f[3] for f in init if f[0] == 'contained' and f[1] == o]
            if len(containers) > 0:
                container = containers[0]
                if container in hack:
                    joints.extend(hack[container])
        if lit[0]in ['on', 'in']:
            movable.append(lit[1])
            surface = lit[2]
            if surface in hack:
                joints.extend(hack[surface])
    remove_movable = [f[1] for f in init if f[0] == 'graspable' and f[1] not in movable]
    remove_joints = [f[1] for f in init if f[0] == 'joint' and f[1] not in joints]

    new_init = []
    removed_init = []
    for lit in init:
        result = True
        for term in lit[1:]:
            if term in remove_joints + remove_movable:
                result = False
                removed_init.append(lit)
                break
        if result:
            new_init.append(lit)

    print(f'\nagent.get_smaller_world | keeping only joints {joints} and movable {movable}')
    print(f'agent.get_smaller_world | removing init {removed_init}\n')
    summarize_facts(new_init, world, name='Facts modified for a smaller problem')
    print_goal(goal)

    sub_problem = PDDLProblem(p.domain_pddl, p.constant_map, p.stream_pddl, p.stream_map, new_init, goal)
    return sub_problem


def add_relevant_facts_given_goals(facts, goals, removed):
    """
    given a list of goal literals and a list of removed init literals,
    find the init literals that has the same elements as those in goal literals
    """
    start = time.time()

    all_elements = defaultdict(list)
    for fact in removed:
        for element in fact[1:]:
            if '=' in str(element):
                all_elements[element].append(fact)

    useful_elements = []
    for literal in goals + facts:
        if literal[0] == 'not':
            continue
        for element in literal[1:]:
            if '=' in str(element) and element not in useful_elements:
                useful_elements.append(element)

    used_elements = []
    added_facts = []
    while len(useful_elements) > 0:
        element = useful_elements.pop(0)
        used_elements.append(element)
        found_facts = all_elements[element]
        for fact in found_facts:
            if fact not in added_facts:
                if fact not in facts:
                    added_facts.append(fact)
                for ele in fact[1:]:
                    if '=' in str(ele) and ele not in useful_elements + used_elements \
                            and not str(ele).startswith('g') and not str(ele).startswith('hg'):
                        useful_elements.append(ele)

    print('\n' + '\n\t'.join([str(f) for f in goals]) +
          f'\n->\tadded facts in {round(time.time() - start, 3)}\n\t' +
          '\n\t'.join([str(f) for f in added_facts]) + '\n')
    facts += added_facts
