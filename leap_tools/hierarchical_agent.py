import sys
from os.path import join, abspath, dirname, isdir, isfile
from os import listdir, pardir
RD = abspath(join(dirname(__file__), pardir, pardir))
# sys.path.append(join(RD, pardir, 'cognitive-architectures', 'pddlgym'))

from cogarch_tools.processes.pddlstream_agent import *

from pybullet_tools.stream_agent import log_goal_plan_init, pddlstream_from_state_goal
from pybullet_tools.logging_utils import print_lists

hpn_kwargs = dict(domain_modifier='atbconf,canmove', exp_name='hpn')
hpn_kwargs = dict(domain_modifier='atbconf,canmove,basemotion', exp_name='hpn')
hpn_kwargs.update(dict(separate_base_planning=True))


class HierarchicalAgent(PDDLStreamAgent):

    def _replan_preprocess(self, **kwargs):
        pass

    def _replan_postprocess(self, env, preimage):
        self.env = env
        self.process_hierarchical_env(env, self.plan, self.domain_pddl)
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

    def get_refinement_goal_init(self, action):
        # facts = observation.get_facts(self.env.init_preimage)
        op = self.env_execution.to_literal(action)
        preimage = self.preimages_after_op[op]

        ## TODO: hack for refining the incomplete move_base action
        if action.name == 'move_base':
            goals = [['AtBConf', action.args[1]]]
        else:
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
                    if useful not in facts and useful not in self.state_facts:
                        facts.append(useful)
                        print('\tadded useful fact', useful)

        ## remove irrelevant facts
        ignore = ['basemotion', 'btraj', 'cfreeapproachpose', 'cfreeposepose', 'not']
        removed = []
        for f in self.state_facts:
            if f[0] in self.env_execution.domain.predicates and f[0] not in ignore and \
                    not self.env_execution.domain.predicates[f[0]].is_derived:
                if f[0] == 'pose' and (
                        ('atpose', f[1], f[2]) not in self.state_facts and ('atpose', f[1], f[2]) not in goals):
                    print('removed irrelevant pose', f)
                    removed.append(f)
                    continue
                if f[0] == 'bconf' and ('atbconf', f[1]) not in self.state_facts:
                    print('removed irrelevant bconf', f)
                    removed.append(f)
                    continue
                if f[0] == 'reach' and ('bconf', f[3]) not in self.state_facts:
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
        goals_check = copy.deepcopy([g for g in goals if g[0] == 'on'])
        goal_on = [g for g in goals if g[0] == 'on']
        if len(goal_on) > 0:
            goal_on = goal_on[0]
            pose = self.on_map[goal_on]
            goals_check += [('pose', goal_on[1], pose)]

        add_relevant_facts_given_goals(facts, goals_check, removed)

        ## some basic fact is missing due to previuos steps
        facts, ignored_goals = self.add_missing_preconditions(action, facts, goals[1:])
        goals = [g for g in goals if g not in ignored_goals]

        return goals, facts

    def add_missing_preconditions(self, action, facts, goals=[]):
        accepted_additions = ['canmove', 'canungrasp', 'cangrasphandle', 'canpick', 'not', '=']
        missing_preconds, ignored_goals, _ = self.check_action_preconditions(action, facts, goals, verbose=True)
        facts += [f for f in missing_preconds if f[0] in accepted_additions or \
                  (len(f) == 2 and isinstance(f[1], tuple) and f[1][0] in accepted_additions)]
        return facts, ignored_goals

    def check_action_preconditions(self, action, facts, goals, verbose=False):
        env = self.env_execution

        action_name = get_original_action_name(action.name)
        action = Action(action_name, action.args)

        print('\n'+'-'*50+'\ncheck_action_preconditions', action)
        action_literal = env.to_literal(action)
        info = env.identify_missing_facts(action_literal)
        if verbose:
            env.print_log(info)

        ## method 1
        preconditions, effects = env.ground_operator_preconds_effects(action_literal)
        del_effects = [env.from_literal(a) for a in preconditions if a.predicate.name != 'identical']
        add_effects = [env.from_literal(a) for a in effects]
        print_lists([(del_effects, 'preconditions'), (add_effects, 'effects')])

        ## method 2
        add_effects_2 = [env.from_literal(a) for a in info['add_effects']]
        add_effects += [a for a in add_effects_2 if a not in add_effects]
        del_effects_2 = [env.from_literal(a) for a in info['del_effects'] + info['need_preconditions']]
        for a in del_effects_2:
            if a[0] == 'not':
                a = a[1]
            if a[0] != 'identical' and a not in del_effects:
                del_effects.append(a)

        variables = {}
        for fact in facts:
            for arg in fact[1:]:
                if arg not in variables:
                    variables[arg] = []
                variables[arg].append(fact)

        missing_preconditions = []
        for precond in del_effects:
            if precond not in facts:
                print(f'\t! precondition not met\t {precond}')
                missing_preconditions.append(precond)

        ignored_goals = []
        for goal in goals:
            goal_mod = tuple([goal[0].lower()] + list(goal[1:]))
            if goal[0] == 'not':
                ignored_goals.append(goal)
                print(f'\t! goal ignored\t {goal}')
            elif goal_mod not in add_effects:
                print(f'\t! goal not met\t {goal}')
                for arg in goal[1:]:
                    if arg not in variables:
                        print(f'\t! variable {arg} in goal is not the same in init')

        print('-'*50)
        return missing_preconditions, ignored_goals, del_effects

    def refine_plan(self, action, observation, **kwargs):
        from pybullet_tools.logging_utils import myprint
        from pddlstream.algorithms.algorithm import reset_globals

        action_name = get_original_action_name(action.name)
        original_action = Action(action_name, action.args)

        self.refinement_count += 1
        myprint(f'\n## {self.refinement_count}th refinement problem')

        ## update goal and init
        cache = self.remove_unpickleble_attributes()
        goals, facts = self.get_refinement_goal_init(action)
        facts = self.object_reducer(facts, goals=goals)
        self.recover_unpickleble_attributes(cache)

        ## skip planning if the subgoal has been achieved
        missing_preconditions = None
        if '--no-' in action.name:
            missing_preconditions, _, preimage = self.check_action_preconditions(action, facts, goals)

        if missing_preconditions is not None and len(missing_preconditions) == 0:
            plan = [original_action]
            env = None
            time_log = {'planning': 0, 'preimage': 0}
            time_log.update(log_goal_plan_init(goals[1:], plan, preimage))
            print('refine_plan.skip replaning')
            # self._update_state(original_action)

        else:

            ## only two-level
            domain_pddl = self.domain_pddl
            predicates = action.name.split('--no-')[1:]
            if len(predicates) == 1 or True:  ## TODO: better way to schedule the postponing
                domain_modifier = None
            else:
                domain_modifier = initialize_domain_modifier(predicates[:-1])

            ## update the PDDLStream problem using domain pddl
            pddlstream_problem = pddlstream_from_state_goal(
                observation.state, goals, custom_limits=self.custom_limits,
                domain_pddl=domain_pddl, stream_pddl=self.stream_pddl, facts=facts, verbose=True
            )
            pddlstream_problem = self.robot.modify_pddl(pddlstream_problem)

            args = [a for a in action.args if type(a).__name__ in ['int', 'tuple', 'str'] and '?' not in str(a)]
            self.pddlstream_kwargs['skeleton'] = [[action_name] + args]

            sub_problem = pddlstream_problem
            sub_state = observation.state
            # sub_problem = get_smaller_world(pddlstream_problem, observation.state.world)

            ## get new plan, by default it's using the original domain file
            plan, env, knowledge, time_log, preimage = self.solve_pddlstream(
                sub_problem, sub_state, domain_pddl=domain_pddl,
                domain_modifier=domain_modifier,
                **self.pddlstream_kwargs, **kwargs)  ## observation.objects

            ## save the failures
            failures_file = join(VISUALIZATIONS_PATH, 'log.json')
            if isdir(VISUALIZATIONS_PATH) and isfile(failures_file):
                shutil.move(failures_file, join(VISUALIZATIONS_PATH, f'log_{self.refinement_count}.json'))

        print('------------------------ \nRefined plan:', plan)
        if plan is not None:
            self.plan = plan + self.plan
            add_facts = [s for s in set(preimage) if s not in self.state_facts]
            self.static_facts += [f for f in add_facts if f[0].lower() in ['cleaned', 'cooked', 'seasoned']]

            ## need to have here because it may have just been refining and no action yet
            # self.state_facts += self.static_facts
            # self.state_facts = list(set(self.state_facts))
            self.record_time(time_log)

            print('\nnew plan:')
            [print('  ', p.name) for p in self.plan]
            print('\nadded facts:')
            [print('  ', p) for p in sorted([str(f) for f in add_facts])]
            print('\n')

            if env is not None:
                self.envs.append(env)
                self.process_hierarchical_env(env, plan, domain_pddl)

            if self.failed_count is not None:
                print('self.failed_count is not None')
                sys.exit()

            return plan
        else:
            if plan is None:
                self.save_stats(solved=False)
                self.plan = None
            print('failed to refine plan! exiting...')
            sys.exit()
            # if self.failed_count == None:
            #     self.failed_count = len(goals) - 1
            # else:
            #     self.failed_count -= 1
            # self.refine_plan(action, observation)

    def process_hierarchical_env(self, env, plan, domain_pddl):

        ## add new continuous vars to env_exeution so from_literal can be used
        self.env_execution.add_objects(env)

        self.preimages_after_op.update(env.get_all_preimages_after_op())

        index = 0
        print('\nupdated library of preimages:')
        for action in plan:
            index += 1
            op = self.env_execution.to_literal(action)
            preimage = self.preimages_after_op[op]
            goals = [self.env_execution.from_literal(n) for n in preimage]
            print(f"\n{index}\t{action}")
            print('   eff:\n\t'+f'\n\t'.join([str(g) for g in goals if g[0] != 'not']))
            g2 = [str(g) for g in goals if g[0] == 'not']
            if len(g2) > 0:
                print('   ' + f'\n   '.join(g2))


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


def get_original_action_name(action_name):
    if '--' in action_name:
        action_name = action_name.split('--')[0]
    return action_name