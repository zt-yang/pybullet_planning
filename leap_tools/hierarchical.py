import sys
import os
from os.path import join, dirname, abspath
ROOT_DIR = abspath(join(dirname(__file__), os.pardir))
sys.path.extend([
    join(ROOT_DIR, '..'),
    join(ROOT_DIR, '..', 'pddlgym'),
    join(ROOT_DIR, '..', 'pddlstream'),
    join(ROOT_DIR, '..', 'bullet', 'pybullet-planning'),
    join(ROOT_DIR, '..', 'bullet', 'pybullet-planning', 'examples'),
    '/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages',  ## to use imageio-ffmpeg
    # join(ROOT_DIR, '..', 'pddlstream'),
    # join(ROOT_DIR, '..', 'pddlstream', 'examples', 'pybullet', 'utils'),
])
import shutil
import copy
import time
from pprint import pprint

from pddlgym.core import PDDLEnv, _check_domain_for_strips
from pddlgym.parser import PDDLDomainParser
from pddlgym.inference import find_satisfying_assignments
from pddlgym.structs import Literal, LiteralConjunction, \
    Predicate, Type, TypedEntity, Exists, ForAll, LiteralDisjunction, DerivedPredicate
from pddlgym.spaces import LiteralSpace, LiteralSetSpace

from pddlstream.language.constants import Action
from pddlstream.algorithms.algorithm import parse_problem

from world_builder.entities import Object

from pybullet_tools.logging_utils import summarize_facts
from pybullet_tools.pr2_primitives import Pose, Grasp, Conf, APPROACH_DISTANCE
from pybullet_tools.pr2_utils import TOP_HOLDING_LEFT_ARM
from pybullet_tools.utils import get_unit_vector, multiply, unit_quat
from pybullet_tools.logging_utils import myprint as print, summarize_state_changes, print_list

from leap_tools.hierarchical_utils import get_dynamic_literals, get_required_pred, get_effects, \
    DEFAULT_TYPE, get_preconds, find_poses, get_updated_facts, find_on_pose, \
    fix_facts_due_to_loaded_agent_state, remove_unparsable_lines


class PDDLProblemTranslator():
    def __init__(self, pddlstream_problem, domain, init=None, objects=[]):
        self.problem_fname = 'dontknow'
        self.domain_name = domain.domain_name
        self.types = domain.types
        self.predicates = domain.predicates
        self.action_names = domain.actions
        self.uses_typing = not ("default" in self.types)
        self.constants = domain.constants
        self.bodies = objects

        self._translate_problem(pddlstream_problem, init=init, objects=objects)
        self.movables = self._get_movables(self.initial_state)

    def _get_movables(self, init):
        movables = []
        for fact in init:
            if fact.predicate.name == 'pose':
                body = fact.variables[0]
                if body not in movables:
                    movables.append(body)
        return movables

    def _translate_problem(self, pddlstream_problem, init=None, objects=[]):
        sys.path.extend([
            join(ROOT_DIR, '..', 'pddlstream'),
            join(ROOT_DIR, '..', 'pddlstream', 'examples', 'pybullet', 'utils'),
        ])
        # self.objects = list(BODY_TO_OBJECT.keys())
        # self.objects.extend(list(ROBOT_TO_OBJECT.keys()))
        self.objects = objects
        self.object_mapping = {str(k): k for k in self.objects}
        self.objects = {str(k): DEFAULT_TYPE for k in self.object_mapping}

        if init is None:
            init = pddlstream_problem.init

        self.initial_state = [self.to_literal(fact) for fact in init if '=' not in fact[0]]  ## cost
        self.goal = LiteralConjunction([self.to_literal(fact) for fact in pddlstream_problem.goal[1:]])

    def add_object(self, arg_name, arg_value):
        self.objects[arg_name] = DEFAULT_TYPE
        self.object_mapping[arg_name] = arg_value

    def to_literal(self, literal_list):
        params = self.objects
        lst = []
        for n in literal_list:
            if not isinstance(n, Object):
                lst.append(str(n).lower())
            else:
                lst.append(str(n.body).lower())
        pred, args = lst[0], lst[1:]
        typed_args = []
        # Validate types against the given params dict.
        if pred not in self.predicates:
            # print("Adding predicate {} to domain".format(pred))
            l = len(lst[1:])
            self.predicates[pred] = Predicate(pred, l, [DEFAULT_TYPE for i in range(l)])
            # print("Predicate {} is not defined".format(pred))
        # assert pred in self.predicates, "Predicate {} is not defined".format(pred)
        if self.predicates[pred].arity != len(args):
            print('self.predicates[pred].arity', self.predicates[pred].arity)
        assert self.predicates[pred].arity == len(args), pred
        for i, arg in enumerate(args):
            ## add object in to self.objects and self.object_mapping
            if arg not in params:
                self.add_object(arg, literal_list[i+1])
            assert arg in params, "Argument {} is not in the params".format(arg)
            if isinstance(params, dict):
                typed_arg = TypedEntity(arg, params[arg])
            else:
                typed_arg = params[params.index(arg)]
            typed_args.append(typed_arg)
        return self.predicates[pred](*typed_args)

    def from_literal(self, literal):
        lst = [literal.predicate.name]
        lst.extend([self.object_mapping[n.name] for n in literal.variables])
        lst = tuple(lst)
        if literal.is_negative or literal.is_anti:
            lst = ('not', lst)
        return lst


class PDDLStreamEnv(PDDLEnv):
    def __init__(self, domain_file, pddlstream_problem, init=None, objects=[],
                 domain_modifier=None, separate_base_planning=False, temp_dir='temp'):
        self._state = None
        self._state_original = None
        self._problem_dir = None
        self._render = None
        self._raise_error_on_invalid_action = False
        self.operators_as_actions = True
        self._problem_index_fixed = False
        self._problem_idx = 0
        self._problem_index_fixed = True

        #------------------- construct the PDDLStream Env --------------------------
        self._domain_file = domain_file
        self._pddlstream_problem = pddlstream_problem
        pddlgym_domain_file = remove_unparsable_lines(domain_file, temp_dir)
        self.domain = self.load_domain(pddlgym_domain_file, self.operators_as_actions, domain_modifier)
        self.domain_original = self.load_domain(pddlgym_domain_file, self.operators_as_actions, None)

        self.separate_base_planning = separate_base_planning
        self.last_bconf = None
        # print('* '*20 + 'print operators' + ' *'*20)
        # pprint(self.domain.operators)
        # print('* '*40)
        self.static_literals, self.externals = self.load_streams(pddlstream_problem)
        self.dynamic_literals = get_dynamic_literals(self.domain)
        self.tests = self.load_tests()
        # self.domain = self.remove_static_preconditions(self.domain, pddlstream_problem)
        self.problems = [PDDLProblemTranslator(self._pddlstream_problem, self.domain,
                                               init=init, objects=objects)]
        self._init_literals = []
        self._last_literals = init
        #---------------------------------------------------------------------------

        # Determine if the domain is STRIPS
        self._domain_is_strips = _check_domain_for_strips(self.domain)
        self._inference_mode = "csp" if self._domain_is_strips else "prolog"

        # Initialize action space with problem-independent components
        actions = list(self.domain.actions)
        self.action_predicates = [self.domain.predicates[a] for a in actions]
        self._dynamic_action_space = False
        self._action_space = LiteralSpace(self.action_predicates,
                                            type_to_parent_types=self.domain.type_to_parent_types)

        # Initialize observation space with problem-independent components
        self._observation_space = LiteralSetSpace(
            set(self.domain.predicates.values()) - set(self.action_predicates),
            type_hierarchy=self.domain.type_hierarchy,
            type_to_parent_types=self.domain.type_to_parent_types)

        # self._axioms = check_domain([domain_file], verbose=False)['axioms']

        ## for getting extended plan
        self.derived_literals = []
        self.derived_assignments = {}
        self.derived = {}
        self.on_map = {}
        self.on_map_inv = {}
        self.poses_at_action = []  ## each item is {obj : pose}
        self.poses_for_traj = {}  ## unsafetraj: poses
        self.test_replacements = {}  ## unsafetraj: bodies

        self.useful_variables = None

    @staticmethod
    def load_domain(domain_file, operators_as_actions=True, domain_modifier=None):
        domain = PDDLDomainParser(domain_file,
                                  expect_action_preds=(not operators_as_actions),
                                  operators_as_actions=operators_as_actions)
        if domain_modifier is not None:
            domain = domain_modifier(domain)
        domain = get_required_pred(domain)

        return domain

    @staticmethod
    def load_streams(pddlstream_problem):
        from pddlstream.algorithms.algorithm import parse_problem

        domain_line = pddlstream_problem.domain_pddl.split('\n')[0]
        if 'pr2' in domain_line:
            from pybullet_tools.stream_agent import get_stream_info
            stream_info = get_stream_info()
        else:  ## if 'feg' in domain_line:
            from pybullet_tools.flying_gripper_agent import get_stream_info
            stream_info = get_stream_info()

        evaluations, goal_exp, _, externals = parse_problem(
            pddlstream_problem, stream_info=stream_info, unit_costs=True, unit_efforts=True)
        # streams, functions, negative, optimizers = partition_externals(externals, verbose=True)
        static_literals = {}
        for stream in externals:
            for t in stream.certified:
                if t[0] not in static_literals:
                    static_literals[t[0]] = []
                static_literals[t[0]].append(stream)
        return static_literals, externals

    def load_tests(self):
        """ check which derived axioms depend on existential dev """
        tests = {}
        for name, pred in self.domain.predicates.items():
            if not pred.is_derived: continue
            if isinstance(pred.body, Exists):
                for literal in pred.body.body.literals:  ## Cfree...
                    if literal.predicate.name in self.static_literals:
                        for stream in self.static_literals[literal.predicate.name]:
                            if 'test-' in stream.name:
                                tests[name] = stream.name
        return tests

    ## discarded
    def remove_static_preconditions(self, domain, pddlstream_problem, verbose=True):
        """ delete the static facts generated by streams from preconditions """
        # from pddlgym.downward_translate.pddl.conditions import Conjunction

        if verbose: print('------------- REMOVING STATIC PRECONDITIONS -------------')
        removed_preds = []

        ## find certified predicates
        evaluations, goal_exp, _, externals = parse_problem(
            pddlstream_problem, stream_info=stream_info, unit_costs=True, unit_efforts=True)
        # streams, functions, negative, optimizers = partition_externals(externals, verbose=True)
        for stream in externals:
            removed_preds.extend([t[0] for t in stream.certified])

        ## find derived predicates that depend on certified predicates
        for name, pred in domain.predicates.items():
            if not pred.is_derived: continue
            for literal in pred.body.positive.body.literals:
                if literal.predicate.name in removed_preds and name not in removed_preds:
                    if verbose: print(f' ignore derived {name} because of {literal}')
                    removed_preds.append(name)

        ## ignore those predicates from operator preconditions and
        self.operators = copy.deepcopy(domain.operators)
        new_operators = {}
        for name, operator in domain.operators.items():
            new_preconds = []
            for cond in operator.preconds.literals:  ## is an Atom
                if cond.predicate.name in removed_preds or 'unsafe' in cond.predicate.name:
                    if verbose: print(f' in operator {name}, precondition {cond} is ignored')
                else:
                    new_preconds.append(cond)
            new_operator = copy.deepcopy(operator)
            new_operator.preconds = LiteralConjunction(new_preconds)
            new_operators[name] = new_operator
        domain.operators = new_operators
        if verbose: print('---------------------------------------------------------')
        return domain

    def _find_existed_variables(self, pred, state_literals, assignment, verbose=False):
        """ return dict of param: var assignments """
        if pred.name in self.new_assignments:
            new_assignments = self.new_assignments[pred.name]
        else:
            # if pred.name == 'closedjoint':
            #     verbose = True
            # verbose = True ## debug
            new_assignments = find_satisfying_assignments(
                state_literals, pred.body.body,
                type_to_parent_types=self.domain.type_to_parent_types,
                constants=self.domain.constants,
                mode="prolog",
                max_assignment_count=99999,
                pred=pred,
                INFER_TYPING=True,
                verbose=verbose,
            )
            self.new_assignments[pred.name] = new_assignments

        ## now it just choose the first one
        if len(new_assignments) == 0:
            print(f'\n\nhierarchical._find_existed_variables({pred}, {assignment}) found no new_assignments\n\n')
            new_assignments = {}
        else:
            new_assignments = [n for n in new_assignments if set(assignment.values()).issubset(set(n.values()))][0]
        return new_assignments

    def _handle_derived_literals(self, state):
        """ make sure the derived literals are saved in env """

        ## -------------- added by YANG ------------------------------------------
        self.derived_literals = []  ## grounded literals
        self.derived_assignments = {}  ## grounded literal : assignments
        self.new_assignments = {}  ## pre.name : assignments with existential vars
        ## ------------------------------------------------------------------------

        # first remove any old derived literals since they're outdated
        to_remove = set()
        for lit in state.literals:
            if lit.predicate.is_derived:
                to_remove.add(lit)
        state = state.with_literals(state.literals - to_remove)
        while True:  # loop, because derived predicates can be recursive
            new_derived_literals = set()
            for pred in self.domain.predicates.values():
                if not pred.is_derived:
                    continue

                DEBUG = False
                ## skip existential crisis to save time in preimage computation
                # if isinstance(pred.body, Exists) and len(pred.body.variables) > 1:
                if isinstance(pred.body, Exists) and \
                        'unsafe' in pred.name.lower() or 'obstacle' in pred.name.lower():
                    # print('\n\nexistential crisis', pred)
                    DEBUG = True
                    continue
                # if pred.name.lower() == 'robinroom':
                #     DEBUG = True

                literals = [l for l in state.literals if 'uniqueoptvalue' not in str(l)]
                assignments = find_satisfying_assignments(
                    literals, pred.body,
                    type_to_parent_types=self.domain.type_to_parent_types,
                    constants=self.domain.constants, mode="prolog",
                    max_assignment_count=99999,
                    pred=pred,  ## for printing debug .pl and for resolving existential crisis
                    verbose=DEBUG,
                    INFER_TYPING=True
                )

                for assignment in assignments:
                    objects = [assignment[param_type(param_name)]
                               for param_name, param_type in zip(pred.param_names, pred.var_types)]
                    derived_literal = pred(*objects)
                    if derived_literal not in state.literals:
                        new_derived_literals.add(derived_literal)

                        ## --------------------- added by YANG -----------------------
                        if derived_literal not in self.derived_literals:
                            self.derived_literals.append(derived_literal)

                            ## needs to find the assignments of things that exist
                            if isinstance(pred.body, Exists):
                                assignment.update(self._find_existed_variables(pred, literals, assignment))
                            self.derived_assignments[derived_literal] = assignment
                        ## ------------------------------------------------------------

            if new_derived_literals:
                state = state.with_literals(state.literals | new_derived_literals)
            else:  # terminate
                break
        return state

    def to_literal(self, literal_list):
        if isinstance(literal_list, Action):
            name, args = literal_list
            literal_list = [name]
            literal_list.extend(args)
        return self._problem.to_literal(literal_list)

    def add_object(self, arg_name, arg_value):
        self._problem.add_object(arg_name, arg_value)

    def add_objects(self, env):
        self._problem.object_mapping.update(env._problem.object_mapping)
        self._problem.objects.update(env._problem.objects)

    def from_literal(self, literal):
        tup = self._problem.from_literal(literal)
        if tup[0] in self.domain.operators:
            return Action(tup[0], tup[1:])
        return tup

    ## discarded
    def _get_applied_streams(self, name, mapping):
        applied_stream_candidates = []
        for stream in self.streams[name]:

            ## TODO : this mapping string is wrong
            # mapping = {operator.params[i]: action.variables[i] for i in range(len(operator.params))}
            mapping_str = {k.name: v for k, v in mapping.items()}

            inputs = []
            for ip in stream.inputs:
                if ip in mapping_str:
                    inputs.append(mapping_str[ip])
                else:
                    inputs.append(ip)
            outputs = []
            for ip in stream.outputs:
                if ip in mapping_str:
                    outputs.append(mapping_str[ip])
                else:
                    outputs.append(ip)
            stream.inputs = inputs
            stream.outputs = outputs
            applied_stream_candidates.append(stream)

        return applied_stream_candidates

    def _resolve_cond(self, cond, need_preconditions, applied_axioms, need_tests,
                      use_original_domain=False, verbose=False):
        """
        example outputs for [move_base]
            need_preconditions:
                [basemotion(q0=(2.0, 5.0, 0.2, 3.142):default,c80=t(4, 23):default,q800=(1.305, 4.525, 0.971, 3.043):default),
                Notidentical(q0=(2.0, 5.0, 0.2, 3.142):default,q800=(1.305, 4.525, 0.971, 3.043):default),
                atbconf(q0=(2.0, 5.0, 0.2, 3.142):default)]

        example outputs for [grasp_handle]
            need_preconditions:
                [aconf(left:default,aq976=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223):default),
                atposition((9, 1):default,pstn9=0.0:default),
                ataconf(left:default,aq976=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223):default),
                Notpulled((9, 1):default),
                kingrasphandle(left:default,(9, 1):default,pstn9=0.0:default,hg824=(-0.584, 0.111, 0.234, -3.142, -0.0, 0.0):default,q448=(1.571, 4.728, 0.469, -2.943):default,aq544=(0.005, 0.805, 0.667, -1.452, 3.579, -0.715, -0.803):default,c240=t(7, 89):default),
                Notunsafeatraj(c240=t(7, 89):default)]
            need_tests:
                [atposition((9, 1):default,pstn9=0.0:default)]
        """
        pred = cond.predicate  ## self.domain.predicates[name]
        if use_original_domain:
            domain = self.domain
            state = self._state
        else:
            domain = self.domain_original
            state = self._state_original

        ## don't add tests
        if pred.is_derived and pred.name not in self.tests:
            ## add both the grounded derived predicate,
            # but also the assigned variables in its preconditions
            applied_axioms.append((cond, self.derived_assignments))

            ## TODO: for now, don't ground things in (exist ) or (or (exist ))
            if not isinstance(pred.body, Exists):
                params = [TypedEntity(pred.param_names[i], pred.var_types[i]) for i in range(len(pred.param_names))]
                mapping = {params[i]: cond.variables[i] for i in range(len(params))}

                literals = pred.body.literals if isinstance(pred.body, LiteralDisjunction) else pred.body.body.literals
                for lit in literals:
                    ## TODO: investigate whether need to consider this
                    if isinstance(lit, Exists):
                        continue
                    n = lit.predicate.name
                    args = []
                    for v in lit.variables:
                        if v in mapping:
                            args.append(mapping[v])
                        else:
                            args.append(v)
                    grounded = domain.predicates[n](*args)
                    if grounded not in state.literals:
                        need_preconditions.append(grounded)
                    else:
                        if verbose: print(f'grounded precondition to derived {cond} already in facts:  {grounded}')

        elif cond.positive.predicate.name not in ['canmove', 'identical']:
            need_tests.append(cond)

        return need_preconditions, applied_axioms, need_tests

    def ground_literal(self, literal, mapping):
        if not hasattr(literal, 'predicate'):
            return None
        name = literal.predicate.name
        args = [mapping[n] for n in literal.variables]

        grounded_literal = self.domain.predicates[name](*args)
        grounded_literal.is_anti = literal.is_anti
        grounded_literal.is_negative = literal.is_negative
        grounded_literal.predicate.is_negative = literal.predicate.is_negative
        grounded_literal._update_variable_caches()

        return grounded_literal

    def get_effects(self, mapping, operator, state):
        add_effects = [self.ground_literal(n, mapping) for n in get_effects(operator)]
        del_effects = [literal for literal in add_effects if literal.is_anti]
        # for del_eff in del_effects:
        #     del_eff.is_anti = False
        add_effects = [n for n in add_effects if n not in state.literals and n not in del_effects]
        return add_effects, del_effects

    def identify_missing_facts(self, action, verbose=True, update_state=True, use_original_domain=False):
        """ find all facts that should be generated by streams or axioms to make action applicable """

        action_name = action.predicate.name
        if use_original_domain:
            domain = self.domain_original
            if '--' in action_name:
                action_name = action_name.split('--')[0]
            state = self._state_original
        else:
            domain = self.domain
            state = self._state
        operator = domain.operators[action_name]
        mapping = {operator.params[i]: action.variables[i] for i in range(len(operator.params))}

        ## ground the preconditions and effects
        need_preconditions = [self.ground_literal(n, mapping) for n in operator.preconds.literals]
        need_preconditions = [n for n in need_preconditions if n is not None and n not in state.literals]
        add_effects, del_effects = self.get_effects(mapping, operator, state)

        info = {
            'need_preconditions': need_preconditions,
            'add_effects': add_effects,
            'del_effects': del_effects
        }

        ## update the state with the missing facts
        if update_state:
            if not use_original_domain:
                self._state = state.with_literals(state.literals | frozenset(need_preconditions))
                self.last_obs = self._state
            else:
                print(f'identify_missing_facts.use_original_domain({action})')
                self.print_log(info)

            # canmoves = [(l.is_anti, l.is_negative, l.predicate.is_anti, l.predicate.is_negative)
            #             for l in self._state.literals if 'canmove' in l.predicate.name]
            # print(f'\n\n\n\t!! before step() canmove ({len(canmoves)})', canmoves)

        ## resolve all preconditions to facts
        resolved_preconditions = []
        applied_streams = []
        applied_axioms = []
        need_tests = []
        while len(set(need_preconditions)) != len(set(resolved_preconditions)):

            for cond in need_preconditions:
                if cond in resolved_preconditions: continue
                resolved_preconditions.append(cond)
                name = cond.predicate.name

                ## add the grounded streams/ axioms the extended plan
                if name in self.static_literals:
                    continue  ## don't care about how those streams are generated
                    # applied_streams = self._get_applied_streams(name, mapping)

                else:  ## missing predicates
                    need_preconditions, applied_axioms, need_tests = self._resolve_cond(
                        cond, need_preconditions, applied_axioms, need_tests,
                        use_original_domain=use_original_domain, verbose=verbose)

        info['applied_axioms'] = applied_axioms
        info['need_tests'] = need_tests  ## now includes tests
        info['need_preconditions'] = [n for n in info['need_preconditions'] if n not in need_tests]
        if len(applied_streams) > 0:  ## no longer used because it's unnecessary
            info['applied_streams'] = applied_streams

        return info

    def update_original_state(self, info, action, title='get_extended_plan'):
        """ use the add_effects and del_effects """
        info['del_effects'] = [f[1] for f in info['del_effects'] if f[0] == 'not']
        add_literals = [self.to_literal(f) for f in info['add_effects']]
        del_literals = [self.to_literal(f) for f in info['del_effects']]
        literals = [f for f in self._state_original.literals | frozenset(add_literals) if f not in del_literals]

        self._state_original_last = self._state_original
        self._state_original = self._state_original.with_literals(literals)

        from_bconf_last = [f for f in self._state_original_last.literals if f.predicate.name == 'atbconf'][0]
        from_bconf = [f for f in self._state_original.literals if f.predicate.name == 'atbconf'][0]

        print(f'\nupdate_original_state.{title} {action}')
        print(f'\tfrom_bconf_last:\t{from_bconf_last}')
        print(f'\tfrom_bconf:\t{from_bconf}\n')

    def step(self, action, identify_missing_action=False):

        ## first find the missing preconditions for applying the action
        info = self.identify_missing_facts(action)
        if identify_missing_action:
            info_original = self.identify_missing_facts(action, use_original_domain=True)
            info['original'] = info_original

        ## step() would add derived literals that the previous function doesn't detect,
        ## because things like `on` just happens even it's not required for the next step
        ## but required for a future step, like `season`
        obs, reward, done = super().step(action)[:3]  ## pddlgym version changed

        ## include the changes and needed facts in debug_info
        info['add'] = list(obs.literals - self.last_obs.literals)
        derived = [n for n in info['add'] if n not in info['add_effects']]
        info['triggered_axioms'] = [(n, self.derived_assignments) for n in self.derived_literals if n in derived]
        info['del'] = list(self.last_obs.literals - obs.literals)
        info['action'] = action

        ## atbconf action
        if identify_missing_action:
            info = self.get_missing_actions(info_original, info)

        ## this is very special purpose
        self.on_map.update(find_on_pose([n for n in derived if n.predicate.name == 'on'],
                                        [n for n in info['add_effects'] if n.predicate.name == 'atpose']))

        self.last_obs = obs
        self.action_logs.append(info)

        return obs, reward, done, info

    def get_missing_actions(self, info_original, info):
        """ hack to help PDDLStream use the minimum set of streams by feeding it subgoals (e.g. atbconf) """
        action_literals = []
        actions = []
        missing_info = []

        ## when the required fact is not in abstract state and current state
        missing_preconditions = [pre for pre in info_original['need_tests'] if pre not in info['need_tests']]
        missing_preconditions = [m for m in missing_preconditions if m not in self._state_original.literals]
        print()
        print_list(missing_preconditions, 'get_missing_actions.missing_preconditions')
        if len(missing_preconditions) > 0:
            # init = self._state_original.literals  ## TODO: has problems because action sequence is wrong
            for pre in missing_preconditions:
                name = pre.predicate.name
                if name == 'atbconf':
                    # if self.last_bconf is None:
                    #     from_bconf = [f for f in init if f.predicate.name == 'atbconf'][0].variables[0]
                    # else:
                    #     from_bconf = self.last_bconf

                    from_bconf = [f for f in self._state_original.literals if f.predicate.name == 'atbconf'][0].variables[0]
                    if from_bconf == pre.variables[0]:
                        continue
                    self.last_bconf = pre.variables[0]
                    args = [from_bconf, pre.variables[0], TypedEntity('?t', DEFAULT_TYPE)]
                    operator = [a for a in self.action_predicates if a.name == 'move_base'][0]
                    action_literal = operator(*args)
                    action_literals.append(action_literal)

                    for arg in args:
                        if arg not in self._problem.objects:
                            self.add_object(str(arg), arg)
                    self.add_object('?t', '?t')

                    action_info = self.identify_missing_facts(action_literal, update_state=False)
                    action_info = {k: [self.from_literal(l) for l in action_info[k]] for k in ['add_effects', 'del_effects']}
                    missing_info.append(action_info)

                    actions.append(self.from_literal(action_literal))

        info.update({
            'missing_action_literals': action_literals,
            'missing_actions': actions,
            'missing_info': missing_info
        })

        return info

    def reset(self):
        start_time = time.time()
        obs, debug_info = super().reset()
        self._state_original = copy.deepcopy(self._state)
        print('prolog time', round(time.time() - start_time, 3))
        self.last_obs = obs
        self.action_logs = []
        return obs, debug_info

    def ground_operator_preconds_effects(self, op, preimage=[]):
        """
        :param op:  a grounded operator
        :return:    lists of grounded literals
        """
        add_list = []
        del_list = []
        action = self.domain.operators[op.predicate.name]
        mapping = {action.params[i]: op.variables[i] for i in range(len(op.variables))}
        for cond in get_preconds(action):
            if isinstance(cond, ForAll):  ## HACK for OR[Notoftype(), In()]
                if isinstance(cond, LiteralDisjunction) and len(cond.body.literals) == 2:
                    notoftype, something = cond.body.literals
                    oftype = mapping[notoftype.variables[1]].name
                    os = [f[1] for f in self.init_preimage if f[0].lower() == 'oftype' and f[2] == oftype.name]
                    for o in os:
                        new_mapping = copy.deepcopy(mapping)
                        new_mapping[TypedEntity('?o', oftype.var_type)] = TypedEntity(o, oftype.var_type)
                        add_list.append(something.predicate(*[new_mapping[n] for n in something.variables]))
                else:
                    print('hierarchical.regress | not isinstance(cond, LiteralDisjunction) '
                          'and len(cond.body.literals) == 2:', cond, cond.body.literals)
            else:
                args = [mapping[n] for n in cond.variables]
                gounded_cond = cond.predicate(*args)
                if len(args) > 0:
                    add_list.append(gounded_cond)
                elif gounded_cond in preimage:
                    del_list.append(gounded_cond)

        for eff in get_effects(action):
            del_list.append(eff.predicate(*[mapping[n] for n in eff.variables]))

        return add_list, del_list

    def regress(self, preimage, op):
        tests = []
        add_list = []
        del_list = []

        if isinstance(op, tuple) and len(op) == 2:  ## (op, assignments) in derived axioms
            ## assignments are necessary especially for ground variables in Exists
            op, assignments = op
            if op in assignments: ## with assignments, e.g. ON
                assignment = assignments[op]
                for literal in get_preconds(op.predicate):
                    for n in literal.variables:
                        if n not in assignment:
                            print(f'\n\nhierarchical.regress: {n} not in {assignment}\n\n')
                    add_list.append(literal.predicate(*[assignment[n] for n in literal.variables]))
            else: ## e.g. UNSAFEAPPROACH
                tests.append(op)
            del_list.append(op)

        elif op.predicate.name in self.domain.operators:  ## operator
            results = self.ground_operator_preconds_effects(op, preimage)
            add_list += results[0]
            del_list += results[1]

        # else:  ## axioms
        #     pred = self.domain.predicates[op.predicate.name]
        #     params = [TypedEntity(pred.param_names[i], pred.var_types[i]) for i in range(len(pred.param_names))]
        #     mapping = {params[i]: op.variables[i] for i in range(len(op.variables))}

        for d in del_list:
            if d in preimage:
                preimage.remove(d)
            if d in self.on_map_inv and self.on_map_inv[d] in preimage:
                preimage.remove(self.on_map_inv[d])
        for a in add_list:
            if a.predicate.name in self.tests:  ## a derived literal that depends on a stream test
                tests.append(a)
            if a not in preimage and a.predicate.name not in self.static_literals: ## not a static fact
                preimage.append(a)

        preimage += [t for t in tests if t not in preimage]
        return preimage, tests

    def update_poses(self, op, facts):
        ## the poses in preimages, may exist in the form of 'atpose' or 'on'
        if isinstance(op, tuple): op = op[0]
        if op.predicate.name == 'on' and op in self.on_map:
            facts.append(self.on_map[op])
        poses = find_poses(facts)

        ## update the steps afterwards, until it's changed
        for body, pose in poses.items():
            for i in range(len(self.poses_at_action)):
                j = len(self.poses_at_action) - 1 - i
                previous = self.poses_at_action[j]
                if body not in previous:
                    self.poses_at_action[j][body] = pose
                else:
                    break
        self.poses_at_action.append(poses)
        return poses

    # def _make_cfree_traj(self, t, o, p):
    #     literal =

    def compute_preimgae(self, extended_plan, verbose=True):

        conditional_tests = []
        all_preimages = []
        all_tests = []
        all_poses = []
        test_map = {}

        def print_preimages(op, preimage, tests=None, poses=None, replaced=None):
            if isinstance(op, tuple): op = op[0]
            op_str = str(op).replace(':default', '')
            to_print = f'\nStep {step_idx} | op = {op_str}'

            literals = {n: [] for n in ['dynamic', 'static', 'dev']}
            for n in preimage:
                if 'unsafe' in n.predicate.name:
                    literals['dev'].append(n)
                elif n.predicate.name in self.dynamic_literals:
                    literals['dynamic'].append(n)
                else:
                    literals['static'].append(n)
            if poses is not None:
                literals['poses'] = poses
            if tests is not None:
                literals['tests_new'] = tests
            if replaced is not None:
                literals['tests_replaced'] = replaced

            for k, v in literals.items():
                v_str = str(v).replace(':default', '')
                to_print += f'\n       {k} ({len(v)}) = {v_str}'
            print(to_print)

        self.pre_images = {}  ## of actions
        self.actions_after = {}
        preimage = copy.deepcopy(self._problem.goal.literals)
        print(f'Step {len(extended_plan)} | goal = {preimage}')
        last_action = None
        for i in range(len(extended_plan)):
            step_idx = len(extended_plan) - 1 - i
            op = extended_plan[step_idx]
            preimage, tests = self.regress(preimage, op)

            test_map.update({t: i for t in tests})
            # preimage = preimage + tests

            poses = self.update_poses(op, copy.deepcopy(preimage))

            ## preimages of actions are to be saved as refinement goals
            if isinstance(op, Literal):  ## op.predicate.name in self.action_predicates:
                self.pre_images[op] = copy.deepcopy(preimage) ## + copy.deepcopy(tests)
                self.actions_after[op] = last_action
                last_action = op

            if verbose:
                print_preimages(op, preimage)

            ## save and print in the end because poses will be propagated back
            all_preimages.append(copy.deepcopy(preimage))
            all_tests.append(tests)
            all_poses.append(poses)

        print('-' * 50)

        for i in range(len(extended_plan)):
            step_idx = len(extended_plan) - 1 - i
            op = extended_plan[step_idx]
            preimage = all_preimages[i]
            tests = all_tests[i]
            poses = all_poses[i]
            replaced = self.replace_tests(op, preimage, tests, poses)

            if verbose:
                print_preimages(op, preimage, tests, poses, replaced)

        return preimage, conditional_tests

    def replace_tests(self, op, preimage, tests, poses):
        if isinstance(op, tuple): op = op[0]

        replacements = {
            'UnsafePose': 'PoseObstacle',
            'UnsafeApproach':  'ApproachObstacle',
            'UnsafeATraj': 'ATrajObstacle' ## 'cfreetrajpose' 'SafeATraj'.lower()  ##
        }
        for old_test, new_test in replacements.items():
            old_test = old_test.lower()
            new_test = new_test.lower()

            unsafetraj = [n for n in tests if n.predicate.name == old_test]
            if len(unsafetraj) > 0:
                unsafetraj = unsafetraj[0]
                self.poses_for_traj[unsafetraj] = poses

            all_tests = [n for n in preimage if n.predicate.name == old_test]
            for test in all_tests:
                if test in self.poses_for_traj:
                    that = self.poses_for_traj[test]
                    diff = [k for k in poses if poses[k] == that[k]]
                    pred = self.domain.predicates[new_test]
                    if len(diff) < len(self._problem.movables):

                        ## --- method 1: (SafeATraj ?t ?o ?p)
                        # diff = [pred(*[t, k, that[k]]) for k in diff] ## (SafeATraj ?t ?o ?p)

                        ## --- method 2: (SafeATraj ?t ?o) with forall-imply
                        # diff = [pred(*[t, k]) for k in diff]
                        # self.test_replacements[test] = diff

                        ## --- method 3: (not (ATrajObstacle ?t ?o))
                        pp = []
                        for k in diff:
                            old_vars = copy.deepcopy(test.variables)
                            old_vars.append(k)
                            # if k in old_vars: continue
                            literal = pred(*old_vars)
                            literal.is_negative = True
                            literal.predicate.is_negative = True
                            literal._update_variable_caches()
                            pp.append(literal)
                        self.test_replacements[test] = pp

        preimage = copy.deepcopy(preimage)
        replaced = []
        for k, v in self.test_replacements.items():
            if k in preimage:
                preimage.remove(k)
                replaced.extend(v)
        preimage.extend(replaced)

        ## TODO: axioms shouldn't be in self.pre_images
        if op in self.pre_images:
            self.pre_images[op] = preimage

        return replaced

    def get_preimage(self, num):
        return list(self.pre_images.values())[len(self.pre_images)-1-num]

    def get_preimage_after(self, op):

        ## the op can be an axiom, thus skip
        if op not in self.actions_after: return None

        plan = list(self.pre_images.keys())
        images = list(self.pre_images.values())

        if plan.index(op) == 0:  ## the last abstract action of the plan
            solution1 = self._problem.goal.literals
        else:
            action = self.actions_after[op]
            step = self.from_literal(action)
            if step not in self.after_images.keys():
                new_list = [str(e) for e in self.after_images.keys()]
                idx_print = new_list.index(str(step)) + 1
                # print('\n'.join([str(k) for k in self.after_images.keys()]))
            else:
                idx_print = list(self.after_images.keys()).index(step) + 1
            print(f'\nusing preimage of Step {idx_print} {step}')
            idx = plan.index(action)
            solution1 = [n for n in images[idx] if (n.predicate.name in self.dynamic_literals)]

        return solution1

    def get_all_preimages_after_op(self):
        results = {op: self.get_preimage_after(op) for op in self.pre_images.keys()}
        return {k: v for k, v in results.items() if v is not None}

    def literals_to_facts(self, literals):
        return [self.from_literal(l) for l in literals
                if not l.predicate.is_derived and l.predicate.name != '=']

    def get_extended_plan(self, abstract_plan, init, verbose=True):
        ## not all applied axioms are useful for achieving the goal
        extended_plan = [] ## copy.deepcopy(self.derived_literals)
        updated_plan = []
        index = 0

        print('\n ========= Computing Extended Plan ======== \n')
        self._state_original_last = self._state_original

        self.after_images = {}
        self.derived = {0: self.derived_assignments}  ## the assignment after init facts
        for step in abstract_plan:
            action = self.to_literal(step)
            obs, reward, done, debug_info = self.step(action, identify_missing_action=True)
            after_image = self.literals_to_facts(obs.literals)
            debug_info_missing = {
                k: debug_info[k] for k in ['missing_actions', 'missing_info', 'missing_action_literals']
            }
            debug_info = {k: v for k, v in debug_info.items() if k not in debug_info_missing}

            ## some missing atbconf actions
            extended_plan.extend(debug_info_missing['missing_action_literals'])
            for i in range(len(debug_info_missing['missing_actions'])):
                added_step = debug_info_missing['missing_actions'][i]
                added_step_literal = debug_info_missing['missing_action_literals'][i]
                missing_info = debug_info_missing['missing_info'][i]
                index += 1

                facts = [self.from_literal(f) for f in self._state_original.literals]
                self.after_images[added_step] = get_updated_facts(facts, missing_info)
                self.derived[index] = {}

                self.update_original_state(missing_info, added_step_literal, title=f'step {index} missing_action')
                # missing_info['del_effects'] = [f[1] for f in missing_info['del_effects'] if f[0] == 'not']
                # add_literals = [self.to_literal(f) for f in missing_info['add_effects']]
                # del_literals = [self.to_literal(f) for f in missing_info['del_effects']]
                # literals = [f for f in self._state_original.literals | frozenset(add_literals) if f not in del_literals]
                # self.update_original_state(literals, added_step_literal, title='get_extended_plan')

                if verbose:
                    print(f'\nStep {index}', done)
                    self.print_log(debug_info_missing)
            updated_plan.extend(debug_info_missing['missing_actions'])

            extended_plan.extend(debug_info['applied_axioms'])  ## axioms needed for applying the operator
            extended_plan.append(action)  ## operators
            extended_plan.extend(debug_info['triggered_axioms'])  ## axioms triggered after applying the operator
            index += 1
            if verbose:
                print(f'\nStep {index}', done)
                self.print_log(debug_info)
                print('-'*50)
            self.after_images[step] = after_image
            self.derived[index] = self.derived_assignments  ## the assignment after each step
            effects_info = {k: [self.from_literal(l) for l in debug_info['original'][k]] for k in ['add_effects', 'del_effects']}
            self.update_original_state(effects_info, action, title=f'step {index} abstract_action')
            updated_plan.append(step)

        print('\n ========= ----------------------- ======== \n')
        summarize_facts([self.from_literal(n) for n in self._state.literals], name='Facts after executing the plan')
        if self.useful_variables is None:
            variables = [n for n in self._state.literals if n.predicate.name in ['pose', 'grasp']]
            self.useful_variables = {n.variables[-1].name: n for n in variables}

        ## the axioms used in the last step may be useful for achieving the goal
        for literal in self.derived_literals:
            if literal in self._problem.goal.literals:
                if (literal, self.derived_assignments) not in extended_plan:
                    extended_plan.append((literal, self.derived_assignments))
        # print('\n'.join([str(n) for n in extended_plan]))

        print('\n-------------- on map ---------------')
        pprint(self.on_map)
        self.on_map_inv = {v: k for k, v in self.on_map.items()}
        print('--------------------------------------\n')
        return extended_plan, done, updated_plan

    def set_init_preimage(self, preimage):
        self.init_preimage = []
        self.add_init_preimage(preimage)
        self.init_preimage = self.literals_to_facts(self._init_literals)

    def add_init_preimage(self, facts):
        allowed = ['atraj', 'bconf', 'not', 'pose', 'grasp', 'supported', 'kin',
                   'traj', 'seconf', 'handdlegrasp', 'kinhandle']
        facts = [f for f in facts if f not in self.init_preimage and \
                 # (f[0].lower() in allowed
                 #  # or 'cfree' in f[0].lower()  ## need to compute at replanning time
                 #  ) and \
                 not (f[0] == 'not' and f[1][0] == '=')] ##
        self.init_preimage.extend(facts)

    @staticmethod
    def print_log(debug_info):
        new_info = {}
        for k, v in debug_info.items():
            if 'axioms' in k:
                new_info[k] = [t[0] for t in v]
            else:
                new_info[k] = v
        pprint(new_info, width=150)


class PDDLStreamForwardEnv(PDDLStreamEnv):
    """ only adding and deleting effects,
        consider no preconditions and no derived axioms """
    def _handle_derived_literals(self, state):
        return state

    def step(self, action):
        action = self.to_literal(action)
        obs, reward, done, debug_info = super().step(action)
        facts = [self.from_literal(l) for l in obs.literals]

        if action.predicate.name in ['place', 'arrange']:
            facts = fix_facts_due_to_loaded_agent_state(facts)
        added, deled = summarize_state_changes(facts, self._last_literals, title='PDDLStreamForwardEnv')
        self._last_literals = facts
        return added, deled


def check_preimage(pddlstream_problem, plan, preimage, domain_pddl, init=[], **kwargs):
    """ called right after making a plan """
    # pddl_file = join(DOMAIN_DIR, domain_pddl)

    env = PDDLStreamEnv(domain_pddl, pddlstream_problem, **kwargs)
    env.reset()
    i0 = env._state.literals

    env._state = env._state.with_literals(i0 | frozenset([env.to_literal(f) for f in preimage]))
    env._state_original = copy.deepcopy(env._state)
    env._init_literals = copy.deepcopy(env._state.literals)

    extended_plan, _, plan = env.get_extended_plan(plan, init)

    env.set_init_preimage(preimage)
    preimage, tests = env.compute_preimgae(extended_plan)
    # goal = env.get_preimage(1)
    # print('\nGoal before the second action', goal)

    return env, plan

