import random

from os.path import join, abspath, basename
import copy
import numpy as np
import time
import json
import sys

from pybullet_tools.utils import WorldSaver
from pybullet_tools.bullet_utils import open_joint
from pybullet_tools.general_streams import Position

from world_builder.init_utils import get_potential_placements

from pigi_tools.data_utils import get_plan_skeleton, get_indices, get_action_elems, \
    get_successful_plan, modify_plan_with_body_map, load_planning_config, get_old_actions, \
    get_joint_name_chars, modify_skeleton_with_body_map

sys.path.append('/home/yang/Documents/fastamp')

MODELS_PATH = '/home/yang/Documents/fastamp/test_models'
SKIP = 'skip'


class FeasibilityChecker(object):

    def __init__(self, run_dir, inv_body_map, **kwargs):
        self.run_dir = run_dir
        self.body_map = {v: k for k, v in inv_body_map.items()}
        self.inv_body_map = inv_body_map  ## rerun may have different pybullet body indices
        self._skwargs = {'include_joint': True, 'include_movable': True}
        self._skwargs_old = copy.deepcopy(self._skwargs)
        if isinstance(run_dir, str):
            self._skwargs['indices'] = self._indices = get_indices(run_dir)
            self._skwargs_old = copy.deepcopy(self._skwargs)
            self._skwargs_old['indices'] = self._indices_old = get_indices(run_dir, larger=False, inv_body_map=inv_body_map)
        self._log = {k: [] for k in ['checks', 'run_time']}

    def __call__(self, *args, **kwargs):
        return self.check(*args, **kwargs)

    def _get_indices(self):
        if isinstance(self.run_dir, str):
            return get_indices(self.run_dir)
        return self.run_dir.world.get_indices()

    def _get_gt_skeletons(self):
        plans = get_successful_plan(self.run_dir, skip_multiple_plans=False)
        plans = [modify_plan_with_body_map(plan, self.body_map) for plan in plans]
        return [get_plan_skeleton(plan, **self._skwargs) for plan in plans]

    def _check(self, input):
        raise NotImplementedError('should implement this for FeasibilityChecker')

    def check(self, inputs):
        from pybullet_tools.logging_utils import myprint as print
        if not isinstance(inputs[0], list):
            inputs = [inputs]

        ## change new plan to have old indices
        # inputs = [modify_plan_with_body_map(plan, self.inv_body_map) for plan in inputs]

        predictions = []
        printout = []
        if 'PVT' not in self.__class__.__name__:
            skip = False
            for input in inputs:
                start = time.time()
                prediction = self._check(input) if not skip else True
                predictions.append(prediction)
                plan = get_plan_from_input(input)
                skeleton = get_plan_skeleton(plan, **self._skwargs) ## _old
                # if 'kc' in skeleton:
                #     print('\nkckckckckckckckckckckckckckckckckcv!!!\n')
                if prediction != SKIP:
                    self._log['checks'].append((skeleton, plan, prediction))
                    self._log['run_time'].append(round(time.time() - start, 4))
                if hasattr(self, 'skeleton'):
                    printout.append((skeleton, 'pass' if prediction else f'x ({self.skeleton})'))
                # if isinstance(self, Heuristic) and prediction:
                #     skip = True
            if isinstance(self, Heuristic):
                # remove_body(self._feg)
                # self._feg = None
                for f in self._fluents_original:
                    if f[0].lower() in ['atposition', 'atpose']:
                        f[2].assign()
                if len(self._log['checks']) > 80:
                    print('fc checks more than 80')
                    sys.exit()
        else:
            self._log['sequence'] = []
            start = time.time()
            appeared_skeletons = []
            predictions = self._check(inputs)
            if len(predictions) != len(inputs):
                print('len predictions', len(predictions), '\nlen inputs', len(inputs))
                # import ipdb; ipdb.set_trace()

            new_predictions = predictions

            ## for the logging
            run_time = round(time.time() - start, 4)
            ave_time = (run_time, len(inputs), round(run_time / len(inputs), 4))
            p = {i: predictions[i] for i in range(len(inputs))}
            sorted_predictions = {k: v for k, v in sorted(p.items(), key=lambda item: item[1], reverse=True)}
            for i, prediction in sorted_predictions.items():
                plan = get_plan_from_input(inputs[i])
                sequence = self.sequences[i]
                skeleton = get_plan_skeleton(plan, **self._skwargs)
                if skeleton in appeared_skeletons:
                    prediction = 0
                    new_predictions[i] = prediction
                else:
                    appeared_skeletons.append(skeleton)
                self._log['checks'].append((skeleton, plan, prediction))
                self._log['sequence'].append(sequence)
                self._log['run_time'].append(ave_time)
                pp = [round(prediction, 3), skeleton]
                if prediction == 0:
                    pp.append('x')
                printout.append(tuple(pp))
            predictions = new_predictions
        [print(p) for p in printout]

        # if len(predictions) == 1:
        #     return predictions[0]
        return predictions

    def dump_log(self, json_path, plans_only=False):
        from world_builder.world import State
        from lisdf_tools.lisdf_planning import Problem

        if isinstance(self, Heuristic):
            self.possible_obstacles = {str(k): tuple(v) for k, v in self.possible_obstacles.items()}
            self.pre_actions_and = {k: tuple(v) for k, v in self.pre_actions_and.items()}
            self.pre_actions_or = {k: tuple(v) for k, v in self.pre_actions_or.items()}
            self.not_pre_actions = {k: tuple(v) for k, v in self.not_pre_actions.items()}

        if isinstance(self, LargerWorld):
            self.include_actions = list(self.include_actions)

        with open(json_path, 'w') as f:
            config = {k: v for k, v in self.__dict__.items() if \
                      not k.startswith('_') and not isinstance(v, State) and not isinstance(v, Problem)}
            config['inv_body_map'] = {str(k): v for k, v in config['inv_body_map'].items()}
            if 'args' in config:
                config['args'] = config['args'].__dict__
            self._log['config'] = config
            log = copy.deepcopy(self._log)
            if plans_only:
                log.pop('run_time')
            json.dump(log, f, indent=3)


class PassAll(FeasibilityChecker):

    def __init__(self, run_dir, body_map, shuffle=False):
        super().__init__(run_dir, body_map)
        self.skeletons_gt = self._get_gt_skeletons()
        self.verbose = False

    def _check(self, plan):
        plan = [[i.name] + [str(a) for a in i.args] for i in plan]
        skeleton = get_plan_skeleton(plan, **self._skwargs)
        line = f'\t{skeleton}'
        if skeleton in self.skeletons_gt:
            line += ' *'
        if self.verbose:
            print(line)
        return True


class ShuffleAll(FeasibilityChecker):
    def __init__(self, run_dir, body_map):
        super().__init__(run_dir, body_map)
        random.seed(int(time.time()))

    def _check(self, input):
        return random.uniform(0.5, 1)


class Oracle(FeasibilityChecker):

    def __init__(self, run_dir, body_map):
        super().__init__(run_dir, body_map)
        self.skeletons = self._get_gt_skeletons()
        ## include plan found by None during testing
        # rerun_dir = join(run_dir, 'rerun_2')
        # if isdir(rerun_dir):
        #     for fc in ['None', 'pvt-task']:
        #         plan_file = join(rerun_dir, f'diverse_plan_rerun_fc={fc}.json')
        #         if isfile(plan_file):
        #             new_plan = json.load(open(plan_file, 'r'))['plan']
        #             if new_plan is not None:
        #                 new_skeleton = get_plan_skeleton(new_plan, **self._skwargs)
        #                 if new_skeleton not in self.skeletons:
        #                     self.skeletons.append(new_skeleton)
        #                     print(f'add new skeleton {new_skeleton} by {fc} rerun to oracle')
        print(f'\nOracle feasibility checker - {self.skeletons}\n')

    def _check(self, plan):
        plan = [[i.name] + [str(a) for a in i.args] for i in plan]
        skeleton = get_plan_skeleton(plan, **self._skwargs)
        print(f'\t[Oracle(FeasibilityChecker)]\t{skeleton}')
        return skeleton in self.skeletons


class LargerWorld(FeasibilityChecker):
    """ remove actions associated with added objects and lookup previous labels."""

    def __init__(self, run_dir, body_map):
        super().__init__(run_dir, body_map)
        from fastamp_utils import get_plans_labels
        data = get_plans_labels(run_dir, using_larger_labels=False)
        new_skeletons = [modify_skeleton_with_body_map(s, body_map) for s in data['skeletons']]
        self.skeletons = {new_skeletons[i]: data['labels'][i] for i in range(len(data['plans']))}
        for i in range(len(data['plans'])):
            print('\t', data['labels'][i], '\t', data['skeletons'][i])
        print(f'\nLargerWorld FC - {len(self.skeletons)} plans, '
              f'{data["num_successful_plans"]} success\n')

        config = load_planning_config(run_dir)
        if 'body_to_name_new' not in config:
            print('\n\nbody_to_name_new not in config', run_dir)
        self.body_to_name_new = config['body_to_name_new']
        self._body_to_name_old = config['body_to_name']

        self.old_actions = get_old_actions(self.skeletons)
        include_actions = []
        for k, v in self.body_to_name_new.items():
            if v in self._body_to_name_old.values():
                continue
            k = eval(k)
            if isinstance(k, tuple) and len(k) == 2:
                include_actions.append(f'l{get_joint_name_chars(v)}')
            elif isinstance(k, int) and 'minifridge' not in v and 'cabinettop' not in v:
                include_actions.append(f'k({k})')
        self.include_actions = set(include_actions)

        self._skips = []

    def _check(self, plan):
        plan = [[i.name] + [str(a) for a in i.args] for i in plan]
        skeleton = get_plan_skeleton(plan, **self._skwargs)
        actions = [a + ')' for a in skeleton.split(')')[:-1]]

        result = SKIP
        after_line = ''

        ## plan must contain new objects
        if len(set(actions).intersection(self.include_actions)) > 0:

            ## actions apart from the new objects must be in the dataset
            actions = [a for a in actions if a in self.old_actions]
            skeleton_new = ''.join(actions)
            if skeleton_new in self.skeletons:
                result = self.skeletons[skeleton_new]
                if skeleton_new != skeleton:
                    result = False
            after_line = '->\t'+skeleton_new

        print('\t', result, '\t', skeleton, after_line)
        return result

###################################################################################################


rename = {'pick': 'k', 'place': 'c', 'pull_door_handle': 'l'}
shorter = lambda a: '-'.join([rename[a.name], str(a.args[1])]) \
    if isinstance(a[1], list) else '-'.join([rename[a[0]], str(a[1])])
shorter_plan = lambda actions: f"({', '.join([shorter(a) for a in actions if a.name in rename])})"


class Heuristic(FeasibilityChecker):

    def __init__(self, initializer, body_map):
        state, goals, init = initializer
        super().__init__(state, body_map)
        self._skwargs_old['indices'] = state.world.get_indices()
        self._state = state
        self._robot = state.robot
        self._fluents = state.get_fluents(only_fluents=True)
        self._fluents_original = copy.deepcopy(self._fluents)
        # self._feg = None
        self._verbose = True
        self._reachability_kwargs = dict(max_attempts=10, debug=False, visualize=False, verbose=self._verbose)

        self.potential_placements = get_potential_placements(goals, init)
        self.possible_obstacles = self._robot.possible_obstacles
        self.pre_actions_and = {}
        self.pre_actions_or = {}
        self.not_pre_actions = {}
        ## { body | (body, None, link): [ body | (body, joint) ] }

    def _check(self, plan):
        """ each checker returns either False or the next relaxed state """
        from pybullet_tools.logging_utils import myprint as print

        def print_plan(plan):
            print('plan\t', shorter_plan(plan))

        def print_result(plan, result):
            text = 'passed' if result else 'failed'
            print(f'----------------------------- {text} -----------------------------')
            print_plan(plan)
            print('------------------------------------------------------------------\n')

        verbose = self._verbose or True
        checkers = {
            'pick': self._check_pick,
            'place': self._check_place,
            'pull_door_handle': self._check_pull,
        }
        if verbose:
            print('\n-------------- heuristic plan feasibility checking -------------')
            print_plan(plan)
        with WorldSaver(self._state.objects):
            plan_so_far = []
            for i in range(len(plan)):
                action = plan[i]
                if action.name in ['move_base']:
                    continue
                if action.name in checkers:
                    key = shorter(action)
                    title = f'\t\t[{i}] {key} '

                    ## we know the pre_actions will enable the action
                    if key in self.pre_actions_and or key in self.pre_actions_or:
                        if self._check_pre_action(action, plan_so_far):
                            if verbose:
                                print(title+'passed according to pre-action')
                            plan_so_far.append(key)
                            continue
                        else:
                            if verbose:
                                print(title+'failed according to pre-action')
                                print('    pre-actions-and', self.pre_actions_and)
                                print('    pre-actions-or', self.pre_actions_or)
                                print_result(plan, False)
                            return False

                    ## we know the pre_actions will not enable the action
                    if key in self.not_pre_actions:
                        ## we know it wont work
                        if self._check_pre_action(action, plan_so_far):
                            if verbose:
                                print(title+'failed according to not pre-action')
                                print('    not_pre_actions', self.not_pre_actions)
                                print_result(plan, False)
                            return False

                    ## test the action
                    if not checkers[action.name](action.args, plan_so_far):
                        if verbose:
                            print(title+'failed')
                            print_result(plan, False)
                        return False
                    # self._cache[key] = True
                    if verbose:
                        print(title+'passed')
                    plan_so_far.append(key)
        if verbose:
            print_result(plan, True)
        return True

    def _check_pre_action(self, action, plan_so_far: []):
        stepers = {
            'pick': self._step_pick,
            'pull_door_handle': self._step_pull,
        }
        if self._verbose:
            print('\tplan_so_far\t', tuple(plan_so_far))

        # if self.pre_actions[shorter(action)] is None:
        #     return False

        passed = True
        key = shorter(action)
        if shorter(action) in self.pre_actions_and:
            ## return True if all have appeared in previous actions
            for pre_action in self.pre_actions_and[key]:
                if pre_action not in plan_so_far:
                    passed = False
                    break
            if passed and action.name in stepers:
                stepers[action.name](action.args)

        elif shorter(action) in self.pre_actions_or:
            ## return True if any have appeared in previous actions
            passed = False
            for pre_action in self.pre_actions_or[key]:
                if pre_action in plan_so_far:
                    passed = True
                    break
            if passed and action.name in stepers:
                stepers[action.name](action.args)

        elif shorter(action) in self.not_pre_actions:
            ## return True if we know will fail
            found_something_new = False
            wont_work = self.not_pre_actions[key]
            for previous_action in plan_so_far:
                if previous_action not in wont_work:
                    found_something_new = True
                    break
            passed = not found_something_new

        return passed

    def _update_pre_action(self, action_name, body, result, plan_so_far):
        from pybullet_tools.logging_utils import myprint as print
        key = shorter((action_name, body))
        if result:
            if key not in self.pre_actions_and:
                if key not in self.not_pre_actions:
                    self.pre_actions_and[key] = set()
                else:
                    something_right = set()
                    for action in plan_so_far:
                        if action not in self.not_pre_actions[key]:
                            something_right.add(action)
                    if body in self.possible_obstacles:
                        movables = [k for k in self.potential_placements if \
                                    self.potential_placements[k] == self.potential_placements[body]]
                    else:
                        movables = [body]
                    print(f'found pre_actions_or({key}) from plan_so_far {something_right} for movables {movables}')
                    for movable in movables:
                        new_key = shorter((action_name, movable))
                        if new_key not in self.pre_actions_or:
                            self.pre_actions_or[new_key] = set()
                        self.pre_actions_or[new_key] = self.pre_actions_or[new_key].union(something_right)
                        print(f'update pre_actions_or{new_key}', self.pre_actions_or[new_key])
            else:
                print('Warning: pre-action already exists', key, result)
        else:
            if key not in self.pre_actions_and:
                if body in self.possible_obstacles and len(self.possible_obstacles[body]) > 0:
                    self.pre_actions_and[key] = set([shorter(('pick', o)) for o in self.possible_obstacles[body]])
                else:
                    if key not in self.not_pre_actions:
                        self.not_pre_actions[key] = set()
                    self.not_pre_actions[key] = self.not_pre_actions[key].union(set(plan_so_far))
                    print(f'add to not-pre-action({key}) from plan_so_far', set(plan_so_far))
            else:
                print('Warning: pre-action already exists', key, result)

    def _check_pick(self, args, plan_so_far):
        """ whether body is reachable by the robot """
        from pybullet_tools.logging_utils import myprint as print
        body = args[1].value
        fluents = [f for f in self._fluents if not (f[0] == 'AtPose' and f[1] == body)]
        result = self._robot.check_reachability(body, self._state, fluents=fluents, **self._reachability_kwargs)
        self._update_pre_action('pick', body, result, plan_so_far)
        print('_check_pick | pre-actions-and', self.pre_actions_and)
        print('_check_pick | pre-actions-or', self.pre_actions_or)
        print('_check_pick | not pre-actions', self.not_pre_actions)
        if result:
            self._step_pick(args)
        return result

    def _step_pick(self, args):
        ## the resulting state is that the object is removed from the state
        body = args[1].value
        self._fluents = [f for f in self._fluents if not (f[0] != 'AtPose' and f[1] == body)]

    def _check_place(self, args, plan_so_far):
        """ whether the surface is reachable by PR2 arm """
        from pybullet_tools.logging_utils import myprint as print
        movable = args[1].value
        fluents = [f for f in self._fluents if not (f[0] == 'AtPose' and f[1] == movable)]
        if movable not in self.potential_placements:
            self._update_pre_action('place', movable, True, plan_so_far)
            return True
        body_link = self.potential_placements[movable]

        # if self._feg is None:
        #     x, y, _ = get_aabb_center(self._robot.aabb())
        #     z = self._robot.aabb().upper[2] + 0.1
        #     self._feg = create_gripper_robot(custom_limits=self._robot.custom_limits,
        #                                      initial_q=(x, y, z, 0, 0, 0))
        # result = self._feg.check_reachability_space(body_link, self._state, fluents=self._fluents,
        #                                             verbose=self._verbose)

        result = self._robot.check_reachability_space(body_link, self._state, body=movable, fluents=fluents,
                                                      **self._reachability_kwargs)
        for k, v in self.potential_placements.items():
            if v == body_link:
                self._update_pre_action('place', movable, result, plan_so_far)
        # print('_check_place | possible obstacles', self.possible_obstacles)
        print('_check_place | pre-actions-and', self.pre_actions_and)
        print('_check_place | pre-actions-or', self.pre_actions_or)
        print('_check_place | not pre-actions', self.not_pre_actions)
        return result

    def _check_pull(self, args, plan_so_far):
        self._step_pull(args)
        return True

    def _step_pull(self, args):
        ## door is assigned to a position that's fully open
        body_joint = b, j = args[1].value
        new_pstn = open_joint(b, j, hide_door=True, return_pstn=True)
        new_pstn = Position(body_joint, value=new_pstn)
        self._fluents = [f for f in self._fluents if not (f[0] == 'AtPosition' and f[1] == body_joint)]
        self._fluents.append(('AtPosition', body_joint, new_pstn))


##############################################################################


class Random(FeasibilityChecker):

    def __init__(self, p_feasible=0.5):
        super().__init__()
        np.random.seed(time.time())
        self.p_feasible = p_feasible

    def _check(self, input):
        return np.random.rand() > self.p_feasible
    # feasibility_checker = lambda *args: False # Reject all
    # feasibility_checker = lambda *args: True # Accept all
    # feasibility_checker = lambda *args: np.random.random() # Randomize


class PVT(FeasibilityChecker):

    def __init__(self, run_dir, body_map, pt_path=None, task_name=None, mode='pvt', scoring=False):
        super().__init__(run_dir, body_map)
        from test_piginet import get_model, DAFAULT_PT_NAME, TASK_PT_NAMES, \
            PT_NEWER, get_args, TASK_PT_STAR
        from fastamp_utils import get_facts_goals_visuals

        """ get data """
        self.run_dir = run_dir

        """ get model """
        if pt_path is None:
            pt_name = DAFAULT_PT_NAME
            if task_name is not None:
                if 'task*' in mode and task_name in TASK_PT_STAR:
                    pt_name = TASK_PT_STAR[task_name]
                elif task_name in TASK_PT_NAMES:
                    pt_name = TASK_PT_NAMES[task_name]
            elif mode in PT_NEWER:
                pt_name = PT_NEWER[mode]
            pt_path = join(MODELS_PATH, pt_name)
        self.pt_path = abspath(pt_path)

        self.args = args = get_args(basename(pt_path))
        self.data = get_facts_goals_visuals(run_dir, mode=args.input_mode, img_mode=args.image_mode, links_only=True)
        self.data['indices'] = get_indices(run_dir, larger=True)
        self.skeletons = self._get_gt_skeletons()

        self._model = get_model(pt_path, args)
        self._model.eval()
        self.scoring = scoring
        self.sequences = []
        print('\n\nPVT model loaded from', pt_path, '\n\n')

    def _check(self, inputs):
        from text_utils import ACTION_NAMES
        from dataset.datasets import get_dataset, collate
        import torch.nn as nn
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        args = self.args
        self.sequences = []

        indices = self.data['indices']
        dataset = []
        index = 0
        for input in inputs:
            data = copy.deepcopy(self.data)
            plan = []
            for a in input:
                elems = get_action_elems(a.args)
                elems = [indices[e] if e in indices else e for e in elems]
                if 'grasp' in ACTION_NAMES[a.name]:
                    continue
                plan.append([ACTION_NAMES[a.name]] + elems)
            data['run_name'] = self.run_dir
            data['plan'] = plan
            data['index'] = index
            data['skeleton'] = get_plan_skeleton(plan, **self._skwargs)
            if 'kc' in data['skeleton'] or 'kc' in self.skeletons[0]:
                print('\nkckckckckck -> ', data['skeleton'], self.skeletons[0])
            label = 1 if data['skeleton'] in self.skeletons else 0
            dataset.append((data, label))
            index += 1

        # import ipdb; ipdb.set_trace()

        base_2 = np.ceil(np.log2(len(inputs)))
        bs = 32 if 'to_storage' in self.run_dir else 128
        bs = min(2 ** int(base_2), bs) ## 128
        Dataset = get_dataset('pigi')
        data_loader = torch.utils.data.DataLoader(
            Dataset(dataset, test_time=True, num_images=args.num_img_channels),
            batch_size=bs, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate
        )
        all_predictions = []
        for inputs, labels in data_loader:
            with torch.set_grad_enabled(False):
                outputs = self._model(inputs)
                # labels = labels.flatten(0).to(device, non_blocking=True)
                if self.scoring:
                    predictions = nn.Sigmoid()(outputs).cpu().numpy()
                else:
                    predictions = nn.Sigmoid()(outputs).round().cpu().bool().numpy()
                all_predictions.extend([p.item() for p in predictions])
                self.sequences.extend(self._model.sequences)

        if len(inputs) == 1:
            return predictions[0].item()
        # skeletons = [d['skeleton'] for d in inputs[1]]
        # scores = {i: (skeletons[i], all_predictions[i]) for i in range(len(skeletons))}
        return all_predictions


##################################################

# TODO: FeasibilityScorer interface

class Shuffler(FeasibilityChecker):
    def _check(self, optimistic_plan):
        score = np.random.rand()
        return score


class Sorter(FeasibilityChecker):
    def _check(self, optimistic_plan):
        score = 1. / (1 + len(optimistic_plan))
        return score # Larger has higher priority

##################################################


class ChooseSkeleton(FeasibilityChecker):

    def __init__(self, run_dir, body_map, allowed_plan):
        super().__init__(run_dir, body_map)
        self.allowed_skeleton = get_plan_skeleton(allowed_plan, **self._skwargs)
        self.verbose = True

    def _check(self, plan):
        plan = [[i.name] + [str(a) for a in i.args] for i in plan]
        skeleton = get_plan_skeleton(plan, **self._skwargs)
        line = f'\tChooseSkeleton(FeasibilityChecker)\t{skeleton}'
        result = False
        if skeleton == self.allowed_skeleton:
            line += ' *'
            result = True
        else:
            line += f'\tself.allowed_skeleton = {self.allowed_skeleton}'
        if self.verbose:
            print(line)
        return result

##################################################


def get_plan_from_input(input):
    plan = []
    for action in input:
        plan.append([action.name] + [str(e) for e in action.args])
    return plan
