import numpy as np
import time
import torch
import json
import copy
import sys
from os.path import join, abspath, dirname
sys.path.append(join('..', 'pybullet_planning', 'fastamp'))
# print('\nsys.path', sys.path)

device = "cuda" if torch.cuda.is_available() else "cpu"


class FeasibilityChecker(object):

    def __init__(self, run_dir, **kwargs):
        self.run_dir = run_dir
        self._log = {k: [] for k in ['checks', 'run_time']}

    def __call__(self, *args, **kwargs):
        return self.check(*args, **kwargs)

    def _check(self, input):
        raise NotImplementedError('should implement this for FeasibilityChecker')

    def check(self, inputs):
        from fastamp.fastamp_utils import get_plan_skeleton, get_indices
        if not isinstance(inputs[0], list):
            inputs = [inputs]

        predictions = []
        indices = get_indices(self.run_dir)
        if 'PVT' not in self.__class__.__name__:
            for input in inputs:
                start = time.time()
                prediction = self._check(input)
                predictions.append(prediction)
                plan = get_plan_from_input(input)
                skeleton = get_plan_skeleton(plan, indices)
                self._log['checks'].append((skeleton, plan, prediction))
                self._log['run_time'].append(round(time.time() - start, 4))
        else:
            printout = []
            start = time.time()
            predictions = self._check(inputs)
            run_time = round(time.time() - start, 4)
            ave_time = (run_time, len(inputs), round(run_time / len(inputs), 4))
            p = {i: predictions[i] for i in range(len(inputs))}
            sorted_predictions = {k: v for k, v in sorted(p.items(), key=lambda item: item[1], reverse=True)}
            for i, prediction in sorted_predictions.items():
                plan = get_plan_from_input(inputs[i])
                skeleton = get_plan_skeleton(plan, indices)
                self._log['checks'].append((skeleton, plan, prediction))
                self._log['run_time'].append(ave_time)
                printout.append((round(prediction, 3), skeleton))
            [print(p) for p in printout]

        if len(predictions) == 1:
            return predictions[0]
        return predictions

    def dump_log(self, json_path):
        with open(json_path, 'w') as f:
            config = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
            if 'args' in config:
                config['args'] = config['args'].__dict__
            self._log['config'] = config
            json.dump(self._log, f, indent=3)


class PassAll(FeasibilityChecker):

    def __init__(self, run_dir):
        super().__init__(run_dir)

    def _check(self, input):
        return True


class Oracle(FeasibilityChecker):

    def __init__(self, run_dir, correct):
        super().__init__(run_dir)
        self.correct = correct
        print(f'\nOracle feasibility checker\n', '\n'.join([str(c) for c in correct]))

    def _check(self, input):
        if len(input) != len(self.correct):
            return False
        for i in range(len(input)):
            action = [input[i].name] + list(input[i].args)
            for j in range(len(self.correct[i])):
                if '=' not in self.correct[i][j]:
                    # if len(action) != len(self.correct[i]):
                    #     print('len(input[i]) != len(self.correct[i])', action, self.correct[i])
                    if str(action[j]) != self.correct[i][j]:
                        # print(i, self.correct[i], '\n', action, '\n')
                        return False
        print('pass', input)
        return True


class Random(FeasibilityChecker):

    def __init__(self, p_feasible=0.5):
        super().__init__()
        np.random.seed(time.time())
        self.p_feasible = p_feasible

    def _check(self, input):
        return np.random.rand() > self.p_feasible


class PVT(FeasibilityChecker):

    def __init__(self, run_dir, pt_path=None, task_name=None, mode='pvt', scoring=False):
        super().__init__(run_dir)
        from fastamp.test_piginet import get_model, DAFAULT_PT_NAME, args, TASK_PT_NAMES, \
            PT_NEWER
        from fastamp.fastamp_utils import get_facts_goals_visuals, get_successful_plan, \
            get_action_elems, get_plans
        args.input_mode = 'pigi'

        """ get data """
        self.run_dir = run_dir
        self.args = args
        self.data = get_facts_goals_visuals(run_dir, mode=args.image_mode, links_only=True)
        plan_gt = get_successful_plan(run_dir, self.data['indices'], self.data['continuous'])[0]
        self.plan_gt = [get_action_elems(a) for a in plan_gt]
        # plan_gt, continuous = get_plans(run_dir, self.data['indices'], self.data['continuous'])
        # self.data['continuous'].update(continuous)
        # self.plan_gt = [get_action_elems(a) for a in plan_gt[0]]

        """ get model """
        if pt_path is None:
            pt_name = DAFAULT_PT_NAME
            if task_name is not None and task_name in TASK_PT_NAMES:
                pt_name = TASK_PT_NAMES[task_name]
            elif mode in PT_NEWER:
                pt_name = PT_NEWER[mode]
            pt_path = join(dirname(abspath(__file__)), '..', 'fastamp', 'models', pt_name)

        self.pt_path = abspath(pt_path)
        self._model = get_model(pt_path)
        self.scoring = scoring
        print('\n\nPVT model loaded from', pt_path, '\n\n')

    def _check(self, inputs):
        from fastamp.text_utils import ACTION_NAMES
        from fastamp.datasets import get_dataset, collate
        from fastamp.fastamp_utils import get_action_elems
        import torch.nn as nn
        args = self.args

        dataset =[]
        for input in inputs:
            data = copy.deepcopy(self.data)
            plan = []
            for a in input:
                elems = get_action_elems(a.args)
                elems = [data['indices'][e] if e in data['indices'] else e for e in elems]
                plan.append([ACTION_NAMES[a.name]] + elems)
            data['plan'] = plan
            label = 1 if plan == self.plan_gt else 0
            dataset.append((data, label))

        base_2 = np.ceil(np.square(len(inputs)))
        bs = min(2 ** int(base_2), 64)
        Dataset = get_dataset(args.input_mode)
        data_loader = torch.utils.data.DataLoader(
            Dataset(dataset),
            batch_size=bs, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate
        )
        for inputs, labels in data_loader:
            with torch.set_grad_enabled(False):
                outputs = self._model(inputs)
                # labels = labels.flatten(0).to(device, non_blocking=True)
                if self.scoring:
                    predictions = nn.Sigmoid()(outputs).cpu().numpy()
                else:
                    predictions = nn.Sigmoid()(outputs).round().cpu().bool().numpy()
        if len(inputs) == 1:
            return predictions[0].item()
        return [p.item() for p in predictions]


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

def get_plan_from_input(input):
    plan = []
    for action in input:
        plan.append([action.name] + [str(e) for e in action.args])
    return plan
