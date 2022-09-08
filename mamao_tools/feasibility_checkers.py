import numpy as np
import time
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


class FeasibilityChecker(object):

    def __init__(self, **kwargs):
        raise NotImplementedError('should implement this for FeasibilityChecker')

    def __call__(self, *args, **kwargs):
        return self.check(*args, **kwargs)

    def check(self, input):
        raise NotImplementedError('should implement this for FeasibilityChecker')


class Oracle(FeasibilityChecker):

    def __init__(self, correct):
        self.correct = correct

    def check(self, input):
        if len(input) != len(self.correct):
            return False
        for i in range(len(input)):
            action = [input[i].name] + list(input[i].args)
            for j in range(len(self.correct[i])):
                if '=' not in self.correct[i][j]:
                    # if len(action) != len(self.correct[i]):
                    #     print('len(input[i]) != len(self.correct[i])', action, self.correct[i])
                    if str(action[j]) != self.correct[i][j]:
                        print(f'\n\nOracle feasibility checker | correct', self.correct[i],
                              'input', action, '\n\n')
                        return False
        return True


class Random(FeasibilityChecker):

    def __init__(self):
        np.random.seed(time.time())

    def check(self):
        return np.random.rand() > 0.5


class PVT(FeasibilityChecker):

    def __init__(self, run_dir, pt_path=None):
        import sys
        from os.path import join, abspath, dirname, isdir, isfile
        sys.path.append(join('..', 'pybullet_planning', 'fastamp'))
        print('\nsys.path', sys.path)
        from fastamp.test_piginet import get_model, DAFAULT_PT_NAME, args
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
            pt_path = join(dirname(abspath(__file__)), '..', 'fastamp', 'models', DAFAULT_PT_NAME)
        self.pt_path = pt_path
        self.model = get_model(pt_path)
        print('PVT model loaded from', pt_path)

    def check(self, input):
        from fastamp.text_utils import ACTION_NAMES
        from fastamp.datasets import get_dataset, collate
        from fastamp.fastamp_utils import get_action_elems
        import torch.nn as nn
        args = self.args
        data = self.data
        plan = []
        for a in input:
            elems = get_action_elems(a.args)
            elems = [data['indices'][e] if e in data['indices'] else e for e in elems]
            plan.append([ACTION_NAMES[a.name]] + elems)
        data['plan'] = plan
        label = 1 if plan == self.plan_gt else 0

        Dataset = get_dataset(args.input_mode)
        data_loader = torch.utils.data.DataLoader(
            Dataset([(data, label)]),
            batch_size=1, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate
        )
        prediction = True
        for inputs, labels in data_loader:
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                labels = labels.flatten(0).to(device, non_blocking=True)
                prediction = nn.Sigmoid()(outputs).round().cpu().squeeze().bool().numpy().item()
        # import ipdb; ipdb.set_trace()
        return prediction

