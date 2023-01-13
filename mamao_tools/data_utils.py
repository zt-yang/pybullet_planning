import json
import shutil
from os import listdir, getcwd
from os.path import join, isfile, isdir, abspath
import untangle
import copy

import sys
sys.path.append('/home/yang/Documents/fastamp')

if 'zhutiany' in getcwd():
    DATASET_PATH = '/home/zhutiany/Documents/mamao-data'
elif 'yang' in getcwd():
    DATASET_PATH = '/home/yang/Documents/fastamp-data'
else: ## not tested
    DATASET_PATH = '../../fastamp-data'


def get_feasibility_checker(run_dir, mode, diverse=False):
    from .feasibility_checkers import PassAll, ShuffleAll, Oracle, PVT, Heuristic
    if mode == 'binary':
        mode = 'pvt'
        diverse = False
    if mode == 'None':
        return PassAll(run_dir)
    elif mode == 'shuffle':
        return ShuffleAll(run_dir)
    elif mode == 'oracle':
        plan = get_successful_plan(run_dir)
        return Oracle(run_dir, plan)
    elif mode == 'heuristic':
        return Heuristic(run_dir)
    elif 'pvt-task' in mode:
        task_name = abspath(run_dir).replace(DATASET_PATH, '').split('/')[1]
        return PVT(run_dir, mode=mode, task_name=task_name, scoring=diverse)
    elif mode.startswith('pvt'):
        return PVT(run_dir, mode=mode, scoring=diverse)
    return None


def organize_dataset(task_name):
    out_path = join(DATASET_PATH, task_name)
    names = [eval(l) for l in listdir(out_path) if isdir(join(out_path, l))]
    if len(names) == 0: return
    index = max(names)
    missing = list(set(range(index)) - set(names))
    if len(missing) == 0: return
    missing.sort()
    top = index
    moved = []
    for i in range(len(missing)):
        old_dir = join(out_path, str(top))
        while not isdir(old_dir) or str(top) in missing:
            top -= 1
            old_dir = join(out_path, str(top))
        if top in moved:
            break
        print(f'---- moving {top} to {missing[i]}')
        top -= 1
        shutil.move(old_dir, join(out_path, str(missing[i])))
        moved.append(missing[i])


def get_indices_from_config(run_dir):
    config = json.load(open(join(run_dir, 'planning_config.json'), 'r'))
    if 'body_to_name' in config:
        return {k: v for k, v in config['body_to_name'].items()} ## .replace('::', '%')
    return False


def process_value(vv, training=True):
    from pybullet_tools.utils import quat_from_euler
    vv = list(vv)
    ## convert quaternion to euler angles
    if len(vv) == 6:
        quat = quat_from_euler(vv[3:])
        vv = vv[:3] + list(quat)
    return vv


def get_successful_plan(run_dir, indices={}, continuous={}):
    plan = []
    with open(join(run_dir, 'plan.json'), 'r') as f:
        data = json.load(f)[0]
        actions = data['plan']
        if actions == 'FAILED':
            return None
        vs, inv_vs = get_variables(data['init'])
        for a in actions:
            name = a[a.index("name='")+6: a.index("', args=(")]
            # if 'grasp_handle' in name:
            #     continue
            args = a[a.index("args=(")+6:-2].replace("'", "")
            new_args = parse_pddl_str(args, vs=vs, inv_vs=inv_vs, indices=indices)
            plan.append([name] + new_args)
    return plan


def parse_pddl_str(args, vs, inv_vs, indices={}):
    """ parse a string of string, int, and tuples into a list """

    ## replace those tuples with placeholders that doesn't have ', ' or ' '
    for string, sub in inv_vs.items():
        if string in args:
            args = args.replace(string, sub)

    if ',' in args:
        """  from plan.json
        e.g. 'left', 7, p1=(3.255, 4.531, 0.762, 0.0, -0.0, 2.758), g208=(0, 0.0, 0.304, -3.142, 0, 0),
             q624=(3.959, 4.687, 0.123, -1.902), c528=t(7, 60), wconf64 """
        args = args.split(', ')

    else:
        """  from problem.pddl
        e.g. pose veggiecauliflower p0=(3.363, 2.794, 0.859, 0.0, -0.0, 1.976) """
        args = args.split(' ')

    ## replace those placeholders with original values
    new_args = []
    for arg in args:
        if arg in indices:
            new_args.append(indices[arg])
        elif 'idx=' in arg:
            idx = int(eval(arg.replace('idx=', '')))
            new_args.append(vs[idx])
        else:
            new_args.append(arg)
    return new_args


def get_plan(run_dir, indices={}, continuous={}, plan_json=None):
    if indices == {}:
        indices = get_indices_from_config(run_dir)
    if plan_json is not None:
        plan = json.load(open(plan_json, 'r'))['plan']
    else:
        plan = get_successful_plan(run_dir, indices)

    ## add the continuous mentioned in plan
    new_continuous = {}
    for a in plan:
        for arg in a[1:]:
            if '=' in arg:
                name, value = arg.split('=')
                value = value.replace('t(', '(')
                if name not in continuous and name not in new_continuous:
                    v = list(eval(value)) if ('(' in value) else [eval(value)]
                    if len(v) != 2:  ## sampled trajectory
                        new_continuous[name] = process_value(v)
    return plan, new_continuous


def get_indices_from_log(run_dir):
    indices = {}
    with open(join(run_dir, 'log.txt'), 'r') as f:
        lines = [l.replace('\n', '') for l in f.readlines()[3:40]]
        lines = lines[:lines.index('----------------')]
    for line in lines:
        elems = line.split('|')
        body = elems[0].rstrip()
        name = elems[1].strip()
        name = name[name.index(':')+2:]
        if 'pr2' in body:
            body = body.replace('pr2', '')
        indices[body] = name
    return indices


def get_indices_from_config(run_dir):
    config = json.load(open(join(run_dir, 'planning_config.json'), 'r'))
    if 'body_to_name' in config:
        return config['body_to_name']
    return False


def get_indices(run_dir):
    result = get_indices_from_config(run_dir)
    if not result:
        return get_indices_from_log(run_dir)
    return result


def get_instance_info(run_dir, world=None):
    if world is None:
        world = get_lisdf_xml(run_dir)
    instances = {}
    for m in world.include:
        if 'mobility.urdf' in m.uri.cdata:
            uri = m.uri.cdata.replace('/mobility.urdf', '')
            uri = uri[uri.index('models/') + 7:]
            index = uri[uri.index('/') + 1:]
            if index == '00001':
                index = 'VeggieCabbage'
            instances[m['name']] = index
    return instances


def exist_instance(run_dir, instance):
    instances = get_instance_info(run_dir)
    return list(instances.values()).count(instance) > 0


def get_lisdf_xml(run_dir):
    return untangle.parse(join(run_dir, 'scene.lisdf')).sdf.world


def get_action_elems(list_of_elem):
    return [str(e) for e in list_of_elem if '#' not in str(e) and \
                    '=' not in str(e) and str(e) not in ['left', 'right', 'hand', 'None']]


def get_plan_skeleton(plan, indices={}, include_joint=True, include_movable=False):
    from text_utils import ACTION_ABV, ACTION_NAMES
    joints = [k for k, v in indices.items() if "::" in v]
    movables = [k for k, v in indices.items() if "::" not in v]
    joint_names = [v for v in indices.values() if "::" in v]
    movable_names = [v for v in indices.values() if "::" not in v]
    inv_joints = {v: k for k, v in indices.items() if v in joint_names}
    inv_movables = {v: k for k, v in indices.items() if v in movable_names}
    inv_indices = {v: k for k, v in indices.items()}

    if isinstance(plan[0], str):
        plan = get_plan_from_strings(plan, indices=indices)

    def get_action_abv(a):
        if len(a) == 0:
            print('what is a', a)
        if a[0] in ACTION_ABV:
            skeleton = ACTION_ABV[a[0]]
        else:
            ABV = {ACTION_NAMES[k]: v for k, v in ACTION_ABV.items()}
            skeleton = ABV[a[0]]
        aa = []
        for e in a:
            aa.append(indices[e] if e in indices else e)
        a = copy.deepcopy(aa)
        if len(skeleton) > 0:
            ## contains 'minifridge::joint_2' or '(4, 1)'
            if include_joint:
                def get_joint_name_chars(o):
                    body_name, joint_name = o.split('::')
                    if not joint_name.startswith('left') and 'left' in joint_name:
                        joint_name = joint_name[0] + 'l'
                    elif not joint_name.startswith('right') and 'right' in joint_name:
                        joint_name = joint_name[0] + 'r'
                    else:
                        joint_name = joint_name[0]
                    return f"({body_name[0]}{joint_name})"
                skeleton += ''.join([get_joint_name_chars(o) for o in a[1:] if o in joint_names])
                skeleton += ''.join([f"({indices[o][0]}{indices[o][-1]})" for o in a[1:] if o in joints])
            ## contains 'veggiepotato' or '3'
            if include_movable:
                skeleton += ''.join([f"({inv_movables[o]})" for o in a[1:] if o in movable_names])
                skeleton += ''.join([f"({o})" for o in a[1:] if o in movables])
        return skeleton
    return ''.join([get_action_abv(a) for a in plan])


def get_init_tuples(run_dir):
    from fastamp_utils import get_init, get_objs
    lines = open(join(run_dir, 'problem.pddl'), 'r').readlines()
    objs = get_objs(lines)
    init = get_init(lines, objs, get_all=True)
    return init


def get_lisdf_xml(run_dir):
    return untangle.parse(join(run_dir, 'scene.lisdf')).sdf.world


def get_instance_info(run_dir, world=None):
    if world is None:
        world = get_lisdf_xml(run_dir)
    instances = {}
    for m in world.include:
        if 'mobility.urdf' in m.uri.cdata:
            uri = m.uri.cdata.replace('/mobility.urdf', '')
            uri = uri[uri.index('models/') + 7:]
            index = uri[uri.index('/') + 1:]
            if index == '00001':
                index = 'VeggieCabbage'
            instances[m['name']] = index
    return instances


def exist_instance(model_instances, instance):
    return list(model_instances.values()).count(instance) > 0


def get_fc_record(run_dir, fc_classes=[], diverse=True, rerun_subdir=None):
    prefix = 'diverse_' if diverse else ''
    pass_fail = {}
    indices = get_indices(run_dir)
    rerun_dir = join(run_dir, rerun_subdir) if rerun_subdir is not None else run_dir
    for fc_class in fc_classes:
        pas = []
        fail = []
        log_file = join(rerun_dir, f"{prefix}fc_log={fc_class}.json")
        plan_file = join(rerun_dir, f"{prefix}plan_rerun_fc={fc_class}.json")

        if isfile(log_file) and isfile(plan_file):
            log = json.load(open(log_file, 'r'))
            for aa in log['checks']:
                plan, prediction = aa[-2:]
                skeleton = get_plan_skeleton(plan, indices=indices)
                note = f"{skeleton} ({round(prediction, 4)})"
                if prediction and prediction > 0.5:
                    pas.append(note)
                else:
                    fail.append(note)

            result = json.load(open(plan_file, 'r'))
            plan = result['plan']
            planning_time = round(result['planning_time'], 2)
            if len(pas) > 0 or len(fail) > 0:
                if plan is not None:
                    plan = get_plan_skeleton(plan, indices=indices)
                    t_skeletons = [sk[:sk.index(' (')] for sk in pas]
                    num_FP = t_skeletons.index(plan) if plan in t_skeletons else len(pas)
                else:
                    num_FP = None
                pass_fail[fc_class] = (fail, pas, [plan], planning_time, num_FP)
    return pass_fail


def get_variables_from_pddl_objects(init):
    vs = []
    for i in init:
        vs.extend([a for a in i[1:] if ',' in a])
    return list(set(vs))


def get_variables_from_pddl(facts, objs):
    new_objs = copy.deepcopy(objs) + ['left', 'right']
    new_objs.sort(key=len, reverse=True)
    vs = []
    for f in facts:
        f = f.replace('\n', '').replace('\t', '').strip()[1:-1]
        if ' ' not in f or f.startswith(';'):
            continue
        f = f[f.index(' ')+1:]
        for o in new_objs:
            f = f.replace(o, '')
        f = f.strip()
        if len(f) == 0:
            continue

        if f not in vs:
            found = False
            for v in vs:
                if v in f:
                    found = True
            if not found and 'wconf' not in f:
                vs.append(f)
    return vs


def get_variables(init, objs=None):
    if isinstance(init[0], str):
        vs = get_variables_from_pddl(init, objs)
    else:
        vs = get_variables_from_pddl_objects(init)

    return vs, {vs[i]: f'idx={i}' for i in range(len(vs))}


def get_plan_from_strings(actions, vs, inv_vs, indices={}):
    from text_utils import ACTION_NAMES
    plan = []
    for a in actions:
        name = a[a.index("name='") + 6: a.index("', args=(")]
        args = a[a.index("args=(") + 6:-2].replace("'", "")
        new_args = parse_pddl_str(args, vs=vs, inv_vs=inv_vs, indices=indices)
        plan.append([ACTION_NAMES[name]] + new_args)
    return plan


def parse_pddl_str(args, vs, inv_vs, indices={}):
    """ parse a string of string, int, and tuples into a list """

    ## replace those tuples with placeholders that doesn't have ', ' or ' '
    for string, sub in inv_vs.items():
        if string in args:
            args = args.replace(string, sub)

    if ',' in args:
        """  from plan.json
        e.g. 'left', 7, p1=(3.255, 4.531, 0.762, 0.0, -0.0, 2.758), g208=(0, 0.0, 0.304, -3.142, 0, 0),
             q624=(3.959, 4.687, 0.123, -1.902), c528=t(7, 60), wconf64 """
        args = args.split(', ')

    else:
        """  from problem.pddl
        e.g. pose veggiecauliflower p0=(3.363, 2.794, 0.859, 0.0, -0.0, 1.976) """
        args = args.split(' ')

    ## replace those placeholders with original values
    new_args = []
    for arg in args:
        if 'idx=' in arg:
            idx = int(eval(arg.replace('idx=', '')))
            arg = vs[idx]
        if arg in indices:
            new_args.append(indices[arg])
        else:
            new_args.append(arg)
    return new_args


def get_successful_plan(run_dir, indices={}):
    plans = []
    ## default best plan is in 'plan.json'
    with open(join(run_dir, 'plan.json'), 'r') as f:
        data = json.load(f)[0]
        actions = data['plan']
        if actions == 'FAILED':
            return None
        vs, inv_vs = get_variables(data['init'])
        plan = get_plan_from_strings(actions, vs=vs, inv_vs=inv_vs, indices=indices)
        plans.append(plan)
    solutions = get_multiple_solutions(run_dir, indices=indices)
    if len(solutions) > 1:
        for solution in solutions:
            if len(solution) < len(plans[0]):
                plans = [solution] + plans
    return plans


def get_multiple_solutions(run_dir, indices={}):
    all_plans = []
    solutions_file = join(run_dir, 'multiple_solutions.json')
    if isfile(solutions_file):
        with open(solutions_file, 'r') as f:
            data = json.load(f)
            for d in data:
                ## those failed attempts
                if 'optimistic_plan' in d:
                    plan = d['optimistic_plan'][1:-1].split('Action')
                    plan = ['Action'+p[:-2] for p in plan[1:]]
                    plan = get_plan_from_strings(plan, indices=indices)
                elif 'rerun_dir' in d:
                    plan = d['plan']
                all_plans.append(plan)
    return all_plans


def save_multiple_solutions(plan_dataset, indices=None, run_dir=None,
                            file_path='multiple_solutions.json'):
    if indices is None and run_dir is not None:
        indices = get_indices(run_dir)
    first_solution = None
    min_len = 10000
    solutions_log = []
    save = False
    for i, (opt_solution, real_solution) in enumerate(plan_dataset):
        stream_plan, (opt_plan, preimage), opt_cost = opt_solution
        plan = None
        score = 0
        if real_solution is not None:
            plan, cost, certificate = real_solution
            if plan is not None and len(plan) > 0:
                if first_solution is None:
                    first_solution = real_solution
                    min_len = len(plan)
                score = round(0.5 + min_len / (2*len(plan)), 3)
        skeleton = ''
        ## in HPN mode
        if '-no--' in str(opt_plan):
            save = True
        if save:
            skeleton = get_plan_skeleton(opt_plan, indices=indices)
            log = {
                'optimistic_plan': str(opt_plan),
                'skeleton': str(skeleton),
                'plan': [str(a) for a in plan] if plan is not None else None,
                'score': score
            }
            solutions_log.append(log)
        print(f'\n{i + 1}/{len(plan_dataset)}) Optimistic Plan: {opt_plan}\n'
              f'Skeleton: {skeleton}\nPlan: {plan}')
    if save:
        with open(file_path, 'w') as f:
            json.dump(solutions_log, f, indent=3)
    if first_solution is None:
        first_solution = None, 0, []
    return first_solution


def has_text_utils():
    try:
        import text_utils
    except ImportError:
        return False
    return True
