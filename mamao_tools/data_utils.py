import json
from os import listdir
from os.path import join, isfile, isdir, abspath
import untangle
import copy


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


def get_plan_skeleton(plan, indices={}):
    from text_utils import ACTION_ABV, ACTION_NAMES
    def get_action_abv(a):
        if isinstance(a, str):
            name = a[a.index("name='")+6: a.index("', args=(")]
            return ACTION_ABV[name]
        else:
            if a[0] in ACTION_ABV:
                skeleton = ACTION_ABV[a[0]]
                if hasattr(a, 'args'):
                    a = a.args
            else:
                ABV = {ACTION_NAMES[k]: v for k, v in ACTION_ABV.items()}
                skeleton = ABV[a[0]]
            aa = []
            for e in a:
                aa.append(indices[str(e)] if str(e) in indices else str(e))
            if len(skeleton) > 0:
                skeleton += ''.join([f"({o[0]}{o[-1]})" for o in aa[1:] if '::' in o])
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
    if len(indices) == 0:
        indices = get_indices(run_dir)
    with open(join(run_dir, 'plan.json'), 'r') as f:
        data = json.load(f)[0]
        actions = data['plan']
        vs, inv_vs = get_variables(data['init'])
        plan = get_plan_from_strings(actions, vs=vs, inv_vs=inv_vs, indices=indices)
    return [plan]