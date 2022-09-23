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
    from fastamp.text_utils import ACTION_ABV, ACTION_NAMES
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
    from fastamp.fastamp_utils import get_init, get_objs
    lines = open(join(run_dir, 'problem.pddl'), 'r').readlines()
    objs = get_objs(lines)
    init = get_init(lines, objs, get_all=True)
    return init

