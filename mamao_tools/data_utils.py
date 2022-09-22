import json
from os import listdir
from os.path import join, isfile, isdir, abspath
import untangle


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