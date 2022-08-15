import json
from os import listdir
from os.path import join, isfile, isdir, abspath


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

    indices = {}
    sub_dir = join(run_dir, 'seg_images')
    if not isdir(sub_dir):
        sub_dir = join(run_dir, 'rgb_images')
    if not isdir(sub_dir):
        sub_dir = join(run_dir, 'depth_maps')
    if not isdir(sub_dir):
        return get_indices_from_log(run_dir)

    ## get from the name of images
    for f in listdir(sub_dir):
        if '[' in f:
            name = f[f.index(']_')+2: f.index('.png')].replace('::', '%')
            id = f[f.index('[')+1: f.index(']_')]
            indices[id] = name
    return indices
