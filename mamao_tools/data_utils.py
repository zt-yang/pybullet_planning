import os
from os import listdir
from os.path import join, isfile, isdir, abspath


def get_indices(run_dir):
    indices = {}
    sub_dir = join(run_dir, 'depth_maps')
    if not isdir(sub_dir):
        sub_dir = join(run_dir, 'rgb_images')
    for f in listdir(sub_dir):
        if '[' in f:
            name = f[f.index(']_')+2: f.index('.png')].replace('::', '%')
            id = f[f.index('[')+1: f.index(']_')]
            indices[id] = name
    return indices
