import warnings
warnings.filterwarnings('ignore')

import sys
from os.path import join, abspath, dirname, isdir, isfile


def absjoin(*args):
    return abspath(join(*args))


pbp_path = absjoin(dirname(__file__), '..')
workspace_path = absjoin(pbp_path, '..', '..')
IS_COGARCH = 'cognitive-architectures' in workspace_path

sys.path.extend([
    pbp_path,
    absjoin(pbp_path, '..', 'lisdf'),
    absjoin(pbp_path, '..', 'pddlstream'),
    absjoin(workspace_path, 'fastamp'),
])

ASSET_PATH = absjoin(pbp_path, '..', 'assets')
EXP_PATH = absjoin(pbp_path, '..', 'test_cases')
OUTPUT_PATH = absjoin(pbp_path, '..', 'outputs')
MAMAO_DATA_PATH = absjoin(workspace_path, 'fastamp-data')

if IS_COGARCH:
    KITCHEN_WORLD_PATH = absjoin(workspace_path, '..', 'jupyter-worlds')
    EXP_PATH = absjoin(KITCHEN_WORLD_PATH, 'test_cases')


def modify_file_by_project(file_path):
    name, suffix = file_path.split('.')
    name += '_cogarch' if IS_COGARCH else '_kitchen'
    return '.'.join([name, suffix])
