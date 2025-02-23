import socket
import os
from os.path import join, abspath, dirname, isdir, isfile, expanduser

abs_join = lambda *args, **kwargs: abspath(join(*args, **kwargs))

""" ------ inside pybullet_planning -------- """
HOME_DIR = expanduser("~")
current_dir = abspath(dirname(__file__))
PBP_PATH = abs_join(current_dir, '..')
DATA_CONFIG_PATH = abs_join(PBP_PATH, 'data_generator', 'configs')

""" ------ useful directories -------- """
PROJECT_PATH = abs_join(PBP_PATH, '..')
DATABASES_PATH = abs_join(PBP_PATH, 'databases')
TEMP_PATH = abs_join(PROJECT_PATH, 'temp')
ASSET_PATH = abs_join(PROJECT_PATH, 'assets')
EXP_PATH = abs_join(PROJECT_PATH, 'experiments')
OUTPUT_PATH = abs_join(PROJECT_PATH, 'outputs')

""" ------ involving other projects ------ """
workspace_path = abs_join(PBP_PATH, '..', '..')
IS_COGARCH = 'cognitive-architectures' in workspace_path
if IS_COGARCH:
    workspace_path = abs_join(workspace_path, '..')
KITCHEN_WORLD = abs_join(workspace_path, 'kitchen-worlds')
PARTNET_PATH = abs_join(workspace_path, 'dataset')

## Caelan
# if socket.gethostname() == 'cgarrett-dt':
#     KITCHEN_WORLD = abs_join(os.pardir, os.pardir, os.pardir, 'kitchen-worlds')
#     PARTNET_PATH = abs_join(os.pardir, os.pardir, 'dataset')
#     ASSET_PATH = abs_join(os.pardir, 'assets')