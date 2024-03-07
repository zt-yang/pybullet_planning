import socket
import os
from os.path import join, abspath, dirname, isdir, isfile

abs_join = lambda *args, **kwargs: abspath(join(*args, **kwargs))

current_dir = abspath(dirname(__file__))
PBP_PATH = join(current_dir, '..')
TEMP_PATH = join(PBP_PATH, '..', 'temp')
ASSET_PATH = join(PBP_PATH, '..', 'assets')
DATA_CONFIG_PATH = join(PBP_PATH, 'data_generator', 'configs')

## different projects
workspace_path = abs_join(PBP_PATH, '..', '..')
if 'cognitive-architectures' in workspace_path:
    workspace_path = abs_join(workspace_path, '..')
KITCHEN_WORLD = abs_join(workspace_path, 'kitchen-worlds')
PARTNET_PATH = abs_join(workspace_path, '..', 'dataset')

## Caelan
# if socket.gethostname() == 'cgarrett-dt':
#     KITCHEN_WORLD = abs_join(os.pardir, os.pardir, os.pardir, 'kitchen-worlds')
#     PARTNET_PATH = abs_join(os.pardir, os.pardir, 'dataset')
#     ASSET_PATH = abs_join(os.pardir, 'assets')