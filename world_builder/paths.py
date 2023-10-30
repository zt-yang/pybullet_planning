import socket
import os
from os.path import join, abspath, dirname, isdir, isfile

abs_join = lambda *args, **kwargs: abspath(join(*args, **kwargs))

current_dir = abspath(dirname(__file__))
pbp_path = abs_join(current_dir, '..')
ASSET_PATH = abs_join(pbp_path, '..', 'assets')

workspace_path = abs_join(pbp_path, '..', '..')
KITCHEN_WORLD = abs_join(workspace_path, 'kitchen-worlds')
PARTNET_PATH = abs_join(workspace_path, '..', 'dataset')

## Caelan
# if socket.gethostname() == 'cgarrett-dt':
#     KITCHEN_WORLD = abs_join(os.pardir, os.pardir, os.pardir, 'kitchen-worlds')
#     PARTNET_PATH = abs_join(os.pardir, os.pardir, 'dataset')
#     ASSET_PATH = abs_join(os.pardir, 'assets')