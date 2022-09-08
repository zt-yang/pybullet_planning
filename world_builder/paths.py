import socket
import os
from os.path import join, abspath, dirname, isdir, isfile

current_dir = abspath(dirname(__file__))
abs_join = lambda *args, **kwargs: abspath(join(*args, **kwargs))

## NVIDIA desktop or NGC
if 'zhutiany' in current_dir or 'mamao' in current_dir:
    KITCHEN_WORLD = abs_join('..', '..', '..', 'kitchen-worlds')
    PARTNET_PATH = abs_join('..', '..', 'data', 'partnet_mobility_v0', 'dataset')
    # ASSET_PATH = absjoin('..', '..', '..', 'kitchen-worlds', 'assets')
    ASSET_PATH = abs_join('..', 'assets')
elif 'yang' in current_dir or 'mamao' in current_dir:
    KITCHEN_WORLD = abs_join('..', '..', '..', 'kitchen-worlds')
    ASSET_PATH = abs_join('..', '..', '..', 'kitchen-worlds', 'assets')
    ASSET_PATH = abs_join('..', 'assets')
## Caelan
elif socket.gethostname() == 'cgarrett-dt':
    KITCHEN_WORLD = abs_join(os.pardir, os.pardir, os.pardir, 'kitchen-worlds')
    PARTNET_PATH = abs_join(os.pardir, os.pardir, 'dataset')
    ASSET_PATH = abs_join(os.pardir, 'assets')
## Macbook Pro
else:
    KITCHEN_WORLD = abs_join('..', '..', '..', 'PyBullet', 'kitchen-worlds')
    PARTNET_PATH = abs_join('..', '..', 'dataset')
    ASSET_PATH = abs_join('..', 'assets')
