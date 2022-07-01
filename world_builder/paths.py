from os.path import join, abspath, dirname, isdir, isfile

current_dir = abspath(dirname(__file__))

## NVIDIA desktop
if 'zhutiany' in current_dir:
    KITCHEN_WORLD = join('..', '..', '..', 'kitchen-worlds')
    PARTNET_PATH = join('..', '..', 'data', 'partnet_mobility_v0', 'dataset')

## Macbook Pro
else:
    KITCHEN_WORLD = join('..', '..', '..', 'PyBullet', 'kitchen-worlds')
    PARTNET_PATH = join('..', '..', 'dataset')