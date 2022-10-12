import warnings
warnings.filterwarnings('ignore')

import sys
from os.path import join, abspath, dirname, isdir, isfile

pbp_path = abspath(join(dirname(__file__), '..'))
workspace_path = abspath(join(pbp_path, '..', '..'))

sys.path.extend([
    pbp_path,
    join(pbp_path, '..', 'lisdf'),
    join(pbp_path, '..', 'pddlstream'),
    join(workspace_path, 'fastamp'),
])

ASSET_PATH = join(pbp_path, '..', 'assets')
EXP_PATH = join(pbp_path, '..', 'test_cases')
OUTPUT_PATH = join(pbp_path, '..', 'outputs')
MAMAO_DATA_PATH = join(workspace_path, 'fastamp-data')