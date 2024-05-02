import os
import sys
from os.path import join, abspath, dirname, isdir, isfile
R = abspath(join(dirname(__file__), os.pardir))
sys.path.extend([R] + [join(R, name) for name in ['pybullet_planning', 'lisdf', 'pddlstream']])

from pybullet_tools.utils import read_pickle, set_renderer, wait_for_user, PI

from tutorials.test_grasps import test_grasps, test_handle_grasps_counter


def test_spot_grasps():
    kwargs = dict(categories=['VeggieCabbage'], skip_grasps=False, base_q=(0, 0, 0, 0))
    kwargs['categories'] = ['Food']

    ## --- step 1: find tool_from_hand transformation
    debug_kwargs = dict(verbose=True, test_rotation_matrix=True, skip_grasp_index=1,
                        test_translation_matrix=False)

    ## --- step 2: find tool_from_root transformation (multiple rotations may look correct,
    #               but only one works after running IR - IK)
    ## (1.571, 3.142, -1.571) (1.571, 3.142, 1.571) (1.571, 3.142, 0) (-1.571, 0, 0)
    debug_kwargs = dict(verbose=True, test_attachment=True, visualize=False, retain_all=False)

    ## --- step 3: verify all grasps generated for one object
    # debug_kwargs = dict(verbose=True, test_attachment=False, visualize=True, retain_all=True)

    ## --- step 4: verify top_grasp_tolerance filtering
    debug_kwargs = dict(verbose=True, visualize=False, retain_all=False, top_grasp_tolerance=PI/4)

    test_grasps('spot', **debug_kwargs, **kwargs)
    # test_grasps('pr2', **debug_kwargs, **kwargs)
    # test_grasps('feg', **debug_kwargs, **kwargs)


if __name__ == '__main__':
    test_spot_grasps()
    # test_rummy_handle_grasps()
    # test_nvidia_kitchen_domain_spot()
