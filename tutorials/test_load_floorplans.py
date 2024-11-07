import sys
from os.path import join, abspath, dirname
from os import pardir
RD = abspath(join(dirname(__file__), pardir, pardir))
sys.path.extend([join(RD), join(RD, 'pddlstream'), join(RD, 'pybullet_planning'), join(RD, 'lisdf')])

from pybullet_tools.utils import PI, wait_if_gui, set_camera_pose

from problem_sets.problem_utils import create_world
from cogarch_tools.cogarch_utils import parse_agent_args, init_pybullet_client

from world_builder.loaders import load_floor_plan
from world_builder.loaders_nvidia_kitchen import load_kitchen_floor_plan

from robot_builder.robot_builders import build_robot_from_args

from tutorials.test_utils import get_test_world


def test_load_floating_gripper_in_kitchen(random_instance=True):
    """ a fixed kitchen layout with random instance of objects (e.g. microwave, trashcan) """
    args = parse_agent_args(config='config_dev.yaml', seed=None)
    init_pybullet_client(args)

    world = create_world(args)
    robot = build_robot_from_args(world, robot_name='feg', initial_q=[1.5, 7, 0.7, 0, -PI / 2, 0])

    surfaces = {
        'counter': {
            'front_right_stove': ['BraiserBody'],
            'hitman_tmp': ['Microwave'],  ## counter on the left
            'indigo_tmp': [],  ## counter on the right ## 'BraiserLid'
        }
    }
    floor = load_kitchen_floor_plan(world, plan_name='kitchen_v2.svg', surfaces=surfaces, random_instance=random_instance)
    world.remove_object(floor)
    set_camera_pose((3, 7, 2.5), (0, 7, 0.5))
    wait_if_gui()


def test_load_spot_in_office():
    """ have bug now with loading spot """
    world = get_test_world(robot='spot', semantic_world=True, width=1980, height=1238)
    load_floor_plan(world, plan_name='office_1.svg', debug=True, spaces=None, surfaces=None,
                    random_instance=False, verbose=True)
    wait_if_gui()


if __name__ == '__main__':
    test_load_floating_gripper_in_kitchen()
    # test_load_spot_in_office()
