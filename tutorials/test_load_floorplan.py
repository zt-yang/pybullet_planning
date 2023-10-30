from pybullet_tools.utils import wait_unlocked
from world_builder.loaders import load_floor_plan
from tutorials.test_utils import get_test_world


def test_load_floorplan(plan_name):
    world = get_test_world(robot='spot', semantic_world=True, width=1980, height=1238)
    load_floor_plan(world, plan_name=plan_name, DEBUG=True, spaces=None, surfaces=None,
                    RANDOM_INSTANCE=False, verbose=True)
    wait_unlocked()


if __name__ == '__main__':
    for plan_name in ['office_1.svg']:
        test_load_floorplan(plan_name)