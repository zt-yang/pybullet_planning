from __future__ import print_function
import config

from pybullet_tools.utils import connect, draw_pose, unit_pose, link_from_name, load_pybullet, load_model, \
    sample_aabb, AABB, set_pose, quat_from_euler, HideOutput, get_aabb_extent, unit_quat, remove_body, \
    set_camera_pose, wait_unlocked, disconnect, wait_if_gui, create_box, get_aabb, get_pose, draw_aabb
from pybullet_tools.bullet_utils import nice, set_camera_target_body
from world_builder.loaders import create_house_floor, create_table, create_movable
from world_builder.loaders_partnet_kitchen import sample_kitchen_sink, sample_full_kitchen

from tutorials.test_utils import get_test_world


def test_sink_configuration(robot, pause=False):
    world = get_test_world(robot=robot, semantic_world=True)
    floor = create_house_floor(world, w=10, l=10, x=5, y=5)
    target = None
    for x in range(1, 5):
        for y in range(1, 5):
            base = sample_kitchen_sink(world, floor=floor, x=2*x, y=2*y)[1]
            mx, my, mz = base.aabb().upper
            ny = base.aabb().lower[1]
            aabb = AABB(lower=(mx - 0.3, ny, mz), upper=(mx, my, mz + 0.1))
            draw_aabb(aabb)
            if x == 4 and y == 3:
                target = base
            if pause:
                set_camera_target_body(base, dx=0.1, dy=0, dz=1.5)
                wait_unlocked()
    set_camera_target_body(target, dx=2, dy=0, dz=4)
    wait_unlocked()


def test_kitchen_configuration(robot):
    world = get_test_world(robot=robot, semantic_world=True, initial_xy=(1.5, 4))
    sample_full_kitchen(world)


if __name__ == '__main__':
    robot = None

    """ --- procedurally generated kitchen counters --- """
    test_sink_configuration(robot, pause=True)
    # test_kitchen_configuration(robot)
