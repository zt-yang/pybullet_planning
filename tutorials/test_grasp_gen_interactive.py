from __future__ import print_function
import sys
from os.path import join, abspath, dirname, isdir, isfile
from os import listdir, pardir
RD = abspath(join(dirname(__file__), pardir, pardir))
sys.path.extend([join(RD), join(RD, 'pddlstream'), join(RD, 'pybullet_planning'), join(RD, 'lisdf')])


from pybullet_tools.utils import PI, draw_pose, unit_pose, get_pose, set_pose, invert, multiply, \
    quat_from_euler
from pybullet_tools.camera_utils import set_camera_target_body
from pybullet_tools.grasp_utils import get_grasp_db_file, get_hand_grasps
from pybullet_tools.pose_utils import change_pose_interactive
from pybullet_tools.stream_tests import visualize_grasps

from world_builder.world import State
from world_builder.world_utils import load_asset, get_instance_name, get_partnet_doors, draw_body_label

from tutorials.test_utils import get_test_world, get_instances, filter_instances, \
    load_model_instance, get_model_path, get_y_gap


def run_interactive_grasp_gen(robot='feg', categories=[], given_instances=None, **kwargs):
    """ load an object and generate grasps interactively
    pip install pynput
    """
    world = get_test_world(robot, initial_q=[0, 0, 0, 0, -PI / 2, 0], **kwargs)
    draw_pose(unit_pose(), length=10)
    robot = world.robot
    problem = State(world, grasp_types=robot.grasp_types)
    gripper = robot.get_gripper(arm=robot.arms[0], visual=True)

    ## setting rotations work for normal movables, but not cloned gripper
    # _, test_obj, _ = load_model_instance('Bottle', '3558', scale=0.15, location=(0, 0, 1))
    # change_pose_interactive(test_obj, name, se3=True)

    for i, cat in enumerate(categories):
        instances = filter_instances(cat, given_instances)
        for id, scale in instances.items():
            if isinstance(id, tuple):
                cat, id = id
            path, body, _ = load_model_instance(cat, id, scale=scale, location=(0, 0))
            name = f'{cat.lower()}#{id}'
            world.add_body(body, name, get_instance_name(abspath(path)))
            draw_body_label(body, id, offset=(0, -0.2, 0.1))
            set_camera_target_body(body, dx=0.5, dy=-0.5, dz=0.5)
            body_pose = get_pose(body)  ## ((0.0, 0.0, 0.5), (0.0, 0.0, 1.0, 0))

            def set_pose_fn(gripper, grasp):
                robot.set_gripper_pose(body_pose, grasp, body=body)

            # wait_if_gui("Start interactive pose generation?")
            init_pose = ((0.0, 0.37, -0.37), (0.0, 0.0, 0.0, 1.0))
            set_pose_fn(gripper, init_pose)
            # grasp = change_pose_interactive(gripper, name, se3=True, set_pose_fn=set_pose_fn, init_pose=init_pose)
            # gripper_pose = ((-0.33, 0.01, 1.011), (0.271, 0.653, 0.653, -0.271))
            grasp = ((0.0, 0.51, -0.33), (0.383, 0.0, 0.0, 0.924))

            grasps = robot.make_grasps('hand', 'hand', body, [grasp], default_w=0.4, collisions=True)
            visualize_grasps(problem, grasps, body_pose, pause_each=True, retain_all=False, test_attachment=False)


if __name__ == '__main__':
    robot = 'feg'  ## 'pr2'
    run_interactive_grasp_gen(robot, ['DinerChair'], given_instances=['100568'])
