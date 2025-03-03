import math
import time
import numpy as np

from pybullet_tools.utils import connect, draw_pose, unit_pose, link_from_name, load_pybullet, load_model, \
    sample_aabb, AABB, set_pose, quat_from_euler, HideOutput, get_aabb_extent, unit_quat, remove_body, \
    set_camera_pose, wait_unlocked, disconnect, wait_if_gui, create_box, get_aabb, get_pose, draw_aabb, multiply, \
    Pose, get_link_pose, get_joint_limits, WHITE, RGBA, set_all_color, RED, GREEN, set_renderer, add_text, \
    Point, set_random_seed, set_numpy_seed, reset_simulation, joint_from_name, PI, wait_for_user, \
    get_joint_name, get_link_name, dump_joint, INF, ConfSaver, pairwise_link_collision, LockRenderer
from pybullet_tools.bullet_utils import nice, \
    draw_fitted_box, open_joint, dump_json
from pybullet_tools.pose_utils import sample_random_pose
from pybullet_tools.camera_utils import set_camera_target_body

from robot_builder.composed_robot import ComposedRobot

from pybullet_tools.tracik import IKSolver


## -------------------------------------------------------------------------


def test_squatting_down(robot, lower, upper):
    for k in np.linspace(lower, upper, 20):
        robot.set_base_conf([0, 0, float(k), 0])
        time.sleep(0.1)
    wait_for_user()


def test_whole_body_ik(robot, tool_link, box_range_aabb=None, first_joint='torso_lift_joint',
                       box=None, grasp_pose=None, given_box_pose=None, verbose=False,
                       num_success=INF, max_tries=20):
    if box is None or grasp_pose is None:
        box = create_box(0.05, 0.05, 0.075, color=(1, 0, 0, 1))
        grasp_pose = ((0, 0, 0.2), quat_from_euler((0, math.pi/2, 0)))

    body_solver = IKSolver(robot.body, tool_link=tool_link, first_joint=first_joint,
                           custom_limits=robot.custom_limits)

    joint_state, leg_conf = None, None

    while num_success > 0 and max_tries > 0:
        box_pose = sample_random_pose(box_range_aabb) if given_box_pose is None else given_box_pose
        gripper_pose = multiply(box_pose, grasp_pose)
        # print(box_pose)
        if verbose:
            print(f'\ngripper_pose = {nice(gripper_pose)}')

        if isinstance(robot, ComposedRobot):
            set_renderer(False)
            robot.open_arms()
        for conf in body_solver.generate(gripper_pose):
            if conf is None:
                break
            joint_state = dict(zip(body_solver.joints, conf))
            joint_values = {}
            for i, value in joint_state.items():
                if i == 0:
                    continue
                joint_name = get_joint_name(robot.body, i)
                joint_values[i] = (joint_name, value)

            with ConfSaver(robot.body):
                body_solver.set_conf(conf)
                collided = robot.check_arm_body_collisions()
            if collided:
                if verbose:
                    print('\n\n self-collision!')
                break

            torso_lift_joint = joint_from_name(robot.body, 'torso_lift_joint')
            leg_conf = robot.solve_leg_conf(joint_state[torso_lift_joint], verbose=False)
            if leg_conf is None:
                if verbose:
                    print('\n\n failed leg ik!')
                break

            if verbose:
                for i in range(len(leg_conf.values)):
                    index = leg_conf.joints[i]
                    value = leg_conf.values[i]
                    joint_values[index] = (get_joint_name(robot.body, index), value)
                joint_values = dict(sorted(joint_values.items()))
                for i, (joint_name, value) in joint_values.items():
                    print('\t', i, '\t', joint_name, '\t', round(value, 3))

            with LockRenderer():
                set_pose(box, box_pose)
                body_solver.set_conf(conf)
                leg_conf.assign()
                if isinstance(robot, ComposedRobot):
                    robot.assign_attachments()
                # set_camera_target_body(box, distance=1)
            set_renderer(True)

            # wait_for_user()

            # print('success')
            num_success -= 1
            break
        max_tries -= 1
    return joint_state, leg_conf


# def test_reachability(robot):
#     world = get_test_world(robot=robot, semantic_world=True, custom_limits=((-4, -4), (4, 4)))
#     robot = world.robot
#     state = State(world, grasp_types=robot.grasp_types)
#
#     for w, xy in [(0.3, (0, 0)), (0.5, (2, 2))]:
#         table1 = create_table(world, w=w, xy=xy)
#         movable1 = create_movable(world, table1, xy=xy)
#         result = robot.check_reachability(movable1, state)
#         print('w', w, result)
#
#     wait_unlocked()
