from __future__ import print_function

import os
import sys
import time
import numpy as np
from pprint import pprint

from pybullet_tools.pr2_streams import get_pull_door_handle_motion_gen as get_pull_drawer_handle_motion_gen
from pybullet_tools.pr2_streams import get_pull_door_handle_motion_gen as get_turn_knob_handle_motion_gen
from pybullet_tools.pr2_streams import get_stable_gen, Position, get_pose_in_space_test, \
    get_marker_grasp_gen, get_bconf_in_region_test, get_pull_door_handle_motion_gen, \
    get_pull_marker_random_motion_gen, get_ik_ungrasp_gen, get_pose_in_region_test, sample_joint_position_gen

from pybullet_tools.pr2_primitives import get_group_joints, get_base_custom_limits, Pose, Conf, \
    get_ik_ir_gen, move_cost_fn, Attach, Detach, Clean, Cook, \
    get_gripper_joints, GripperCommand, Simultaneous, create_trajectory
from pybullet_tools.general_streams import get_grasp_list_gen, get_contain_list_gen, get_handle_grasp_list_gen, \
    get_handle_grasp_gen, get_compute_pose_kin, sample_joint_position_closed_gen
from pybullet_tools.bullet_utils import set_camera_target_body, \
    nice, BASE_LIMITS, initialize_collision_logs, collided
from pybullet_tools.pr2_problems import create_pr2
from pybullet_tools.pr2_utils import create_gripper, set_group_conf
from pybullet_tools.utils import get_client, \
    Pose, get_bodies, pairwise_collision, get_pose, point_from_pose, set_renderer, get_joint_name, \
    remove_body, LockRenderer, WorldSaver, wait_if_gui, SEPARATOR, safe_remove, ensure_dir, \
    get_distance, get_max_limit, BROWN, BLUE, WHITE, TAN, GREY, YELLOW, GREEN, BLACK, RED, CLIENTS, wait_unlocked
from pybullet_tools.flying_gripper_utils import get_se3_joints, get_cloned_se3_conf

from pddlstream.language.constants import AND

from world_builder.entities import Object


def process_debug_goals(state, goals, init):
    if isinstance(goals, tuple):
        test, args = goals
        if test == 'test_handle_grasps':
            goals = test_handle_grasps(state, args)
        elif test == 'test_grasps':
            goals = test_grasps(state, args)
        elif test == 'test_grasp_ik':
            goals = test_grasp_ik(state, init, args)
        elif test == 'test_pose_gen':
            goals, ff = test_pose_gen(state, init, args)
            init += ff
        elif test == 'test_relpose_inside_gen':
            goals, ff = test_relpose_inside_gen(state, init, args)
            init += ff
        elif test == 'test_joint_closed':
            goals = test_joint_closed(state, init, args)
        elif test == 'test_door_pull_traj':
            goals = test_door_pull_traj(state, init, args)
        elif test == 'test_reachable_pose':
            goals = test_reachable_pose(state, init, args)
        elif test == 'test_at_reachable_pose':
            goals = test_at_reachable_pose(init, args)
        elif test == 'test_pose_kin':
            goals = test_rel_to_world_pose(init, args)
        else:
            # test_initial_region(state, init)
            # test_marker_pull_bconfs(state, init)
            # test_pulling_handle_ik(state)
            # test_drawer_open(state, goals)
            print('\n\n\npr2_agent.pddlstream_from_state_goal | didnt implement', goals)
            sys.exit()

    goal = [AND]
    goal += goals
    if len(goals) > 0:
        if goals[0][0] == 'AtBConf':
            init += [('BConf', goals[0][1])]
        elif goals[0][0] == 'AtSEConf':
            init += [('SEConf', goals[0][1])]
        elif goals[0][0] == 'AtPosition':
            init += [('Position', goals[0][1], goals[0][2]), ('IsOpenedPosition', goals[0][1], goals[0][2])]
        elif goals[0][0] == 'AtGrasp':
            init += [('Grasp', goals[0][2], goals[0][3])]
        elif goals[0][0] == 'AtHandleGrasp':
            init += [('HandleGrasp', goals[0][2], goals[0][3])]
        elif goals[0][0] == 'AtMarkerGrasp':
            init += [('MarkerGrasp', goals[0][2], goals[0][3])]

        if goal[-1] == ("not", ("AtBConf", "")):
            atbconf = [i for i in init if i[0].lower() == "AtBConf".lower()][0]
            goal[-1] = ("not", atbconf)

    return goal


###############################################################################


def test_initial_region(state, init):
    world = state.world
    if world.name_to_body('hallway') is not None:
        robot = state.robot
        marker = world.name_to_body('marker')
        funk = get_bconf_in_region_test(robot)
        funk2 = get_pose_in_region_test(robot)
        bq = [i for i in init if i[0].lower() == "AtBConf".lower()][0][-1]
        p = [i for i in init if i[0].lower() == "AtPose".lower() and i[1] == marker][0][-1]

        for location in ['hallway', 'storage', 'kitchen']:
            answer1 = funk(bq, world.name_to_body(location))
            answer2 = funk2(marker, p, world.name_to_body(location))
            print(f"RobInRoom({location}) = {answer1}\tInRoom({marker}, {location}) = {answer2}")
        print('---------------\n')


def test_marker_pull_bconfs(state, init):
    world = state.world
    funk = get_pull_marker_random_motion_gen(state)  ## is a from_fn
    o = world.name_to_body('marker')
    p1 = [i for i in init if i[0].lower() == "AtPose".lower() and i[1] == o][0][2]
    g = test_marker_pull_grasps(state, o)
    bq1 = [i for i in init if i[0].lower() == "AtBConf".lower()][0][1]
    p2, bq2, t = funk('left', o, p1, g, bq1)
    rbb = create_pr2()
    set_group_conf(rbb, 'base', bq2.values)


def test_marker_pull_grasps(state, marker, visualize=False):
    funk = get_marker_grasp_gen(state)
    grasps = funk(marker) ## funk(cart) ## is a previous version
    if visualize:
        robot = state.robot
        cart = state.world.BODY_TO_OBJECT[marker].grasp_parent
        for grasp in grasps:
            gripper_grasp = robot.visualize_grasp(get_pose(marker), grasp[0].value)
            set_camera_target_body(gripper_grasp, dx=0, dy=-1, dz=0)
            print('collision with marker', pairwise_collision(gripper_grasp, marker))
            print('collision with cart', pairwise_collision(gripper_grasp, cart))
            remove_body(gripper_grasp)
    print('test_marker_pull_grasps:', grasps)
    return grasps


def test_handle_grasps(state, name='hitman_drawer_top_joint', visualize=False, verbose=False):
    if isinstance(name, str):
        body_joint = state.world.name_to_body(name)
    elif isinstance(name, tuple):
        body_joint = name
        name = state.world.BODY_TO_OBJECT[body_joint].name
    else:
        raise NotImplementedError(name)

    name_to_object = state.world.name_to_object
    funk = get_handle_grasp_list_gen(state, num_samples=24, visualize=visualize, verbose=verbose)
    outputs = funk(body_joint)
    if visualize:
        body_pose = name_to_object(name).get_handle_pose()
        visualize_grasps_by_quat(state, outputs, body_pose, verbose=verbose)
    print(f'test_handle_grasps ({len(outputs)}): {outputs}')
    arm = state.robot.arms[0]
    goals = [("AtHandleGrasp", arm, body_joint, outputs[0][0])]
    return goals


def test_grasps(state, name='cabbage', visualize=True):
    set_renderer(True)
    title = 'pr2_agent.test_grasps | '
    if isinstance(name, str):
        body = state.world.name_to_body(name)
    else: ## if isinstance(name, Object):
        body = name
    robot = state.robot

    funk = get_grasp_list_gen(state, verbose=True, visualize=True, RETAIN_ALL=False, top_grasp_tolerance=None)  ## PI/4
    outputs = funk(body)

    if 'left' in robot.joint_groups:
        if visualize:
            body_pose = get_pose(body)
            print('body_pose', nice(body_pose))
            visualize_grasps(state, outputs, body_pose)
        print(f'{title}grasps:', outputs)
        goals = [("AtGrasp", 'left', body, outputs[0][0])]

    elif 'hand' in robot.joint_groups:
        from pybullet_tools.bullet_utils import collided

        g = outputs[0][0]
        gripper = robot.visualize_grasp(get_pose(g.body), g.approach, body=g.body, width=g.grasp_width)
        gripper_conf = get_cloned_se3_conf(robot, gripper)
        goals = [("AtGrasp", 'hand', body, g)]
        if visualize:
            set_renderer(True)
            goals = [("AtSEConf", Conf(robot, get_se3_joints(robot), gripper_conf))]
            # body_collided = len(get_closest_points(gripper, 2)) != 0
            # body_collided_2 = collided(gripper, [2], world=state.world, verbose=True, tag=title)
            # print(f'{title}collision between gripper {gripper} and object {g.body}: {body_collided}')
        else:
            remove_body(gripper)
    else:
        raise NotImplementedError(robot.joint_groups)
    return goals


def visualize_grasps(state, outputs, body_pose, RETAIN_ALL=False, collisions=False, pause_each=False,
                     TEST_ATTACHMENT=False):
    robot = state.robot
    colors = [BROWN, BLUE, WHITE, TAN, GREY, YELLOW, GREEN, BLACK, RED]
    color_names = ['BROWN', 'BLUE', 'WHITE', 'TAN', 'GREY', 'YELLOW', 'GREEN', 'BLACK', 'RED']

    def visualize_grasp(grasp, index=0):
        w = grasp.grasp_width
        if RETAIN_ALL:
            idx = index % len(colors)
            print(' grasp.value', nice(grasp.value), 'color', color_names[idx])
            gripper_grasp = robot.visualize_grasp(body_pose, grasp.value, body=grasp.body,
                                                  color=colors[idx], width=w, new_gripper=True)
            if collisions and collided(gripper_grasp, state.obstacles, verbose=True):
                remove_body(gripper_grasp)
                return None
            # set_camera_target_body(gripper_grasp, dx=0.5, dy=0.5, dz=0.5)
        else:
            gripper_grasp = robot.visualize_grasp(body_pose, grasp.value, body=grasp.body, color=GREEN, width=w)
            gripper_approach = robot.visualize_grasp(body_pose, grasp.approach, color=BROWN)

            if TEST_ATTACHMENT:
                set_renderer(True)
                set_camera_target_body(gripper_grasp, dx=0.5, dy=0.5, dz=0.5)
                attachment = grasp.get_attachment(robot, robot.arms[0], visualize=True)

            # set_camera_target_body(gripper_approach, dx=0, dy=-1, dz=0)
            remove_body(gripper_grasp)
            remove_body(gripper_approach)
            # weiyu: debug
            # robot.remove_grippers()
            return None

        return gripper_grasp

    # if not isinstance(outputs, types.GeneratorType):
    #     for i in range(len(outputs)):
    #         visualize_grasp(outputs[i][0], index=i)
    # else:

    i = 0
    gripper_grasp = None
    for grasp in outputs:
        output = visualize_grasp(grasp[0], index=i)
        if output is not None:
            gripper_grasp = output
            i += 1
        if pause_each:
            wait_if_gui()
    if i > 0:
        set_camera_target_body(gripper_grasp, dx=0.5, dy=0.5, dz=0.5)

    # if RETAIN_ALL:
    #     wait_if_gui()


def visualize_grasps_by_quat(state, outputs, body_pose, verbose=False):
    robot = state.robot
    colors = [BROWN, BLUE, WHITE, TAN, GREY, YELLOW, GREEN, BLACK, RED]
    all_grasps = {}
    for i in range(len(outputs)):
        grasp = outputs[i][0]
        quat = nice(grasp.value[1])
        if quat not in all_grasps:
            all_grasps[quat] = []
        all_grasps[quat].append(grasp)

    j = 0
    set_renderer(True)
    for k, v in all_grasps.items():
        print(f'{len(v)} grasps of quat {k}')
        visuals = []
        color = colors[j%len(colors)]
        for grasp in v:
            gripper = robot.visualize_grasp(body_pose, grasp.value, color=color, verbose=verbose,
                                            width=grasp.grasp_width, body=grasp.body)
            visuals.append(gripper)
        j += 1
        wait_if_gui()
        # set_camera_target_body(gripper, dx=0, dy=0, dz=1)
        for visual in visuals:
            remove_body(visual)


def test_grasp_ik(state, init, name='cabbage', visualize=True):
    goals = test_grasps(state, name, visualize=False)
    body, grasp = goals[0][-2:]
    robot = state.robot
    custom_limits = robot.get_custom_limits()
    pose = [i for i in init if i[0].lower() == "AtPose".lower() and i[1] == body][0][-1]

    funk = get_ik_ir_gen(state, verbose=visualize, custom_limits=custom_limits
                         )('left', body, pose, grasp)
    print('test_grasp_ik', body, pose, grasp)
    next(funk)

    return goals


def test_pulling_handle_ik(problem):
    name_to_body = problem.name_to_body

    # ## --------- test pulling handle ik
    # for bq in [(1.583, 6.732, 0.848), (1.379, 7.698, -2.74), (1.564, 7.096, 2.781)]:
    #     custom_limits = get_base_custom_limits(problem.robot, BASE_LIMITS)
    #     robot = problem.robot
    #     drawer = name_to_body('hitman_drawer_top_joint')
    #     funk = get_handle_grasp_gen(problem)
    #     grasp = funk(drawer)[0][0]
    #     funk = get_pull_handle_motion_gen(problem)
    #     position1 = Position(name_to_body('hitman_drawer_top_joint'))
    #     position2 = Position(name_to_body('hitman_drawer_top_joint'), 'max')
    #     bq1 = Conf(robot, get_group_joints(robot, 'base'), bq)
    #     t = funk('left', drawer, position1, position2, grasp, bq1)

    ## --------- test modified pulling handle ik
    for bq in [(1.583, 6.732, 0.848), (1.379, 7.698, -2.74), (1.564, 7.096, 2.781)]:
        custom_limits = get_base_custom_limits(problem.robot, BASE_LIMITS)
        robot = problem.robot
        drawer = name_to_body('hitman_drawer_top_joint')
        funk = get_handle_grasp_gen(problem)
        grasp = funk(drawer)[0][0]
        funk = get_pull_drawer_handle_motion_gen(problem, extent='max')
        position1 = Position(name_to_body('hitman_drawer_top_joint'))
        position2 = Position(name_to_body('hitman_drawer_top_joint'), 'max')
        bq1 = Conf(robot, get_group_joints(robot, 'base'), bq)
        t = funk('left', drawer, grasp, bq1)
        break


def test_pose_gen(problem, init, args):
    o, s = args
    pose = [i for i in init if i[0].lower() == "AtPose".lower() and i[1] == o][0][-1]
    if isinstance(o, Object):
        o = o.body
    funk = get_stable_gen(problem)(o, s)
    p = next(funk)[0]
    print(f'test_pose_gen({o}, {s}) | {p}')
    pose.assign()
    return [('AtPose', o, p)], [('Pose', o, p), ('Supported', o, p, s)]


def test_relpose_inside_gen(problem, init, args):
    o, s = args
    pose = [i for i in init if i[0].lower() == "AtPose".lower() and i[1] == o][0][-1]
    if isinstance(o, Object):
        o = o.body
    outputs = get_contain_list_gen(problem, relpose=True)(o, s)
    p = outputs[0][0]
    print(f'test_relpose_inside_gen({o}, {s}) | {p}')
    pose.assign()
    return [('AtRelPose', o, p, s)], [('RelPose', o, p, s)]


def test_joint_closed(problem, init, o):
    pstn = [i for i in init if i[0].lower() == "AtPosition".lower() and i[1] == o][0][-1]
    funk = sample_joint_position_gen(closed=True)
    funk = sample_joint_position_closed_gen()
    goal = []
    for (pstn1,) in funk(o, pstn):
        print(pstn1)
        goal = [('AtPosition', o, pstn1)]
    return goal


def test_door_pull_traj(problem, init, o):
    pst1 = [f[2] for f in init if f[0].lower() == 'atposition' and f[1] == o][0]
    funk1 = sample_joint_position_gen()
    pst2 = funk1(o, pst1)[0][0]

    funk2 = get_handle_grasp_gen(problem, visualize=True)
    # g = funk2(o)[0][0]
    grasps = funk2(o)

    q1 = [f[1] for f in init if f[0].lower() == 'atseconf'][0]
    funk3 = get_pull_door_handle_motion_gen(problem, visualize=False, verbose=False)
    for i in range(len(grasps)):
        (g,) = grasps[i]
        print(f'\n!!!! pr2_agent.test_door_pull_traj | grasp {i}: {nice(g.value)}')
        result = funk3('hand', o, pst1, pst2, g, q1)
        if result != None:
            [q2, cmd] = result
            return [("AtPosition", o, pst2)]
    print('\n\n!!!! cant find any handle grasp that works for', o)
    sys.exit()


def test_reachable_pose(state, init, o):
    from pybullet_tools.flying_gripper_utils import get_reachable_test
    robot = state.robot
    funk = get_reachable_test(state, custom_limits=robot.custom_limits)
    p = [f[2] for f in init if f[0].lower() == "AtPose".lower() and f[1] == o][0]
    q = [f[1] for f in init if f[0].lower() == 'AtSEConf'.lower()][0]

    outputs = get_grasp_list_gen(state)(o)
    for (g,) in outputs:
        result = funk(o, p, g, q)
        if result: return True
    return False


def test_at_reachable_pose(init, o):
    p = [f[2] for f in init if f[0].lower() == "AtPose".lower() and f[1] == o][0]
    return [('AtReachablePose', o, p)]


def test_rel_to_world_pose(init, args):
    attached, supporter = args
    world = '@world'
    funk = get_compute_pose_kin()
    print(f'test_rel_to_world_pose({attached}, {supporter})')

    rp = [f[2] for f in init if f[0].lower() == "AtRelPose".lower() and f[1] == supporter and f[-1] == world][0]
    p2 = [f[2] for f in init if f[0].lower() == "AtPose".lower() and f[1] == world][0]
    p1 = funk(supporter, rp, world, p2)[0]
    print(p1)

    rp = [f[2] for f in init if f[0].lower() == "AtRelPose".lower() and f[1] == attached and f[-1] == supporter][0]
    p1 = funk(attached, rp, supporter, p1)
    print(p1)
    return [('Holding', 'left', attached)]
