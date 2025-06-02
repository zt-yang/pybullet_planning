from __future__ import print_function

import os
import sys
import time
import numpy as np
from pprint import pprint

from pybullet_tools.mobile_streams import get_pull_door_handle_motion_gen as get_pull_drawer_handle_motion_gen
from pybullet_tools.mobile_streams import get_pull_door_handle_motion_gen as get_turn_knob_handle_motion_gen
from pybullet_tools.mobile_streams import get_pull_door_handle_motion_gen, get_ik_ungrasp_gen
from pybullet_tools.pr2_streams import get_stable_gen, Position, get_pose_in_space_test, \
    get_marker_grasp_gen, get_bconf_in_region_test, \
    get_pull_marker_random_motion_gen, get_pose_in_region_test, sample_joint_position_gen

from pybullet_tools.pr2_primitives import get_group_joints, get_base_custom_limits, Pose, Conf, \
    get_ik_ir_gen, move_cost_fn, Attach, Detach, Clean, Cook, \
    get_gripper_joints, GripperCommand, Simultaneous, create_trajectory
from pybullet_tools.general_streams import get_grasp_list_gen, get_contain_list_gen, get_handle_grasp_list_gen, \
    get_handle_grasp_gen, get_compute_pose_kin, sample_joint_position_closed_gen, get_contain_gen, \
    get_stable_list_gen, get_above_pose_gen, get_nudge_grasp_list_gen
from pybullet_tools.grasp_utils import enumerate_rotational_matrices, \
    enumerate_translation_matrices, test_transformations_template

from pybullet_tools.bullet_utils import colors, color_names, \
    nice, BASE_LIMITS, initialize_collision_logs, collided
from pybullet_tools.camera_utils import set_camera_target_body
from pybullet_tools.pr2_problems import create_pr2
from pybullet_tools.pr2_utils import create_pr2_gripper, set_group_conf
from pybullet_tools.utils import get_client, quat_from_euler, remove_handles, PI, wait_for_user, \
    Pose, get_bodies, pairwise_collision, get_pose, point_from_pose, set_renderer, get_joint_name, \
    remove_body, LockRenderer, WorldSaver, wait_if_gui, SEPARATOR, safe_remove, ensure_dir, \
    get_distance, get_max_limit, BROWN, BLUE, WHITE, TAN, GREY, YELLOW, GREEN, BLACK, RED, CLIENTS, wait_unlocked
from pybullet_tools.flying_gripper_utils import get_se3_joints, get_cloned_se3_conf

from pddlstream.language.constants import AND

from world_builder.entities import Object
from world_builder.actions import PULL_UNTIL, NUDGE_UNTIL


def process_debug_goals(state, goals, init):
    if isinstance(goals, tuple):
        test, args = goals
        ff = []
        if not isinstance(test, str):
            goals = test(state, args)
        elif test == 'test_handle_grasps':
            goals = test_handle_grasps(state, args)
        elif test == 'test_object_grasps':
            goals = test_object_grasps(state, args)
        elif test == 'test_nudge_grasps':
            goals = test_nudge_grasps(state, args)
        elif test == 'test_nudge_back_grasps':
            goals = test_nudge_grasps(state, args, nudge_back=True)
        elif test == 'test_loaded_grasp_offset':
            goals = test_loaded_grasp_offset(state, args)
        elif test == 'test_grasp_ik':
            goals = test_grasp_ik(state, init, args)
        elif test == 'test_pose_gen':
            goals, ff = test_pose_gen(state, init, args)
        elif test == 'test_pose_inside_gen':
            goals, ff = test_pose_inside_gen(state, init, args)
        elif test == 'test_relpose_inside_gen':
            goals, ff = test_relpose_inside_gen(state, init, args)
        elif test == 'test_pose_above_gen':
            goals = test_pose_above_gen(state, init, args)
        elif test == 'test_joint_open':
            goals, ff = test_joint_open(init, args)
        elif test == 'test_joint_closed':
            goals, ff = test_joint_closed(init, args)
        elif test == 'test_pull_nudge_joint_positions':
            goals = test_pull_nudge_joint_positions(state, args)
        elif test == 'test_door_pull_traj':
            goals = test_door_pull_traj(state, init, args)
        elif test == 'test_reachable_bconf':
            goals, ff = test_reachable_bconf(state, init, args)
        elif test == 'test_reachable_pose':
            goals = test_reachable_pose(state, init, args)
        elif test == 'test_at_reachable_pose':
            goals = test_at_reachable_pose(init, args)
        elif test == 'test_pose_kin':
            goals = test_rel_to_world_pose(init, args)
        elif test == 'test_marker_grasp':
            goals = test_marker_grasp(state, init, args)
        elif test == 'test_marker_pull':
            goals = test_marker_pull(state, init, args)
        else:
            # test_initial_region(state, init)
            # test_pulling_handle_ik(state)
            # test_drawer_open(state, goals)
            print('\n\n\nstream_tests.pddlstream_from_state_goal | didnt implement', goals)
            sys.exit()
        init += ff

    goal = [AND]
    goal += goals
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


def test_marker_pull(state, init, o, visualize=False):
    p1 = [i for i in init if i[0].lower() == "AtPose".lower() and i[1] == o][0][2]
    grasps = get_marker_grasps(state, o, visualize=visualize)
    grasp = grasps[-1][0]

    funk = get_pull_marker_random_motion_gen(state)
    bq1 = [i for i in init if i[0].lower() == "AtBConf".lower()][0][1]
    p2, bq2, t = funk('left', o, p1, grasp, bq1)
    rbb = create_pr2()
    set_group_conf(rbb, 'base', bq2.values)



def test_marker_grasp(state, init, o, visualize=False):
    p1 = [i for i in init if i[0].lower() == "AtPose".lower() and i[1] == o][0][2]
    grasps = get_marker_grasps(state, o, visualize=visualize)
    grasp = grasps[-1][0]

    # funk = get_pull_marker_random_motion_gen(state)
    # bq1 = [i for i in init if i[0].lower() == "AtBConf".lower()][0][1]
    # p2, bq2, t = funk('left', o, p1, grasp, bq1)
    # rbb = create_pr2()
    # set_group_conf(rbb, 'base', bq2.values)
    goals = [("AtMarkerGrasp", 'left', o, grasp)]
    return goals


def get_marker_grasps(state, marker, visualize=False):
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
    return grasps


def test_handle_grasps(state, arg='hitman_drawer_top_joint', visualize=False, verbose=False):
    """ visualize both grasp pose and approach pose """
    body_joint, name = get_body_joint_and_name(state, arg)

    name_to_object = state.world.name_to_object
    funk = get_handle_grasp_list_gen(state, num_samples=24, visualize=True, retain_all=True,
                                     verbose=verbose)
    outputs = funk(body_joint)
    if visualize:
        body_pose = name_to_object(name).get_handle_pose()
        visualize_grasps_by_quat(state, outputs, body_pose, verbose=verbose)
    print(f'test_handle_grasps ({len(outputs)}): {outputs}')
    arm = state.robot.arms[0]
    goals = [("AtHandleGrasp", arm, body_joint, outputs[0][0])]
    return goals


def get_body_joint_and_name(state, name):
    if isinstance(name, str):
        body_joint = state.world.name_to_body(name)
    elif isinstance(name, tuple):
        body_joint = name
        name = state.world.BODY_TO_OBJECT[body_joint].name
    else:
        raise NotImplementedError(name)
    return body_joint, name


def test_nudge_grasps(state, arg='chewie_door_right_joint', nudge_back=False,
                      visualize=False, retain_all=False, verbose=False):
    body_joint, name = get_body_joint_and_name(state, arg)

    name_to_object = state.world.name_to_object
    funk = get_nudge_grasp_list_gen(state, nudge_back=nudge_back, num_samples=2,
                                    visualize=False, retain_all=retain_all, verbose=verbose)
    outputs = funk(body_joint)
    if visualize:
        body_pose = name_to_object(name).get_handle_pose()
        visualize_grasps_by_quat(state, outputs, body_pose, verbose=verbose)
    print(f'test_nudge_grasps({nudge_back}) -> generated len = {len(outputs)}: {outputs}')
    pred = "NudgedDoor" if not nudge_back else "NudgedBackDoor"
    goals = [(pred, body_joint)]
    return goals


def test_pull_nudge_joint_positions(state, joints=None):
    funk_pull = sample_joint_position_gen(num_samples=2, p_max=PULL_UNTIL)
    funk_nudge = sample_joint_position_gen(num_samples=2, p_max=NUDGE_UNTIL)
    world = state.world

    if joints is None:
        joints = world.cat_to_bodies('door', get_all=True)

    for joint in joints:
        pstn1 = Position(joint)
        name = world.get_debug_name(joint)
        print(f'\njoint {name}')
        for (pstn2,) in funk_pull(joint, pstn1):
            print(f'\tfunk_pull({pstn1}) -> {pstn2}')
            for (pstn3,) in funk_nudge(joint, pstn2):
                print(f'\t\tfunk_nudge({pstn2}) -> {pstn3}')
    sys.exit()


def test_loaded_grasp_offset(state, args, test_translations=False, test_rotations=False,
                             show_debug_triads=True):
    rotations = [(0, 0, -1.571), (0, 0, 1.571), (0, 0, 0), (0, 0, 3.142)]  ## , skip_until=22
    rotations = [(0, 0, 0)]  ## tested on bananas
    translations = [(0, 0, 0)]
    translations = [(-0.01, 0, -0.03), (0, -0.01, -0.03)]  ## (0, 0, -0.03),
    skip_until = None  ## None

    if test_translations:
        translations = enumerate_translation_matrices(x=0.03, negations=True)
    if test_rotations:
        rotations = enumerate_rotational_matrices(return_list=True)

    def funk(t, r):
        offset = (t, quat_from_euler(r))
        remove_handles(state.world.robot.debug_handles)
        test_object_grasps(state, args, visualize=True, debug=False, randomize=False,
                           loaded_offset=offset, debug_triads=show_debug_triads)

    title = 'stream_tests.test_loaded_grasp_offset'
    return test_transformations_template(rotations, translations, funk, title, skip_until=skip_until)


def test_object_grasps(state, name='cabbage', visualize=True, debug=False, debug_triads=False,
                       loaded_offset=None, randomize=True):
    """
    visualize = True:   to see all grasps for selecting the grasp index to plan for
    debug = True:       show the grasp to plan for
    """
    if visualize:
        set_renderer(True)
    title = 'stream_tests.test_grasps | '
    if isinstance(name, tuple):
        name, top_grasp_tolerance = name
    if isinstance(name, str):
        body = state.world.name_to_body(name)
    else:  ## if isinstance(name, Object):
        body = name
    robot = state.robot

    stream_kwargs = state.world.stream_kwargs
    stream_kwargs_here = {k: stream_kwargs[k] for k in ['top_grasp_tolerance', 'use_all_grasps']}

    funk = get_grasp_list_gen(state, verbose=True, visualize=visualize, retain_all=False, randomize=randomize,
                              loaded_offset=loaded_offset, debug=debug_triads, **stream_kwargs_here)
    outputs = funk(body)

    if 'left_gripper' in robot.joint_groups:
        if visualize:
            body_pose = get_pose(body)
            print('body_pose', nice(body_pose))
            with LockRenderer(True):
                all_grippers = visualize_grasps(state, outputs, body_pose)
        print(f'{title}grasps:', outputs)

        k = 2 if 'rummy' in robot.name else 0  ## 2, 5 for top, 8 for side
        if visualize:
            wait_if_gui('all grasps')
            with LockRenderer(True):
                for jj, gripper in enumerate(all_grippers):
                    if jj == k: continue
                    remove_body(gripper)

            ## visualize approach pose
            grasp = outputs[k][0]
            gripper_approach = robot.visualize_grasp(body_pose, grasp.approach, body=grasp.body,
                                                     color=BROWN, new_gripper=True)

            if debug:
                wait_if_gui('chosen grasp')
            remove_body(all_grippers[k])
            remove_body(gripper_approach)
        goals = [("AtGrasp", 'left', body, outputs[k][0])]

    elif 'hand_gripper' in robot.joint_groups:
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


def visualize_grasps(state, outputs, body_pose, retain_all=True, collisions=False, pause_each=False,
                     test_attachment=False, color=None, verbose=False, **kwargs):
    robot = state.robot
    all_grippers = []

    def visualize_grasp(grasp, gripper_color=color, index=0):
        w = grasp.grasp_width
        if retain_all:
            if gripper_color is None:
                idx = index % len(colors)
                print(f' {index}\tgrasp.value', nice(grasp.value), 'color', color_names[idx])
                gripper_color = colors[idx]

            if verbose:
                print(f'\nstream_tests.visualize_grasps | '
                      f'\trobot.visualize_grasp({nice(body_pose)}, ({nice(grasp.value)}):'
                      f'\t{nice(robot.tool_from_hand)}\t', kwargs)

            gripper_grasp = robot.visualize_grasp(body_pose, grasp.value, body=grasp.body,
                                                  color=gripper_color, width=w, new_gripper=True, **kwargs)
            if pause_each:
                wait_if_gui()
            if collisions and collided(gripper_grasp, state.obstacles, verbose=True):
                remove_body(gripper_grasp)
                return None
            # set_camera_target_body(gripper_grasp, dx=0.5, dy=0.5, dz=0.5)
        else:
            gkwargs = dict(body=grasp.body, width=w, new_gripper=True)
            gripper_grasp = robot.visualize_grasp(body_pose, grasp.value, color=GREEN, **gkwargs)
            gripper_approach = robot.visualize_grasp(body_pose, grasp.approach, color=BROWN, **gkwargs)

            if test_attachment:
                set_renderer(True)
                set_camera_target_body(gripper_grasp, dx=0.5, dy=0.5, dz=0.5)
                attachment = grasp.get_attachment(robot, robot.arms[0], visualize=True)
            if pause_each:
                wait_if_gui()

            # set_camera_target_body(gripper_approach, dx=0, dy=-1, dz=0)
            remove_body(gripper_grasp)
            remove_body(gripper_approach)
            return None

        return gripper_grasp

    # if not isinstance(outputs, types.GeneratorType):
    #     for i in range(len(outputs)):
    #         visualize_grasp(outputs[i][0], index=i)
    # else:

    i = 0
    gripper_grasp = None
    for grasp in outputs:
        if isinstance(grasp, tuple):  ## from grasp_gen
            grasp = grasp[0]
        output = visualize_grasp(grasp, index=i)
        if output is not None:
            gripper_grasp = output
            all_grippers.append(gripper_grasp)
            i += 1
    if i > 0:
        set_camera_target_body(gripper_grasp, dx=0.5, dy=0.5, dz=0.5)

    # if retain_all:
    #     wait_if_gui()
    # robot.hide_cloned_grippers()  ## put gripper below floor
    return all_grippers


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
    visuals = []
    for k, v in all_grasps.items():
        print(f'{len(v)} grasps of quat {k}')
        color = colors[j % len(colors)]
        for grasp in v:
            kwargs = dict(body=grasp.body, verbose=verbose, width=grasp.grasp_width, new_gripper=True)
            gripper = robot.visualize_grasp(body_pose, grasp.value, color=color, **kwargs)
            gripper_app = robot.visualize_grasp(body_pose, grasp.approach, color=GREY, **kwargs)
            visuals.extend([gripper, gripper_app])
        j += 1

    wait_for_user('visualize_grasps_by_quat')
    # set_camera_target_body(gripper, dx=0, dy=0, dz=1)
    for visual in visuals:
        remove_body(visual)


## ------------------------------------------------------------------


def test_grasp_ik(state, init, name='cabbage', visualize=True):
    goals = test_object_grasps(state, name, visualize=False)
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


def test_pose_gen(state, init, args, num_samples=30, visualize=True, debug=True):
    if len(args) == 2:
        o, s = args
        just_pick = False
    elif len(args) == 3:
        a, o, s = args
        just_pick = True
    else:
        assert False, 'test_pose_gen | Invalid number of arguments'

    pose = [i for i in init if i[0].lower() == "AtPose".lower() and i[1] == o][0][-1]
    if isinstance(o, Object):
        o = o.body
    results = get_stable_list_gen(state, num_samples=num_samples, visualize=visualize)(o, s)

    p = None
    for (p, ) in results:
        if not debug:
            break
        p.assign()
        time.sleep(0.5)

    if visualize and not debug:
        p.assign()
        wait_unlocked()

    print(f'test_pose_gen({o}, {s}) | {p}')
    # if just_pick:
    #     p.assign()
    #     return [('Holding', a, o)], []  ## failed, can't just change initial state here
    # else:
    pose.assign()
    return [('AtPose', o, p)], [('Pose', o, p), ('Supported', o, p, s)]


def test_pose_inside_gen(problem, init, args):
    o, s = args
    pose = [i for i in init if i[0].lower() == "AtPose".lower() and i[1] == o][0][-1]
    if isinstance(o, Object):
        o = o.body
    funk = get_contain_gen(problem)(o, s)
    p = next(funk)[0]
    print(f'test_pose_inside_gen({o}, {s}) | {p}')
    pose.assign()
    return [('AtPose', o, p)], [('Pose', o, p), ('Contained', o, p, s)]


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


def test_pose_above_gen(problem, init, args):
    obj, region = args
    obj_pose = [i for i in init if i[0].lower() == "AtPose".lower() and i[1] == obj][0][-1]
    region_pose = [i for i in init if i[0].lower() == "AtPose".lower() and i[1] == region][0][-1]
    outputs = get_above_pose_gen(problem)(region, region_pose, obj)
    for (p, ) in outputs:
        print(f'test_pose_above_gen({obj}, {region}) | {p}')
        p.assign()
        set_renderer(True)
        wait_unlocked()
        break
    obj_pose.assign()
    return [['SprinkledTo', obj, region]]


def test_joint_open(init, o):
    pstn1 = [i for i in init if i[0].lower() == "AtPosition".lower() and i[1] == o][0][-1]
    funk = sample_joint_position_gen(visualize=True)
    goal = []
    new_facts = []
    for (pstn2,) in funk(o, pstn1):
        print(pstn2)
        goal = [('AtPosition', o, pstn2)]
        new_facts = [('IsSampledPosition', o, pstn1, pstn2)]
        break
    return goal, new_facts


def test_joint_closed(init, o):
    pstn1 = [i for i in init if i[0].lower() == "AtPosition".lower() and i[1] == o][0][-1]
    funk = sample_joint_position_gen(to_close=True)
    funk = sample_joint_position_closed_gen()
    goal = []
    new_facts = []
    for (pstn2,) in funk(o, pstn1):
        print(pstn2)
        goal = [('AtPosition', o, pstn2)]
        new_facts = [('IsSampledPosition', o, pstn1, pstn2)]
    return goal, new_facts


def test_door_pull_traj(problem, init, o):
    pst1 = [f[2] for f in init if f[0].lower() == 'atposition' and f[1] == o][0]
    funk1 = sample_joint_position_gen()
    for (pst2, ) in funk1(o, pst1):

        funk2 = get_handle_grasp_gen(problem, visualize=True)
        # g = funk2(o)[0][0]
        grasps = funk2(o)

        q1 = [f[1] for f in init if f[0].lower() == 'atseconf'][0]
        funk3 = get_pull_door_handle_motion_gen(problem, visualize=False, verbose=False)
        for i in range(len(grasps)):
            (g,) = grasps[i]
            print(f'\n!!!! stream_tests.test_door_pull_traj | grasp {i}: {nice(g.value)}')
            result = funk3('hand', o, pst1, pst2, g, q1)
            if result is not None:
                [q2, cmd] = result
                return [("AtPosition", o, pst2)]
        print('\n\n!!!! cant find any handle grasp that works for', o)
        break
    sys.exit()


def test_reachable_bconf(state, init, args):
    a, o = args
    p = [f[2] for f in init if f[0].lower() == "AtPose".lower() and f[1] == o][0]
    bq = [f[1] for f in init if f[0].lower() == 'AtBConf'.lower()][0]

    funk = get_grasp_list_gen(state, verbose=True, visualize=True, retain_all=False,
                              top_grasp_tolerance=None)
    outputs = funk(o)
    for (g,) in outputs:
        body_pose = get_pose(o)
        print('body_pose', nice(body_pose))
        visualize_grasps(state, [(g,)], body_pose)
        return [('Reach', a, o, p, g, bq)], [("Grasp", o, g)]
    return None


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

