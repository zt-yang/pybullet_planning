#!/usr/bin/env python

from __future__ import print_function
import os
from pprint import pprint
import json
import pickle
import shutil
from os import listdir
from os.path import join, abspath, dirname, isdir, isfile, basename
import numpy as np
import sys
from collections import namedtuple

from pybullet_tools.bullet_utils import query_yes_no, has_srl_stream, running_in_pycharm
from pybullet_tools.logging_utils import print_pink, print_green, print_blue
from pybullet_tools.camera_utils import adjust_camera_pose, set_camera_target_body, \
    make_observation_collages
from pybullet_tools.pose_utils import ObjAttachment, has_getch
from pybullet_tools.utils import reset_simulation, VideoSaver, wait_unlocked, get_aabb_center, load_yaml, \
    set_camera_pose, get_aabb_extent, set_camera_pose2, invert, WorldSaver
from pybullet_tools.stream_agent import FAILED
from pybullet_tools.logging_utils import print_debug

from lisdf_tools.lisdf_utils import make_furniture_transparent
from lisdf_tools.lisdf_loader import load_lisdf_pybullet
from lisdf_tools.lisdf_planning import Problem
from lisdf_tools.image_utils import make_composed_image_multiple_episodes, images_to_gif

from world_builder.entities import add_robot_cameras
from world_builder.actions import apply_commands, get_primitive_actions, get_action_name, adapt_action
from world_builder.paths import PBP_PATH

from data_generator.run_utils import copy_dir_for_process, process_all_tasks

from pigi_tools.data_utils import get_plan, get_body_map, get_multiple_solutions, add_to_planning_config, \
    load_planning_config, get_world_aabb, check_unrealistic_placement_z, get_goals

Action = namedtuple('Action', ['name', 'args'])

REPLAY_CONFIG_DEBUG = join(PBP_PATH, 'pigi_tools', 'configs', 'replay_debug.yaml')
GYM_IMAGES_DIR = join(PBP_PATH, '..', 'gym_images')


def get_pkl_run(run_dir, verbose=True):
    if basename(run_dir) == 'collisions.pkl':
        pkl_file = run_dir
        run_dir = dirname(run_dir)
    else:
        pkl_file = 'commands.pkl'
        if run_dir.endswith('.pkl'):
            pkl_file = basename(run_dir)
            run_dir = run_dir[:-len(pkl_file) - 1]
            rerun_dir = basename(run_dir)
            run_dir = run_dir[:-len(rerun_dir) - 1]
            pkl_file = join(rerun_dir, pkl_file)

    exp_dir = copy_dir_for_process(run_dir, tag='replaying', verbose=verbose)
    if basename(pkl_file) == 'collisions.pkl':
        plan = None
    elif basename(pkl_file) != 'commands.pkl':
        plan_json = join(run_dir, pkl_file).replace('commands', 'plan').replace('.pkl', '.json')
        plan = get_plan(run_dir, plan_json=plan_json)
    else:
        ## if there are reran versions
        plan = get_plan(run_dir, skip_multiple_plans=True)
    commands = pickle.load(open(join(exp_dir, pkl_file), "rb"))
    return exp_dir, run_dir, commands, plan


def check_if_exist_rerun(run_dir, world, commands, plan):
    indices = world.get_indices()
    multiple_solutions = get_multiple_solutions(run_dir, indices=indices, commands_too=True)
    if len(multiple_solutions) > 1:
        plan, path = multiple_solutions[0]
        commands_file = join(path, 'commands.pkl')
        if not isfile(commands_file):
            return None
        commands = pickle.load(open(commands_file, "rb"))
    return commands, plan


##########################################################################################


def load_basic_plan_commands(world, exp_dir, run_dir, verbose=False, load_attach=True):
    problem = Problem(world)
    if verbose:
        world.summarize_all_objects()
    body_map = get_body_map(run_dir, world, larger=False)
    if load_attach:
        load_attachments(run_dir, world, body_map=body_map)
    maybe_hpn = 'hpn' in run_dir
    plan = get_plan(run_dir, skip_multiple_plans=True, maybe_hpn=maybe_hpn)
    commands = pickle.load(open(join(exp_dir, 'commands.pkl'), "rb"))
    return problem, commands, plan, body_map


def load_pigi_data(run_dir, use_gui=True, width=1440, height=1120, verbose=False):
    """ for replaying """
    exp_dir = copy_dir_for_process(run_dir, tag='replaying', verbose=verbose)
    print_debug(f"load_pigi width = {width} height = {height}")
    world = load_lisdf_pybullet(exp_dir, use_gui=use_gui, width=width, height=height, verbose=False)
    problem, commands, plan, body_map = load_basic_plan_commands(world, exp_dir, run_dir,
                                                                 verbose=verbose, load_attach=False)
    return world, problem, exp_dir, run_dir, commands, plan, body_map


def load_pigi_data_complex(run_dir_ori, use_gui=True, width=1440, height=1120, verbose=False):
    exp_dir, run_dir, commands, plan = get_pkl_run(run_dir_ori, verbose=verbose)

    # load_lisdf_synthesizer(exp_dir)
    larger_world = 'rerun' in run_dir_ori and '/tt_' in run_dir_ori
    world = load_lisdf_pybullet(exp_dir, use_gui=use_gui, width=width, height=height,
                                verbose=False, larger_world=larger_world)

    problem = Problem(world)
    if verbose:
        world.summarize_all_objects()
    body_map = get_body_map(run_dir, world, larger=False)
    result = check_if_exist_rerun(run_dir, world, commands, plan)
    if result is None:
        print(run_dir_ori, 'does not have rerun commands.pkl')
        reset_simulation()
        shutil.rmtree(exp_dir)
        return
    commands, plan = result

    return world, problem, exp_dir, run_dir, commands, plan, body_map


#############################################################################################


def load_attachments(run_dir, world, body_map):
    config = load_planning_config(run_dir)
    if 'attachments' not in config:
        return {}
    attachments = config['attachments']
    for child, (parent, parent_link, grasp_pose) in attachments.items():
        body = body_map[child]
        world.attachments[body] = ObjAttachment(parent, parent_link, grasp_pose, body)


def load_replay_config_yaml_file(conf_path):
    c = load_yaml(conf_path)
    for key in ['pybullet', 'gym']:
        if key in c:
            c.update(c.pop(key))
    return c


def load_replay_conf(conf_path, **kwargs):
    c = load_replay_config_yaml_file(conf_path)
    for k, v in c.items():
        if v == 'none':
            c[k] = None

    if c['step_by_step']:
        c['save_mp4'] = False

    if c['given_path']:
        # c['visualize_collisions'] = True
        c['parallel'] = False
        if 'rerun' in c['given_path']:
            c['save_jpg'] = False
            c['save_composed_jpg'] = False
            c['gif'] = False
    else:
        c['parallel'] = c['parallel'] and c['save_jpg'] and not c['preview_scene']

    if c['mp4_side_view'] or c['mp4_top_view'] or c['light_conf'] is not None:
        c['save_mp4'] = True
        c['save_gif'] = False
        if c['light_conf'] is not None:
            c['width'] = 3840
            c['height'] = 2160
        if c['mp4_side_view'] or c['mp4_top_view']:
            c['width'] = 1920
            c['height'] = 1080
            c['light_conf'] = dict(direction=np.asarray([0, -1, 0]), intensity=np.asarray([1, 1, 1]))

    if c['save_observation_mp4']:
        c['save_mp4'] = False
        c['save_gif'] = True

    for k, v in kwargs.items():
        c[k] = v

    c['save_jpg'] = c['save_jpg'] or c['save_composed_jpg'] or c['save_gif']
    c['use_gym'] = c['use_gym'] and has_srl_stream() and not running_in_pycharm()
    # c['save_mp4'] = c['save_mp4'] or c['use_gym']
    c['step_by_step'] = c['step_by_step'] and has_getch()
    # c['cases'] = get_sample_envs_for_rss(task_name=c['task_name'], count=None)

    return c


def run_replay(config_yaml_file, load_data_fn=load_pigi_data, **kwargs):
    """
    when use_gym=True, if ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory
        export LD_LIBRARY_PATH=/home/zhutiany/miniconda3/envs/cogarch/lib
    """
    c = load_replay_conf(config_yaml_file, **kwargs)

    def process(run_dir_ori):
        return run_one(run_dir_ori, load_data_fn=load_data_fn, **c)

    def _case_filter(run_dir_ori):
        case_kwargs = dict(given_path=c['given_path'], cases=c['cases'], check_collisions=c['check_collisions'],
                           save_jpg=c['save_jpg'], save_gif=c['save_gif'],
                           skip_if_processed_recently=c['skip_if_processed_recently'], check_time=c['check_time'])
        return case_filter(run_dir_ori, **case_kwargs)

    process_all_tasks(process, c['task_name'], parallel=c['parallel'], cases=c['cases'],
                      path=c['given_path'], dir=c['given_dir'], case_filter=_case_filter)


def set_replay_camera_pose(world, run_dir, camera_kwargs, camera_point, target_point):
    if camera_kwargs is not None:
        if 'camera_point' in camera_kwargs:
            set_camera_pose(camera_kwargs['camera_point'], camera_kwargs['target_point'])
        elif 'camera_point_begin' in camera_kwargs:
            set_camera_pose(camera_kwargs['camera_point_begin'], camera_kwargs['target_point_begin'])
    else:
        set_camera_pose(camera_point, target_point)
        return
    #
    #     planning_config = load_planning_config(run_dir)
    #     if False and 'obs_camera_pose' in planning_config:
    #         ### not working
    #         camera_pose = planning_config['obs_camera_pose']
    #         # camera_pose = adjust_camera_pose(camera_pose)
    #         set_camera_pose2(camera_pose)
    #     else:
    #         aabb = world.get_world_aabb()
    #         center = get_aabb_center(aabb)
    #         extent = get_aabb_extent(aabb)
    #         camera_point = center + extent / 2
    #         set_camera_pose(camera_point, center)


def clean_list(lst):
    ## for actions like sprinkle, both parts of the trajectory consists of MoveArmAction
    new_lst = []
    for i, elem in enumerate(lst):
        if not (i > 0 and elem == new_lst[-1]):
            new_lst.append(elem)
    return new_lst


def get_segmented_commands(world, commands, plan, time_log=None):
    """
    if time_log is None, segment commands by actions
    if time_log is not None, segment commands by subproblems, time_log is a list of dicts: e.g.
        {
            "planning": 3.6821,
            "goal": [["openedjoint([(10, 1)])"]],
            "goal_original": [["openedjoint","fridge#1::fridge_door"]],
            "plan_skeleton": [
                "move_base()",
                "grasp_pull_ungrasp_handle((10, 1))"
            ]
        },
    """
    segmented_commands = []
    actions, continuous = plan

    i = 0
    action_commands = [
        (a[0], clean_list(get_action_name(aa) for aa in get_primitive_actions(Action(a[0], a[1:]), world, teleport=True)))
        for a in actions
    ]
    current_action_name, current_command_names = action_commands[i]
    last_command_name = None
    current_commands = []
    for command in commands:
        name = get_action_name(command)
        if name != last_command_name:
            if last_command_name == current_command_names[0]:
                if len(current_command_names) > 1:
                    current_command_names = current_command_names[1:]
                else:
                    segmented_commands.append((current_action_name, current_commands, ([actions[i]], continuous)))
                    i += 1
                    if i == len(action_commands):
                        break
                    current_action_name, current_command_names = action_commands[i]
                    current_commands = [current_commands[-1]]
            last_command_name = name
        current_commands.append(command)
    if i != len(action_commands):
        segmented_commands.append((current_action_name, current_commands, ([actions[i]], continuous)))

    if time_log is not None:
        index = 0
        new_segmented_commands = []
        for j, log in enumerate(time_log):
            new_subproblem_name = '-'.join(log["goal"])
            new_commands = []
            new_plan = []
            n = len(log["plan_skeleton"])
            for i in range(n):
                if i+index >= len(segmented_commands):
                    break
                current_action_name, current_commands, (current_plan, _) = segmented_commands[i+index]
                new_commands.extend(current_commands)
                new_plan.extend(current_plan)
            print(j, new_subproblem_name, len(new_plan), [p[0] for p in new_plan])
            new_segmented_commands.append((new_subproblem_name, new_commands, (new_plan, continuous)))
            index += n
        segmented_commands = new_segmented_commands

    return segmented_commands


def run_one(run_dir_ori, load_data_fn=load_pigi_data, task_name=None, given_path=None, given_dir=None, cases=None,
            parallel=False, skip_if_processed_recently=False, check_time=None, preview_scene=False, step_by_step=False,
            action_by_action=False, use_gym=False, auto_play=True, verbose=False, width=1280, height=800, fx=600,
            time_step=0.02, make_robot_and_links_transparent=False,
            camera_point=(8.5, 2.5, 3), target_point=(0, 2.5, 0), camera_kwargs=None, camera_movement=None,
            light_conf=None, check_collisions=False, cfree_range=0.1, visualize_collisions=False,
            evaluate_quality=False, save_jpg=False, save_composed_jpg=False, save_gif=False,
            save_animation_json=False, save_mp4=False, save_segmented_mp4=False,
            save_observation_mp4=False, save_observation_pcd=False,
            frame_gap=3, mp4_side_view=False, mp4_top_view=False, shape_json=None, return_frames=False):
    """ specific to gym replay:
            mp4_side_view/mp4_top_view: overwrites camera_point and target_point
            frame_gap: subsample the image frames
    """

    world, problem, exp_dir, run_dir, commands, plan, body_map = load_data_fn(
        run_dir_ori, use_gui=not use_gym, width=width, height=height, verbose=verbose
    )

    ## add observation cameras
    if save_observation_mp4:
        from robot_builder.robots import PR2Robot
        camera_matrix = world.camera.camera_matrix
        world.robot_cameras = add_robot_cameras(world.robot, PR2Robot.camera_frames, camera_matrix=camera_matrix)

    ## load objects transparent
    if make_robot_and_links_transparent:
        make_furniture_transparent(world, exp_dir, lower_tpy=0.5, upper_tpy=0.2)

    with WorldSaver():
        commands = [adapt_action(command, problem, plan, verbose=False) for command in commands]
    set_replay_camera_pose(world, run_dir, camera_kwargs, camera_point, target_point)

    # set_camera_target_body(4, dx=0.8, dy=-0.8, dz=1.5)  ## braiser lid
    # set_camera_target_body(3, dx=1, dy=0.1, dz=1.5)  ## close cabinet door
    # set_camera_target_body(8, dx=-0.2, dy=0.2, dz=1.7)  ## sink
    if preview_scene:
        wait_unlocked()

    ## for LISDF-LOADER JS visualization
    if use_gym and save_animation_json:
        save_animation_json_in_gym(run_dir_ori, problem, commands, body_map=None)
        sys.exit()

    ## save the initial scene image in pybullet
    zoomin_kwargs = dict(width=width//4, height=height//4, fx=fx//2)
    if not check_collisions and save_jpg:
        save_initial_scene_in_pybullet(run_dir_ori, world, zoomin_kwargs, save_composed_jpg=save_composed_jpg,
                                       save_gif=save_gif, width=width, height=height, fx=fx)

    if use_gym:
        gym_kwargs = dict(width=width, height=height, camera_point=camera_point, target_point=target_point,
                          camera_kwargs=camera_kwargs, camera_movement=camera_movement, light_conf=light_conf,
                          save_jpg=save_jpg, save_gif=save_gif, save_mp4=save_mp4, frame_gap=frame_gap,
                          save_segmented_mp4=save_segmented_mp4, mp4_side_view=mp4_side_view, mp4_top_view=mp4_top_view,
                          save_observation_mp4=save_observation_mp4, save_observation_pcd=save_observation_pcd,
                          shape_json=shape_json, return_frames=return_frames)
        filenames = run_one_in_isaac_gym(exp_dir, run_dir, world, problem, commands, plan, body_map, **gym_kwargs)
        if return_frames:
            reset_simulation()
            return exp_dir, filenames

    ## pybullet
    else:
        if save_mp4:
            if save_segmented_mp4:
                time_log = process_text_for_animation(run_dir, world.body_to_name, get_info_path(run_dir))
                segmented_commands = get_segmented_commands(world, commands, plan, time_log)
                video_dir = join(run_dir, 'videos')
                os.makedirs(video_dir, exist_ok=True)
                for i, (action_name, short_commands, short_plan) in enumerate(segmented_commands):
                    mp4_name = f'replay_{i}_{action_name}.mp4'
                    attachments = save_mp4_in_pybullet(run_dir, problem, short_commands, short_plan, mp4_name=mp4_name,
                                                       time_step=time_step)
                    problem.world.attachments = attachments
                    print_green(f"{i}\t{attachments}\n")
                    shutil.move(join(run_dir, mp4_name), join(video_dir, mp4_name))
            else:
                save_mp4_in_pybullet(run_dir, problem, commands, plan, time_step=time_step)

        else:
            if save_segmented_mp4:  ## debug
                time_log = process_text_for_animation(run_dir, world.body_to_name, get_info_path(run_dir))
                segmented_commands = get_segmented_commands(world, commands, plan, time_log)
            run_one_in_pybullet(run_dir, run_dir_ori, world, problem, commands, plan, body_map, task_name=task_name,
                                step_by_step=step_by_step, action_by_action=action_by_action, auto_play=auto_play,
                                check_collisions=check_collisions, cfree_range=cfree_range, visualize_collisions=visualize_collisions,
                                evaluate_quality=evaluate_quality, zoomin_kwargs=zoomin_kwargs, time_step=time_step,
                                save_jpg=save_jpg, save_composed_jpg=save_composed_jpg, save_gif=save_gif,
                                save_observation_mp4=save_observation_mp4, verbose=verbose)

    reset_simulation()
    shutil.rmtree(exp_dir)


#######################################################################################################


def save_animation_json_in_gym(run_dir_ori, problem, commands, body_map=None):
    from isaac_tools.gym_utils import record_actions_in_gym
    wconfs = record_actions_in_gym(problem, commands, body_map=body_map, return_wconf=True)
    with open(join(run_dir_ori, 'animation.json'), 'w') as f:
        json.dump(wconfs, f, indent=2)


def save_initial_scene_in_pybullet(run_dir_ori, world, zoomin_kwargs, save_composed_jpg=False, save_gif=False,
                                   width=1280, height=800, fx=600):
    viz_dir = join(run_dir_ori, f'images')
    for sud in range(len(world.camera_kwargs)):
        viz_dir = join(run_dir_ori, f'zoomin_{sud}'.replace('_0', ''))
        world.add_camera(viz_dir, img_dir=viz_dir, **zoomin_kwargs, **world.camera_kwargs[sud])
        world.visualize_image(index='initial', rgb=True)

    ## for a view of the whole scene
    if save_composed_jpg or save_gif:
        world.add_camera(viz_dir, width=width // 2, height=height // 2, fx=fx // 2, img_dir=viz_dir,
                         camera_point=(6, 4, 2), target_point=(0, 4, 1))
        world.make_transparent(world.robot.body, transparency=0)


def save_mp4_in_pybullet(run_dir, problem, commands, plan, mp4_name='replay.mp4', time_step=0.025):
    video_path = join(run_dir, mp4_name)
    with VideoSaver(video_path):
        attachments = apply_commands(problem, commands, time_step=time_step, verbose=False, plan=plan)
    print('saved mp4 to', abspath(video_path))
    return attachments


def run_one_in_pybullet(run_dir, run_dir_ori, world, problem, commands, plan, body_map, task_name=None,
                        step_by_step=False, action_by_action=False, auto_play=True, check_collisions=False,
                        cfree_range=0.1, time_step=0.02, visualize_collisions=False, evaluate_quality=False,
                        zoomin_kwargs=None, save_jpg=False, save_composed_jpg=False, save_gif=False,
                        save_observation_mp4=False, verbose=False):
    run_name = basename(run_dir)
    if not auto_play:
        wait_unlocked()

    answer = True
    if not auto_play:
        answer = query_yes_no(f"start replay {run_name}?", default='yes')
    if answer:
        time_step = 2e-5 if save_jpg else time_step
        time_step = None if step_by_step else time_step
        results = apply_commands(problem, commands, time_step=time_step, verbose=verbose, plan=plan,
                                 body_map=body_map, save_composed_jpg=save_composed_jpg, save_gif=save_gif,
                                 save_observation_mp4=save_observation_mp4,
                                 check_collisions=check_collisions, cfree_range=cfree_range,
                                 action_by_action=action_by_action, visualize_collisions=visualize_collisions)

    if check_collisions:
        new_data = {'cfree': results} if results else {'cfree': cfree_range}
        ## another way to be bad data is if the place pose is not realistic
        if new_data['cfree'] == cfree_range:
            result = check_unrealistic_placement_z(world, run_dir)
            if result:
                new_data['cfree'] = result
        add_to_planning_config(run_dir, new_data)
        if results:
            print('COLLIDED', run_dir)

    else:

        if save_composed_jpg:
            episodes = results
            h, w, _ = episodes[0][0][0][0].shape
            crop = (0, h // 3 - h // 30, w, 2 * h // 3 - h // 30)
            make_composed_image_multiple_episodes(episodes, join(world.img_dir, 'composed.jpg'),
                                                  verbose=verbose, crop=crop)

        if save_gif:

            """ animation of robot perspective """
            if save_observation_mp4:
                collage_images = make_observation_collages(results)
                gif_path = join(run_dir_ori, f'observations.gif')
                images_to_gif(world.img_dir, gif_path, collage_images)

            else:
                """ animation of how objects move """
                episodes = results['scene']
                h, w, _ = episodes[0].shape
                crop = (0, h // 3 - h // 30, w, 2 * h // 3 - h // 30)
                gif_path = join(run_dir_ori, f'replay.gif')
                images_to_gif(world.img_dir, gif_path, episodes, crop=crop)

        # if SAVE_COMPOSED_JPG or SAVE_GIF:
        #     world.camera = world.cameras[0]

        """ images at zoom-in camera poses defined in `{data_dir}/planning_config.json` at key "camera_zoomins" """
        if save_jpg:
            for sud in range(len(world.camera_kwargs)):
                viz_dir = join(run_dir_ori, f'zoomin_{sud}'.replace('_0', ''))
                world.add_camera(viz_dir, img_dir=viz_dir, **zoomin_kwargs, **world.camera_kwargs[sud])
                world.visualize_image(index='final', rgb=True)
                print('\n... saved jpg in', viz_dir)
            # world.visualize_image(index='final', rgb=True, **world.camera_kwargs)

    if evaluate_quality:
        answer = query_yes_no(f"delete this run {run_name}?", default='no')
        if answer:
            root_dir = dirname(dirname(run_dir_ori))
            new_dir = join(root_dir, 'impossible', f"{task_name}_{run_name}")
            shutil.move(run_dir, new_dir)
            print(f"moved {run_dir} to {new_dir}")

    # wait_if_gui('replay next run?')
    return results


def run_one_in_isaac_gym(exp_dir, run_dir, world, problem, commands, plan, body_map, width=1280, height=800,
                         camera_point=(8.5, 2.5, 3), target_point=(0, 2.5, 0), camera_kwargs=None,
                         camera_movement=None, light_conf=None, save_jpg=True, save_gif=False,
                         save_mp4=False, save_segmented_mp4=False, mp4_side_view=False, mp4_top_view=False,
                         save_observation_mp4=True, save_observation_pcd=True,
                         shape_json=None, frame_gap=3, return_frames=False):
    from isaac_tools.gym_utils import load_lisdf_isaacgym, record_actions_in_gym, set_camera_target_body

    title = 'run_one_in_isaac_gym\t'

    print_debug(f"world.robot.__class__.__name__ = {world.robot.__class__.__name__}")

    gym_world = load_lisdf_isaacgym(abspath(exp_dir), camera_width=width, camera_height=height,
                                    camera_point=camera_point, target_point=target_point)
    set_camera_target_body(gym_world, run_dir)

    save_name = basename(exp_dir).replace('temp_', '')

    #####################################################

    ## the aforegiven camera_point and target_point is for
    # set better camera view for making gif screenshots
    removed = []
    if mp4_side_view or mp4_top_view:
        removed, camera_kwargs = get_gym_actors_to_ignore(
            gym_world, world, run_dir, mp4_side_view=mp4_side_view, mp4_top_view=mp4_top_view, shape_json=shape_json
        )

    if camera_kwargs is not None:
        camera_point = camera_kwargs['camera_point']
        camera_target = camera_kwargs['camera_target']
        print_green(f"\ncamera_point = {camera_point},\ttarget_point = {target_point}\n")
        gym_world.set_viewer_target(camera_point, camera_target)
        gym_world.set_camera_target(gym_world.cameras[0], camera_point, camera_target)
        print_pink(f'setting camera at camera_point = {camera_point}\tcamera_target = {camera_target}')
        if 'sink' in run_dir:
            gym_world.simulator.set_light(
                direction=np.asarray([0, -1, 0.2]),  ## [0, -1, 0.2]
                intensity=np.asarray([1, 1, 1]))
        if False:
            img_file = gym_world.get_rgba_image(gym_world.cameras[0])
            from PIL import Image
            im = Image.fromarray(img_file)
            im.save(join('gym_images', save_name + '.png'))
            sys.exit()

    if light_conf is not None:
        gym_world.simulator.set_light(**light_conf)

    if save_jpg:
        des_path = join(run_dir, 'gym_scene.png')
        print('moved jpg to {}'.format(des_path))
        shutil.move(join(exp_dir, 'gym_scene.png'), des_path)

    if save_gif or save_mp4:

        gym_kwargs = dict(body_map=body_map, time_step=0, verbose=False, save_gif=save_gif, save_mp4=save_mp4,
                          ignore_actors=removed, camera_movement=camera_movement)
        if save_segmented_mp4:
            time_log = process_text_for_animation(run_dir, world.body_to_name, get_info_path(run_dir))
            segmented_commands = get_segmented_commands(world, commands, plan, time_log)
            print('\n'+'-'*50)
            print([(k[0], len(k[1])) for k in segmented_commands])
            print('-'*50+'\n')
            all_gif_path = []
            for i, (action_name, short_commands, short_plan) in enumerate(segmented_commands):
                gif_path = join(run_dir, f'gym_replay__{i}__{action_name}.gif')
                img_dir = join(exp_dir, f'gym_images__{i}__{action_name}')
                attachments = record_actions_in_gym(problem, short_commands, gym_world, plan=short_plan,
                                                    img_dir=img_dir, gif_path=gif_path, frame_gap=1, **gym_kwargs)
                problem.world.attachments = attachments
                print_pink(f"\nafter record_actions_in_gym to {gif_path}, containing {len(short_commands)} commands,"
                           f"resulting in {len(attachments)} attachments")
                if len(short_commands) > 0:
                    all_gif_path.append(gif_path)
            gif_path = all_gif_path

        else:
            gif_path = join(run_dir, 'gym_replay.gif')
            img_dir = join(exp_dir, 'gym_images')
            filenames = record_actions_in_gym(problem, commands, gym_world, plan=plan, img_dir=img_dir,
                                              frame_gap=frame_gap, gif_path=gif_path, return_frames=return_frames,
                                              **gym_kwargs)
            print(filenames)
            if return_frames:
                del (gym_world.simulator)
                return filenames
        # gym_world.wait_if_gui()

        if mp4_side_view:
            save_name = save_name + '_side'
        if mp4_top_view:
            save_name = save_name + '_top'

        os.makedirs(GYM_IMAGES_DIR, exist_ok=True)
        new_file = join(GYM_IMAGES_DIR, save_name+'.gif')

        if save_gif:
            # shutil.copy(gif_path, join(run_dir, basename(gif_path)))
            print('moved gif to {}'.format(join(run_dir, new_file)))
            shutil.move(gif_path, new_file)

        if save_mp4:
            move_file_to_demo_folder = False
            if save_segmented_mp4:
                new_dir = join(GYM_IMAGES_DIR, save_name)
                os.makedirs(new_dir, exist_ok=True)
                for i, path in enumerate(gif_path):
                    path = path.replace('.gif', '.mp4')
                    if isfile(path):
                        new_mp4_name = join(new_dir, basename(path))
                        shutil.move(path, new_mp4_name)
                        print(f'{title} | step {i}: moved {path} to {new_mp4_name}')
                    else:
                        print(f'{title} | step {i}: skipping {path} because the action has no commands')

                shutil.move(get_info_path(run_dir), get_info_path(new_dir))

            elif move_file_to_demo_folder:
                mp4_name = gif_path.replace('.gif', '_scene.mp4')
                new_mp4_name = new_file.replace('.gif', '_scene.mp4')
                shutil.move(mp4_name, new_mp4_name)
                print(f'moved mp4 to {new_mp4_name}')

    del (gym_world.simulator)


def process_text_for_animation(run_dir, body_to_name, json_path):
    def translate_literal(goal):
        elems, found_objects = translate_objects(goal[1:])
        return [goal[0]] + elems, found_objects

    def translate_objects(lst):
        elems = []
        found_objects = {}
        for e in lst:
            if isinstance(e, list):
                e = tuple(e)
            if e in body_to_name:
                object_name = body_to_name[e].replace('#1', '')
                if '::' in object_name:
                    object_name = object_name.split('::')[1]
                found_objects[e] = object_name
                e = object_name
            elems.append(e)
        return elems, found_objects

    def extract_value(k, log):
        v = log[k]
        if k == 'goal':
            if isinstance(v[0], str) and '([' in v[0]:
                elems, found_objects = extract_value('goal_original', log)
                pred, args = v[0][:-2].split('([')  ## 'openedjoint([(10, 1)])'
                found_objects = dict(sorted(found_objects.items(), key=lambda x: len(x[1]), reverse=True))
                for index, object_name in found_objects.items():
                    args = args.replace(str(index), object_name)
                return [pred] + args.split(', '), found_objects
            else:
                return translate_literal(v)
        elif k == 'goal_original':
            return translate_literal(log[k][0])
        else:
            return log[k], {}

    time_log = json.load(open(join(run_dir, 'time.json'), 'r'))
    print(f'loaded {len(time_log)} subproblems')
    time_log = [
        {k: extract_value(k, log)[0] for k in ['planning', 'goal', 'goal_original', 'plan_skeleton']}
        for log in time_log[:-1] if log['plan'] != FAILED
    ]
    print(f'processed {len(time_log)} subproblems')

    with open(json_path, 'w') as f:
        json.dump(time_log, f, indent=2)
    return time_log


#######################################################################################################


def merge_all_wconfs(all_wconfs):
    longest_command = max([len(wconf) for wconf in all_wconfs])
    whole_wconfs = []
    for t in range(longest_command):
        whole_wconf = {}
        for num in range(len(all_wconfs)):
            if t < len(all_wconfs[num]):
                whole_wconf.update(all_wconfs[num][t])
            elif t < 2 * len(all_wconfs[num]):
                whole_wconf.update(all_wconfs[num][2 * len(all_wconfs[num]) - t - 1])
            elif t < 3 * len(all_wconfs[num]):
                whole_wconf.update(all_wconfs[num][t - 2 * len(all_wconfs[num])])
        whole_wconfs.append(whole_wconf)
    return whole_wconfs


def replay_all_in_gym(width=1440, height=1120, num_rows=5, num_cols=5, world_size=(6, 6), verbose=False,
                      frame_gap=6, debug=False, loading_effect=False, save_gif=True, save_mp4=False,
                      camera_motion=None):
    from test_utils import get_dirs_camera
    from isaac_tools.gym_utils import load_envs_isaacgym, record_actions_in_gym, \
        update_gym_world_by_wconf, save_gym_run, interpolate_camera_pose
    from tqdm import tqdm

    img_dir = join('gym_images')
    gif_name = 'gym_replay_batch_gym.gif'
    # if isdir(img_dir):
    #     shutil.rmtree(img_dir)
    # os.mkdir(img_dir)

    data_dir = 'test_full_kitchen_100' if loading_effect else 'test_full_kitchen_sink'
    ori_dirs, camera_point_begin, target_point, camera_kwargs = get_dirs_camera(
        num_rows, num_cols, world_size, data_dir=data_dir, camera_motion=camera_motion)
    lisdf_dirs = [copy_dir_for_process(ori_dir, verbose=verbose) for ori_dir in ori_dirs]
    num_worlds = min([len(lisdf_dirs), num_rows * num_cols])

    ## translate commands into world_confs
    all_wconfs = []

    if loading_effect:
        ### load all gym_worlds and return all wconfs
        gym_world, offsets, all_wconfs = load_envs_isaacgym(lisdf_dirs, num_rows=num_rows, num_cols=num_cols,
                                                            world_size=world_size, loading_effect=True, verbose=verbose,
                                                            camera_point=camera_point_begin, target_point=target_point)
    else:
        for i in tqdm(range(num_worlds)):
            exp_dir, run_dir, commands, plan = get_pkl_run(lisdf_dirs[i], verbose=verbose)
            world = load_lisdf_pybullet(exp_dir, use_gui=debug,
                                        width=width, height=height, verbose=False)
            body_map = get_body_map(run_dir, world, larger=False)
            problem = Problem(world)
            wconfs = record_actions_in_gym(problem, commands, plan=plan, return_wconf=True,
                                           world_index=i, body_map=body_map)
            all_wconfs.append(wconfs)
            reset_simulation()
            if debug:
                wait_unlocked()

        ## load all scenes in gym
        gym_world, offsets = load_envs_isaacgym(lisdf_dirs, num_rows=num_rows, num_cols=num_cols, world_size=world_size,
                                                camera_point=camera_point_begin, target_point=target_point, verbose=verbose)

    # img_file = gym_world.get_rgba_image(gym_world.cameras[0])
    # from PIL import Image
    # im = Image.fromarray(img_file)
    # im.save(join('gym_images', 'replay_all_in_gym' + '.png'))
    # sys.exit()

    all_wconfs = merge_all_wconfs(all_wconfs)
    print(f'\n\nrendering all {len(all_wconfs)} frames')

    all_wconfs = all_wconfs[:600]

    ## update all scenes for each frame
    filenames = []
    for i in tqdm(range(len(all_wconfs))):
        interpolate_camera_pose(gym_world, i, len(all_wconfs), camera_kwargs)
        update_gym_world_by_wconf(gym_world, all_wconfs[i], offsets=offsets)

        # ## just save the initial state
        # img_file = gym_world.get_rgba_image(gym_world.cameras[0])
        # from PIL import Image
        # im = Image.fromarray(img_file)
        # im.save(join('gym_images', 'replay_all_in_gym' + '.png'))
        # sys.exit()

        if i % frame_gap == 0:
            img_file = gym_world.get_rgba_image(gym_world.cameras[0])
            filenames.append(img_file)

    save_gym_run(img_dir, gif_name, filenames, save_gif=save_gif, save_mp4=save_mp4)


def generated_recentely(file, skip_if_processed_recently=False, check_time=None):
    result = False
    if isfile(file):
        if skip_if_processed_recently:
            last_modified = os.path.getmtime(file)
            if last_modified > check_time:
                result = True
        else:
            result = True
    return result


def case_filter(run_dir_ori, given_path=None, cases=None, check_collisions=False,
                save_jpg=False, save_gif=False, skip_if_processed_recently=False, check_time=None):
    """ whether to process this run """
    if cases is not None or given_path is not None:
        return True

    result = True
    if check_collisions:
        config, mod_time = load_planning_config(run_dir_ori, return_mod_time=True)
        if 'cfree' in config:  ## and config['cfree']:
            if mod_time > 1674746024:
                return False
            if isinstance(config['cfree'], float):  ## check if unrealistic
                return True
            # if isinstance(config['cfree'], str) and exist_instance(run_dir_ori, '100015') \
            #         and 'braiser' in config['cfree']:
            #     return True
            result = False
        return result

    recent_kwargs = dict(skip_if_processed_recently=skip_if_processed_recently, check_time=check_time)
    if save_jpg:
        # run_num = eval(run_dir_ori[run_dir_ori.rfind('/')+1:])
        # if 364 <= run_num < 386:
        #     return True
        viz_dir = join(run_dir_ori, 'zoomin')
        if isdir(viz_dir):
            enough = len([a for a in listdir(viz_dir) if '.png' in a]) > 1
            file = join(viz_dir, 'rgb_image_final.png')
            result = not generated_recentely(file, **recent_kwargs) or not enough
    if result:
        return result

    if save_gif:
        file = join(run_dir_ori, 'replay.gif')
        result = not generated_recentely(file, **recent_kwargs)

    multiple_solutions_file = join(run_dir_ori, 'multiple_solutions.json')
    # if not result and isfile(multiple_solutions_file):
    #     plans = json.load(open(multiple_solutions_file, 'r'))
    #     if len(plans) == 2 and 'rerun_dir' in plans[0]:
    #         print('dont skip multiple solutions', run_dir_ori)
    #         result = True
    return result


##############################################################################


def get_gym_actors_to_ignore(gym_world, world, run_dir, mp4_side_view=False, mp4_top_view=False, shape_json=None):
    removed = []
    world_aabb = get_world_aabb(run_dir, shape_json=shape_json)
    x, y, _ = get_aabb_center(world_aabb)
    if mp4_side_view:
        x2 = 8
        z2 = 3
    if mp4_top_view:
        x2 = x + 0.1
        z2 = 9
        x2 = x + 4.5
        z2 = 7
        trashcan = (np.array([-4, -4, 8]), np.array([0, 0, 0, 1]))
        goals = get_goals(run_dir)

        ## remove all cabinetupper
        removed.extend(world.remove_from_gym_world('cabinetupper', gym_world, trashcan))

        ## remove cabinettop if not in goal
        if 'cabinettop' not in goals[-1][-1]:
            removed.extend(world.remove_from_gym_world('cabinettop', gym_world, trashcan))
        else:
            name_one = goals[-1][-1].split('::')[0]
            exceptions = [name_one, f"{name_one}_filler"]
            removed.extend(world.remove_from_gym_world('cabinettop', gym_world, trashcan,
                                                       exceptions=exceptions))
        if 'storedinspace' == goals[-1][0]:
            x2 = x + 3

        ## remove the shelf if bottles are not to be moved
        if not (goals[-1][0] == 'storedinspace' and goals[-1][1] == '@bottle'):
            removed.extend(world.remove_from_gym_world('shelf', gym_world, trashcan))
            count_in = []
            count_out = []
            for name in world.name_to_body:
                if name in ['floor1'] or '::' in name:
                    continue
                print('checking', name)
                pose = gym_world.get_pose(gym_world.get_actor(name))
                if pose[0][2] > 1.5 and ('bottle' in name or 'medicine' in name):
                    count_in.append(name)
                else:
                    count_out.append(name)
            print('count_in', count_in)
            print('count_out', count_out)
            if len(count_in) > 0:
                removed.extend(world.remove_from_gym_world(count_in[0], gym_world, trashcan,
                                                           exceptions=count_out))

    ## camera_point = [5.3, 4.7, 9]	camera_target = [2.3, 4.7, 1]
    ## at camera_point = [5.3, 4.723, 9]	camera_target = [2.3, 4.723, 1]
    camera_kwargs = dict(camera_point=[x2, y, z2], camera_target=[x, y, 1])
    return removed, camera_kwargs


def get_info_path(run_dir):
    return join(run_dir, 'segmented_commands_info.json')


##################################################################################


def get_dirs_camera(num_rows=5, num_cols=5, world_size=(6, 6), data_dir=None, camera_motion=None):
    ## load all dirs
    ori_dirs = []
    if world_size == (4, 8):
        ori_dirs = get_sample_envs_full_kitchen(num_rows * num_cols, data_dir=data_dir)

    camera_target = None
    target_point_begin = None
    target_point_final = None
    if num_rows == 1 and num_cols == 1:

        if world_size == (4, 8):
            camera_point = (6, 4, 6)
            camera_target = (0, 4, 0)
            camera_point_begin = (6, 4, 6)
            camera_point_final = (12, 4, 12)

    elif num_rows == 2 and num_cols == 1:

        if world_size == (4, 8):
            camera_point = (24, 4, 10)
            camera_target = (0, 4, 0)
            camera_point_begin = (16, 4, 6)
            camera_point_final = (24, 4, 10)

    elif num_rows == 4 and num_cols == 4:

        if world_size == (4, 8):
            camera_point = (32, 8, 10)
            camera_target = (0, 24, 0)
            camera_point_begin = (16, 16, 2)
            camera_point_final = (32, 16, 10)

    elif num_rows == 5 and num_cols == 5:
        ori_dirs = get_sample_envs_for_corl()

        if world_size == (6, 6):
            mid = num_rows * 6 // 2
            camera_point = (45, 15, 10)
            camera_target = (0, 15, 0)
            camera_point_final = (mid + 35, 15, 14)
            camera_point_begin = (mid + 9, 15, 4)
            camera_target = (mid, 15, 0)

    elif num_rows == 8 and num_cols == 3:

        if world_size == (4, 8):
            if camera_motion == 'zoom':
                camera_target = (4*6, 8*2, 0)
                camera_point_begin = (4*8-1.5, 8*3-4, 2)
                camera_point_final = (4*8+3, 8*3, 4)
            elif camera_motion == 'spotlight':
                camera_target = (4*4, 8*1.5, 1)
                camera_point_begin = (4*4, 8*1.5, 2.5)
                camera_point_final = 3.5

    elif num_rows == 10 and num_cols == 10:

        if world_size == (4, 8):
            ## bad
            camera_target = (0, 48, 0)
            camera_point_begin = (16, 40, 6)
            camera_point_final = (40, 32, 16)

    elif num_rows == 14 and num_cols == 14:
        ori_dirs = get_sample_envs_200()

        if world_size == (6, 6):
            y = 42 + 3
            camera_point_begin = (67, y, 3)
            camera_point_final = (102, y, 24)
            camera_target = (62, y, 0)

    elif num_rows == 16 and num_cols == 16:

        if world_size == (4, 8):
            camera_target = (5*11, 8*4-4, 0)
            camera_point_begin = (5*13-1, 8*4-6, 2)
            camera_point_final = (5*16, 8*2, 12)

    elif num_rows == 32 and num_cols == 8:

        if world_size == (4, 8):
            if camera_motion == 'zoom':
                camera_target = (5*(11+16), 8*4-4, 0)
                camera_point_begin = (5*(12+16), 8*4-6, 2)
                camera_point_final = (5*(16+16), 8*2, 12)
            elif camera_motion == 'pan':
                target_point_begin = (5*(32-4), 8*3-4, 0)
                target_point_final = (5*(16-4), 8*3-4, 0)
                camera_point_begin = (5*32, 8*1, 12)
                camera_point_final = (5*16, 8*1, 12)

    if target_point_begin is None and target_point_final is None:
        target_point_begin = target_point_final = camera_target
    if camera_target is None:
        camera_target = target_point_begin

    kwargs = dict(camera_point_begin=camera_point_begin,
                  camera_point_final=camera_point_final,
                  target_point_begin=target_point_begin,
                  target_point_final=target_point_final)
    return ori_dirs, camera_point_begin, camera_target, kwargs