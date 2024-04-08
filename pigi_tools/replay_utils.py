#!/usr/bin/env python

from __future__ import print_function
import os
import json
import pickle
import shutil
from os import listdir
from os.path import join, abspath, dirname, isdir, isfile, basename
import numpy as np
import sys

from pybullet_tools.bullet_utils import query_yes_no, has_srl_stream
from pybullet_tools.camera_utils import adjust_camera_pose
from pybullet_tools.pose_utils import ObjAttachment, has_getch
from pybullet_tools.utils import reset_simulation, VideoSaver, wait_unlocked, get_aabb_center, load_yaml, \
    set_camera_pose, get_aabb_extent, set_camera_pose2, invert

from lisdf_tools.lisdf_loader import load_lisdf_pybullet
from lisdf_tools.lisdf_planning import Problem
from lisdf_tools.image_utils import make_composed_image_multiple_episodes, images_to_gif

from world_builder.actions import apply_actions
from data_generator.run_utils import copy_dir_for_process, process_all_tasks

from pigi_tools.data_utils import get_plan, get_body_map, get_multiple_solutions, add_to_planning_config, \
    load_planning_config, get_world_aabb, check_unrealistic_placement_z, get_goals


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


def load_pigi_data(run_dir, use_gui=True, width=1440, height=1120, verbose=False):
    """ for replaying """

    exp_dir = copy_dir_for_process(run_dir, tag='replaying', verbose=verbose)
    plan = get_plan(run_dir, skip_multiple_plans=True)
    commands = pickle.load(open(join(exp_dir, 'commands.pkl'), "rb"))
    world = load_lisdf_pybullet(exp_dir, use_gui=use_gui, width=width, height=height, verbose=False)

    problem = Problem(world)
    if verbose:
        world.summarize_all_objects()
    body_map = get_body_map(run_dir, world, larger=False)
    load_attachments(run_dir, world, body_map=body_map)

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
        world.ATTACHMENTS[body] = ObjAttachment(parent, parent_link, grasp_pose, body)


def load_replay_conf(conf_path):
    c = load_yaml(conf_path)
    for k, v in c.items():
        if v == 'none':
            c[k] = None

    # from types import SimpleNamespace
    # conf = SimpleNamespace(**kwargs)

    c['save_jpg'] = c['save_jpg'] or c['save_composed_jpg'] or c['save_gif']
    c['use_gym'] = c['use_gym'] and has_srl_stream()
    c['step_by_step'] = c['step_by_step'] and has_getch()
    # c['cases'] = get_sample_envs_for_rss(task_name=c['task_name'], count=None)

    if c['step_by_step']:
        c['save_mp4'] = False

    if c['given_path']:
        c['visualize_collisions'] = True
        c['parallel'] = False
    else:
        if 'rerun' in c['given_path']:
            c['save_jpg'] = False
            c['save_composed_jpg'] = False
            c['gif'] = False
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

    return c


def run_replay(config_yaml_file, load_data_fn):
    c = load_replay_conf(config_yaml_file)

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

        planning_config = load_planning_config(run_dir)
        if False and 'obs_camera_pose' in planning_config:
            ### not working
            camera_pose = planning_config['obs_camera_pose']
            # camera_pose = adjust_camera_pose(camera_pose)
            set_camera_pose2(camera_pose)
        else:
            aabb = world.get_world_aabb()
            center = get_aabb_center(aabb)
            extent = get_aabb_extent(aabb)
            camera_point = center + extent / 2
            set_camera_pose(camera_point, center)


def run_one(run_dir_ori, load_data_fn=load_pigi_data, task_name=None, given_path=None, given_dir=None, cases=None,
            parallel=False, skip_if_processed_recently=False, check_time=None, preview_scene=False, step_by_step=False,
            use_gym=False, auto_play=True, verbose=False, width=1280, height=800, fx=600, time_step=0.02,
            camera_point=(8.5, 2.5, 3), target_point=(0, 2.5, 0), camera_kwargs=None, camera_movement=None,
            light_conf=None, check_collisions=False, cfree_range=0.1, visualize_collisions=False,
            evaluate_quality=False, save_jpg=False, save_composed_jpg=False, save_gif=False,
            save_animation_json=False, save_mp4=False, mp4_side_view=False, mp4_top_view=False):

    world, problem, exp_dir, run_dir, commands, plan, body_map = load_data_fn(
        run_dir_ori, use_gui=not use_gym, width=width, height=height, verbose=verbose
    )
    set_replay_camera_pose(world, run_dir, camera_kwargs, camera_point, target_point)
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
        run_one_in_isaac_gym(exp_dir, run_dir, world, problem, commands, plan, body_map, width=width, height=height,
                             camera_point=camera_point, target_point=target_point, camera_kwargs=camera_kwargs,
                             camera_movement=camera_movement, light_conf=light_conf, save_gif=save_gif,
                             save_mp4=save_mp4, mp4_side_view=mp4_side_view, mp4_top_view=mp4_top_view)

    elif save_mp4:
        save_mp4_in_pybullet(run_dir, problem, commands, plan, time_step=time_step)

    else:
        run_one_in_pybullet(run_dir, run_dir_ori, world, problem, commands, plan, body_map, task_name=task_name,
                            step_by_step=step_by_step, auto_play=auto_play, check_collisions=check_collisions,
                            cfree_range=cfree_range, visualize_collisions=visualize_collisions,
                            evaluate_quality=evaluate_quality, zoomin_kwargs=zoomin_kwargs, time_step=time_step,
                            save_jpg=save_jpg, save_composed_jpg=save_composed_jpg, save_gif=save_gif, verbose=verbose)

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


def save_mp4_in_pybullet(run_dir, problem, commands, plan, time_step=0.025):
    video_path = join(run_dir, 'replay.mp4')
    with VideoSaver(video_path):
        apply_actions(problem, commands, time_step=time_step, verbose=False, plan=plan)
    print('saved to', abspath(video_path))


def run_one_in_pybullet(run_dir, run_dir_ori, world, problem, commands, plan, body_map, task_name=None,
                        step_by_step=False, auto_play=True, check_collisions=False, cfree_range=0.1, time_step=0.02,
                        visualize_collisions=False, evaluate_quality=False, zoomin_kwargs=None,
                        save_jpg=False, save_composed_jpg=False, save_gif=False, verbose=False):
    run_name = basename(run_dir)
    if not auto_play:
        wait_unlocked()
    answer = True
    if not auto_play:
        answer = query_yes_no(f"start replay {run_name}?", default='yes')
    if answer:
        time_step = 2e-5 if save_jpg else time_step
        time_step = None if step_by_step else time_step
        results = apply_actions(problem, commands, time_step=time_step, verbose=verbose, plan=plan,
                                body_map=body_map, save_composed_jpg=save_composed_jpg, save_gif=save_gif,
                                check_collisions=check_collisions, cfree_range=cfree_range,
                                visualize_collisions=visualize_collisions)

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

        """ animation of how objects move """
        if save_gif:
            episodes = results
            h, w, _ = episodes[0].shape
            crop = (0, h // 3 - h // 30, w, 2 * h // 3 - h // 30)
            gif_name = 'replay.gif'
            images_to_gif(world.img_dir, gif_name, episodes, crop=crop)

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


def run_one_in_isaac_gym(exp_dir, run_dir, world, problem, commands, plan, body_map, width=1280, height=800,
                         camera_point=(8.5, 2.5, 3), target_point=(0, 2.5, 0), camera_kwargs=None, camera_movement=None,
                         light_conf=None, save_gif=False, save_mp4=False, mp4_side_view=False, mp4_top_view=False):
    from isaac_tools.gym_utils import load_lisdf_isaacgym, record_actions_in_gym, set_camera_target_body

    gym_world = load_lisdf_isaacgym(abspath(exp_dir), camera_width=width, camera_height=height,
                                    camera_point=camera_point, target_point=target_point)
    set_camera_target_body(gym_world, run_dir)

    save_name = basename(exp_dir).replace('temp_', '')

    #####################################################

    ## set better camera view for making gif screenshots
    removed = []
    if mp4_side_view or mp4_top_view:
        removed, camera_kwargs = get_gym_actors_to_ignore(
            gym_world, world, run_dir, mp4_side_view=False, mp4_top_view=False
        )

    if camera_kwargs is not None:
        camera_point = camera_kwargs['camera_point']
        camera_target = camera_kwargs['camera_target']
        gym_world.set_viewer_target(camera_point, camera_target)
        gym_world.set_camera_target(gym_world.cameras[0], camera_point, camera_target)
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

    #####################################################

    img_dir = join(exp_dir, 'gym_images')
    gif_name = 'gym_replay.gif'
    os.mkdir(img_dir)
    gif_name = record_actions_in_gym(problem, commands, gym_world, img_dir=img_dir, body_map=body_map,
                                     time_step=0, verbose=False, plan=plan, save_gif=save_gif, gif_name=gif_name,
                                     save_mp4=save_mp4, camera_movement=camera_movement,
                                     ignore_actors=removed)
    # gym_world.wait_if_gui()

    if mp4_side_view:
        save_name = save_name + '_side'
    if mp4_top_view:
        save_name = save_name + '_top'
    new_file = join('gym_images', save_name + '.gif')
    if save_gif:
        # shutil.copy(join(exp_dir, gif_name), join(run_dir, gif_name))
        print('moved gif to {}'.format(join(run_dir, new_file)))
        shutil.move(join(exp_dir, gif_name), new_file)
    if save_mp4:
        mp4_name = gif_name.replace('.gif', '.mp4')
        new_mp4_name = new_file.replace('.gif', '.mp4')
        shutil.move(join(mp4_name), new_mp4_name)
        print('moved mp4 to {}'.format(join(new_mp4_name)))
    del (gym_world.simulator)


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
            world = load_lisdf_pybullet(exp_dir, use_gui=not USE_GYM or debug,
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


def get_gym_actors_to_ignore(gym_world, world, run_dir, mp4_side_view=False, mp4_top_view=False):
    removed = []
    world_aabb = get_world_aabb(run_dir)
    x, y, _ = get_aabb_center(world_aabb)
    if mp4_side_view:
        x2 = 8
        z2 = 3
    if mp4_top_view:
        x2 = x + 0.1
        z2 = 9
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
        if 'storedinspace' == goals[-1][0] or True:
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
    camera_kwargs = dict(camera_point=[x2, y, z2], camera_target=[x, y, 1])
    return removed, camera_kwargs


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