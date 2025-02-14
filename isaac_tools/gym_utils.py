import os.path
import random
import sys
from os.path import join, isdir, abspath, dirname, isfile, basename
import math
import time
from pprint import pprint
from tqdm import tqdm
import shutil
import numpy as np
import json

from world_builder.world_utils import get_camera_zoom_in
from pybullet_tools.bullet_utils import nice, equal, get_datetime
from pybullet_tools.utils import pose_from_tform, get_pose, get_joint_name, get_joint_position, \
    get_movable_joints, Euler, quat_from_euler
from pybullet_tools.logging_utils import print_pink, print_green, print_debug, print_blue, \
    print_cyan, print_red

from isaac_tools.urdf_utils import load_lisdf, test_is_robot
# from isaac_tools.gym_world import create_single_world
from isaac_tools.isaac_gym_world import create_single_world

from lisdf_tools.image_utils import images_to_mp4, images_to_gif
from pigi_tools.data_utils import load_planning_config, get_worlds_aabb, \
    get_instance_info, exist_instance

ASSET_PATH = join(dirname(__file__), '..', 'assets')


def load_one_world(gym_world, lisdf_dir, offset=None, loading_effect=False,
                   robots=True, world_index=None, assets=None, 
                   update_viewer=True, test_camera_pose=False, save_obj_shots=False,
                   debug_gym_installation=True, **kwargs):
    movable_keywords = ['bottle', 'medicine', 'veggie', 'meat', 'braiser']
    cabinet_keywards = ['cabinettop', 'cabinettall', 'minifridge', 'dishwasher', 'microwavehanging'] ## , 'ovencounter'
    loading_objects = {}
    loading_positions = {}

    def is_loading_objects(name, keywords):
        for k in keywords:
            if k in name:
                return True
        return False

    if save_obj_shots:
        load_obj_shots_bg(gym_world)

    print_debug('... started loading objects')

    for name, path, scale, is_fixed, pose, positions in load_lisdf(lisdf_dir, robots=robots, **kwargs):
        is_robot = test_is_robot(name)
        if test_camera_pose and not is_robot:
            continue

        ## for rendering zoom up only
        # if 'mm_braiser_' in lisdf_dir and ('cabinettop' in name or 'cabinetupper' in name or 'shelf_lower' in name):
        #     continue

        if world_index is not None:
            name = f'w{world_index}_{name}'

        pose = pose_from_tform(pose)
        if 'pr2' in name:
            pose = (np.array([0, 0, 0]), np.array([0, 0, 0, 1]))
        if offset is not None:
            pose = (pose[0] + offset, pose[1])

        ## for rendering zoom up only
        # if 'mm_braiser_' in lisdf_dir and ('medicine' in name or 'bottle' in name) and pose[0][2] > 1.5:
        #     continue

        if loading_effect and is_loading_objects(name, movable_keywords):
            drop_height = 0.4
            if 'braiserlid' in name:
                drop_height += 0.1
            loading_objects[name] = pose[0][2], drop_height
            pose = (pose[0] + np.array([0, 0, drop_height]), pose[1])

        color = None
        if isinstance(path, tuple):
            (w, l, h), color = path
            asset = gym_world.simulator.box_asset(w, length=l, height=h, fixed_base=True)
        else:
            ## cache assets so we can create more actors
            if assets is None or path not in assets:
                asset = gym_world.simulator.load_asset(
                    asset_file=path, root=None, fixed_base=is_fixed or is_robot,  # y_up=is_robot,
                    gravity_comp=is_robot, collapse=False, vhacd=False
                )
                if assets is not None:
                    assets[path] = asset
            else:
                asset = assets[path]

        # if verbose:
        #     print(f"Name: {name} | Fixed: {is_fixed} | Scale: {scale:.3f} | Path: {path}")
        actor = gym_world.create_actor(asset, name=name, scale=scale)
        if color is not None:
            gym_world.set_color(actor, color)

        if positions is not None:
            joint_positions = gym_world.get_joint_positions(actor)
            joint_names = gym_world.get_joint_names(actor)
            joint_positions = {joint_names[i]: joint_positions[i] for i in range(len(joint_names))}
            joint_positions.update(positions)
            joint_positions = list(joint_positions.values())

            if loading_effect and is_loading_objects(name, cabinet_keywards):
                opened_positions = []
                jj = gym_world.get_joint_properties(actor)
                if jj['hasLimits'].any():
                    for i in range(len(jj['upper'])):
                        pstn = joint_positions[i]
                        lower = jj['lower'][i]
                        upper = jj['upper'][i]
                        # if upper - lower > 1.57:
                        #     upper = lower + 1.57
                        upper = lower + 1.57 * random.uniform(0.25, 1)
                        if pstn == upper:
                            opened_positions.append((lower, pstn, -1))
                        elif pstn == lower:
                            opened_positions.append((upper, pstn, 1))
                        else:
                            opened_positions.append((pstn, pstn, 0))
                loading_positions[name] = opened_positions
                joint_positions = [p[0] for p in opened_positions]
            gym_world.set_joint_positions(actor, joint_positions)

        if save_obj_shots:
            img_file = join('gym_images', 'obj', f'obj-{name}.png')
            take_obj_shot(gym_world, actor, img_file, pose)

        gym_world.set_pose(actor, pose)

        # if debug_gym_installation:
        #     gym_world.simulator.wait_for_duration()
        #     sys.exit()

    if update_viewer:
        gym_world.simulator.update_viewer()

    print_debug('... finished loading objects')
    return assets, (loading_objects, loading_positions)


def load_lisdf_isaacgym(lisdf_dir, robots=True, pause=False, loading_effect=False,
                        camera_point=(8.5, 2.5, 3), target_point=(0, 2.5, 0), return_wconf=False,
                        camera_width=2560, camera_height=1600, load_cameras=False,
                        save_obj_shots=False, **kwargs):

    if 'full_kitchen' in lisdf_dir or 'mm_' in lisdf_dir or 'tt_' in lisdf_dir:
        config = load_planning_config(lisdf_dir)
        if 'world_aabb' not in config:
            world_aabb = get_worlds_aabb([lisdf_dir])
        else:
            world_aabb = config['world_aabb']
        y = (world_aabb[1][1] + world_aabb[0][1]) / 2
        camera_point = (8, y, 3)
        target_point = (0, y, 1)

    print_pink(f"\ncamera_point = {camera_point},\ttarget_point = {target_point},"
               f"\tcamera_width = {camera_width},\tcamera_height = {camera_height}\n")

    # TODO: Segmentation fault - possibly cylinders & mimic joints
    gym_world = create_single_world(use_gpu=False, spacing=5.)
    gym_world.set_viewer_target(camera_point, target=target_point)

    camera = gym_world.create_camera(width=camera_width, height=camera_height, fov=60)
    gym_world.set_camera_target(camera, camera_point, target_point)

    loading = load_one_world(gym_world, lisdf_dir, robots=robots,
                             loading_effect=loading_effect, save_obj_shots=save_obj_shots, **kwargs)

    ## add floor for one world
    asset = gym_world.simulator.box_asset(50, length=50, height=0.1, fixed_base=True)
    actor = gym_world.create_actor(asset, name='floor', scale=1)
    gym_world.set_color(actor, (0.4, 0.4, 0.4, 1))
    gym_world.set_pose(actor, (np.array([0, 0, -0.04]), np.array([0, 0, 0, 1])))

    if save_obj_shots:
        gym_world.set_camera_target(camera, camera_point, target_point)

    if load_cameras:
        ## load planning conf to get camera poses
        path = '/home/yang/Documents/cognitive-architectures/bullet/assets/models/RGBCamera/102523/mobility.urdf'
        asset = gym_world.simulator.load_asset(
            asset_file=path, root=None, fixed_base=True,
            gravity_comp=False, collapse=False, vhacd=False)
        gym_world.simulator.set_light(direction=np.asarray([1, 1, 1]),
                                      intensity=np.asarray([0.5, 0.5, 0.5]))

        config = load_planning_config(lisdf_dir)
        for i in range(1, len(config['camera_kwargs'])):
            args = config['camera_kwargs'][i]
            x1, _, z1 = args['camera_point']
            x2, _, z2 = args['target_point']
            pitch = - np.arctan((z2 - z1) / (x2 - x1))
            pose = (args['camera_point'], quat_from_euler(Euler(pitch=pitch)))
            actor = gym_world.create_actor(asset, name=f'camera_{i}', scale=0.1)
            gym_world.set_pose(actor, pose)

        for j in range(len(config['camera_zoomins'])):
            args = config['camera_zoomins'][j]
            pose = gym_world.get_pose(gym_world.get_actor(args['name']))
            dx, _, dz = args['d']
            d = np.asarray(args['d'])
            point = pose[0] + d
            if 'sink' in args['name'] or 'braiser' in args['name']:
                point[2] = 1.79
            else:
                point[2] = point[2] * 0.8
            pitch = - np.arctan(point[2] / dx)
            pose = (point, quat_from_euler(Euler(pitch=pitch)))
            actor = gym_world.create_actor(asset, name=f'camera_{i+1+j}', scale=0.1)
            gym_world.set_pose(actor, pose)

        ## looking from the right side for the paper
        gym_world.set_camera_target(camera, (6, 9, 3), (0, 4, 1))  ## from right
        gym_world.set_camera_target(camera, (6, -2.5, 3), (0, 2.5, 1))  ## from left
        # gym_world.set_camera_target(camera, (8, 3.5, 3), (0, 3.5, 1))

    if loading_effect:
        wconfs = record_gym_world_loading_objects(gym_world, loading, return_wconf=return_wconf, **kwargs)
        if return_wconf:
            return wconfs
    else:
        img_file = join(lisdf_dir, 'gym_scene.png')
        gym_world.save_image(camera, image_type='rgb', filename=img_file)
        if load_cameras:
            run_name = basename(lisdf_dir).replace('temp_', '')
            shutil.copy(img_file, join('gym_images', f'{run_name}_cameras.png'))

    if pause:
        gym_world.wait_if_gui()
    return gym_world


def record_gym_world_loading_objects(gym_world, loading_objects, world_index=None,
                                     img_dir=None, verbose=False,  ## for generating gif
                                     return_wconf=True, offset=None):  ## for loading multiple
    frames = 140
    dt = 0.01
    g = 9.8
    w0 = 0.5
    camera = gym_world.cameras[0]
    filenames = []
    frame_gap = 1
    gif_name = 'gym_scene.gif'

    loading_objects, loading_positions = loading_objects

    h_starts = {k: v[0]+v[1] for k, v in loading_objects.items()}
    h_ends = {k: v[0] for k, v in loading_objects.items()}
    delays = {k: random.uniform(0, 20) for k in h_ends}
    delays[f'w{world_index}_braiserbody#1'] = d = random.uniform(0, 15)
    delays[f'w{world_index}_braiserlid#1'] = d + random.uniform(2, 5)

    delays.update({k: [random.uniform(0, 20) for v in vv] for k, vv in loading_positions.items()})
    a = {k: [random.uniform(g/2, g) for v in vv] for k, vv in loading_positions.items()}

    wconfs = []
    for i in range(frames):
        t = i * dt
        moved = []
        wconf = {}
        for actor in gym_world.get_actors():
            name = gym_world.get_actor_name(actor)
            if world_index is not None and f"w{world_index}_" not in name:
                continue
            wconf[name] = {'joint_state': {}}
            pose = gym_world.get_pose(actor)

            if name in h_ends:
                t0 = delays[name] * dt
                if t > t0:
                    if offset is not None:
                        pose = (pose[0] + offset, pose[1])
                    if pose[0][2] > h_ends[name]:
                        new_z = h_starts[name] - 0.5 * g * (t-t0) ** 2
                        if new_z < h_ends[name]:
                            new_z = h_ends[name]
                        else:
                            moved.append(name)
                        pose = (pose[0] + np.array([0, 0, new_z - pose[0][2]]), pose[1])

                    if not return_wconf:
                        gym_world.set_pose(actor, pose)
            wconf[name]['pose'] = pose

            joint_positions = gym_world.get_joint_positions(actor)
            if len(joint_positions) > 0:
                joint_names = gym_world.get_joint_names(actor)
                if name in loading_positions:
                    new_positions = []
                    for j in range(len(joint_positions)):
                        t0 = delays[name][j] * dt
                        start, end, direction = loading_positions[name][j]
                        if t > t0 and direction != 0:
                            pstn = start - (-1)**direction * (w0 * (t - t0) - 0.5 * a[name][j] * (t-t0) ** 2)
                            if (direction > 0 and pstn < end) or (direction < 0 and pstn):
                                pstn = end
                            else:
                                moved.append((name, j, start, end, '->', pstn))
                            new_positions.append(pstn)
                        else:
                            new_positions.append(joint_positions[j])
                    if not return_wconf:
                        gym_world.set_joint_positions(actor, new_positions)
                    joint_positions = new_positions
                wconf[name]['joint_state'] = {joint_names[i]: joint_positions[i] for i in range(len(joint_positions))}

        if verbose:
            print('t:', i, 'Moved:', moved)

        if return_wconf:
            # wconf = {f"w{world_index}_{k}": v for k, v in wconf.items()}
            wconfs.append(wconf)
        else:
            gym_world.simulator.update_viewer()
            if i % frame_gap == 0:
                img_file = gym_world.get_rgba_image(camera)
                filenames.append(img_file)
    if return_wconf:
        return wconfs
    images_to_gif(img_dir, gif_name, filenames)
    return gif_name


def update_gym_world_by_pb_world(gym_world, pb_world, pause=False, verbose=False, offset=None, ignore_actors=[]):
    for actor in gym_world.get_actors():
        name = gym_world.get_actor_name(actor)
        if name in ignore_actors or name not in pb_world.name_to_body:
            continue
        body = pb_world.name_to_body[name]
        pose = get_pose(body)

        if offset is not None:
            pose = (pose[0] + offset, pose[1])
        gym_world.set_pose(actor, pose)
        if verbose:
            print(f'Name: {name} | Actor: {actor} | Body: {body}')

        joint_state = {}
        for joint in get_movable_joints(body):
            joint_name = get_joint_name(body, joint)
            position = get_joint_position(body, joint)
            joint_state[joint_name] = position
            if verbose:
                print(f'Joint: {joint_name} | Position: {position}')
        joints = gym_world.get_joint_names(actor)
        positions = list(map(joint_state.get, joints))
        gym_world.set_joint_positions(actor, positions)

    if offset is not None or True:
        gym_world.simulator.update_viewer()
    if pause:
        gym_world.wait_if_gui()


def update_gym_world_by_wconf(gym_world, wconf, offsets=None):
    for actor in gym_world.get_actors():
        name = gym_world.get_actor_name(actor)
        if name not in wconf:  ## because some videos are shorter
            continue

        world_index = eval(name[1:name.index('_')])
        data = wconf[name]

        pose = data['pose']
        if offsets is not None:
            pose = (pose[0] + offsets[world_index], pose[1])
        gym_world.set_pose(actor, pose)

        joint_state = data['joint_state']
        joints = gym_world.get_joint_names(actor)
        positions = list(map(joint_state.get, joints))
        gym_world.set_joint_positions(actor, positions)

    gym_world.simulator.update_viewer()


def load_envs_isaacgym(ori_dirs, robots=True, pause=False, num_rows=5, num_cols=5, world_size=(6, 6),
                       camera_point=(34, 12, 10), target_point=(0, 12, 0), verbose=False,
                        # camera_width=2560, camera_height=1600,
                        camera_width=3840, camera_height=2160,
                       loading_effect=False, **kwargs):
    from tqdm import tqdm

    gym_world = create_single_world(use_gpu=True, spacing=0)
    gym_world.set_viewer_target(camera_point, target=target_point)

    camera = gym_world.create_camera(width=camera_width, height=camera_height, fov=60)
    gym_world.set_camera_target(camera, camera_point, target_point)

    num_worlds = num_rows * num_cols
    scale = 1.25 ## 0.75 if world_size[0] > 12 else 1
    offsets = []
    all_wconfs = []
    assets = {}
    for i in tqdm(range(num_worlds)):
        ori_dir = ori_dirs[i]
        row = i // num_rows
        col = i % num_rows
        offset = np.asarray([col * world_size[0]*scale, row * world_size[1]*scale, 0])
        offsets.append(offset)
        if verbose:
            print('------------------\n', ori_dir.replace('/home/yang/Documents/', ''), '\n')
            print(f'... load_envs_isaacgym | row, col = {(row, col)}, offset = {offset}')
        
        assets, loading = load_one_world(gym_world, ori_dir, offset=offset, robots=robots,
            update_viewer=False, world_index=i, loading_effect=loading_effect, assets=assets, **kwargs)
        if loading_effect:
            wconf = record_gym_world_loading_objects(gym_world, loading, world_index=i, 
                                                     return_wconf=True, **kwargs)
            all_wconfs.append(wconf)
    gym_world.simulator.update_viewer()

    # task_name = os.path.basename(task_dir)
    # img_file = os.path.join(task_dir, '..', f'{task_name}_gym_scene.png')
    # camera = gym_world.create_camera(width=camera_width, height=camera_height, fov=60)
    # gym_world.set_camera_target(camera, point_from, point_to)
    # gym_world.save_image(camera, image_type='rgb', filename=img_file)

    if pause:
        gym_world.wait_if_gui()
    if loading_effect:
        return gym_world, offsets, all_wconfs
    return gym_world, offsets


##################################################################


def load_obj_shots_bg(gym_world, vertical=True, background=(1, 1, 1, 1)):
    if vertical:
        w, l, h = 0.5, 8, 3
        x, y, z = -5, -5, 1
    else:
        w, l, h = 6, 6, 0.5
        x, y, z = -4, -4, 0
    asset = gym_world.simulator.box_asset(w, length=l, height=h, fixed_base=True)
    actor = gym_world.create_actor(asset, name='backdrop', scale=1)
    gym_world.set_pose(actor, ((x, y, z), (0, 0, 0, 1)))
    gym_world.set_color(actor, background)
    gym_world.simulator.set_light(
        direction=np.asarray([-1, -1, -1]),
        intensity=np.asarray([0.5, 0.5, 0.5]))


def take_obj_shot(gym_world, actor, img_file, pose, direction='horizonal'):
    camera = gym_world.cameras[-1]
    if direction == 'horizonal':
        to_point = [-4, -4, 1]
        from_point = [-1, -4, 1.5]
    elif direction == 'diagonal':
        to_point = [-4, -4, 1]
        from_point = [-1, -4, 3]
    else:
        to_point = [-4, -4, 1]
        from_point = [-3.9, -4, 4]

    gym_world.set_pose(actor, (to_point, pose[1]))

    if 'CabinetTop_' in img_file:
        d = 0
        if 'CabinetTop_00001' in img_file:
            d = -0.4
        if 'CabinetTop_00002' in img_file:
            d = 0.4
        if 'CabinetTop_00003' in img_file:
            d = -0.5
        to_point[1] += d
        from_point[1] += d

    gym_world.set_camera_target(camera, from_point, to_point)
    gym_world.save_image(camera, image_type='rgb', filename=img_file)


##################################################################


def record_actions_in_gym(problem, commands, gym_world=None, img_dir=None, gif_path='gym_replay.gif',
                          time_step=0.5, verbose=False, plan=None, return_wconf=False,
                          world_index=None, body_map=None, save_gif=True, save_mp4=False,
                          camera_movement=None, ignore_actors=None,
                          frame_gap=3, return_frames=False): ## 3 for single world, 5 for longer horizon
    """ act out the whole plan and event in the world without observation or replaning
    generate videos from scene camera and robot camera
    """
    from world_builder.actions import adapt_action, apply_commands
    from world_builder.world import State
    from world_builder.actions import Action, AttachObjectAction
    from pybullet_tools.utils import wait_for_duration

    os.makedirs(img_dir)
    if commands is None:
        return
    state_event = State(problem.world)

    num_cameras = len(gym_world.cameras)
    camera_names = ['scene', 'head', 'wrist']
    filenames_by_view = {k: [] for k in camera_names}
    # camera = None if gym_world is None else gym_world.cameras[0]

    wconfs = []
    for i, action in enumerate(commands):
        if verbose:
            print(i, action)
        if 'tachObjectAction' in str(action):
            if action.body in body_map:
                action.body = body_map[action.body]
        # action = adapt_action(action, problem, plan, verbose=verbose)
        if action is None:
            continue
        state_event = action.transition(state_event.copy())

        if return_wconf:
            wconfs.append(problem.world.get_wconf(world_index=world_index,
                                                  attachments=state_event.attachments))
        else:
            if isinstance(action, AttachObjectAction):
                print(action.grasp)
            wait_for_duration(time_step)

            """ update gym world """
            update_gym_world_by_pb_world(gym_world, problem.world, ignore_actors=ignore_actors)

            for j in range(num_cameras):
                camera = gym_world.cameras[j]
                camera_name = camera_names[j]
                if j == 0:
                    interpolate_camera_pose(gym_world, i, len(commands), camera_movement)

                view_dir = join(img_dir, camera_name)
                if i == 0:
                    os.makedirs(view_dir)

                if img_dir is not None and (frame_gap == 0 or i % frame_gap == 0):
                    img_file = join(view_dir, f'{i}.npy')
                    # gym_world.get_rgba_image(camera, image_type='rgb', filename=img_file)  ##
                    img = gym_world.get_rgba_image(camera)
                    with open(img_file, 'wb') as f:
                        np.save(f, img)
                    filenames_by_view[camera_name].append(img_file)

    if return_wconf:
        return wconfs

    for camera_name, filenames in filenames_by_view.items():
        if len(filenames) > 0:
            print_debug(f"\ncamera name = {camera_name}\tsaved {len(filenames)} images to make mp4")
            if return_frames:
                return filenames
            this_gif_path = gif_path.replace('.gif', f'_{camera_name}.gif')
            print_green(f'\tcombining numpy files like {filenames[0]}\n\t to make gif {this_gif_path}')
            save_gym_run(img_dir, this_gif_path, filenames, save_gif=save_gif, save_mp4=save_mp4)
    return state_event.attachments


def save_gym_run(img_dir, gif_path, filenames, save_gif=True, save_mp4=True):
    if save_gif:
        images_to_gif(img_dir, gif_path, filenames)
        print('created gif {}'.format(gif_path))
    if save_mp4:
        mp4_path = join(gif_path.replace('.gif', '.mp4'))  ## f'_{get_datetime()}.mp4'
        images_to_mp4(filenames, mp4_path=mp4_path)
        print_green('\tcreated mp4 {} with {} frames\n'.format(mp4_path, len(filenames)))
        return filenames


def interpolate_camera_pose(gym_world, i, length, camera_movement):
    if camera_movement is None:
        return
    target_point_begin = camera_movement['target_point_begin']
    target_point_final = camera_movement['target_point_final']
    camera_point_begin = camera_movement['camera_point_begin']
    camera_point_final = camera_movement['camera_point_final']

    if camera_point_final != camera_point_begin:
        if isinstance(camera_point_final, int):
            ## rotate camera around point begin by radius of camera_point_final
            direction = 1
            if camera_point_final < 0:
                direction = -1
            if i > length / 2:
                i = length - i
            dx = camera_point_final * math.sin(2 * math.pi * direction * i / length)
            dy = camera_point_final * math.cos(2 * math.pi * direction * i / length)
            offset = [dx, dy, 0]
            camera_point = tuple([camera_point_begin[j] + offset[j] for j in range(3)])
        else:
            camera_point = tuple(
                [camera_point_begin[j] + (camera_point_final[j] - camera_point_begin[j]) * i / length
                 for j in range(3)])
    else:
        camera_point = camera_point_begin

    if target_point_final != target_point_begin:
        target_point = tuple(
            [target_point_begin[j] + (target_point_final[j] - target_point_begin[j]) * i / length
             for j in range(3)])
    else:
        target_point = target_point_begin

    gym_world.set_camera_target(gym_world.cameras[0], camera_point, target_point)


def set_camera_target_body(gym_world, run_dir):
    camera_zoomin = get_camera_zoom_in(run_dir)
    if camera_zoomin is not None:
        name = camera_zoomin['name']
        dx, dy, dz = camera_zoomin['d']
        if 'sink' in name:
            dy = dx
            dx = -dx
            if exist_instance(run_dir, '102379'):
                dz += 0.345
        elif 'braiser' in name:
            dx = -dx
        actor = gym_world.get_actor(name)
        pose = gym_world.get_pose(actor)

        camera_target = pose[0]
        instances = get_instance_info(run_dir)
        print(f'\n gym_utils.set_camera_target_body({name}) | {instances[name]} \n')
        if instances[name] in ['00001']:
            camera_target[1] -= 0.4
            camera_target[0] += 0.4
        elif instances[name] in ['00002']:
            camera_target[1] += 0.4
        elif instances[name] in ['00003']:
            camera_target[1] -= 0.5
        camera_point = camera_target + np.array([dx, dy, dz])

        gym_world.set_viewer_target(camera_point, target=camera_target)
        gym_world.set_camera_target(gym_world.cameras[0], camera_point, camera_target)


if __name__ == "__main__":
    ## broken, need to copy test dir to test_cases folder for relative asset paths to be found
    lisdf_dir = '/home/caelan/Programs/interns/yang/kitchen-worlds/test_cases/tt_one_fridge_pick_2'
    lisdf_dir = '/home/yang/Documents/fastamp-data/tt_two_fridge_in/4'
    world = load_lisdf_isaacgym(lisdf_dir, pause=True)
