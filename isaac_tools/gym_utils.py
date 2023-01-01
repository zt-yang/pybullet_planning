import os.path
import random
import sys
from os.path import join, isdir, abspath, dirname
import time
from pprint import pprint
import numpy as np

from pybullet_tools.bullet_utils import nice, equal
from pybullet_tools.utils import pose_from_tform, get_pose, get_joint_name, get_joint_position, get_movable_joints
from isaac_tools.urdf_utils import load_lisdf, test_is_robot

ASSET_PATH = join(dirname(__file__), '..', 'assets')

"""
# Note to myself on setting up slr_stream $ IsaacGym
1. clone git@gitlab.com:nvidia_srl/caelan/srl_stream.git
2. download isaac gym from https://developer.nvidia.com/isaac-gym/download, 
    a. follow instruction in docs/install.html to install isaacgym 
        `(cd ~/Documents/isaacgym/python; pip install -e .)`
3. add python path to srl_stream (srl_stream/src)
4. `pip install setuptools_scm trimesh`
5. run from terminal. It will hang if ran in pycharm
    `cd tests; python test_gym.py`
"""


def load_one_world(gym_world, lisdf_dir, offset=None, loading_effect=False,
                   robots=True, world_index=None, assets=None, 
                   update_viewer=True, test_camera_pose=False, **kwargs):
    movable_keywords = ['bottle', 'medicine', 'veggie', 'meat', 'braiser']
    cabinet_keywards = ['cabinettop', 'cabinettall', 'minifridge', 'dishwasher', 'microwavehanging'] ## , 'ovencounter'
    loading_objects = {}
    loading_positions = {}

    def is_loading_objects(name, keywords):
        for k in keywords:
            if k in name:
                return True
        return False

    for name, path, scale, is_fixed, pose, positions in load_lisdf(lisdf_dir, robots=robots, **kwargs):
        is_robot = test_is_robot(name)
        if test_camera_pose and not is_robot:
            continue

        if world_index is not None:
            name = f'w{world_index}_{name}'

        pose = pose_from_tform(pose)
        if offset is not None:
            pose = (pose[0] + offset, pose[1])

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
                    gravity_comp=is_robot, collapse=False, vhacd=False)
                if assets is not None:
                    assets[path] = asset
            else:
                asset = assets[path]

        # if verbose:
        #     print(f"Name: {name} | Fixed: {is_fixed} | Scale: {scale:.3f} | Path: {path}")
        actor = gym_world.create_actor(asset, name=name, scale=scale)
        if color is not None:
            gym_world.set_color(actor, color)

        gym_world.set_pose(actor, pose)
        if positions is not None and 'pr2' not in name:
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
    if update_viewer:
        gym_world.simulator.update_viewer()
    return assets, (loading_objects, loading_positions)


def load_lisdf_isaacgym(lisdf_dir, robots=True, pause=False, loading_effect=False,
                       camera_point=(8.5, 2.5, 3), camera_target=(0, 2.5, 0), return_wconf=False,
                        camera_width=2560, camera_height=1600, **kwargs):
    sys.path.append('/home/yang/Documents/playground/srl_stream/src')
    from srl_stream.gym_world import create_single_world, default_arguments

    # TODO: Segmentation fault - possibly cylinders & mimic joints
    gym_world = create_single_world(args=default_arguments(use_gpu=False), spacing=5.)
    gym_world.set_viewer_target(camera_point, target=camera_target)

    loading = load_one_world(gym_world, lisdf_dir, robots=robots,
                             loading_effect=loading_effect, **kwargs)
    camera = gym_world.create_camera(width=camera_width, height=camera_height, fov=60)
    gym_world.set_camera_target(camera, camera_point, camera_target)

    if loading_effect:
        wconfs = record_gym_world_loading_objects(gym_world, loading, return_wconf=return_wconf, **kwargs)
        if return_wconf:
            return wconfs
    else:
        img_file = os.path.join(lisdf_dir, 'gym_scene.png')
        gym_world.save_image(camera, image_type='rgb', filename=img_file)

    if pause:
        gym_world.wait_if_gui()
    return gym_world


def record_gym_world_loading_objects(gym_world, loading_objects, world_index=None,
                                     img_dir=None, verbose=False,  ## for generating gif
                                     return_wconf=True, offset=None):  ## for loading multiple
    frames = 150
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
    delays['braiserbody#1'] = random.uniform(0, 15)
    delays['braiserlid#1'] = random.uniform(15, 25)

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


def update_gym_world_by_pb_world(gym_world, pb_world, pause=False, verbose=False, offset=None):
    for actor in gym_world.get_actors():
        name = gym_world.get_actor_name(actor)
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
    if offset is not None:
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
                       camera_point=(34, 12, 10), camera_target=(0, 12, 0),
                        # camera_width=2560, camera_height=1600,
                        camera_width=3840, camera_height=2160,
                       loading_effect=False, **kwargs):
    sys.path.append('/home/yang/Documents/playground/srl_stream/src')
    from srl_stream.gym_world import create_single_world, default_arguments

    gym_world = create_single_world(args=default_arguments(use_gpu=True), spacing=0)
    gym_world.set_viewer_target(camera_point, target=camera_target)

    camera = gym_world.create_camera(width=camera_width, height=camera_height, fov=60)
    gym_world.set_camera_target(camera, camera_point, camera_target)

    num_worlds = num_rows * num_cols
    offsets = []
    all_wconfs = []
    assets = {}
    for i in range(num_worlds):
        ori_dir = ori_dirs[i]
        print('------------------\n', ori_dir.replace('/home/yang/Documents/', ''), '\n')
        row = i // num_rows
        col = i % num_rows
        offset = np.asarray([col * world_size[0]*0.75, row * world_size[1]*0.75, 0])
        offsets.append(offset)
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


def init_isaac_world():
    from isaacgym import gymapi, gymutil
    args = gymutil.parse_arguments()

    # initialize gym
    gym = gymapi.acquire_gym()

    # configure sim
    sim_params = gymapi.SimParams()
    sim_params.dt = dt = 1.0 / 60.0
    if args.physics_engine == gymapi.SIM_FLEX:
        pass
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu

    sim_params.use_gpu_pipeline = False
    if args.use_gpu_pipeline:
        print("WARNING: Forcing CPU pipeline.")

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

    # add ground plane
    plane_params = gymapi.PlaneParams()
    gym.add_ground(sim, plane_params)

    # create viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** Failed to create viewer")
        quit()

    return gym, sim, viewer


def record_actions_in_gym(problem, commands, gym_world=None, img_dir=None, gif_name='gym_replay.gif',
                          time_step=0.5, verbose=False, plan=None, return_wconf=False, world_index=None):
    """ act out the whole plan and event in the world without observation/replanning """
    from world_builder.actions import adapt_action, apply_actions
    from world_builder.world import State
    from world_builder.actions import Action, AttachObjectAction
    from pybullet_tools.utils import wait_for_duration

    if commands is None:
        return
    state_event = State(problem.world)
    camera = None if gym_world is None else gym_world.cameras[0]
    filenames = []
    frame_gap = 3
    wconfs = []
    for i, action in enumerate(commands):
        if verbose:
            print(i, action)
        action = adapt_action(action, problem, plan)
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
            update_gym_world_by_pb_world(gym_world, problem.world)
            if img_dir is not None and i % frame_gap == 0:
                # img_file = join(img_dir, f'{i}.png')
                # gym_world.get_rgba_image(camera, image_type='rgb', filename=img_file)  ##
                img_file = gym_world.get_rgba_image(camera)
                filenames.append(img_file)

    if return_wconf:
        return wconfs
    images_to_gif(img_dir, gif_name, filenames)
    return gif_name


def images_to_gif(img_dir, gif_name, filenames):
    import imageio
    start = time.time()
    gif_file = join(img_dir, gif_name)
    print(f'saving to {abspath(gif_file)} with {len(filenames)} frames')
    with imageio.get_writer(gif_file, mode='I') as writer:
        for filename in filenames:
            # image = imageio.imread(filename)
            writer.append_data(filename)

    print(f'saved to {abspath(gif_file)} with {len(filenames)} frames in {round(time.time() - start, 2)} seconds')
    return gif_file


def images_to_mp4(images=[], img_dir='images', mp4_name='video.mp4'):
    import cv2
    import os

    fps = 20
    if isinstance(images[0], str):
        images = [img for img in os.listdir(img_dir) if img.endswith(".png")]
        frame = cv2.imread(os.path.join(img_dir, images[0]))
    else:
        frame = images[0]
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') ## cv2.VideoWriter_fourcc(*'XVID') ## cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(mp4_name, fourcc, fps, (width, height))

    for image in images:
        if isinstance(images[0], str):
            image = cv2.imread(os.path.join(img_dir, image))
        elif isinstance(images[0], np.ndarray) and image.shape[-1] == 4:
            image = image[:, :, :3]
            image = image[...,[2,1,0]].copy() ## RGB to BGR for cv2
        video.write(image)

    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    ## broken, need to copy test dir to test_cases folder for relative asset paths to be found
    lisdf_dir = '/home/caelan/Programs/interns/yang/kitchen-worlds/test_cases/tt_one_fridge_pick_2'
    lisdf_dir = '/home/yang/Documents/fastamp-data/tt_two_fridge_in/4'
    world = load_lisdf_isaacgym(lisdf_dir, pause=True)
