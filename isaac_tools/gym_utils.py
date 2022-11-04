import os.path
import sys
from os.path import join, isdir, abspath, dirname
import time
import numpy as np

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


def load_one_world(gym_world, lisdf_dir, offset=None, robots=True, world_index=None,
                   update_viewer=True, **kwargs):
    for name, path, scale, is_fixed, pose, positions in load_lisdf(lisdf_dir, robots=robots, **kwargs):
        if 'veggiegreenpepper' in name or 'meatturkeyleg' in name or 'veggietomato' in name:
            print('!!! skipping', name)
            continue
        is_robot = test_is_robot(name)
        asset = gym_world.simulator.load_asset(
            asset_file=path, root=None, fixed_base=is_fixed or is_robot,  # y_up=is_robot,
            gravity_comp=is_robot, collapse=False, vhacd=False)
        if world_index is not None:
            name = f'w{world_index}_{name}'
        print(f"Name: {name} | Fixed: {is_fixed} | Scale: {scale:.3f} | Path: {path}")
        actor = gym_world.create_actor(asset, name=name, scale=scale)
        pose = pose_from_tform(pose)
        if offset is not None:
            pose = (pose[0] + offset, pose[1])
        gym_world.set_pose(actor, pose)
        if positions is not None and 'pr2' not in name:
            joint_names = gym_world.get_joint_names(actor)
            joint_positions = gym_world.get_joint_positions(actor)
            joint_positions = {joint_names[i]: joint_positions[i] for i in range(len(joint_names))}
            joint_positions.update(positions)
            gym_world.set_joint_positions(actor, list(joint_positions.values()))
    if update_viewer:
        gym_world.simulator.update_viewer()


def load_lisdf_isaacgym(lisdf_dir, robots=True, pause=False, num_rows=5, num_cols=5,
                       camera_point=(8.5, 2.5, 3), camera_target=(0, 2.5, 0),
                        camera_width=2560, camera_height=1600, **kwargs):
    sys.path.append('/home/yang/Documents/playground/srl_stream/src')
    from srl_stream.gym_world import create_single_world, default_arguments

    # TODO: Segmentation fault - possibly cylinders & mimic joints
    gym_world = create_single_world(args=default_arguments(use_gpu=False), spacing=5.)
    gym_world.set_viewer_target(camera_point, target=camera_target)

    load_one_world(gym_world, lisdf_dir, robots=robots, **kwargs)

    img_file = os.path.join(lisdf_dir, 'gym_scene.png')
    camera = gym_world.create_camera(width=camera_width, height=camera_height, fov=60)
    gym_world.set_camera_target(camera, camera_point, camera_target)
    gym_world.save_image(camera, image_type='rgb', filename=img_file)

    if pause:
        gym_world.wait_if_gui()
    return gym_world


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


def update_gym_world_by_wconf(gym_world, wconf, offset=None):
    for actor in gym_world.get_actors():
        name = gym_world.get_actor_name(actor)
        if name not in wconf:
            continue

        data = wconf[name]
        if not data['is_static']:
            pose = data['pose']
            if offset is not None:
                pose = (pose[0] + offset, pose[1])
            gym_world.set_pose(actor, pose)

        if 'joint_state' in data:
            joint_state = data['joint_state']
            joints = gym_world.get_joint_names(actor)
            positions = list(map(joint_state.get, joints))
            gym_world.set_joint_positions(actor, positions)

    gym_world.simulator.update_viewer()


def load_envs_isaacgym(ori_dirs, robots=True, pause=False, num_rows=5, num_cols=5,
                       camera_point=(34, 12, 10), camera_target=(0, 12, 0),
                        camera_width=2560, camera_height=1600, **kwargs):
    sys.path.append('/home/yang/Documents/playground/srl_stream/src')
    from srl_stream.gym_world import create_single_world, default_arguments

    gym_world = create_single_world(args=default_arguments(use_gpu=False), spacing=5.)
    gym_world.set_viewer_target(camera_point, target=camera_target)

    world_size = 6
    num_worlds = min(num_rows * num_cols, 24)
    offsets = []
    for i in range(num_worlds):
        ori_dir = ori_dirs[i]
        print('------------------\n', ori_dir, '\n\n')
        row = i // num_rows
        col = i % num_rows
        print(row, col)
        offset = np.asarray([row * world_size, col * world_size, 0])
        offsets.append(offset)
        load_one_world(gym_world, ori_dir, offset=offset, robots=robots,
                       update_viewer=False, world_index=i, **kwargs)
    gym_world.simulator.update_viewer()

    # task_name = os.path.basename(task_dir)
    # img_file = os.path.join(task_dir, '..', f'{task_name}_gym_scene.png')
    # camera = gym_world.create_camera(width=camera_width, height=camera_height, fov=60)
    # gym_world.set_camera_target(camera, point_from, point_to)
    # gym_world.save_image(camera, image_type='rgb', filename=img_file)

    if pause:
        gym_world.wait_if_gui()
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
            wconfs.append(problem.world.get_wconf(world_index=world_index))
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
    gif_file = join(img_dir, '..', gif_name)
    print(f'saving to {abspath(gif_file)} with {len(filenames)} frames')
    with imageio.get_writer(gif_file, mode='I') as writer:
        for filename in filenames:
            # image = imageio.imread(filename)
            writer.append_data(filename)

    print(f'saved to {abspath(gif_file)} with {len(filenames)} frames in {round(time.time() - start, 2)} seconds')


if __name__ == "__main__":
    ## broken, need to copy test dir to test_cases folder for relative asset paths to be found
    lisdf_dir = '/home/caelan/Programs/interns/yang/kitchen-worlds/test_cases/tt_one_fridge_pick_2'
    lisdf_dir = '/home/yang/Documents/fastamp-data/tt_two_fridge_in/4'
    world = load_lisdf_isaacgym(lisdf_dir, pause=True)
