import numpy as np
from os.path import dirname, basename
import sys
import time

try:
    from isaacgym import gymapi, gymutil
except ImportError:
    print("export LD_LIBRARY_PATH=/home/zhutianyang/miniconda3/envs/kitchen/lib (or wherever you installed isaacgym)")
    sys.exit()


from pybullet_tools.utils import elapsed_time, INF
from pybullet_tools.logging_utils import print_debug, print_cyan, print_red, print_blue


def create_single_world(use_gpu=False, spacing=1.0, **kwargs):
    """
    Creates and returns a single IsaacGymWorld instance.
    Parameters:
        use_gpu (bool): Indicates whether to use the GPU.
        spacing (float): Determines the environment dimensions.
        kwargs: Optional keyword arguments, e.g. 'camera_point' and 'target_point'
                to set the default viewer camera.
    Returns:
        world (IsaacGymWorld): The world object, with its 'simulator' attribute pointing to itself.
    """
    world = IsaacGymWorld(use_gpu=use_gpu, spacing=spacing)
    default_camera_point = kwargs.get('camera_point', [2.0, 2.0, 2.0])
    default_target_point = kwargs.get('target_point', [0.0, 0.0, 0.0])
    world.set_viewer_target(default_camera_point, default_target_point)
    # Alias the simulator attribute to itself so that old references in gym_utils works.
    world.simulator = world
    return world


def init_isaac_world(use_gpu):
    args = gymutil.parse_arguments()

    # initialize gym
    gym = gymapi.acquire_gym()

    # configure sim
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

    # add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(z=1)
    gym.add_ground(sim, plane_params)

    # create viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** Failed to create viewer")
        quit()

    return gym, sim, viewer


class IsaacGymWorld(object):
    """
      - Environment creation (add_env)
      - Actor creation, pose and color setting (create_actor, set_pose, get_pose, set_color)
      - Access to actor information (get_actors, get_actor_name, joint state methods)
      - Camera creation and targeting (create_camera, set_camera_target, set_viewer_target)
      - Viewer updating, image saving and waiting in GUI mode (update_viewer, save_image, wait_if_gui)
    """
    def __init__(self, use_gpu=True, spacing=1.0):
        self.gym, self.sim, self.viewer = init_isaac_world(use_gpu)
        self.envs = []
        self.actor_data = {}      # maps actor handle to metadata (name, asset, color, etc.)
        self.cameras = []         # list of camera sensors
        self.add_env(spacing=spacing)

    def add_env(self, num_per_row=1, spacing=1.0):
        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        env = self.gym.create_env(self.sim, lower, upper, num_per_row)
        self.envs.append(env)
        return env

    def set_viewer_target(self, camera_point, target):
        """
        Set the viewer camera so that it looks from `camera_point` toward `target_point`.
        Both should be iterables of three numbers.
        """
        cp = gymapi.Vec3(*camera_point)
        tp = gymapi.Vec3(*target)
        self.gym.viewer_camera_look_at(self.viewer, None, cp, tp)

    def create_actor(self, asset, name="actor", scale=1.0, pose=None, env_index=0):
        """
        Create an actor from an asset and add it to the specified environment.
        The pose is a tuple (position, orientation) where position is a (3,) np.array and
        orientation is a (4,) np.array (quaternion in x,y,z,w order).
        """
        env = self.envs[env_index]
        if pose is None:
            pose = (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))
        transform = gymapi.Transform()
        transform.p = gymapi.Vec3(*pose[0].tolist())
        transform.r = gymapi.Quat(*pose[1].tolist())
        actor = self.gym.create_actor(env, asset, transform, name, 0, 1)
        if scale is not None:
            self.gym.set_actor_scale(env, actor, scale)
        self.actor_data[actor] = {"name": name, "scale": scale, "asset": asset}
        return actor

    def set_pose(self, actor, pose, env_index=0):
        """
        Set the pose (position and orientation) for an actor.
        """
        env = self.envs[env_index]
        transform = gymapi.Transform()
        transform.p = gymapi.Vec3(*pose[0])
        transform.r = gymapi.Quat(*pose[1])
        # self.gym.set_actor_transform(env, actor, transform)
        base = self.gym.get_actor_root_rigid_body_handle(env, actor)
        self.gym.set_rigid_transform(env, base, transform)

    def get_pose(self, actor, env_index=0):
        """
        Return the (position, orientation) of an actor.
        """
        env = self.envs[env_index]
        transform = self.gym.get_actor_transform(env, actor)
        position = np.array([transform.p.x, transform.p.y, transform.p.z])
        # Note: The quaternion is in (x, y, z, w) order.
        quaternion = np.array([transform.r.x, transform.r.y, transform.r.z, transform.r.w])
        return (position, quaternion)

    def set_color(self, actor, color, link_index=None, env_index=0):
        """
        Set the rigid body color of an actor.
        Color should be a 3-tuple or 4-tuple (RGBA). Here we use only RGB.
        """
        env = self.envs[env_index]
        if len(color) == 4:
            r, g, b, a = color
        else:
            r, g, b = color
        self.actor_data[actor]['color'] = color
        self.gym.set_rigid_body_color(env, actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(r, g, b))

    def get_actors(self):
        """
        Return the list of actor handles.
        """
        return list(self.actor_data.keys())

    def get_actor_name(self, actor):
        """
        Return the name of the given actor.
        """
        return self.actor_data.get(actor, {}).get("name", "unknown")

    def get_joint_positions(self, actor, env_index=0):
        """
        Retrieve the (degree-of-freedom) joint positions for an actor.
        (Requires that the asset exposes DOFs.)
        """
        env = self.envs[env_index]
        # This assumes the IsaacGym API returns a numpy array of DOF positions.
        return self.gym.get_actor_dof_states(env, actor, gymapi.STATE_POS)['pos']

    def get_joint_names(self, actor):
        """
        Retrieve the joint names from the asset used by the actor.
        """
        asset = self.actor_data.get(actor, {}).get("asset", None)
        if asset is not None:
            return self.gym.get_asset_dof_names(asset)
        return []

    def make_actor_dof_states(self, actor, env_index=0):
        """
        Initiate an empty DOF joint positions for an actor.
        """
        env = self.envs[env_index]
        state = self.gym.get_actor_dof_states(env, actor, gymapi.STATE_POS)
        state['pos'] = np.zeros_like(state['pos'])
        state['vel'] = np.zeros_like(state['vel'])
        return state

    def set_joint_positions(self, actor, positions, env_index=0):
        """
        Set the DOF joint positions for an actor.
        """
        if len(self.get_joint_names(actor)) == 0:
            return
        env = self.envs[env_index]
        dof_state = self.make_actor_dof_states(actor, env_index)
        dof_state['pos'] = positions
        self.gym.set_actor_dof_states(env, actor, dof_state, gymapi.STATE_POS)

    ## ==================================================================================================

    def create_camera(self, width=640, height=480, fov=60, env_index=0):
        """
        Create a camera sensor in the specified environment.
        """
        env = self.envs[env_index]
        camera_options = gymapi.CameraProperties()
        camera_options.width = width
        camera_options.height = height
        camera_options.horizontal_fov = fov
        camera = self.gym.create_camera_sensor(env, camera_options)
        self.cameras.append(camera)
        return camera

    def set_camera_target(self, camera, camera_point, target_point, env_index=0):
        """
        Set a camera sensor's location and target point.
        """
        env = self.envs[env_index]
        cp = gymapi.Vec3(*camera_point)
        tp = gymapi.Vec3(*target_point)
        self.gym.set_camera_location(camera, env, cp, tp)

    def save_image(self, camera, filename, image_type='rgb', env_index=0):
        """
        Save an image from the specified camera sensor.
        """
        env = self.envs[env_index]

        image = None
        if image_type == 'rgb':
            image_type = gymapi.ImageType.IMAGE_COLOR
            image = self.get_rgba_image(camera, env_index)
        elif image_type == 'depth':
            image_type = gymapi.ImageType.IMAGE_DEPTH
        elif image_type == 'seg':
            image_type = gymapi.ImageType.IMAGE_SEGMENTATION
        else:
            image_type = gymapi.ImageType.IMAGE_COLOR

        if image is None:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)

            image = self.gym.get_camera_image(self.sim, env, camera, image_type)
            print_blue(f"\t\timage.shape={image.shape}")

        from PIL import Image
        im = Image.fromarray(image)
        im.save(filename)

    ## ===================================================================================

    def update_viewer(self):
        """
        Update viewer: advances graphics and synchronizes time.
        """
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

    def wait_if_gui(self):
        """
        Wait for user input when running with a viewer.
        """
        self.update_viewer()
        input("Press Enter to continue...")

    def wait_for_duration(self, duration=INF, update_delay=1e-2):
        start_time = time.time()
        while elapsed_time(start_time) < duration:
            self.update_viewer()
            time.sleep(update_delay)

    def load_asset(self, asset_file, root=None, fixed_base=True,
                   disable_gravity=True, **kwargs):
        """
        Convenience method to load an asset given its file name.
        """
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = fixed_base
        asset_options.disable_gravity = disable_gravity
        asset_options.flip_visual_attachments = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.armature = 0.01
        if root is None:
            root = dirname(asset_file)
            asset_file = basename(asset_file)
        return self.gym.load_asset(self.sim, root, asset_file, asset_options)

    def box_asset(self, width, length=None, height=None, fixed_base=False):
        """
        Create a simple box asset.
        """
        if length is None:
            length = width
        if height is None:
            height = width
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = fixed_base
        asset = self.gym.create_box(self.sim, width, length, height, asset_options)
        return asset

    def set_light(self, intensity=None, ambient=None, direction=None, light_index=0):
        """
        Set the light parameters in the simulation.
        """
        if intensity is None:
            intensity = np.array([1.0, 1.0, 1.0])
        if ambient is None:
            ambient = np.array([0.5, 0.5, 0.5])
        if direction is None:
            direction = np.array([1.0, 1.0, 1.0])
        intensity_vec = gymapi.Vec3(*intensity.tolist())
        ambient_vec = gymapi.Vec3(*ambient.tolist())
        direction_vec = gymapi.Vec3(*direction.tolist())
        self.gym.set_light_parameters(self.sim, light_index, intensity_vec, ambient_vec, direction_vec)

    def get_rgba_image(self, camera, env_index=0):
        """
        Retrieve an RGBA image from a camera sensor.
        """
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        env = self.envs[env_index]
        image = self.gym.get_camera_image(self.sim, env, camera, gymapi.ImageType.IMAGE_COLOR)
        height, _ = image.shape
        reshaped_image = image.reshape([height, -1, 4])  ## (720, 5120) --> (720, 1280, 4)
        return reshaped_image

    def get_joint_properties(self, actor, env_index=0):
        """
        Retrieve joint properties (such as limits) for an actor.
        """
        env = self.envs[env_index]
        return self.gym.get_actor_dof_properties(env, actor)
