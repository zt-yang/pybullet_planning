import math
import numpy as np
from isaacgym import gymapi, gymutil

import pathlib
import os
from os import listdir
from os.path import join, isfile, isdir, abspath

from utils import init_isaac_world

ASSET_ROOT = join(pathlib.Path(__file__), '..', '..', '..', 'assets', 'models')


class IsaacGymWorld(object):
    """ API for rendering a static scene and taking pictures from multiple view points """
    def __init__(self):
        # initialize gym, configure sim, add a ground plane
        self.gym, self.sim, self.viewer = init_isaac_world()
        self.envs = []
        self.actor_handles = []
        self.num_actors = {}

    def add_env(self):
        # set up the env grid
        num_per_row = 2
        spacing = 1
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
        self.envs.append(env)
        return env

    def add_camera(self, camera_point=(17.2, 2.0, 16), target_point=(5, -2.5, 13)):
        # position the camera
        cam_pos = gymapi.Vec3(camera_point[0], camera_point[1], camera_point[2])
        cam_target = gymapi.Vec3(target_point[0], target_point[1], target_point[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def load_asset(self, file_name):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.use_mesh_materials = True
        asset_options.mesh_normal_mode = True
        self.gym.load_asset(self.sim, ASSET_ROOT, file_name, asset_options)

    def add_asset_to_env(self, env, asset, i):
        # add actor
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 1.32, 0.0)
        pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

        actor_handle = self.gym.create_actor(env, asset, pose, "actor", i, 1)
        self.actor_handles.append(actor_handle)

def load_lisdf(gym, lisdf_path):
    pass


def load_urdf(gym, sim, file_name):
    pass


# # cache useful handles
# envs = []
# actor_handles = []
#
# for i, asset in enumerate(assets):
#     # create env
#     env = gym.create_env(sim, env_lower, env_upper, num_per_row)
#     envs.append(env)
#
#     # add actor
#     pose = gymapi.Transform()
#     pose.p = gymapi.Vec3(0.0, 1.32, 0.0)
#     pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
#
#     actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
#     actor_handles.append(actor_handle)
#
# while not gym.query_viewer_has_closed(viewer):
#     # step the physics
#     gym.simulate(sim)
#     gym.fetch_results(sim, True)
#
#     # update the viewer
#     gym.step_graphics(sim)
#     gym.draw_viewer(viewer, sim, True)
#
#     # Wait for dt to elapse in real time.
#     # This synchronizes the physics simulation with the rendering rate.
#     gym.sync_frame_time(sim)
#
# print("Done")
#
# gym.destroy_viewer(viewer)
# gym.destroy_sim(sim)
