import numpy as np
import math

import pybullet as p
from world_builder.world import Agent
from world_builder.actions import MoveAction, TeleportAction, TurnAction, DriveAction, MoveInSE3Action
from pybullet_tools.bullet_utils import clip_delta, multiply2d, invert2d, nice, get_point_distance
from pybullet_tools.camera_utils import get_pose2d
from pybullet_tools.pose_utils import draw_pose2d, MIN_DISTANCE
from pybullet_tools.utils import get_sample_fn, get_difference_fn, aabb_overlap, set_pose, BodySaver, \
    pairwise_collisions, get_aabb, plan_joint_motion, randomize, waypoints_from_path, plan_nonholonomic_motion, \
    get_nonholonomic_distance_fn, get_distance_fn, PoseSaver, get_extend_fn, get_nonholonomic_extend_fn, \
    PI, add_parameter, add_button, Pose, Point, Euler, adjust_path, get_com_pose


APPROX_DIFFERENTIAL = True


class MotionAgent(Agent):
    # TODO: rename nonholonomic to differential drive
    requires_conf = requires_poses = True

    def __init__(self, world, reversible=False, monitor=False, **kwargs):
        super(MotionAgent, self).__init__(world, **kwargs)
        self.reversible = reversible
        self.monitor = monitor
        self.holonomic = not self.world.drive
        self.teleport = self.world.teleport
        self.sample_fn = get_sample_fn(self.robot, self.robot.joints, custom_limits=self.robot.custom_limits)
        self.difference_fn = get_difference_fn(self.robot, self.robot.joints)
        if APPROX_DIFFERENTIAL or self.holonomic:
            #self.distance_fn = lambda *args: np.reciprocal(self.duration_fn(*args))
            self.distance_fn = get_distance_fn(self.robot, self.robot.joints, weights=self.robot.weights)
            self.extend_fn = get_extend_fn(self.robot, self.robot.joints, resolutions=self.robot.resolutions)
        else:
            linear_tol = 1e-3 # 1e-3 | 1e-6
            angular_tol = math.radians(1e-1) # 1e-3 | 1e-6
            self.distance_fn = get_nonholonomic_distance_fn(self.robot, self.robot.joints, reversible=self.reversible,
                                                            linear_tol=linear_tol, angular_tol=angular_tol,
                                                            weights=self.robot.weights)
            self.extend_fn = get_nonholonomic_extend_fn(self.robot, self.robot.joints, reversible=self.reversible,
                                                        linear_tol=linear_tol, angular_tol=angular_tol,
                                                        resolutions=self.robot.resolutions)
        #self.distance_fn = get_duration_fn(self.robot, self.robot.joints, velocities=self.max_velocities)

        self.path = None
        self.handles = []
        # TODO: memory

    def update(self, observation):
        # TODO: check whether the states have changed significantly since last
        observation.assign()
        return set(observation.obstacles) - {self.robot}

    def sample_goal(self, obstacles):
        with BodySaver(self.robot):
            while True:
                target_conf = self.sample_fn()
                self.robot.set_positions(target_conf)
                if not pairwise_collisions(self.robot, obstacles, max_distance=MIN_DISTANCE):
                    return target_conf

    ## ------ moved to world.set_path(path)
    # def set_path(self, path):
    #     remove_handles(self.handles)
    #     self.path = path
    #     if self.path is None:
    #         return self.path
    #     if isinstance(self.path[0], tuple):  ## pr2
    #         #path = adjust_path(self.robot, self.robot.joints, self.path)
    #         self.handles.extend(draw_pose2d_path(self.path[::4], length=0.05))
    #         #wait_if_gui()
    #     else:  ## SE3 gripper
    #         path = [p.values for p in path]
    #         self.handles.extend(draw_pose3d_path(path, length=0.05))
    #     return self.path

    def sample_path(self, obstacles):
        #start_conf = self.robot.get_positions()
        goal_conf = self.sample_goal(obstacles)
        self.handles = draw_pose2d(goal_conf, z=0.)
        #self.robot.set_positions(start_conf)
        with BodySaver(self.robot):
            plan_motion = plan_joint_motion if self.holonomic else plan_nonholonomic_motion
            # TODO: could have some tolerance for the "two boundary problem"
            kwargs = {} if self.holonomic else {'reversible': self.reversible}
            # TODO: plan with decreasing max_distance
            path = plan_motion(self.robot, self.robot.joints, goal_conf,
                               obstacles=obstacles, self_collisions=False,  #max_distance=MIN_DISTANCE/2.,
                               weights=self.robot.weights, resolutions=self.robot.resolutions,
                               disabled_collisions=self.robot.disabled_collisions, **kwargs)
        return self.set_path(path)

    def check_path(self, path, obstacles):
        if path is None:
            return False
        with BodySaver(self.robot):
            for conf in randomize(path):
                self.robot.set_positions(conf)
                if pairwise_collisions(self.robot, obstacles, max_distance=MIN_DISTANCE / 2.): # Make sure no regions
                    return False
            return True
        # TODO: pickup at the back of bot

    def follow_path(self, current_conf):
        # TODO: also see SS-Replan
        # TODO: discard negligible displacements
        # TODO: maybe only control for x, y positions and don't worry about orientations at all
        # TODO: ensure that we move in the correct direction using the dot product with the average waypoint direction
        # curve = interpolate_path(self.robot, self.robot.joints, path)
        self.robot.set_positions(current_conf)
        if not self.path:
            return None

        if self.teleport:
            action = TeleportAction(self.path[-1])
            self.set_path(path=[])
            return action

        indices = list(range(len(self.path)))
        for i in reversed(indices):
            if self.distance_fn(current_conf, self.path[i]) < 1e-2:
                # TODO: track the error
                nearby_index = i
                self.path = self.path[nearby_index+1:]
                break
        else:
            closest_index = min(indices, key=lambda i: self.distance_fn(current_conf, self.path[i]))
            self.path = self.path[closest_index:]
        if not self.path:
            return None
        new_path = [current_conf] + list(self.extend_fn(current_conf, self.path[0])) + self.path[1:]
        new_path = adjust_path(self.robot, self.robot.joints, new_path)
        waypoints = waypoints_from_path(new_path, tolerance=1e-6)
        if len(waypoints) <= 1:
            return None # TODO: return true
        target_conf = waypoints[1]
        move_action = self.move_action(current_conf, target_conf)
        # print(f'            to {target_conf} by taking {move_action}')
        return move_action

    def move_action(self, current_conf, target_conf, velocity_scale=0.95):
        if len(current_conf) == 6:
            return MoveInSE3Action(target_conf)
        target_local = multiply2d(invert2d(current_conf), target_conf)
        #delta = target_local
        delta = self.difference_fn(target_local, np.zeros(3))
        # TODO: don't include wrap around once using adjust_path
        delta = clip_delta(delta, velocity_scale * self.max_velocities, self.time_step)
        # new_conf = multiply2d(current_conf, target_local)
        # print(new_conf, target_conf)
        # handles = draw_pose2d(target_conf, z=0.) # + draw_pose2d(current_conf, z=0.)
        if np.allclose(delta[1:], np.zeros(2)):
            return DriveAction(delta[0])
        if np.allclose(delta[:2], np.zeros(2)):
            return TurnAction(delta[2])
        return MoveAction(delta=delta)

    def policy(self, observation):
        obstacles = self.update(observation)
        current_conf = np.array(self.robot.get_positions())
        if self.monitor and not self.check_path(self.path, obstacles):
            self.path = None
        if self.path is None:
            self.sample_path(obstacles)
        if (self.path is None) or (len(self.path) == 0):
            return None
        action = self.follow_path(current_conf)
        if action is None:
            self.path = None
        return action

    def set_pddlstream_problem(self, problem_dict, state):
        pass

    def init_experiment(self, args, **kwargs):
        pass
