import numpy as np
import random

from world_builder.world import Agent
from world_builder.actions import MoveAction, TurnAction, DriveAction
from pybullet_tools.bullet_utils import clip_delta, draw_pose2d, constant_controller, timeout_controller, multiply2d, invert2d, \
    sample_conf
from pybullet_tools.utils import get_difference_fn, remove_handles


class SpasticAgent(Agent):
    def __init__(self, world, **kwargs):
        super(SpasticAgent, self).__init__(world, **kwargs)
        #custom_limits = {joint:  delta * np.array([-1., +1.])
        #                 for joint, delta in zip(self.robot.joints, self.max_delta)}
        #self.sample_fn = get_sample_fn(self.robot, self.robot.joints, custom_limits=custom_limits)
    def policy(self, observation):
        delta = np.random.uniform(-self.max_delta, self.max_delta)
        #delta = self.sample_fn()
        #delta = clip_delta(delta, self.max_velocities, self.time_step)
        return MoveAction(delta=delta)


class WaypointAgent(Agent):
    requires_conf = True
    def __init__(self, world, goals_per_sec=1, **kwargs):
        super(WaypointAgent, self).__init__(world, **kwargs)
        #self.sample_fn = get_sample_fn(self.robot, self.robot.joints, custom_limits=self.robot.custom_limits)
        self.difference_fn = get_difference_fn(self.robot, self.robot.joints)
        self.p_goal = goals_per_sec*self.time_step
        self.goal_conf = None
        self.handles = []
    def policy(self, observation):
        if self.goal_conf is None or (random.random() < self.p_goal):
            #self.goal_conf = self.sample_fn()
            self.goal_conf = sample_conf(self.robot)
            remove_handles(self.handles)
            self.handles = draw_pose2d(self.goal_conf, length=0.05)
        observation.assign()
        current_conf = self.robot.get_positions()
        goal_local = multiply2d(invert2d(current_conf), self.goal_conf)
        delta = self.difference_fn(goal_local, np.zeros(3))
        delta = clip_delta(delta, self.max_velocities, self.time_step)
        return MoveAction(delta=delta)


class SpinningAgent(Agent):
    def __init__(self, world, sign=+1, **kwargs):
        super(SpinningAgent, self).__init__(world, **kwargs)
        self.sign = sign
    def policy(self, observation):
        return TurnAction(self.sign)


class HoldAgent(Agent):
    def __init__(self, world, prob_rotate=0.5, prob_reverse=0.5, **kwargs):
        super(HoldAgent, self).__init__(world, **kwargs)
        self.prob_rotate = prob_rotate
        self.prob_reverse = prob_reverse
        self.controller = None
    def sample_action(self, observation):
        sign = -1 if random.random() < self.prob_reverse else +1
        if random.random() < self.prob_rotate:
            return TurnAction(sign)
        return DriveAction(sign)
    def next_action(self):
        # TODO: quit when the set point is reached
        if self.controller is None:
            return None
        try:
            return next(self.controller)
        except StopIteration:
            return None
    def policy(self, observation):
        action = self.next_action()
        if action is None:
            action = self.sample_action(observation)
            timeout = random.uniform(0.25, 1.25)
            self.controller = timeout_controller(constant_controller(action), timeout=timeout, time_step=self.time_step)
        return self.next_action()
