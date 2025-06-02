from typing import Dict
import math
import random
from os.path import abspath, join
import numpy as np
from pybullet_tools.utils import Pose, Euler, multiply, get_joint_names, Point, get_aabb
from pybullet_tools.bullet_utils import nice
from robot_builder.robots import MobileRobot
from robot_builder.robot_utils import BASE_GROUP, GRIPPER_GROUP, BASE_TORSO_GROUP, create_mobile_robot, \
    ARM_GROUP
from world_builder.paths import PBP_PATH
from leap_tools.hierarchical_agent import HierarchicalAgent


## ---------------------------------------------------------------------------


class ComposedRobot(MobileRobot):

    def __init__(self, robot, end_effectors: Dict, **kwargs):
        self.dual_arm = robot.dual_arm
        self.arms = robot.arms
        super(ComposedRobot, self).__init__(robot.body, base_link=robot.base_link,
                                            use_torso=robot.use_torso, move_base=robot.move_base, **kwargs)

        self.robot = robot
        self.body = robot.body
        self.custom_limits = robot.custom_limits

        self.self_collisions = robot.self_collisions
        self.joint_groups = robot.joint_groups
        self.solve_leg_conf_fn = robot.solve_leg_conf_fn

        self.end_effectors = end_effectors
        self.attachments = [o.attachment for o in end_effectors.values()]

    def assign_attachments(self):
        for attachment in self.attachments:
            attachment.assign()

    def set_base_conf(self, positions):
        self.robot.set_base_conf(positions)
        self.assign_attachments()

    def set_joint_positions(self, joints, positions):
        self.robot.set_joint_positions(joints, positions)
        self.assign_attachments()

    def get_tool_link(self, arm):
        return self.robot.get_tool_link(arm)

    def check_arm_body_collisions(self):
        return self.robot.check_arm_body_collisions()

    def open_arm(self, arm):
        self.robot.open_arm(arm)
        self.assign_attachments()

    def get_carry_conf(self, arm, grasp_type, g):
        return self.robot.get_carry_conf(arm)

    def get_gripper_root(self, arm):
        return self.end_effectors[arm].get_gripper_root(arm)

    # def get_tool_from_hand(self, body):
    #     if self.tool_from_hand is None:
    #         ## door handle
    #         if body is None or isinstance(body, tuple):
    #             tool_from_hand = Pose(point=(0, 0, -0.08), euler=Euler(-math.pi / 2, 0, 0))
    #
    #         ## movable objects
    #         else: ## if isinstance(body, int):
    #             tool_from_hand = Pose(point=(0, 0, -0.05), euler=Euler(math.pi, 0, math.pi/2))
    #     else:
    #         tool_from_hand = self.tool_from_hand
    #     return tool_from_hand
    #     pass

    # def get_grasp_pose(self, body_pose, grasp, arm='right', body=None, verbose=False):
    #     body_pose = self.get_body_pose(body_pose, body=body, verbose=verbose)
    #     tool_from_hand = self.get_tool_from_hand(body)
    #     if verbose:
    #         print('\tbody', body)
    #         print('\tbody_pose', nice(body_pose))
    #         print('\tgrasp', nice(grasp))
    #         print('\tself.tool_from_hand', nice(tool_from_hand), '\n')
    #     return multiply(body_pose, grasp, tool_from_hand)
    #
    # def get_all_arms(self):
    #     from pybullet_tools.pr2_utils import ARM_NAMES
    #     return ARM_NAMES
    #
    # def get_tool_link(self, arm):
    #     return g1_TOOL_LINK.format(side=arm)
    #
    # def get_gripper_root(self, arm):
    #     return g1_GRIPPER_ROOT.format(side=arm)
    #
    # def get_arm_joints(self, arm):
    #     return self.get_group_joints(f"{arm}_{ARM_GROUP}")
    #
    # def get_gripper_joints(self, arm):
    #     return self.get_group_joints(f"{arm}_{GRIPPER_GROUP}")
    #
    # def get_carry_conf(self, arm, g_type=None, g=None):
    #     if arm == 'left':
    #         return g1_LEFT_REST_CONF
    #     return g1_RIGHT_REST_CONF
    #
    # def mod_grasp_along_handle(self, grasp, dl):
    #     # return multiply(grasp, Pose(point=(dl, 0, 0)))
    #     return multiply(grasp, Pose(point=(0, dl, 0)))
    #
    # def get_tool_from_root(self, a):
    #     tool_from_root = Pose(point=(0, 0, -0.12), euler=Euler(0, 0, math.pi/2))
    #     tool_from_root = Pose(point=(0, 0, -0.24), euler=Euler(-math.pi/2, 0, 0))
    #     # print('robot.tool_from_root\t', nice(tool_from_root))
    #     return tool_from_root
    #
    # def open_cloned_gripper(self, gripper, arm=None, width=1):
    #     from g1_tools.g1_utils import open_cloned_gripper
    #     if arm is None:
    #         arm = self.get_arm_from_gripper(gripper)
    #     open_cloned_gripper(self.body, gripper, arm, w=width)
    #
    # def close_cloned_gripper(self, gripper, arm=None):
    #     from g1_tools.g1_utils import close_cloned_gripper
    #     if arm is None:
    #         arm = self.get_arm_from_gripper(gripper)
    #     close_cloned_gripper(self.body, gripper, arm)
    #
    # def open_arm(self, arm):
    #     arm_group = self.get_arm_joints(arm)
    #     conf = {
    #         f"left_{ARM_GROUP}": g1_LEFT_REST_CONF,
    #         f"right_{ARM_GROUP}": g1_RIGHT_REST_CONF,
    #     }[f"{arm}_{ARM_GROUP}"]
    #     self.set_joint_positions(arm_group, conf)
    #
    # def close_until_collision(self, arm, gripper_joints, bodies, **kwargs):
    #     open_conf = [g1_GRIPPER_MAX] * 2
    #     closed_conf = [0] * 2
    #     return super(g1Robot, self).close_until_collision(
    #         arm, gripper_joints, bodies=bodies, open_conf=open_conf,
    #         closed_conf=closed_conf, **kwargs)
