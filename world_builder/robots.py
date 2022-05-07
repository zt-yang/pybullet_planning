import math
from .entities import Robot

from pybullet_tools.utils import get_joint_positions, clone_body, set_all_color, TRANSPARENT, \
    Attachment, link_from_name, get_link_subtree, get_joints, is_movable, multiply, invert, \
    set_joint_positions, set_pose, GREEN, dump_body, get_pose, remove_body, PoseSaver

from pybullet_tools.bullet_utils import equal, nice, get_gripper_direction, set_camera_target_body
from pybullet_tools.pr2_primitives import Conf
from pybullet_tools.pr2_utils import PR2_TOOL_FRAMES, close_until_collision



class PR2Robot(Robot):

    def get_init(self, init_facts=[], conf_saver=None):
        from pybullet_tools.pr2_utils import get_arm_joints, ARM_NAMES, get_group_joints, \
            get_group_conf, get_top_grasps, get_side_grasps

        robot = self.body

        def get_conf(joints):
            if conf_saver == None:
                return get_joint_positions(robot, joints)
            return [conf_saver.conf[conf_saver.joints.index(n)] for n in joints]

        def get_base_conf():
            base_joints = get_group_joints(robot, 'base')
            initial_bq = Conf(robot, base_joints, get_conf(base_joints))
            for fact in init_facts:
                if fact[0] == 'bconf' and equal(fact[1].values, initial_bq.values):
                    return fact[1]
            return initial_bq

        def get_arm_conf(arm):
            arm_joints = get_arm_joints(robot, arm)
            conf = Conf(robot, arm_joints, get_conf(arm_joints))
            for fact in init_facts:
                if fact[0] == 'aconf' and fact[1] == arm and equal(fact[2].values, conf.values):
                    return fact[2]
            return conf

        initial_bq = get_base_conf()
        init = [('BConf', initial_bq), ('AtBConf', initial_bq)]

        for arm in ARM_NAMES:
            conf = get_arm_conf(arm)
            init += [('Arm', arm), ('AConf', arm, conf),
                     ('DefaultConf', arm, conf), ('AtAConf', arm, conf)]
            if arm in ['left']:
                init += [('Controllable', arm)]

        HANDS_EMPTY = {arm: True for arm in ARM_NAMES}
        for arm, empty in HANDS_EMPTY.items():
            if empty: init += [('HandEmpty', arm)]

        return init

    def get_stream_map(self, problem, collisions, custom_limits, teleport):
        from pybullet_tools.pr2_agent import get_stream_map
        return get_stream_map(problem, collisions, custom_limits, teleport)

    def create_gripper(self, arm='left', visual=True):
        from pybullet_tools.pr2_utils import create_gripper
        return create_gripper(self.body, arm=arm, visual=visual)

    def get_gripper_joints(self, gripper_grasp):
        return [joint for joint in get_joints(gripper_grasp) if is_movable(gripper_grasp, joint)]

    def close_cloned_gripper(self, gripper_grasp):
        set_joint_positions(gripper_grasp, self.get_gripper_joints(gripper_grasp), [0] * 4)

    def open_cloned_gripper(self, gripper_grasp):
        set_joint_positions(gripper_grasp, self.get_gripper_joints(gripper_grasp), [0.548] * 4)

    def compute_grasp_width(self, **kwargs):
        from pybullet_tools.pr2_utils import compute_grasp_width
        return compute_grasp_width(self.body, **kwargs)

    def visualize_grasp(self, body_pose, grasp, arm='left', color=GREEN):
        from pybullet_tools.pr2_utils import PR2_GRIPPER_ROOTS
        from pybullet_tools.pr2_primitives import get_tool_from_root

        robot = self.body
        gripper_grasp = self.create_gripper(arm, visual=True)
        self.open_cloned_gripper(gripper_grasp)

        set_all_color(gripper_grasp, color)
        tool_from_root = get_tool_from_root(robot, arm)
        grasp_pose = multiply(multiply(body_pose, invert(grasp)), tool_from_root)
        set_pose(gripper_grasp, grasp_pose)

        direction = get_gripper_direction(grasp_pose)
        print('\n', nice(grasp_pose), direction)
        if direction == None:
            print('new direction')
            return gripper_grasp, True
        if 'down' in direction:
            print('pointing down')
            return gripper_grasp, True

        return gripper_grasp, False

    def get_attachment(self, grasp, arm):
        tool_link = link_from_name(self.body, PR2_TOOL_FRAMES[arm])
        return Attachment(self.body, tool_link, grasp.value, grasp.body)

class FEGripper(Robot):

    def create_gripper(self, arm='hand', visual=True, color=None):
        gripper = clone_body(self.body, visual=False, collision=True)
        if not visual:
            set_all_color(self.body, TRANSPARENT)
        if color != None:
            set_all_color(self.body, color)
        return gripper

    def get_gripper_joints(self, gripper_grasp=None):
        from pybullet_tools.flying_gripper_utils import get_joints_by_group, FINGERS_GROUP
        return get_joints_by_group(self.body, FINGERS_GROUP)

    def open_cloned_gripper(self, gripper, width=1):
        from pybullet_tools.flying_gripper_utils import open_cloned_gripper
        open_cloned_gripper(self.body, gripper, w=width)

    def close_cloned_gripper(self, gripper):
        from pybullet_tools.flying_gripper_utils import close_cloned_gripper
        close_cloned_gripper(self.body, gripper)

    def compute_grasp_width(self, arm, body, grasp_pose, **kwargs):
        body_pose = multiply(self.get_body_pose(body), grasp_pose)
        # gripper = self.create_gripper(arm, visual=True)
        with PoseSaver(self.body):
            gripper = self.body
            set_pose(gripper, body_pose)
            gripper_joints = self.get_gripper_joints()
            width = close_until_collision(gripper, gripper_joints, bodies=[body], **kwargs)
            # remove_body(gripper)
        return width

    def get_body_pose(self, body):
        from pybullet_tools.utils import Pose, Euler
        if isinstance(body, tuple) and len(body[0]) == 3 and len(body[1]) == 4:
            body_pose = body
        else:
            body_pose = get_pose(body)
        return multiply(body_pose, Pose(euler=Euler(math.pi/2, 0, -math.pi/2)))

    def visualize_grasp(self, body_pose, grasp, arm='hand', color=GREEN, width=1):
        body_pose = self.get_body_pose(body_pose)
        gripper = self.create_gripper(arm, visual=True)
        self.open_cloned_gripper(gripper, width)

        set_all_color(gripper, color)
        grasp_pose = multiply(body_pose, grasp) ##
        set_pose(gripper, grasp_pose)

        # set_camera_target_body(gripper, dx=0.5, dy=0.5, dz=0.8)
        return gripper

    def get_init(self, init_facts=[], conf_saver=None):
        from pybullet_tools.flying_gripper_utils import get_se3_joints, get_se3_conf, ARM_NAME
        robot = self.body

        def get_conf(joints):
            if conf_saver == None:
                return get_se3_conf(robot)
            return [conf_saver.conf[conf_saver.joints.index(n)] for n in joints]

        def get_se3_q():
            joints = get_se3_joints(robot)
            initial_q = Conf(robot, joints, get_conf(joints))
            for fact in init_facts:
                if fact[0] == 'seconf' and equal(fact[1].values, initial_q.values):
                    return fact[1]
            return initial_q

        initial_q = get_se3_q()
        arm = ARM_NAME
        return [('SEConf', initial_q), ('AtSEConf', initial_q),
                ('Arm', arm), ('Controllable', arm), ('HandEmpty', arm)]

    def get_stream_map(self, problem, collisions, custom_limits, teleport):
        from pybullet_tools.flying_gripper_agent import get_stream_map
        return get_stream_map(problem, collisions, custom_limits, teleport)

    def get_attachment(self, grasp, arm=None):
        tool_link = link_from_name(self.body, 'panda_hand')
        return Attachment(self.body, tool_link, grasp.value, grasp.body)
