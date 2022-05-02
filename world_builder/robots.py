from .entities import Robot

from pybullet_tools.utils import get_joint_positions, clone_body, set_all_color, TRANSPARENT, \
    Attachment , link_from_name

from pybullet_tools.bullet_utils import equal
from pybullet_tools.pr2_primitives import Conf
from pybullet_tools.pr2_utils import PR2_TOOL_FRAMES

class PR2Robot(Robot):

    def get_init(self, init_facts=[], conf_saver=None):
        from pybullet_tools.pr2_utils import get_arm_joints, ARM_NAMES, get_group_joints, \
            get_group_conf, get_top_grasps, get_side_grasps, create_gripper
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

    def get_attachment(self, grasp, arm):
        tool_link = link_from_name(self.body, PR2_TOOL_FRAMES[arm])
        return Attachment(self.body, tool_link, grasp.value, grasp.body)

class FEGripper(Robot):

    def create_gripper(self, arm=None, visual=True):
        gripper = clone_body(self.body, visual=False, collision=True)
        if not visual:
            set_all_color(self.body, TRANSPARENT)
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
