import math
from .entities import Robot

from pybullet_tools.utils import get_joint_positions, clone_body, set_all_color, TRANSPARENT, \
    Attachment, link_from_name, get_link_subtree, get_joints, is_movable, multiply, invert, \
    set_joint_positions, set_pose, GREEN, dump_body, get_pose, remove_body, PoseSaver, \
    ConfSaver, get_unit_vector, unit_quat

from pybullet_tools.bullet_utils import equal, nice, get_gripper_direction, set_camera_target_body
from pybullet_tools.pr2_primitives import APPROACH_DISTANCE, Conf, Grasp
from pybullet_tools.pr2_utils import PR2_TOOL_FRAMES, close_until_collision

class RobotAPI(Robot):

    def get_init(self, init_facts=[], conf_saver=None):
        raise NotImplementedError('should implement this for RobotAPI!')

    def get_stream_map(self, problem, collisions, custom_limits, teleport):
        raise NotImplementedError('should implement this for RobotAPI!')

    def create_gripper(self, arm='left', visual=True):
        raise NotImplementedError('should implement this for RobotAPI!')

    def get_gripper_joints(self, gripper_grasp):
        raise NotImplementedError('should implement this for RobotAPI!')

    def close_cloned_gripper(self, gripper_grasp):
        raise NotImplementedError('should implement this for RobotAPI!')

    def compute_grasp_width(self, arm, body, grasp_pose, **kwargs):
        raise NotImplementedError('should implement this for RobotAPI!')

    def visualize_grasp(self, body_pose, grasp, arm='left', color=GREEN):
        raise NotImplementedError('should implement this for RobotAPI!')

    def get_attachment(self, grasp, arm):
        raise NotImplementedError('should implement this for RobotAPI!')

    def get_attachment_link(self, arm):
        raise NotImplementedError('should implement this for RobotAPI!')

    def get_carry_conf(self, arm, grasp_type, g):
        raise NotImplementedError('should implement this for RobotAPI!')

    def get_approach_vector(self, arm, grasp_type):
        raise NotImplementedError('should implement this for RobotAPI!')

    def get_approach_pose(self, approach_vector, g):
        raise NotImplementedError('should implement this for RobotAPI!')

    def make_grasps(self, g_type, arm, body, grasps_O, collisions=True):
        app = self.get_approach_vector(arm, g_type)
        grasps_R = [Grasp(g_type, body, g, self.get_approach_pose(app, g),
                          self.get_carry_conf(arm, g_type, g))
                    for g in grasps_O]

        ## filter for grasp width
        filtered_grasps = []
        for grasp in grasps_R:
            grasp_width = self.compute_grasp_width(g_type, get_pose(body), grasp.value,
                                                   body=body) if collisions else 0.0
            if grasp_width is not None:
                grasp.grasp_width = grasp_width
                filtered_grasps.append(grasp)
        return filtered_grasps

class PR2Robot(RobotAPI):

    grasp_types = ['top']

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

    def get_attachment_link(self, arm):
        return link_from_name(self.body, PR2_TOOL_FRAMES.get(arm, arm))

    def get_carry_conf(self, arm, grasp_type, g):
        if grasp_type == 'top':
            return TOP_HOLDING_LEFT_ARM
        if grasp_type == 'side':
            return SIDE_HOLDING_LEFT_ARM

    def get_approach_vector(self, arm, grasp_type):
        if grasp_type == 'top':
            return APPROACH_DISTANCE*get_unit_vector([1, 0, 0])
        if grasp_type == 'side':
            return APPROACH_DISTANCE*get_unit_vector([2, 0, -1])

    def get_approach_pose(self, approach_vector, g):
        return multiply((approach_vector, unit_quat()), g)

class FEGripper(RobotAPI):
    from pybullet_tools.utils import Pose, Euler

    grasp_types = ['hand']
    tool_from_hand = Pose(euler=Euler(math.pi / 2, 0, -math.pi / 2))

    def get_attachment_link(self, arm):
        from pybullet_tools.flying_gripper_utils import TOOL_LINK
        return link_from_name(self.body, TOOL_LINK)

    def create_gripper(self, arm='hand', visual=True, color=None):
        from pybullet_tools.utils import unit_pose
        gripper = clone_body(self.body, visual=False, collision=True)
        set_pose(gripper, unit_pose())
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

    def compute_grasp_width(self, arm, body_pose, grasp, body=None, verbose=False, **kwargs):
        from pybullet_tools.flying_gripper_utils import se3_ik, set_se3_conf, get_se3_conf
        body_pose = self.get_body_pose(body_pose, body=body, verbose=verbose)
        grasp_pose = multiply(body_pose, grasp)
        if verbose:
            print(f'robots.compute_grasp_width | body_pose = {nice(body_pose)} | grasp = {nice(grasp)}')
            print('robots.compute_grasp_width | grasp_pose = multiply(body_pose, grasp) = ', nice(grasp_pose))

        with ConfSaver(self.body):
            conf = se3_ik(self, grasp_pose)
            gripper = self.body
            set_se3_conf(gripper, conf)
            if verbose:
                print(f'robots.compute_grasp_width | gripper_grasp {gripper} | object_pose {nice(body_pose)}'
                      f' | se_conf {nice(get_se3_conf(gripper))} | grasp = {nice(grasp)} ')

            gripper_joints = self.get_gripper_joints()
            if isinstance(body, tuple): body = body[0]
            width = close_until_collision(gripper, gripper_joints, bodies=[body], **kwargs)
            # remove_body(gripper)
        return width

    def get_body_pose(self, body_pose, body=None, verbose=False):
        T = self.tool_from_hand
        title = f'    robot.get_body_pose({nice(body_pose)}, body={body})'

        ## if body_pose is handle link pose and body is (body, joint)
        if body != None and isinstance(body, tuple) and not isinstance(body[0], tuple):
            if verbose: print(f'{title} | return as is')
            return body_pose
            # new_body_pose = multiply(body_pose, invert(T), T)
            # print(f'{title} | multiply(body_pose, invert(self.tool_from_hand), self.tool_from_hand) = {nice(new_body_pose)}')
            # return new_body_pose

        ## if body is given in the place of body_pose
        b = body_pose
        if not (isinstance(b, tuple) and isinstance(b[0], tuple) and len(b[0]) == 3 and len(b[1]) == 4):
            if verbose: print(f'{title} | actually given body')
            body_pose = get_pose(b)

        new_body_pose = multiply(body_pose, T)
        if verbose: print(f'{title} | multiply(body_pose, self.tool_from_hand) = {nice(new_body_pose)}')
        return new_body_pose

    def visualize_grasp(self, body_pose, grasp, arm='hand', color=GREEN, width=1, body=None, verbose=False):
        from pybullet_tools.flying_gripper_utils import se3_ik, set_cloned_se3_conf, get_cloned_se3_conf

        body_pose = self.get_body_pose(body_pose, body=body, verbose=verbose)
        gripper = self.create_gripper(arm, visual=True)
        self.open_cloned_gripper(gripper, width)

        set_all_color(gripper, color)
        grasp_pose = multiply(body_pose, grasp) ##
        if verbose:
            print(f'robots.visualize_grasp | body_pose = {nice(body_pose)} | grasp = {nice(grasp)}')
            print('robots.visualize_grasp | grasp_pose = multiply(body_pose, grasp) = ', nice(grasp_pose))
        # set_pose(self.body, grasp_pose)
        grasp_conf = se3_ik(self, grasp_pose)
        if grasp_conf == None:
            print(f'robots.visualize_grasp | ik failed for {nice(grasp_pose)}')
        set_cloned_se3_conf(self.body, gripper, grasp_conf)

        # set_camera_target_body(gripper, dx=0.5, dy=0.5, dz=0.5)
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

    def get_carry_conf(self, arm, grasp_type, g):
        return g

    def get_approach_vector(self, arm, grasp_type):
        return APPROACH_DISTANCE*get_unit_vector([0, 0, -1])

    def get_approach_pose(self, approach_vector, g):
        return multiply(g, (approach_vector, unit_quat()))
