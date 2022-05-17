import math
from .entities import Robot

from pybullet_tools.utils import get_joint_positions, clone_body, set_all_color, TRANSPARENT, \
    link_from_name, get_link_subtree, get_joints, is_movable, multiply, invert, \
    set_joint_positions, set_pose, GREEN, dump_body, get_pose, remove_body, PoseSaver, \
    ConfSaver, get_unit_vector, unit_quat, get_link_pose, unit_pose, draw_pose, remove_handles, \
    interpolate_poses

from pybullet_tools.bullet_utils import equal, nice, get_gripper_direction, set_camera_target_body, Attachment, \
    BASE_LIMITS
from pybullet_tools.pr2_primitives import APPROACH_DISTANCE, Conf, Grasp, get_base_custom_limits
from pybullet_tools.pr2_utils import PR2_TOOL_FRAMES, PR2_GROUPS, close_until_collision, TOP_HOLDING_LEFT_ARM, \
    SIDE_HOLDING_LEFT_ARM
from pybullet_tools.general_streams import get_handle_link

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

    def compute_grasp_width(self, arm, body_pose, grasp_pose, body=None, verbose=False, **kwargs):
        raise NotImplementedError('should implement this for RobotAPI!')

    def visualize_grasp(self, body_pose, grasp, arm='left', color=GREEN, **kwargs):
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
            grasp_width = self.compute_grasp_width(arm, get_pose(body), grasp.value,
                                                   body=body) if collisions else 0.0
            if grasp_width is not None:
                grasp.grasp_width = grasp_width
                filtered_grasps.append(grasp)
        return filtered_grasps

    def make_attachment(self, grasp, tool_link):
        o = grasp.body
        if isinstance(o, tuple) and len(o) == 2:
            body, joint = o
            link = get_handle_link(o)
            return Attachment(self.body, tool_link, grasp.value, body, child_joint=joint, child_link=link)

        return Attachment(self.body, tool_link, grasp.value, grasp.body)

    def get_custom_limits(self):
        return self.custom_limits


class PR2Robot(RobotAPI):

    grasp_types = ['top']
    joint_groups = ['left', 'right', 'base']

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

    def get_cloned_gripper_joints(self, gripper_grasp):
        return [joint for joint in get_joints(gripper_grasp) if is_movable(gripper_grasp, joint)]

    def get_gripper_joints(self, arm='left'):
        from pybullet_tools.pr2_utils import get_gripper_joints
        return get_gripper_joints(self.body, arm)

    def close_cloned_gripper(self, gripper_grasp):
        joints = self.get_cloned_gripper_joints(gripper_grasp)
        set_joint_positions(gripper_grasp, joints, [0] * 4)

    def open_cloned_gripper(self, gripper_grasp, arm='left'):
        joints = self.get_cloned_gripper_joints(gripper_grasp)
        set_joint_positions(gripper_grasp, joints, [0.548] * 4)

    def compute_grasp_width(self, arm, body_pose, grasp_pose, body=None, verbose=False, **kwargs):
        from pybullet_tools.pr2_utils import compute_grasp_width
        return compute_grasp_width(self.body, arm, body, grasp_pose, **kwargs)

    def visualize_grasp(self, body_pose, grasp, arm='left', color=GREEN, verbose=False, **kwargs):
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
        # if verbose:
        #     if direction == None:
        #         print('new direction')
        #         return gripper_grasp, True
        #     if 'down' in direction:
        #         print('pointing down')
        #         return gripper_grasp, True

        return gripper_grasp

    def get_attachment(self, grasp, arm):
        tool_link = link_from_name(self.body, PR2_TOOL_FRAMES[arm])
        return self.make_attachment(grasp, tool_link)
        # return Attachment(self.body, tool_link, grasp.value, grasp.body)

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

    def get_all_joints(self):
        return sum(PR2_GROUPS.values(), [])

    def get_lisdf_string(self):
        return """
    <include name="pr2">
      <uri>../models/drake/pr2_description/urdf/pr2_simplified.urdf</uri>
      {pose_xml}
    </include>
"""

    def get_positions(self, joint_group='base', roundto=None):
        from pybullet_tools.pr2_utils import get_arm_joints
        if joint_group == 'base':
            joints = self.joints
        else: ## if joint_group == 'left':
            joints = get_arm_joints(self.body, joint_group)
        positions = self.get_joint_positions(joints)
        if roundto == None:
            return positions
        return tuple([round(n, roundto) for n in positions])

    # def get_custom_limits(self):
    #     return get_base_custom_limits(self, BASE_LIMITS)

    def iterate_approach_path(self, arm, gripper, pose_value, grasp, body=None):
        from pybullet_tools.pr2_primitives import get_tool_from_root
        tool_from_root = get_tool_from_root(self.body, arm)
        grasp_pose = multiply(pose_value, invert(grasp.value))
        approach_pose = multiply(pose_value, invert(grasp.approach))
        for tool_pose in interpolate_poses(grasp_pose, approach_pose):
            set_pose(gripper, multiply(tool_pose, tool_from_root))
            # if body is not None:
            #     set_pose(body, multiply(tool_pose, grasp.value))
            yield

class FEGripper(RobotAPI):
    from pybullet_tools.utils import Pose, Euler

    grasp_types = ['hand']
    joint_groups = ['hand']
    tool_from_hand = Pose(euler=Euler(math.pi / 2, 0, -math.pi / 2))

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
        if isinstance(body, tuple): body = body[0]
        with PoseSaver(body):
            body_pose = unit_pose()
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

        ## if body or body_joint is given in the place of body_pose
        b = body_pose
        if not (isinstance(b, tuple) and isinstance(b[0], tuple) and len(b[0]) == 3 and len(b[1]) == 4):
            if isinstance(b, tuple):
                handle_link = get_handle_link(b)
                new_body_pose = body_pose = get_link_pose(b[0], handle_link)
                # new_body_pose = multiply(body_pose, invert(T))
                if verbose: print(f'{title} | actually given (body, joint), multiply(get_link_pose(body, '
                                  f'handle_link), invert(T)) = {nice(new_body_pose)}')
                return new_body_pose
            else:
                body_pose = get_pose(b)
                if verbose: print(f'{title} | actually given body, body_pose = get_pose(b) = {nice(body_pose)}')

        new_body_pose = multiply(body_pose, T)
        if verbose: print(f'{title} | multiply(body_pose, self.tool_from_hand) = {nice(new_body_pose)}')
        return new_body_pose

    def visualize_grasp(self, body_pose, grasp, arm='hand', color=GREEN, width=1, verbose=False,
                        body=None, mod_target=None):
        from pybullet_tools.flying_gripper_utils import se3_ik, set_cloned_se3_conf, get_cloned_se3_conf
        from pybullet_tools.utils import Pose, euler_from_quat
        title = 'robots.visualize_grasp |'

        body_pose = self.get_body_pose(body_pose, body=body, verbose=verbose)
        gripper = self.create_gripper(arm, visual=True)
        self.open_cloned_gripper(gripper, width)
        set_all_color(gripper, color)

        grasp_pose = multiply(body_pose, grasp)  ##
        if verbose:
            handles = draw_pose(grasp_pose, length=0.05)
            print(f'{title} body_pose = {nice(body_pose)} | grasp = {nice(grasp)}')
            print(f'{title} grasp_pose = multiply(body_pose, grasp) = ', nice(grasp_pose))

        grasp_conf = se3_ik(self, grasp_pose, verbose=verbose, mod_target=mod_target)
        if verbose and grasp_conf == None:
            print(f'{title} body_pose = {nice(body_pose)} --> ik failed')

        # if mod_pose == None:
        #     grasp_conf = se3_ik(self, grasp_pose, verbose=verbose)
        #     if grasp_conf == None:
        #         print(f'{title} body_pose = {nice(body_pose)} --> ik failed')
        #
        # ## in case of weird collision body
        # else:
        #     mod_pose = Pose(point=mod_pose[0], euler=euler_from_quat(body_pose[1]))
        #     grasp_pose = multiply(mod_pose, grasp)  ##
        #     grasp_conf = se3_ik(self, grasp_pose, verbose=verbose)
        #     if grasp_conf == None:
        #         print(f'{title} body_pose = {nice(body_pose)} | mod_pose = {nice(mod_pose)} --> ik failed')
        #     else:
        #         grasp_conf = list(multiply(body_pose, grasp)[0]) + list(grasp_conf)[3:]

        if verbose:
            remove_handles(handles)
        if grasp_conf == None:
            return None

        # set_pose(self.body, grasp_pose) ## wrong!
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
        return self.make_attachment(grasp, tool_link)
        # return Attachment(self.body, tool_link, grasp.value, grasp.body)

    def get_attachment_link(self, arm):
        from pybullet_tools.flying_gripper_utils import TOOL_LINK
        return link_from_name(self.body, TOOL_LINK)

    def get_carry_conf(self, arm, grasp_type, g):
        return g

    def get_approach_vector(self, arm, grasp_type):
        return APPROACH_DISTANCE*get_unit_vector([0, 0, -1])

    def get_approach_pose(self, approach_vector, g):
        return multiply(g, (approach_vector, unit_quat()))

    def get_all_joints(self):
        from pybullet_tools.flying_gripper_utils import SE3_GROUP, FINGERS_GROUP
        return SE3_GROUP + FINGERS_GROUP

    def get_lisdf_string(self):
        from pybullet_tools.flying_gripper_utils import FE_GRIPPER_URDF
        return """
    <include name="feg">
      <uri>../models/franka_description/robots/hand_se3.urdf</uri>
      {pose_xml}
    </include>
"""
    def get_positions(self, joint_group='hand', roundto=None):
        from pybullet_tools.flying_gripper_utils import get_se3_conf
        return tuple([round(n, roundto) for n in get_se3_conf(self)])

    def iterate_approach_path(self, arm, gripper, pose_value, grasp, body=None):
        from pybullet_tools.flying_gripper_utils import get_approach_path, set_cloned_se3_conf
        path = get_approach_path(self, body, grasp, custom_limits=self.custom_limits)
        for conf in path:
            set_cloned_se3_conf(self.body, gripper, conf.values)
            yield