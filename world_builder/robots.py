import math
import time
import copy
import random
from os.path import basename
from .entities import Robot

from pybullet_tools.utils import get_joint_positions, clone_body, set_all_color, TRANSPARENT, \
    link_from_name, get_link_subtree, get_joints, is_movable, multiply, invert, LockRenderer, \
    set_joint_positions, set_pose, GREEN, dump_body, get_pose, remove_body, PoseSaver, \
    ConfSaver, get_unit_vector, unit_quat, get_link_pose, unit_pose, draw_pose, remove_handles, \
    interpolate_poses, Pose, Euler, quat_from_euler, set_renderer, get_bodies, get_all_links, PI, \
    WorldSaver, is_darwin, wait_for_user, YELLOW, euler_from_quat, wait_for_duration, \
    wait_unlocked, wait_if_gui, set_renderer

from pybullet_tools.bullet_utils import equal, nice, get_gripper_direction, set_camera_target_body, Attachment, \
    BASE_LIMITS, get_rotation_matrix, collided, query_yes_no
from pybullet_tools.pr2_primitives import APPROACH_DISTANCE, Conf, Grasp, get_base_custom_limits
from pybullet_tools.pr2_utils import PR2_TOOL_FRAMES, PR2_GROUPS, close_until_collision, TOP_HOLDING_LEFT_ARM, \
    SIDE_HOLDING_LEFT_ARM
from pybullet_tools.general_streams import get_handle_link, get_grasp_list_gen, get_contain_list_gen, \
    get_cfree_approach_pose_test, get_stable_list_gen, play_trajectory

from world_builder.utils import load_asset


class RobotAPI(Robot):
    tool_from_hand = unit_pose()

    def __init__(self, body, **kwargs):
        super(RobotAPI, self).__init__(body, **kwargs)
        self.grippers = {}
        self.possible_obstacles = {}  ## body: obstacles
        self.collision_animations = []

    def get_init(self, init_facts=[], conf_saver=None):
        raise NotImplementedError('should implement this for RobotAPI!')

    def get_stream_map(self, problem, collisions, custom_limits, teleport, **kwargs):
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

    def get_gripper(self, arm='left', **kwargs):
        if arm not in self.grippers or self.grippers[arm] not in get_bodies():
            self.grippers[arm] = self.create_gripper(arm=arm, **kwargs)
        return self.grippers[arm]

    def make_grasps(self, g_type, arm, body, grasps_O, collisions=True):
        from pybullet_tools.general_streams import is_top_grasp
        app = self.get_approach_vector(arm, g_type)
        grasps_R = []
        for g in grasps_O:
            if is_top_grasp(self, arm, body, g, get_pose(body)) and False:
                approach = g
            else:
                approach = self.get_approach_pose(app, g)
            grasps_R.append(Grasp(g_type, body, g, approach,
                                  self.get_carry_conf(arm, g_type, g)))

        ## filter for grasp width
        filtered_grasps = []
        for grasp in grasps_R:
            grasp_width = self.compute_grasp_width(arm, get_pose(body), grasp,
                                                   body=body) if collisions else 0.0
            if grasp_width is not None:
                grasp.grasp_width = grasp_width
                filtered_grasps.append(grasp)
        return filtered_grasps

    def make_attachment(self, grasp, tool_link, visualize=False):
        o = grasp.body
        if isinstance(o, tuple) and len(o) == 2:
            body, joint = o
            link = get_handle_link(o)
            return Attachment(self, tool_link, grasp.value, body, child_joint=joint, child_link=link)

        arm = self.arms[0]
        tool_from_root = self.get_tool_from_root(arm)
        child_pose = get_pose(o)
        grasp_pose = self.get_grasp_pose(child_pose, grasp.value, body=o)
        gripper_pose = multiply(grasp_pose, invert(tool_from_root))
        grasp_pose = multiply(invert(gripper_pose), child_pose)
        attachment = Attachment(self, tool_link, grasp_pose, grasp.body)
        if visualize:
            with PoseSaver(attachment.child):
                set_camera_target_body(o)
                wait_for_user('robots.make_attachment | start attachment?')
                attachment.assign()
                set_renderer(True)
                set_camera_target_body(o)
                wait_for_user('robots.make_attachment | correct attachment?')
        return attachment

    def get_custom_limits(self):
        return self.custom_limits

    def get_body_pose(self, body_pose, body=None, verbose=False):
        title = f'    robot.get_body_pose({nice(body_pose)}, body={body})'

        ## if body_pose is handle link pose and body is (body, joint)
        if body is not None and isinstance(body, tuple) and not isinstance(body[0], tuple):
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

        r = get_rotation_matrix(body) if body is not None else self.tool_from_hand
        new_body_pose = multiply(body_pose, r)  ##
        # if verbose: print(f'{title} | multiply(body_pose, self.tool_from_hand) = {nice(new_body_pose)}')
        return new_body_pose

    def get_grasp_pose(self, body_pose, grasp, arm='left', body=None, verbose=False):
        ## those primitive shapes
        if body is not None and isinstance(body, int) and len(get_all_links(body)) == 1:
            from pybullet_tools.pr2_primitives import get_tool_from_root
            tool_from_root = multiply(((0, 0.025, 0.025), unit_quat()), self.tool_from_hand,
                                      get_tool_from_root(self.body, arm))  ##
        ## those urdf files made from one .obj file
        else:
            body_pose = self.get_body_pose(body_pose, body=body, verbose=verbose)
            tool_from_root = ((0, 0, -0.05), quat_from_euler((math.pi / 2, -math.pi / 2, -math.pi)))
        return multiply(body_pose, grasp, tool_from_root)

    def set_spawn_range(self, limits):
        self.spawn_range = limits


class MobileRobot(RobotAPI):

    arms = None

    def __init__(self, body, USE_TORSO=True, **kwargs):
        super(MobileRobot, self).__init__(body, **kwargs)
        self.USE_TORSO = USE_TORSO

    def get_arm_joints(self, arm):
        raise NotImplementedError('should implement this for MobileRobot!')

    def get_group_joints(self, group):
        raise NotImplementedError('should implement this for MobileRobot!')

    def get_all_arms(self):
        return self.arms

    def get_init(self, init_facts=[], conf_saver=None):
        robot = self.body

        def get_conf(joints):
            if conf_saver is None:
                return get_joint_positions(robot, joints)
            return [conf_saver.conf[conf_saver.joints.index(n)] for n in joints]

        def get_base_conf():
            base_joints = self.get_group_joints('base')
            initial_bq = Conf(robot, base_joints, get_conf(base_joints))
            for fact in init_facts:
                if fact[0] == 'bconf' and equal(fact[1].values, initial_bq.values):
                    return fact[1]
            return initial_bq

        def get_base_torso_conf():
            base_joints = self.get_group_joints('base-torso')
            initial_bq = Conf(robot, base_joints, get_conf(base_joints))
            for fact in init_facts:
                if fact[0] == 'btconf' and equal(fact[1].values, initial_bq.values):
                    return fact[1]
            return initial_bq

        def get_arm_conf(arm):
            arm_joints = self.get_arm_joints(arm)
            conf = Conf(robot, arm_joints, get_conf(arm_joints))
            for fact in init_facts:
                if fact[0] == 'aconf' and fact[1] == arm and equal(fact[2].values, conf.values):
                    return fact[2]
            return conf

        initial_bq = get_base_torso_conf() if self.USE_TORSO else get_base_conf()
        init = [('BConf', initial_bq), ('AtBConf', initial_bq)]

        for arm in self.get_all_arms():
            conf = get_arm_conf(arm)
            init += [('Arm', arm), ('AConf', arm, conf), ('AtAConf', arm, conf),
                     ('DefaultAConf', arm, conf), ('HandEmpty', arm)]
            if arm in self.arms:
                init += [('Controllable', arm)]

        return init

    def get_stream_map(self, problem, collisions, custom_limits, teleport,
                       domain_pddl=None, **kwargs):
        from pybullet_tools.pr2_agent import get_stream_map
        return get_stream_map(problem, collisions, custom_limits, teleport, **kwargs)

    def get_stream_info(self, **kwargs):
        from pybullet_tools.pr2_agent import get_stream_info
        return get_stream_info() ## partial=partial, defer=defer


class SpotRobot(MobileRobot):

    arms = ['hand']
    grasp_types = ['top', 'side']  ##
    joint_groups = ['hand', 'base-torso']

    def __init__(self, body, base_link='base', **kwargs):
        from pybullet_tools.spot_utils import SPOT_JOINT_GROUPS
        joints = SPOT_JOINT_GROUPS['base-torso']
        super(SpotRobot, self).__init__(body, base_link=base_link, joints=joints, **kwargs)

    def get_tool_link(self, a='hand'):
        from pybullet_tools.spot_utils import SPOT_TOOL_LINK
        return SPOT_TOOL_LINK

    def get_arm_joints(self, arm):
        return self.get_group_joints('arm')

    def get_group_joints(self, group):
        from pybullet_tools.spot_utils import get_group_joints
        return get_group_joints(self.body, group)

    # def get_stream_map(self, problem, collisions, custom_limits, teleport,
    #                    domain_pddl=None, **kwargs):
    #     from pybullet_tools.pr2_agent import get_stream_map
    #     stream_map = get_stream_map(problem, collisions, custom_limits, teleport, **kwargs)
    #     # stream_map = {k: v for k, v in stream_map.items() if k in {
    #     #     'sample-pose', 'sample-grasp', 'sample-grasp', 'sample-grasp'
    #     # }}
    #     return stream_map

    def create_gripper(self, arm='hand', **kwargs):
        from pybullet_tools.pr2_utils import create_gripper
        from pybullet_tools.spot_utils import SPOT_TOOL_LINK as link_name
        self.grippers[arm] = create_gripper(self.body, arm=arm, link_name=link_name, **kwargs)
        return self.grippers[arm]


class PR2Robot(MobileRobot):

    arms = ['left']
    grasp_types = ['top', 'side']
    joint_groups = ['left', 'right', 'base', 'base-torso']
    tool_from_hand = Pose(euler=Euler(math.pi / 2, 0, -math.pi / 2))
    finger_link = 7  ## for detecting if a grasp is pointing upwards

    def __init__(self, body, DUAL_ARM=False, **kwargs):
        super(PR2Robot, self).__init__(body, **kwargs)
        self.DUAL_ARM = DUAL_ARM
        self.grasp_aconfs = {}
        ## get away with the problem of Fluent stream outputs cannot be in action effects: ataconf

    def get_arm_joints(self, arm):
        from pybullet_tools.pr2_utils import get_arm_joints
        return get_arm_joints(self.body, arm)

    def get_group_joints(self, group):
        from pybullet_tools.pr2_utils import get_group_joints
        return get_group_joints(self.body, group)

    def get_all_arms(self):
        from pybullet_tools.pr2_utils import ARM_NAMES
        return ARM_NAMES

    def add_collision_grasp(self, a, o, g):
        from world_builder.actions import AttachObjectAction
        act = AttachObjectAction(a, g, o)
        to_add = str(act)
        added = [str(c) for c in self.collision_animations]
        if to_add not in added:
            self.collision_animations.append(act)

    def add_collision_conf(self, conf):  ## bconf, aconf
        from world_builder.actions import MoveArmAction
        import pickle
        # joints = bconf.joints + aconf.joints
        # values = bconf.values + aconf.values
        # conf = Conf(bconf.body, joints, values)
        self.collision_animations.append(MoveArmAction(conf))
        with open('collisions.pkl', 'wb') as file:
            pickle.dump(self.collision_animations, file)

    def get_arm_joints(self, arm):
        from pybullet_tools.pr2_utils import get_arm_joints
        return get_arm_joints(self.body, arm)

    def create_gripper(self, arm='left', **kwargs):
        # TODO(caelan): gripper bodies are removed
        from pybullet_tools.pr2_utils import create_gripper
        self.grippers[arm] = create_gripper(self.body, arm=arm, **kwargs)
        return self.grippers[arm]

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

    def compute_grasp_width(self, arm, body_pose, grasp, body=None, verbose=False, **kwargs):
        from pybullet_tools.pr2_utils import get_gripper_joints
        result = None
        with PoseSaver(body):
            with ConfSaver(self.body):
                assignment = self.get_attachment(grasp, arm)
                assignment.assign()
                gripper_joints = get_gripper_joints(self.body, arm)
                result = close_until_collision(self.body, gripper_joints, bodies=[body], **kwargs)
        return result

    # def compute_grasp_width(self, arm, body_pose, grasp, body=None, verbose=False, **kwargs):
    #     from pybullet_tools.pr2_utils import compute_grasp_width
    #     with PoseSaver(body):
    #         return compute_grasp_width(self.body, arm, body, grasp.value, **kwargs)

    # def get_grasp_pose(self, body_pose, grasp, arm='left', body=None, verbose=False):
    #     body_pose = self.get_body_pose(body_pose, body=body, verbose=verbose)
    #     tool_from_root = ((0, 0, -0.05), quat_from_euler((math.pi / 2, -math.pi / 2, -math.pi)))
    #     return multiply(body_pose, grasp, tool_from_root)

    def remove_grippers(self):
        for body in self.grippers.values():
            self.remove_gripper(body)

    def remove_gripper(self, gripper_handle):
        # TODO: update the cache
        remove_body(gripper_handle)

    def set_gripper_pose(self, body_pose, grasp, gripper=None, arm='left', body=None, verbose=False, **kwargs):
        if gripper is None:
            #gripper = self.create_gripper(arm, **kwargs)
            gripper = self.get_gripper(arm, **kwargs)
        self.open_cloned_gripper(gripper)
        grasp_pose = self.get_grasp_pose(body_pose, grasp, arm, body=body, verbose=verbose)
        set_pose(gripper, grasp_pose)
        #return grasp_pose
        return gripper

    def visualize_grasp(self, body_pose, grasp, arm='left', color=GREEN, cache=False,
                        body=None, verbose=False, **kwargs):
        gripper_grasp = self.create_gripper(arm, visual=True)
        if color is not None:
            set_all_color(gripper_grasp, color)
        self.set_gripper_pose(body_pose, grasp, gripper=gripper_grasp, arm=arm, body=body, verbose=verbose)

        # if verbose:
        #     handles = draw_pose(grasp_pose, length=0.05)

        ## ---- identify the direction the gripper is pointing towards
        # direction = get_gripper_direction(grasp_pose)
        # print('\n', nice(grasp_pose), direction)

        # if verbose:
        #     if direction == None:
        #         print('new direction')
        #         return gripper_grasp, True
        #     if 'down' in direction:
        #         print('pointing down')
        #         return gripper_grasp, True

        return gripper_grasp

    def mod_grasp_along_handle(self, grasp, dl):
        return multiply(grasp, Pose(point=(0, dl, 0)))

    def get_attachment(self, grasp, arm, **kwargs):
        tool_link = self.get_attachment_link(arm)
        return self.make_attachment(grasp, tool_link, **kwargs)
        # return Attachment(self.body, tool_link, grasp.value, grasp.body)

    def get_attachment_link(self, arm):
        return link_from_name(self.body, PR2_TOOL_FRAMES.get(arm, arm))

    def get_carry_conf(self, arm, grasp_type, g):
        return TOP_HOLDING_LEFT_ARM
        # if grasp_type == 'top':
        #     return TOP_HOLDING_LEFT_ARM
        # if grasp_type == 'side':
        #     return SIDE_HOLDING_LEFT_ARM

    def get_approach_vector(self, arm, grasp_type, scale=1):
        # return tuple(APPROACH_DISTANCE / 3 * get_unit_vector([0, 0, -1]))
        scale *= 4
        return tuple(scale * APPROACH_DISTANCE / 3 * get_unit_vector([0, 0, -1]))
        # if grasp_type == 'top':
        #     return APPROACH_DISTANCE*get_unit_vector([1, 0, 0])
        # if grasp_type == 'side':
        #     return APPROACH_DISTANCE*get_unit_vector([2, 0, -1])

    def get_approach_pose(self, approach_vector, g):
        return multiply(g, (approach_vector, unit_quat()))

    def get_all_joints(self):
        return sum(PR2_GROUPS.values(), [])

    def get_lisdf_string(self):
        return """
    <include name="{name}">
      <uri>../../assets/models/drake/pr2_description/urdf/pr2_simplified.urdf</uri>
      {pose_xml}
    </include>
"""

    def get_joints_from_group(self, joint_group):
        from pybullet_tools.pr2_utils import get_arm_joints, get_group_joints
        if joint_group == 'base':
            joints = self.joints
        elif joint_group == 'base-torso':
            joints = get_group_joints(self.body, joint_group)
        else: ## if joint_group == 'left':
            joints = get_arm_joints(self.body, joint_group)
        return joints

    def get_positions(self, joint_group='base', roundto=None):
        joints = self.get_joints_from_group(joint_group)
        positions = self.get_joint_positions(joints)
        if roundto == None:
            return positions
        return tuple([round(n, roundto) for n in positions])

    def set_group_positions(self, joint_group, positions):
        joints = self.get_joints_from_group(joint_group)
        assert len(joints) == len(positions)
        self.set_joint_positions(joints, positions)

    # def get_custom_limits(self):
    #     return get_base_custom_limits(self, BASE_LIMITS)

    def get_tool_from_root(self, a):
        from pybullet_tools.pr2_primitives import get_tool_from_root
        return get_tool_from_root(self.body, a)

    def iterate_approach_path(self, arm, gripper, pose_value, grasp, body=None):
        tool_from_root = self.get_tool_from_root(arm)
        grasp_pose = multiply(pose_value, invert(grasp.value))
        approach_pose = multiply(pose_value, invert(grasp.approach))
        for tool_pose in interpolate_poses(grasp_pose, approach_pose):
            set_pose(gripper, multiply(tool_pose, tool_from_root))
            # if body is not None:
            #     set_pose(body, multiply(tool_pose, grasp.value))
            yield

    def get_custom_limits(self):  ## TODO: needs to test this
        custom_limits = get_base_custom_limits(self.body, self.custom_limits)
        return custom_limits

    @property
    def base_group(self):
        return 'base-torso' if self.USE_TORSO else 'base'

    def get_base_joints(self):
        from pybullet_tools.pr2_utils import get_group_joints
        return get_group_joints(self.body, self.base_group)

    def randomly_spawn(self):
        def sample_robot_conf():
            (x1, y1, z1), (x2, y2, z2) = self.spawn_range
            x = random.uniform(x1, x2)
            y = random.uniform(y1, y2)
            yaw = random.uniform(0, math.pi)
            if self.USE_TORSO:
                z1 = max(z1, 0)
                z2 = min(z2, 0.35)
                z = random.uniform(z1, z2)
                return [x, y, z, yaw]
            return [x, y, yaw]

        self.set_positions(sample_robot_conf(), self.get_base_joints())

    def get_tool_link(self, a):
        from pybullet_tools.pr2_utils import PR2_TOOL_FRAMES
        return PR2_TOOL_FRAMES[a]

    def get_base_conf(self):
        from pybullet_tools.pr2_utils import get_group_joints
        base_joints = get_group_joints(self.body, self.base_group)
        q = get_joint_positions(self.body, base_joints)
        return Conf(self.body, base_joints, q)

    def get_arm_conf(self, arm):
        from pybullet_tools.pr2_utils import get_arm_joints
        arm_joints = get_arm_joints(self.body, arm)
        return Conf(self.body, arm_joints)

    def check_reachability(self, body, state, verbose=False, visualize=False, debug=False,
                           max_attempts=10, fluents=[]):
        from pybullet_tools.pr2_primitives import Pose
        from pybullet_tools.pr2_streams import get_ik_gen, get_ik_fn
        from world_builder.entities import Object

        if is_darwin():
            return True

        title = f'          robot.check_reachability for {body}'
        obj = body
        if isinstance(body, Object):
            title += f'({body.body})'
            body = body.body

        start = time.time()
        if verbose:
            print(f'... started {title}', end='\r')

        # context_saver = WorldSaver(bodies=[state.world.robot])
        arm = self.arms[0]
        q = self.get_base_conf()
        aq = self.get_arm_conf('left')
        p = Pose(body, get_pose(body))
        movable_poses = [(f[1], f[2]) for f in fluents if f[0] == 'AtPose']
        if debug:
            for b2, p2 in movable_poses:
                print(f'       {b2} {state.world.BODY_TO_OBJECT[b2].name}\t {p2}')
            # print('       ik_gen debug bottle')
            # p.assign()
            # set_camera_target_body(body)
            # wait_unlocked()

        def restore(result):
            # context_saver.restore()
            # remove_body(state.gripper)
            # state.gripper = None
            q.assign()
            aq.assign()
            p.assign()
            if verbose:
                print(f'... finished {title} -> {result} in {round(time.time() - start, 2)}s')

        with LockRenderer(True):
            funk = get_grasp_list_gen(state, verbose=verbose, visualize=visualize, top_grasp_tolerance=PI / 4)
            test = None
            outputs = funk(obj)

            kwargs = dict(collisions=True, teleport=False)
            funk2 = get_ik_gen(state, max_attempts=max_attempts, ir_only=True, learned=False,
                               custom_limits=state.robot.custom_limits,
                               verbose=verbose, visualize=visualize, **kwargs)

            funk3 = get_ik_fn(state, verbose=verbose, visualize=visualize, **kwargs)
            for (grasp, ) in outputs:
                ## test_approach_path
                result = True
                for b2, p2 in movable_poses:
                    if test is None:
                        test = get_cfree_approach_pose_test(state)
                    result = test(body, p, grasp, b2, p2)
                    if body not in self.possible_obstacles:
                        self.possible_obstacles[body] = set()
                    if not result:
                        self.possible_obstacles[body].add(b2)
                    elif b2 in self.possible_obstacles[body]:
                        self.possible_obstacles[body].remove(b2)
                    if debug:
                        print(title, f'| test_approach_path({body}, {b2})', result)
                if not result:
                    continue

                ## find bconf
                gen = funk2(arm, body, p, grasp)
                try:
                    result = next(gen)
                    if result is not None:
                        (bconf,) = result
                        ## find aconf
                        result = funk3(arm, body, p, grasp, bconf, fluents=fluents)
                        if result is not None:
                            if verbose:
                                print(title, f'succeeded', bconf)

                            if debug and False:
                                answer = query_yes_no(f"play reachability test traj?", default='no')
                                if answer:
                                    (cmd,) = result
                                    bconf.assign()
                                    attachment = grasp.get_attachment(self, arm, visualize=False)
                                    play_trajectory(cmd, p=p, attachment=attachment)

                            restore(True)
                            return True
                        else:
                            if verbose:
                                print(title, f'IK failed')
                except Exception:
                    if verbose:
                        print(title, f'IR failed')
                    pass

            restore(False)
            return False

    def check_reachability_space(self, body_link, state, body=None, fluents=[], num_samples=5, **kwargs):

        for f in fluents:
            if f[0].lower() == 'atposition':
                f[2].assign()

        ## sample three poses of cabbage and check reachability of each
        obj = None
        if body is None:
            body, path = load_asset('VeggieCabbage')[:2]
            obj = state.world.add_body(body, 'marker', path=path)

        if 'space' in state.world.get_type(body_link):
            funk = get_contain_list_gen(state, verbose=False, num_samples=num_samples)
        else:
            funk = get_stable_list_gen(state, verbose=False, num_samples=num_samples)
        outputs = funk(body, body_link)
        result = False
        for j in range(len(outputs)):
            if outputs is None or outputs[j] is None:
                print('\ncheck_reachability_space | outputs', outputs, '\n')
                continue
            set_pose(body, outputs[j][0].value)
            result = result or self.check_reachability(body, state, fluents=fluents, **kwargs)
            if result:
                break

        if not result and body in self.possible_obstacles:
            self.possible_obstacles[body_link] = self.possible_obstacles[body]

        if obj is not None:
            state.world.remove_object(obj)
        return result

    def plan_joint_motion(self, start_conf, end_conf, mp_fn, attachments, arm, pose=None,
                          verbose=False, title='robot.plan_joint_motion', debug=False, **kwargs):
        from pybullet_tools.utils import BodySaver
        from pybullet_tools.pr2_utils import get_arm_joints
        from pybullet_tools.pr2_primitives import create_trajectory, Commands, State

        def debug_trajectory(grasp_path, **kwargs):
            attachment = list(attachments.values())[0] if len(attachments) else None
            mt = create_trajectory(robot, arm_joints, grasp_path)
            cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt])
            play_trajectory(cmd, p=pose, attachment=attachment, **kwargs)

        robot = self
        obss = kwargs['obstacles']
        arm_joints = get_arm_joints(robot, arm)
        set_joint_positions(robot, arm_joints, start_conf)
        grasp_path = mp_fn(robot, arm_joints, end_conf, attachments=attachments.values(), **kwargs)
        if grasp_path is None:

            if debug:
                ## try again without attachments, just for debugging purposes
                set_joint_positions(robot, arm_joints, start_conf)
                grasp_path = mp_fn(robot, arm_joints, end_conf, attachments=[], **kwargs)
                reason = 'robot-world collision'
                if grasp_path is not None:
                    reason = 'object-world collision'
                    ## replay the grasp path to find colliding objects
                    debug_trajectory(grasp_path, obstacles=obss, title='investigate path collision')
                if verbose:
                    print(f'{title}Grasp path failure because of', reason)
                    wait_unlocked()
            return None
        if debug:
            kwargs = dict(obstacles=obss, title='successfully found path')
            if len(attachments) == 0:
                debug_trajectory(grasp_path, **kwargs)
            else:
                debug_trajectory(grasp_path[::-1], **kwargs)
        return grasp_path

    def compute_grasp_aconf(self, arm, obj, pose, grasp, custom_limits={}, grasp_conf_tries=10,
                           verbose=False, title='robot.sample_grasp_aconf', **collided_kwargs):
        from pybullet_tools.pr2_primitives import get_tool_from_root
        from pybullet_tools.ikfast.pr2.ik import pr2_inverse_kinematics
        key = (arm, obj, grasp)
        if key in self.grasp_aconfs:
            print(f'grasp_conf found', key)
            return self.grasp_aconfs[key]
        robot = self
        pose_value = grasp.value

        tool_from_root = get_tool_from_root(robot, arm)
        gripper_pose = multiply(robot.get_grasp_pose(pose.value, pose_value, body=obj), invert(tool_from_root))

        grasp_conf = pr2_inverse_kinematics(robot, arm, gripper_pose, custom_limits=custom_limits) #, upper_limits=USE_CURRENT)
                                                #nearby_conf=USE_CURRENT) # upper_limits=USE_CURRENT,

        count_down = grasp_conf_tries
        while (grasp_conf is None) or collided(robot, **collided_kwargs):
            if verbose:
                if grasp_conf is not None:
                    grasp_conf = nice(grasp_conf)
                print(
                    f'{title}Grasp IK failure | {grasp_conf} <- pr2_inverse_kinematics({robot}, '
                    f'{arm}, {nice(gripper_pose[0])}) | pose {pose}, grasp {grasp}')
            if count_down == 0:
                #if grasp_conf is not None:
                #    print(grasp_conf)
                #    #wait_if_gui()

                return None
            grasp_conf = pr2_inverse_kinematics(robot, arm, gripper_pose, custom_limits=custom_limits)
            count_down -= 1

        if verbose:
            print(f'{title}Grasp IK success | {nice(grasp_conf)} = pr2_inverse_kinematics({robot} '
                  f'{arm}, {nice(gripper_pose[0])}) | pose = {pose}, grasp = {grasp}')
        print(f'grasp_conf after', key, grasp_conf)
        self.grasp_aconfs[key] = grasp_conf
        return grasp_conf


class FEGripper(RobotAPI):

    arms = ['hand']
    grasp_types = ['hand']
    joint_groups = ['hand']
    tool_from_hand = Pose(euler=Euler(math.pi / 2, 0, -math.pi / 2))
    finger_link = 8  ## for detecting if a grasp is pointing upwards

    # def get_pose(self):
    #     from pybullet_tools.flying_gripper_utils import get_se3_conf
    #     return get_se3_conf(self.body)

    def create_gripper(self, arm='hand', visual=True, color=None):
        from pybullet_tools.utils import unit_pose
        if arm in self.grippers:
            gripper = self.grippers[arm]
        else:
            gripper = clone_body(self.body, visual=False, collision=True)
            self.grippers[arm] = gripper
        set_pose(gripper, unit_pose())
        if not visual:
            set_all_color(gripper, TRANSPARENT)
        if color is not None:
            set_all_color(gripper, color)
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

        if isinstance(body, tuple):
            return 0.02
            # body = body[0]

        # with PoseSaver(body):
        body_pose = unit_pose()
        grasp = grasp.value
        grasp_pose = multiply(body_pose, grasp)
        if verbose:
            print(f'robots.compute_grasp_width | body_pose = {nice(body_pose)} | grasp = {nice(grasp)}')
            print('robots.compute_grasp_width | grasp_pose = multiply(body_pose, grasp) = ', nice(grasp_pose))

        with ConfSaver(self.body):
            with PoseSaver(body):
                conf = se3_ik(self, grasp_pose)
                if conf is None:
                    print('\t\t\tFEGripper.conf is None', nice(grasp))
                    return None
                # print('\tFEGripper.compute_grasp_width', nice(grasp))
                gripper = self.body
                set_se3_conf(gripper, conf)
                body_pose = self.get_body_pose(body_pose, body=body, verbose=verbose)
                set_pose(body, body_pose)
                if verbose:
                    print(f'robots.compute_grasp_width | gripper_grasp {gripper} | object_pose {nice(body_pose)}'
                          f' | se_conf {nice(get_se3_conf(gripper))} | grasp = {nice(grasp)} ')

                gripper_joints = self.get_gripper_joints()
                width = close_until_collision(gripper, gripper_joints, bodies=[body], **kwargs)
                # remove_body(gripper)
        return width

    def visualize_grasp(self, body_pose, grasp, arm='hand', color=GREEN, width=1, verbose=False,
                        body=None, mod_target=None):
        from pybullet_tools.flying_gripper_utils import se3_ik, set_cloned_se3_conf, get_cloned_se3_conf
        from pybullet_tools.utils import Pose, euler_from_quat
        title = 'robots.visualize_grasp |'

        body_pose = self.get_body_pose(body_pose, body=body, verbose=verbose)
        gripper = self.create_gripper(arm, visual=True)
        self.open_cloned_gripper(gripper, width)
        set_all_color(gripper, color)

        grasp_pose = multiply(body_pose, grasp)
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
        if grasp_conf is None:
            return None

        # set_pose(self.body, grasp_pose) ## wrong!
        set_cloned_se3_conf(self.body, gripper, grasp_conf)

        # set_camera_target_body(gripper, dx=0.5, dy=0.5, dz=0.5)
        return gripper

    def mod_grasp_along_handle(self, grasp, dl):
        return multiply(grasp, Pose(point=(dl, 0, 0)))

    def get_tool_from_root(self, arm):
        return ((0, 0, -0.05), quat_from_euler((math.pi / 2, -math.pi / 2, -math.pi)))

    def get_init(self, init_facts=[], conf_saver=None):
        from pybullet_tools.flying_gripper_utils import get_se3_joints, get_se3_conf, ARM_NAME
        robot = self.body

        def get_conf(joints):
            if conf_saver is None:
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
        return [('SEConf', initial_q), ('AtSEConf', initial_q), ('OriginalSEConf', initial_q),
                ('Arm', arm), ('Controllable', arm), ('HandEmpty', arm)]

    def get_attachment(self, grasp, arm=None, **kwargs):
        tool_link = link_from_name(self.body, 'panda_hand')
        return self.make_attachment(grasp, tool_link, **kwargs)
        # return Attachment(self.body, tool_link, grasp.value, grasp.body)

    def get_attachment_link(self, arm):
        from pybullet_tools.flying_gripper_utils import TOOL_LINK
        return link_from_name(self.body, TOOL_LINK)

    def get_carry_conf(self, arm, grasp_type, g):
        return g

    def get_approach_vector(self, arm, grasp_type, scale=1):
        return APPROACH_DISTANCE/3 *get_unit_vector([0, 0, -1]) * scale

    def get_approach_pose(self, approach_vector, g):
        return multiply(g, (approach_vector, unit_quat()))

    def get_all_joints(self):
        from pybullet_tools.flying_gripper_utils import SE3_GROUP, FINGERS_GROUP
        return SE3_GROUP + FINGERS_GROUP

    def get_lisdf_string(self):
        from pybullet_tools.flying_gripper_utils import FE_GRIPPER_URDF
        return """
    <include name="{name}">
      <uri>../../assets/models/franka_description/robots/hand_se3.urdf</uri>
      {pose_xml}
    </include>
"""
    def get_positions(self, joint_group='hand', roundto=None):
        from pybullet_tools.flying_gripper_utils import get_se3_conf
        return tuple([round(n, roundto) for n in get_se3_conf(self)])

    def iterate_approach_path(self, arm, gripper, pose_value, grasp, obstacles=[], body=None):
        from pybullet_tools.flying_gripper_utils import get_approach_path, set_cloned_se3_conf
        path = get_approach_path(self, body, grasp, obstacles=obstacles, custom_limits=self.custom_limits)
        if path == None:
            return
        for conf in path:
            set_cloned_se3_conf(self.body, gripper, conf.values)
            yield

    def get_stream_map(self, problem, collisions, custom_limits, teleport, domain_pddl=None, **kwargs):
        if basename(domain_pddl) == 'feg_kitchen_clean.pddl':
            from nsplan_tools.feg_streams import get_stream_map
        else:
            from pybullet_tools.flying_gripper_agent import get_stream_map
        return get_stream_map(problem, collisions, custom_limits, teleport, **kwargs)

    def get_stream_info(self):
        from pybullet_tools.flying_gripper_agent import get_stream_info
        # from pybullet_tools.pr2_agent import get_stream_info
        return get_stream_info()

    def get_base_joints(self):
        from pybullet_tools.flying_gripper_utils import get_se3_joints
        return get_se3_joints(self.body)

    def randomly_spawn(self):
        def sample_robot_conf():
            (x1, y1, z1), (x2, y2, z2) = self.spawn_range
            x = random.uniform(x1, x2)
            y = random.uniform(y1, y2)
            z = random.uniform(z1, z2)
            # yaw = random.uniform(0, math.pi)
            return [x, y, z, 0, -math.pi / 2, 0]

        self.set_positions(sample_robot_conf(), self.get_base_joints())

    def get_initial_q(self):
        init_q = get_joint_positions(self.body, self.get_base_joints())
        # init_q = get_joint_positions(self.body, self.get_base_joints())[:-1]
        # init_q = list(init_q) + [0] * 3
        return init_q

    def check_reachability(self, body, state, fluents=[], obstacles=None, verbose=False, debug=False):
        from pybullet_tools.flying_gripper_utils import get_cloned_se3_conf, plan_se3_motion

        return True

        if obstacles is None:
            obstacles = state.obstacles
        if body in obstacles:
            obstacles = copy.deepcopy(obstacles)
            obstacles.remove(body)

        robot = self.body
        world = state.world
        init_q = self.get_initial_q()
        funk = get_grasp_list_gen(state, collisions=True, visualize=False,
                                  RETAIN_ALL=False, top_grasp_tolerance=math.pi / 4)

        result = False
        body_pose = get_pose(body)
        outputs = funk(body)
        for output in outputs:
            grasp = output[0]
            w = grasp.grasp_width
            gripper_grasp = self.visualize_grasp(body_pose, grasp.value, body=grasp.body, color=GREEN, width=w)
            end_q = get_cloned_se3_conf(robot, gripper_grasp)
            if not collided(gripper_grasp, obstacles, verbose=True, tag='check reachability of movable', world=world):
                if verbose: print('\n... check reachability from', nice(init_q), 'to', nice(end_q))
                path = plan_se3_motion(self, init_q, end_q, obstacles=obstacles,
                                       custom_limits=self.custom_limits)
                if path is not None:
                    if verbose: print('... path found of length', len(path))
                    result = True
                    break
                else:
                    if verbose: print('... no path found', nice(end_q))
            else:
                if verbose: print('... collided', nice(end_q))
            remove_body(gripper_grasp)
        return result

    def check_reachability_space(self, body_link, state, fluents=[], obstacles=None, verbose=False):
        from pybullet_tools.flying_gripper_utils import set_cloned_se3_conf, plan_se3_motion, \
            get_se3_joints

        if obstacles is None:
            obstacles = state.obstacles
        for f in fluents:
            if f[0].lower() == 'atposition':
                f[2].assign()

        robot = self.body
        init_q = self.get_initial_q()
        q = Conf(robot, get_se3_joints(robot), init_q)
        marker = self.create_gripper(color=YELLOW)
        set_cloned_se3_conf(robot, marker, [0] * 6)

        funk = get_contain_list_gen(state, verbose=False, visualize=True)
        gen = funk(marker, body_link)
        count = 4
        result = False
        for output in gen:
            if output is None:
                break
            p = output[0].value
            (x, y, z), quat = p
            end_q = list([x, y, z + 0.1]) + list(euler_from_quat(quat))
            path = plan_se3_motion(robot, init_q, end_q, obstacles=obstacles,
                                   custom_limits=self.custom_limits)
            if verbose: print('\n... check reachability from', nice(init_q), 'to space', nice(end_q))
            if path is not None:
                if verbose: print('... path found of length', len(path))
                result = True
                break
            else:
                if verbose: print('... no path found', nice(end_q))

            if count == 0:
                break
            count -= 1

        remove_body(marker)
        q.assign()
        return result

