import numpy as np
import pybullet as p
import copy

from pybullet_tools.bullet_utils import clip_delta, multiply2d, is_above, nice, open_joint, set_camera_target_robot, \
    toggle_joint, add_attachment, remove_attachment, draw_pose2d_path, \
    draw_pose3d_path
from pybullet_tools.pr2_streams import Position, get_pull_door_handle_motion_gen, \
    LINK_POSE_TO_JOINT_POSITION

from pybullet_tools.utils import str_from_object, get_closest_points, INF, create_attachment, wait_if_gui, \
    get_aabb, get_joint_position, get_joint_name, get_link_pose, link_from_name, PI, Pose, Euler, \
    get_extend_fn, get_joint_positions, set_joint_positions, get_max_limit, get_pose, set_pose, set_color, \
    remove_body, create_cylinder, set_all_static, wait_for_duration, remove_handles, set_renderer, \
    LockRenderer
from pybullet_tools.pr2_utils import PR2_TOOL_FRAMES, get_gripper_joints
from pybullet_tools.pr2_primitives import Trajectory, Command, Conf, Trajectory, Commands
from pybullet_tools.flying_gripper_utils import set_se3_conf

from .world import State


class Action(object): # TODO: command
    def transition(self, state):
        raise NotImplementedError()
    def __repr__(self):
        return '{}{}'.format(self.__class__.__name__, str_from_object(self.__dict__))


class RobotAction(object):
    def __init__(self, robot):
        self.robot = robot


#######################################################


class TeleportAction(Action):
    def __init__(self, conf):
        self.conf = conf
    def transition(self, state):
        joints = state.robot.get_joints()  ## all joints, not just x,y,yqw
        if len(self.conf) == len(joints):
            state.robot.set_positions(self.conf, joints = joints)
        else:
            state.robot.set_pose(self.conf)
        set_camera_target_robot(state.robot)
        return state.new_state()


class MoveAction(Action):
    #def __init__(self, delta_x=0., delta_y=0., delta_yaw=0.):
    def __init__(self, delta): # TODO: pass in the robot
        # TODO: clip or normalize if moving too fast?
        self.delta = delta
    def transition(self, state):
        if self.delta is None:
            return state
        # TODO: could move to the constructor instead
        #new_delta = self.delta
        new_delta = clip_delta(self.delta, state.world.max_velocities, state.world.time_step)
        if not np.allclose(self.delta, new_delta, atol=0., rtol=1e-6):
            print('Warning! Clipped delta from {} to {}'.format(np.array(self.delta), np.array(new_delta)))

        assert len(new_delta) == len(state.robot.joints)
        conf = np.array(state.robot.get_positions())
        new_conf = multiply2d(conf, new_delta)
        # new_conf = [wrap_angle(position) for joint, position in zip(robot.joints, new_conf)]
        state.robot.set_positions(new_conf)
        return state.new_state()


class MoveArmAction(Action):
    def __init__(self, conf):
        self.conf = conf
    def transition(self, state):
        set_joint_positions(state.robot, self.conf.joints, self.conf.values)
        return state.new_state()


class MoveBaseAction(MoveArmAction):
    pass

#######################################################


class DriveAction(MoveAction):
    def __init__(self, delta=0.):
        super(DriveAction, self).__init__(delta=[delta, 0, 0])


class TurnAction(MoveAction):
    def __init__(self, delta=0.):
        super(TurnAction, self).__init__(delta=[0, 0, delta])


#######################################################


class AttachAction(Action):
    attach_distance = 5e-2
    def transition(self, state):
        new_attachments = dict(state.attachments)
        for obj in state.movable:
            collision_infos = get_closest_points(state.robot, obj, max_distance=INF)
            min_distance = min([INF] + [info.contactDistance for info in collision_infos])
            if (obj not in new_attachments) and (min_distance < self.attach_distance):
                attachment = create_attachment(state.robot, state.robot.base_link, obj)
                new_attachments[obj] = attachment
        return state.new_state(attachments=new_attachments)


class ReleaseAction(Action):
    def transition(self, state):
        return state.new_state(attachments={})


######################## Teleop Agent ###############################


class FlipAction(Action):
    # TODO: parent action
    def __init__(self, switch): #, active=True):
        self.switch = switch
        #self.active = active
    def transition(self, new_state):
        if not any(is_above(robot, get_aabb(self.switch)) for robot in new_state.robots):
            return new_state
        new_state.variables['Pressed', self.switch] = not new_state.variables['Pressed', self.switch]
        return new_state


class PressAction(Action):
    def __init__(self, button):
        self.button = button
    def transition(self, new_state):
        # TODO: automatically pass a copy of the state
        if not any(is_above(robot, get_aabb(self.button)) for robot in new_state.robots):
            return new_state
        new_state.variables['Pressed', self.button] = True
        return new_state

class OpenJointAction(Action):
    def __init__(self, affected):
        self.affected = affected
    def transition(self, state):
        for body, joint in self.affected:
            old_pose = get_joint_position(body, joint)
            toggle_joint(body, joint)
            new_pose = get_joint_position(body, joint)
            obj = state.world.BODY_TO_OBJECT[(body, joint)]
            print(f'{(body, joint)} | {obj.name} | limit: {nice((obj.min_limit, obj.max_limit))} | pose: {old_pose} -> {new_pose}')
        return state.new_state()

class PickUpAction(Action):
    def __init__(self, object, gripper='left'):
        self.object = object
        self.gripper = gripper
    def transition(self, state):
        obj = self.object
        tool_link = PR2_TOOL_FRAMES[self.gripper]
        tool_pose = get_link_pose(state.robot, link_from_name(state.robot, tool_link))
        old_pose = obj.get_pose()
        obj.set_pose(tool_pose)
        new_pose = obj.get_pose()
        print(f"{obj.name} is teleported from {nice(old_pose)} to {self.gripper} gripper {nice(new_pose)}")

        state.robot.objects_in_hand[self.gripper] = obj
        new_attachments = add_attachment(state=state, obj=obj, attach_distance=0.1)
        return state.new_state(attachments=new_attachments)

class PutDownAction(Action):
    def __init__(self, surface, gripper='left'):
        self.surface = surface
        self.gripper = gripper
    def transition(self, state):
        obj = state.robot.objects_in_hand[self.gripper]
        state.robot.objects_in_hand[self.gripper] = -1
        print(f'DEBUG1, PutDownAction transition {obj}')
        self.surface.place_obj(obj)
        new_attachments = remove_attachment(state, obj)
        return state.new_state(attachments=new_attachments)

OBJECT_PARTS = {
    'Veggie': ['VeggieLeaf', 'VeggieStem'],
    'Egg': ['EggFluid', 'EggShell']
} ## object.category are all lower case
OBJECT_PARTS = {k.lower():v for k,v in OBJECT_PARTS.items()}

class ChopAction(Action):
    def __init__(self, object):
        self.object = object
    def transition(self, state):
        pose = self.object.get_pose()
        surface = self.object.supporting_surface
        for obj_name in OBJECT_PARTS[self.object.category]:
            part = surface.place_new_obj(obj_name)
            yaw = np.random.uniform(0, PI)
            part.set_pose(Pose(point=pose[0], euler=Euler(yaw=yaw)))
        state.world.remove_object(self.object)
        objects = state.objects
        objects.remove(self.object.body)
        return state.new_state(objects=objects)

class CrackAction(Action):
    def __init__(self, object, surface):
        self.object = object
        self.surface = surface
    def transition(self, state):
        pose = self.object.get_pose()

class ToggleSwitchAction(Action):
    def __init__(self, object):
        self.object = object

class InteractAction(Action):
    def transition(self, state):
        return state


######################## PR2 Agent ###############################

class TeleportObjectAction(Action):
    def __init__(self, arm, grasp, object):
        self.object = object
        self.arm = arm
        self.grasp = grasp
    def transition(self, state):
        old_pose = get_pose(self.object)
        link = link_from_name(state.robot, PR2_TOOL_FRAMES.get(self.arm, self.arm))
        set_pose(self.object, get_link_pose(state.robot, link))
        new_pose = get_pose(self.object)
        print(f"   [TeleportObjectAction] !!!! obj {self.object} is teleported from {nice(old_pose)} to {self.arm} gripper {nice(new_pose)}")
        return state.new_state()

class GripperAction(Action):
    def __init__(self, arm, position=None, extent=None, teleport=False):
        self.arm = arm
        self.position = position
        self.extent = extent  ## 1 means fully open, 0 means fully closed
        self.teleport = teleport

    def transition(self, state):
        robot = state.robot

        ## get width from extent
        if self.extent != None:
            gripper_joint = robot.get_gripper_joints(self.arm)[0]
            self.position = get_max_limit(robot, gripper_joint)

        joints = robot.get_gripper_joints(self.arm)
        start_conf = get_joint_positions(robot, joints)
        end_conf = [self.position] * len(joints)
        if self.teleport:
            path = [start_conf, end_conf]
        else:
            extend_fn = get_extend_fn(robot, joints)
            path = [start_conf] + list(extend_fn(start_conf, end_conf))
        for positions in path:
            set_joint_positions(robot, joints, positions)

        return state.new_state()

class AttachObjectAction(Action):
    def __init__(self, arm, grasp, object, verbose=True):
        self.arm = arm
        self.grasp = grasp
        self.object = object
        self.verbose = verbose
    def transition(self, state):
        link = state.robot.get_attachment_link(self.arm)
        new_attachments = add_attachment(state=state, obj=self.object, parent=state.robot,
                                         parent_link=link, attach_distance=None, verbose=self.verbose)  ## can attach without contact
        for k in new_attachments:
            if k in state.world.ATTACHMENTS:
                state.world.ATTACHMENTS.pop(k)
        return state.new_state(attachments=new_attachments)

class DetachObjectAction(Action):
    def __init__(self, arm, object, verbose=False):
        self.arm = arm
        self.object = object
    def transition(self, state):
        # print(f'bullet.actions | DetachObjectAction | remove {self.object} from state.attachment')
        new_attachments = remove_attachment(state, self.object)
        return state.new_state(attachments=new_attachments)

class JustDoAction(Action):
    def __init__(self, body):
        self.body = body
    def transition(self, state):
        label = self.__class__.__name__.lower().replace('just','').capitalize() + 'ed'
        if label.endswith('eed'): label = label.replace('eed', '')
        state.variables[label, self.body] = True
        if hasattr(state.world, 'BODY_TO_OBJECT'):
            state.world.BODY_TO_OBJECT[self.body].add_text(label)
        return state.new_state()

class JustClean(JustDoAction):
    def __init__(self, body):
        super(JustClean, self).__init__(body)

class JustCook(JustDoAction):
    def __init__(self, body):
        super(JustCook, self).__init__(body)

class JustSeason(JustDoAction):
    def __init__(self, body):
        super(JustSeason, self).__init__(body)

class JustServe(JustDoAction):
    def __init__(self, body):
        super(JustServe, self).__init__(body)

class JustSucceed(Action):
    def __init__(self):
        pass
    def transition(self, state):
        return state.new_state()


class ChangePositions(Action):
    def __init__(self, pstn):
        self.pstn = pstn
    def transition(self, state):
        self.pstn.assign()
        return state.new_state()


class MagicDisappear(Action):
    def __init__(self, body):
        self.body = body
    def transition(self, state):
        state.world.remove_object(state.world.BODY_TO_OBJECT[self.body])
        objects = copy.deepcopy(state.objects)
        if self.body in objects: objects.remove(self.body)
        return state.new_state(objects=objects)


class TeleportObject(Action):
    def __init__(self, body, pose):
        self.body = body
        self.pose = pose
    def transition(self, state):
        self.pose.assign()
        return state.new_state()


class ChangeJointPosition(Action):
    def __init__(self, position):
        self.position = position

    def transition(self, state):
        pst = self.position
        max_position = Position((pst.body, pst.joint), 'max')
        if pst.value == max_position:
            state.variables['Opened', (pst.body, pst.joint)] = True
        self.position.assign()
        return state.new_state()


class ChangeLinkColorEvent(Action):
    def __init__(self, body, color, link=None):
        self.body = body
        self.color = color
        self.link = link
    def transition(self, state):
        set_color(self.body, self.color, self.link)
        return state.new_state()


class CreateCylinderEvent(Action):
    def __init__(self, radius, height, color, pose):
        self.radius = radius
        self.height = height
        self.color = color
        self.pose = pose
        self.body = None
    def transition(self, state):
        objects = state.objects
        if self.body == None:
            self.body = create_cylinder(self.radius, self.height, color=self.color)
            if self.body not in objects: objects.append(self.body)
            set_pose(self.body, self.pose)
            set_all_static()
            print(f'    bullet.actions.CreateCylinderEvent | {self.body} at {nice(self.pose)}')
        return state.new_state(objects=objects) ## we may include the new body in objects, becomes the new state for planning


class RemoveBodyEvent(Action):
    def __init__(self, body=None, event=None):
        self.body = body
        self.event = event
    def transition(self, state):
        body = None
        objects = state.objects
        if self.body != None:
            body = self.body
            remove_body(self.body)
        if self.event != None:
            body = self.event.body
        if body != None:
            remove_body(body)
            objects.remove(body)
            print(f'    bullet.actions.RemoveBodyEvent | {body}')
        return state.new_state(objects=objects)

#######################################################

# class MovePoseAction(Action):
#     def __init__(self, pose):
#         self.pose = pose
#     def transition(self, state):
#         set_pose(self.pose.body, self.pose.value)
#         return state.new_state()


class MoveInSE3Action(Action):
    def __init__(self, conf):
        self.conf = conf
    def transition(self, state):
        set_se3_conf(state.robot, self.conf)
        return state.new_state()

#######################################################


def adapt_action(a, problem, plan, verbose=True):
    if plan is None:
        return a

    if a.__class__.__name__ == 'AttachObjectAction' and isinstance(a.object, tuple):
        robot = problem.world.robot
        plan, continuous = plan
        if len(plan) == 1 and len(plan[0]) > 1:
            plan = plan[0]
        def get_value(string):
            name, tup = string.split('=')
            # value = continuous[name]
            # if name.startswith('pstn'):
            #     value = value[0]
            # return value
            return eval(tup)
        if ' ' in plan[0][0]:
            act = [aa for aa in plan if aa[0].startswith('pull') and aa[2] == problem.world.body_to_name[a.object]][0]
        else:
            act = [aa for aa in plan if aa[0] == 'pull_door_handle' and aa[2] == str(a.object)][0]
        pstn1 = Position(a.object, get_value(act[3]))
        pstn2 = Position(a.object, get_value(act[4]))
        bq1 = get_value(act[6])  ## continuous[act[6].split('=')[0]]
        bq1 = Conf(robot.body, robot.get_base_joints(), bq1)
        aq1 = get_value(act[9]) ## continuous[act[9].split('=')[0]]
        aq1 = Conf(robot.body, robot.get_arm_joints(a.arm), aq1)
        funk = get_pull_door_handle_motion_gen(problem, collisions=False, verbose=verbose)
        # set_renderer(False)
        with LockRenderer(True):
            funk(a.arm, a.object, pstn1, pstn2, a.grasp, bq1, aq1)
        # print(LINK_POSE_TO_JOINT_POSITION)
        set_renderer(True)
    return a


def apply_actions(problem, actions, time_step=0.5, verbose=False, plan=None, body_map=None):
    """ act out the whole plan and event in the world without observation/replanning """
    if actions is None:
        return
    state_event = State(problem.world)
    for i, action in enumerate(actions):
        if verbose:
            print(i, action)
        if 'tachObjectAction' in str(action) and body_map is not None:
            if action.object in body_map:
                action.object = body_map[action.object]
            action.verbose = verbose
        action = adapt_action(action, problem, plan, verbose=verbose)
        if action is None:
            continue
        if isinstance(action, Command):
            print('\n\n\napply_actions found Command', action)
            import sys
            sys.exit()
        elif isinstance(action, Action):
            state_event = action.transition(state_event.copy())
        elif isinstance(action, list):
            for a in action:
                state_event = a.transition(state_event.copy())
        
        if time_step is None:
            wait_if_gui()
        else:
            wait_for_duration(time_step)


def get_primitive_actions(action, world, teleport=False):
    def get_traj(t, sub=4, viz=True):
        world.remove_handles()

        ## get the confs
        [t] = t.commands
        if teleport:
            t = Trajectory([t.path[0]] + [t.path[-1]])

        ## subsample
        path = t.path
        if len(path) > 10:
            path = list(t.path[::sub])
            if t.path[-1] not in path:
                path.append(t.path[-1])
        confs = [conf.values for conf in path]

        ## make the actions and draw the poses
        DoF = len(t.path[0].values)
        if DoF == 3:
            t = [MoveBaseAction(conf) for conf in path]
            if viz:
                world.add_handles(draw_pose2d_path(confs, length=0.05))

        elif DoF == 4:
            t = [MoveBaseAction(conf) for conf in path]
            if viz:
                world.add_handles(draw_pose3d_path(confs, length=0.05))

        elif DoF == 6:
            t = [MoveArmAction(conf) for conf in path]
            if viz:
                world.add_handles(draw_pose3d_path(confs, length=0.05))

        elif DoF == 7:
            t = [MoveArmAction(conf) for conf in path]

        else:
            print('\n\nactions.get_primitive_actions | whats this traj', t)

        return t

    name, args = action
    if 'pull_door_handle' in name or 'pull_articulated_handle' in name:
        if '_attachment' in name:
            o3, p3, p4 = args[-3:]
            args = args[:-3]

        ## FEG
        if len(args) == 8:
            a, o, p1, p2, g, q1, q2, t = args
            new_commands = get_traj(t)

        ## PR2
        else:
            a, o, p1, p2, g, q1, q2, bt, aq1, aq2, at = args[:11]
            new_commands = get_traj(bt)  ## list(get_traj(bt).path)

        ## for controlled event
        events = world.get_events(o)
        if events is not None:
            new_commands += world.get_events(o)

    elif name == 'grasp_pull_handle':
        a, o, p1, p2, g, q1, q2, t1, t2, t3 = args

        ## step 1: grasp handle
        t = get_traj(t1, viz=False)
        close_gripper = GripperAction(a, position=g.grasp_width, teleport=teleport)
        attach = AttachObjectAction(a, g, o)
        new_commands = t + [close_gripper, attach]

        ## step 2: move the handle (door, drawer, knob)
        t = get_traj(t2, viz=False)
        new_commands += t

        ## step 3: ungrasp the handle
        t = get_traj(t3, viz=False)
        open_gripper = GripperAction(a, extent=1, teleport=teleport)
        detach = DetachObjectAction(a, o)
        new_commands += [detach, open_gripper] + t[::-1]

        ## draw the whole traj
        path = t1.commands[0].path + t2.commands[0].path + t3.commands[0].path
        t = Trajectory(path)
        from pybullet_tools.pr2_primitives import State
        cmd = Commands(State(), savers=[], commands=[t])
        get_traj(cmd)

    elif 'move_base' in name or 'pull_' in name:
        if 'move_base' in name:
            q1, q2, t = args[:3]
        elif 'pull_marker' in name:
            a, o, p1, p2, g, q1, q2, o2, p3, p4, t = args
        elif 'pull_drawer_handle' in name:
            a, o, p1, p2, g, q1, q2, t = args
        else:
            print('\n\n actions.get_primitive_actions, not implemented', name)

        new_commands = get_traj(t)

    elif 'move_cartesian' in name:
        q1, q2, t = args[:3]
        t = get_traj(t)
        new_commands = t

    # elif name == 'turn_knob':
    #     a, o, p1, p2, g, q, aq1, aq2, t = args
    #     t = get_traj(t)
    #     new_commands = t + world.get_events(o)

    ## ------------------------------------
    ##    variates of pick
    ## ------------------------------------

    elif 'pick' in name:
        if '_rel' in name:
            a, o, p, rp, o2, p2, g, _, t = args
        else:
            a, o, p, g = args[:4]
            t = args[-1]
        t = get_traj(t)
        close_gripper = GripperAction(a, position=g.grasp_width, teleport=teleport)
        attach = AttachObjectAction(a, g, o)
        new_commands = t + [close_gripper, attach] + t[::-1]

    elif name == 'grasp_handle':
        a, o, p, g, q, aq1, aq2, t = args
        t = get_traj(t)
        close_gripper = GripperAction(a, position=g.grasp_width, teleport=teleport)
        attach = AttachObjectAction(a, g, o)
        new_commands = t + [close_gripper, attach]

    elif name == 'grasp_marker':
        a, o1, o2, p, g, q, t = args
        t = get_traj(t)
        close_gripper = GripperAction(a, position=g.grasp_width, teleport=teleport)
        attach = AttachObjectAction(a, g, o1)
        attach2 = AttachObjectAction(a, g, o2)
        new_commands = t + [close_gripper, attach, attach2]

    elif name == 'pick_hand':
        a, o, p, g, _, t = args[:6]
        t = get_traj(t)
        close_gripper = GripperAction(a, position=g.grasp_width, teleport=teleport)
        attach = AttachObjectAction(a, g, o)
        new_commands = t + [close_gripper, attach] + t[::-1]

    elif name == 'grasp_handle_hand':
        """ DEPRECATED after merging grasp-pull-ungrasp """
        a, o, p, g, _, t = args[:6]
        t = get_traj(t)
        close_gripper = GripperAction(a, position=g.grasp_width, teleport=teleport)
        attach = AttachObjectAction(a, g, o)
        new_commands = t + [close_gripper, attach]

    ## ------------------------------------
    ##    variates of pick
    ## ------------------------------------

    elif name == 'place':
        a, o, p, g, _, t = args[:6]
        t = get_traj(t)
        open_gripper = GripperAction(a, extent=1, teleport=teleport)
        detach = DetachObjectAction(a, o)
        new_commands = t + [detach, open_gripper] + t[::-1]

    elif name == 'ungrasp_handle':
        a, o, p, g, q, aq1, aq2, t = args
        t = get_traj(t)
        detach = DetachObjectAction(a, o)
        open_gripper = GripperAction(a, extent=1, teleport=teleport)
        new_commands = [detach, open_gripper] + t[::-1]

    elif name == 'ungrasp_marker':
        a, o, o2, p, g, q, t = args
        t = get_traj(t)
        detach = DetachObjectAction(a, o)
        detach2 = DetachObjectAction(a, o2)
        open_gripper = GripperAction(a, extent=1, teleport=teleport)
        new_commands = [detach, detach2, open_gripper] + t[::-1]

    elif name == 'place_hand':
        a, o, p, g, _, t = args[:6]
        t = get_traj(t)
        open_gripper = GripperAction(a, extent=1, teleport=teleport)
        detach = DetachObjectAction(a, o)
        new_commands = t + [detach, open_gripper] + t[::-1]

    elif name == 'ungrasp_handle_hand':
        """ DEPRECATED after merging grasp-pull-ungrasp """
        a, o, p, g, _, t = args[:6]
        t = get_traj(t)
        open_gripper = GripperAction(a, extent=1, teleport=teleport)
        detach = DetachObjectAction(a, o)
        new_commands = [detach, open_gripper] + t[::-1]

    ## ------------------------------------
    ##    symbolic high-level actions
    ## ------------------------------------

    elif 'clean' in name:  # TODO: add text or change color?
        body, sink = args[:2]
        new_commands = [JustClean(body)]

    elif 'cook' in name:
        body, stove = args[:2]
        new_commands = [JustCook(body)]

    elif name == 'season':
        body, counter, seasoning = args
        new_commands = [JustSeason(body)]

    elif name == 'serve':
        body, serve, plate = args
        new_commands = [JustServe(body)]

    elif name == 'magic':  ## for testing that UnsafeBTraj takes changes in pose into account
        marker, cart, p1, p2 = args
        new_commands = [MagicDisappear(marker), MagicDisappear(cart)]

    elif name == 'teleport':
        body, p1, p2, w1, w2 = args
        w1.printout()
        w2.printout()
        new_commands = [TeleportObject(body, p2)]

    elif name == 'declare_victory':
        new_commands = [JustSucceed()]

    elif name == 'toggle':
        o, pstn1, pstn2 = args
        new_commands = [ChangePositions(pstn2)]

    elif name in ['declare_store_in_space', 'declare_store_on_surface']:
        new_commands = []

    elif name.startswith('_'):
        new_commands = []

    else:
        print('\n\n havent implement commands for', name)
        raise NotImplementedError(name)

    return new_commands


def repair_skeleton(skeleton):
    from pddlstream.algorithms.constraints import WILD
    operators = {
        'pick_hand': 6,
        'place_hand': 6,
        'grasp_pull_handle': 10,

        'pick': 6,
        'place': 6,
        'declare_store_in_space': 2,
        'grasp_handle': 8,
        'ungrasp_handle': 8,
        'pull_door_handle': 11,
    }
    new_skeleton = []
    for a in skeleton:
        name = a[0]
        args = a[1:]
        if name in operators:
            args = list(args) + [WILD] * (operators[name] - len(args))
            new_skeleton.append(tuple([name, args]))
    return new_skeleton
