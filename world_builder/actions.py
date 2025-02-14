import numpy as np
import pybullet as p
import copy
from pprint import pprint

from pybullet_tools.bullet_utils import clip_delta, multiply2d, nice, open_joint, \
    toggle_joint, query_right_left, collided, equal
from pybullet_tools.pose_utils import is_above, remove_attachment, draw_pose2d_path, add_attachment_in_world, \
    add_attachment, draw_pose3d_path
from pybullet_tools.camera_utils import get_obj_keys_for_segmentation, get_segmask, set_camera_target_robot
from pybullet_tools.pr2_streams import Position
from pybullet_tools.mobile_streams import get_pull_door_handle_motion_gen
from pybullet_tools.utils import str_from_object, get_closest_points, INF, create_attachment, wait_if_gui, \
    get_aabb, get_joint_position, get_joint_name, get_link_pose, link_from_name, PI, Pose, Euler, \
    get_extend_fn, get_joint_positions, set_joint_positions, get_max_limit, get_pose, set_pose, set_color, \
    remove_body, create_cylinder, set_all_static, wait_for_duration, remove_handles, set_renderer, \
    LockRenderer, wait_unlocked, ConfSaver, set_joint_position, get_body_name, joint_controller_hold
from pybullet_tools.pr2_utils import PR2_TOOL_FRAMES, get_gripper_joints
from pybullet_tools.pr2_primitives import Trajectory, Command, Conf, Trajectory, Commands
from pybullet_tools.flying_gripper_utils import set_se3_conf, get_pull_handle_motion_gen
from pybullet_tools.logging_utils import print_debug

from world_builder.world import World, State

from robot_builder.robot_utils import close_until_collision

from lisdf_tools.image_utils import RAINBOW_COLORS, save_seg_mask

PULL_UNTIL = 1.8
NUDGE_UNTIL = 2.3

# pull_actions = ['grasp_handle', 'pull_handle', 'ungrasp_handle']
# pull_with_link_actions = ['grasp_handle', 'pull_handle_with_link', 'ungrasp_handle']
pull_actions = ['grasp_pull_ungrasp_handle']
pull_with_link_actions = ['grasp_pull_ungrasp_handle_with_link']
nudge_actions = ['nudge_door']
pick_place_actions = ['pick', 'place']
pick_arrange_actions = ['pick', 'arrange']
pick_sprinkle_actions = ['pick', 'sprinkle']
pick_place_rel_actions = ['pick_from_supporter', 'place_to_supporter']

attach_joint_actions = ['pull_door_handle', 'pull_handle', 'pull_handle_with_link', 'nudge_door',
                        'grasp_pull_ungrasp_handle', 'grasp_pull_ungrasp_handle_with_link',
                        'grasp_pull_handle']


def get_action_name(action):
    return action.__class__.__name__


class Action(object):  # TODO: command
    def transition(self, state):
        raise NotImplementedError()

    def __repr__(self):
        return '{}{}'.format(get_action_name(self), str_from_object(self.__dict__))


class RobotAction(object):
    def __init__(self, robot):
        self.robot = robot


#######################################################


class TeleportAction(Action):
    def __init__(self, conf):
        self.conf = conf
        print('TeleportAction', nice(conf))

    def transition(self, state):
        joints = state.robot.get_joints()  ## all joints, not just x,y,yqw
        if len(self.conf) == len(joints):
            state.robot.set_positions(self.conf, joints=joints)
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
        set_joint_positions(self.conf.body, self.conf.joints, self.conf.values)
        ## only during cog_run
        if hasattr(state, 'movable') and state.movable is not None and get_action_name(self) == 'MoveArmAction':
            carrying = [a.child for a in state.attachments.values() if a.parent == state.robot]
            carrying = [o.body if hasattr(o, 'body') else o for o in carrying]
            movables = [o for o in state.movable if o not in [state.robot]+carrying]

            ## TODO: make collision checking stricter
            if len(movables) > 0:
                if collided(state.robot, movables, verbose=True, log_collisions=False):
                    line = f'\t[MoveArmAction] Robot collided with objects {movables}'
                    print(line)
                    # wait_if_gui(line)
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


class SetJointPositionAction(Action):
    def __init__(self, body, position):
        self.body = body
        self.position = position.value

    def transition(self, state):
        body, joint = self.body
        set_joint_position(body, joint, self.position)
        return state.new_state()


class GripperAction(Action):
    def __init__(self, arm, position=None, extent=None, teleport=False, verbose=False):
        self.arm = arm
        self.position = position
        self.extent = extent  ## 1 means fully open, 0 means fully closed
        self.teleport = teleport
        self.verbose = verbose

    def get_gripper_path(self, state):
        robot = state.robot
        joints = robot.get_gripper_joints(self.arm)

        start_conf = get_joint_positions(robot, joints)

        ## get width from extent
        if self.extent is not None:
            self.position = robot.get_gripper_position_at_extent(self.arm, self.extent)

        else:  ## if self.position == None:
            bodies = [b for b in state.objects if isinstance(b, int) and b != robot.body]
            joints = robot.get_gripper_joints(self.arm)
            with ConfSaver(robot.body):
                self.position = robot.close_until_collision(self.arm, joints, bodies=bodies)
            if hasattr(self, 'verbose') and self.verbose:
                print(f"   [GripperAction] !!!! gripper {self.arm} is closed to {round(self.position, 3)} until collision")
            # self.position = 0.5  ## cabbage, artichoke
            # self.position = 0.4  ## tomato
            # self.position = 0.2  ## zucchini
            # self.position = 0.14  ## bottle

        end_conf = robot.get_gripper_end_conf(self.arm, self.position)
        if self.teleport:
            path = [start_conf, end_conf]
        else:
            extend_fn = get_extend_fn(robot, joints)
            path = [start_conf] + list(extend_fn(start_conf, end_conf))
        return path

    def transition(self, state):
        robot = state.robot
        joints = robot.get_gripper_joints(self.arm)

        path = self.get_gripper_path(state)
        for positions in path:
            set_joint_positions(robot, joints, positions)

        return state.new_state()


class RevisedAction(Action):
    """ previous version of some actions used 'self.object' instead of self.body """

    def get_body(self):
        return self.body if hasattr(self, 'body') else self.object

    def set_body(self, body):
        """ previous version of this class used 'object' """
        if hasattr(self, 'body'):
            self.body = body
        else:
            self.object = body


class AttachObjectAction(RevisedAction):
    def __init__(self, arm, grasp, body, verbose=True, debug_rc2oc=False):
        self.arm = arm
        self.grasp = grasp
        self.body = body
        self.verbose = verbose
        self.attach_kwargs = dict(debug_rc2oc=debug_rc2oc, attach_distance=None, verbose=False, OBJ=False)

    def transition(self, state):
        parent = state.robot
        link = state.robot.get_attachment_link(self.arm)
        obj = self.get_body()
        if isinstance(obj, int):
            obj = state.world.get_object(self.get_body())
        added_attachments = add_attachment_in_world(state=state, obj=obj, parent=parent, parent_link=link,
                                                    **self.attach_kwargs)
        new_attachments = dict(state.attachments)
        new_attachments.update(added_attachments)
        return state.new_state(attachments=new_attachments)


class DetachObjectAction(RevisedAction):
    def __init__(self, arm, body, supporter=None, verbose=False):
        if verbose: print(f'DetachObjectAction.__init__({body, supporter})')
        self.arm = arm
        self.body = body
        self.supporter = supporter
        self.verbose = verbose

    def transition(self, state):
        body = self.get_body()
        verbose = self.verbose if hasattr(self, 'verbose') else True
        if verbose:
            print(f'DetachObjectAction({body}, supporter={self.supporter})')
        updated_attachments, removed_attachment = remove_attachment(state, obj=body, verbose=verbose)
        new_attachments = {}

        ## knowing where the object is left on/in
        if hasattr(self, 'supporter') and self.supporter is not None:
            # import ipdb; ipdb.set_trace()
            obj = body
            parent = self.supporter
            parent_link = -1
            if isinstance(self.supporter, tuple):
                parent_link = self.supporter[-1]
            if isinstance(state.world, World):  ## hasattr(state.world, 'BODY_TO_OBJECT'):
                if parent not in state.world.BODY_TO_OBJECT:
                    parent = None
                else:
                    parent = state.world.BODY_TO_OBJECT[parent]
                obj = state.world.BODY_TO_OBJECT[body]
            elif isinstance(parent, tuple):
                parent, _, parent_link = parent

            if parent_link is not None:
                new_attachments = add_attachment_in_world(state=state, obj=obj, parent=parent, OBJ=True,
                                                          parent_link=parent_link, verbose=self.verbose)
                updated_attachments.update(new_attachments)

        if isinstance(state.world, World):
            update_world_with_attachment_changes(state.world, updated_attachments, new_attachments, removed_attachment)
        return state.new_state(attachments=updated_attachments)


def update_world_with_attachment_changes(world, updated_attachments, new_attachments, removed_attachment):
    """ change world.attachments and world.ignore_pairs
        if attached to a movable objects, update in ignore_pairs
    """
    title = '[actions.update_world_with_attachment_changes]\t'
    world.attachments = updated_attachments

    def parent_is_movable(a):
        obj = a.parent
        if obj.link is not None:
            obj = world.body_to_object(obj.body)
        return 'movable' in obj.categories

    def get_name(pair):
        return (world.get_debug_name(pair[0]), world.get_debug_name(pair[1]))

    pairs_to_add = [(a.child.body, a.parent.body) for a in new_attachments.values() if parent_is_movable(a)]
    pairs_to_del = [(a.child.body, a.parent.body) for a in [removed_attachment] if parent_is_movable(a)]
    for pair in pairs_to_add:
        world.add_ignored_pair(pair)
        print_debug(f'{title} adding {get_name(pair)} to world.ignored_pairs', 'pink')
    for pair in pairs_to_del:
        world.del_ignored_pair(pair)
        print_debug(f'{title} removing {get_name(pair)} from world.ignored_pairs', 'pink')


class JustDoAction(Action):
    def __init__(self, body):
        self.body = body

    def transition(self, state):
        label = get_action_name(self).lower().replace('just', '').capitalize() + 'ed'
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


## ------------------------------------------------------------


class ChangeLinkColorEvent(Action):
    def __init__(self, body, color, link=None):
        self.body = body
        self.color = color
        self.link = link

    def transition(self, state):
        # import ipdb; ipdb.set_trace()
        set_color(self.body, self.color, self.link)
        attached_objs = [v.child for k, v in state.attachments.items() if k.body == self.body and k.link == self.link]
        if len(attached_objs) > 0:
            for o in attached_objs:
                print(f'[actions.ChangeLinkColorEvent]\tcoloring object {o.debug_name} as it is placed on {(self.body, self.link)}')
                set_color(o.body, self.color, o.link)
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
        if self.body is None:
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


##################################################


class GripperCommand(Command):
    def __init__(self, robot, arm, position, teleport=False):
        self.robot = robot
        self.arm = arm
        self.position = position
        self.teleport = teleport
    def apply(self, state, **kwargs):
        joints = self.robot.get_gripper_joints(self.arm)
        start_conf = get_joint_positions(self.robot, joints)
        end_conf = [self.position] * len(joints)
        if self.teleport:
            path = [start_conf, end_conf]
        else:
            extend_fn = get_extend_fn(self.robot, joints)
            path = [start_conf] + list(extend_fn(start_conf, end_conf))
        for positions in path:
            set_joint_positions(self.robot, joints, positions)
            yield positions
    def control(self, **kwargs):
        joints = self.robot.get_gripper_joints(self.arm)
        positions = [self.position]*len(joints)
        for _ in joint_controller_hold(self.robot, joints, positions):
            yield
    def __repr__(self):
        return '{}({},{},{})'.format(self.__class__.__name__, get_body_name(self.robot),
                                     self.arm, self.position)


#######################################################


def adapt_attach_action(a, problem, plan, verbose=True):

    continuous = {}
    if len(plan) == 2:
        plan, continuous = plan
    # if len(plan) == 1 and len(plan[0]) > 1:
    #     plan = plan[0]

    def get_value(string):
        name, tup = string.split('=')
        # value = continuous[name]
        # if name.startswith('pstn'):
        #     value = value[0]
        # return value
        return eval(tup)

    body = a.get_body()
    robot = problem.world.robot
    body_to_name = problem.world.body_to_name
    if hasattr(problem.world, 'BODY_TO_OBJECT'):
        body_to_name = {eval(k): v for k, v in body_to_name.items()}

    if ' ' in plan[0][0]:
        act = [aa for aa in plan if aa[0].startswith('pull') and aa[2] == body_to_name[body]][0]
    else:
        ## LABEL 1 - this condition limits the search to start pull with the current joint position,
        # but when generating segmented replay videos, the world is still in initial state
        pstn = get_joint_position(body[0], body[1])

        act = [aa for aa in plan if aa[0] in attach_joint_actions and \
               aa[2] in [str(body), body_to_name[body]] and \
               equal(continuous[aa[3].split('=')[0]][0], pstn)]
        title = f'adapt_attach_action({str(body)}|{body_to_name[body]},pstn={pstn})'
        if len(act) == 0:
            print(f'{title} not found in len = {len(plan)}')
            return
        elif len(act) > 1:
            print(f"{title} found {len(act)} matches\n\t"+'\n\t'.join([str(a) for a in act]))

        act = act[0]

    pstn1 = Position(body, get_value(act[3]))
    pstn2 = Position(body, get_value(act[4]))
    bq1 = get_value(act[6])  ## continuous[act[6].split('=')[0]]
    bq1 = Conf(robot.body, robot.get_base_joints(), bq1)
    if 'feg' in robot.name:
        funk = get_pull_handle_motion_gen(problem, collisions=False, verbose=verbose)
        aq1 = None
    else:
        var = act[8] if act[0] == 'nudge_door' else act[-1]  ## pull_handle
        aq1 = get_value(var)  ## continuous[act[9].split('=')[0]]
        aq1 = Conf(robot.body, robot.get_arm_joints(a.arm), aq1)
        funk = get_pull_door_handle_motion_gen(problem, collisions=False, verbose=verbose)

    # set_renderer(False)
    with LockRenderer(True):
        funk(a.arm, body, pstn1, pstn2, a.grasp, bq1, aq1)
    set_renderer(True)

    ## REF 1 - this condition limits the search to start pull with the current joint position,
    # but when generating segmented replay videos, the world is still in initial state
    pstn2.assign()


def adapt_action(a, problem, plan, verbose=True):
    if plan is None:
        return a

    ## find the joint positions to animate
    if get_action_name(a) == 'AttachObjectAction' and isinstance(a.get_body(), tuple):
        with ConfSaver(problem.robot):
            adapt_attach_action(a, problem, plan, verbose=verbose)

    return a


def apply_commands(problem, actions, time_step=0.5, verbose=True, plan=None, body_map=None,
                   save_composed_jpg=False, save_gif=False, save_observation_mp4=False, check_collisions=False,
                   action_by_action=False, cfree_range=0.1, visualize_collisions=False):
    """ act out the whole plan and event in the world without observation/replanning """
    if actions is None:
        return
    world = problem.world
    state_event = State(world)
    episodes = []
    camera_names = ['scene']
    robot_cameras = []
    if save_observation_mp4:
        robot_cameras = world.robot_cameras
        camera_names += list(robot_cameras.keys())
    seg_images = {k: [] for k in camera_names}
    recording = False
    last_name = None
    last_object = None

    if not check_collisions and (save_composed_jpg or save_gif):
        colors = RAINBOW_COLORS
        color_index = 0
        indices = world.get_indices(body_map=body_map, larger=False)
        if save_composed_jpg:
            imgs = world.camera.get_image(segment=True, segment_links=True)
            seg = imgs.segmentationMaskBuffer
            unique = get_segmask(seg)
            obj_keys = get_obj_keys_for_segmentation(indices, unique)
        elif save_gif:
            obj_keys = get_obj_keys_for_segmentation(indices)

    robot = world.robot.body
    objects = world.get_collision_objects()
    ignored_collisions = {robot: []}
    initial_collisions = copy.deepcopy(ignored_collisions)
    expected_pose = None
    cfree_until = None
    # executed_action = []
    # last_trajectory = []
    # current_trajectory = []
    # skipped_last = False
    i = 0
    while i < len(actions):
        action = actions[i]
        name = get_action_name(action)

        # ## TODO: fix this bug in saving commands ------------------------------
        # if name == 'GripperAction':
        #     last_trajectory = [str(a) for a in current_trajectory]
        #     current_trajectory = []
        #
        # if str(action) in executed_action and str(action) not in last_trajectory:
        #     if not name.startswith('Move') and not skipped_last:
        #         pass
        #     else:
        #         print('Skipping already executed action:', action)
        #         skipped_last = True
        #         continue
        # else:
        #     executed_action.append(str(action))
        # skipped_last = False
        # current_trajectory.append(action)
        # ## ---------------------------------------------------------------------

        line = f"{i}/{len(actions)}\t{action}"
        if action_by_action and name == 'GripperAction':
            wait_if_gui(f'Execute {line}?')
        elif verbose:
            print(line)

        if 'GripperAction' in name and check_collisions:
            next_action = actions[i+1]
            next_name = get_action_name(next_action)
            if 'AttachObjectAction' in next_name:
                last_object = next_action.body
                # if isinstance(last_object, str):
                #     last_object = eval(last_object)
                if last_object in body_map:
                    last_object = body_map[last_object]
                if isinstance(last_object, tuple):
                    last_object = last_object[0]
                object_name = world.body_to_name[last_object]
                for a in plan[0]:
                    if 'place' in a[0] and a[2] == object_name:
                        pp = a[3]
                        expected_pose = eval(pp[pp.index('('):])[:3]
                    if 'arrange' in a[0] and a[2] == object_name:
                        pp = a[4]
                        expected_pose = eval(pp[pp.index('('):])[:3]
                ignored_collisions[robot].append(last_object)
                ignored_collisions[last_object] = [robot, last_object]
                cfree_until = i + 6
            else:
                ignored_collisions = copy.deepcopy(initial_collisions)
                expected_pose = None
                cfree_until = i + 6

        record_img = False
        if 'tachObjectAction' in name and body_map is not None:
            body = action.get_body()
            if body in body_map:
                action.set_body(body_map[body])
            last_object = body
            action.verbose = verbose
            if not check_collisions and (save_composed_jpg or save_gif):
                record_img = True
                if 'Attach' in name:
                    color = colors[color_index]
                    color_index += 1

                    body_name = world.body_to_name[body]
                    for b, l in obj_keys[body_name]:
                        set_color(b, color, link=l)
                    recording = body_name

                elif 'Detach' in name:
                    recording = False
                    if save_composed_jpg:
                        episodes.append((seg_images['scene'], isinstance(last_object, int)))
                        seg_images['scene'] = []
        elif 'MoveArm' in name:
            record_img = (i % 5 == 0)
            if i + 1 < len(actions):
                next_action = actions[i + 1]
                next_name = get_action_name(next_action)
                if 'MoveArm' not in next_name:
                    record_img = True
            if 'feg' in problem.robot.name:
                record_img = True
        elif 'MoveBase' in name:
            record_img = i % 2 == 0

        if verbose and name != last_name:
            print(f"{i} {action}")
            last_name = name
            record_img = True

        ###############################################

        ## adapt before using segmented plans for visualization, TODO: check if works
        # action = adapt_action(action, problem, plan, verbose=False)
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

        ###############################################

        if check_collisions:

            ## gripper colliding with object just placed down
            ## or braiser colliding with lid just picked away
            if cfree_until is None or i >= cfree_until:

                for body, ignored in ignored_collisions.items():
                    obstacles = [o for o in objects if o not in ignored]

                    ## object colliding with surface just placed on
                    if body != robot:
                        if expected_pose is not None:
                            pose = get_pose(body)[0]
                            dist = np.linalg.norm(np.array(pose) - np.array(expected_pose))
                            if dist < cfree_range:
                                continue

                    result = collided(body, obstacles, world=world, verbose=True, min_num_pts=3, log_collisions=False)
                    if result:
                        if visualize_collisions:
                            wait_if_gui()
                        return result

        ###############################################

        if recording:
            if not record_img:
                i += 1
                continue

            if save_composed_jpg:
                imgs = world.camera.get_image(segment=True, segment_links=True)
                body_name = recording
                seg_images['scene'].append(save_seg_mask(imgs, obj_keys[body_name], verbose=False))

            elif save_gif:
                if save_observation_mp4:
                    for camera_name, camera in robot_cameras.items():
                        imgs = camera.get_image()
                        seg_images[camera_name].append(imgs.rgbPixels[:, :, :3])
                imgs = world.camera.get_image()
                seg_images['scene'].append(imgs.rgbPixels[:, :, :3])

        ############# keyboard control (if ran in terminal) ##################

        elif time_step is None:
            result = query_right_left()
            if not result:
                if i > 1:
                    i -= 2
                else:
                    i -= 1

        ############# automatically play forward ##################

        else:
            wait_for_duration(time_step)

        i += 1

    if check_collisions:
        return False
    if save_composed_jpg:
        if recording:
            episodes.append((seg_images['scene'], isinstance(last_object, int)))
        return episodes
    if save_gif:
        return seg_images

    return state_event.attachments


def get_primitive_actions(action, world, teleport=False, simulate=False, verbose=True, debug_rc2oc=False):
    """
    usually used to convert action generated by planner to commands for execution and record
    may also be used for segmenting commands given a plan for replay & visualization
    """
    def get_traj(t, sub=4, viz=True):
        if simulate:
            return [t.commands[0]]  ## Commands class has function .control()

        if isinstance(t, str):
            if '=t(4' in t:
                return [MoveBaseAction(None)]
            if '=t(7' in t:
                return [MoveArmAction(None)]
            return Trajectory([])
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

    o = None
    name, args = action
    if name.startswith('pull_handle') or 'pull_articulated_handle' in name:

        ## FEG
        if len(args) == 8:
            a, o, p1, p2, g, q1, q2, t = args
            new_commands = get_traj(t)

        ## PR2
        else:
            a, o, p1, p2, g, q1, q2, t = args[:8]

            if debug_rc2oc:
                rc2oc = world.robot.ROBOT_CONF_TO_OBJECT_CONF
                body, joint = o
                if body in rc2oc:
                    if joint in rc2oc[body]:
                        confs = world.robot.get_rc2oc_confs()
                        for conf in confs:
                            if conf in rc2oc[body][joint]:
                                path = t.commands[0].path
                                DOF = len(path[0].values)
                                group = {3: 'base-torse', 4: 'base-torse', 7: f'{a}_arm'}[DOF]
                                if group in rc2oc[body][joint][conf]:
                                    x = range(len(path))
                                    path = [conf.values for conf in path]
                                    saved = [(conf, round(pstn, 4)) for conf, pstn in rc2oc[body][joint][conf][group].items()][1:]
                                    diff = [round(np.linalg.norm(np.asarray(path[i]) - np.asarray(saved[i][0])), 4) for i in x]
                                    path = [[round(n, 4) for n in conf] for conf in path]
                                    lines = [f"{i}\t{path[i]}\t{saved[i][0]}\t{diff[i]}\t{saved[i][1]}" for i in x]
                                    print(f'[actions.get_primitive_actions] : trajectory | rc2oc | diff | pstn{o}')
                                    print('\t'+'\n\t'.join(lines))

            new_commands = get_traj(t)

    elif name == 'grasp_pull_handle':
        """ grasp, pull, ungrasp together (FEG) """
        a, o, p1, p2, g, q1, q2, t1, t2, t3 = args

        ## step 1: grasp handle
        t = get_traj(t1, viz=False)
        close_gripper = GripperAction(a, teleport=teleport)  ## position=g.grasp_width
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

    elif name.startswith('grasp_pull_ungrasp_handle'):
        """ grasp, pull, ungrasp together (Mobile) """
        a, o, p1, p2, g, q1, q2, bt, aq1, aq2, at = args[:11]

        ## step 1: grasp handle
        at = get_traj(at)
        close_gripper = GripperAction(a, teleport=teleport)  ## , position=g.grasp_width
        attach = AttachObjectAction(a, g, o, debug_rc2oc=debug_rc2oc)
        new_commands = at + [close_gripper, attach]

        ## step 2: move the handle (door, drawer, knob)
        bt = get_traj(bt)
        new_commands += bt

        ## step 3: ungrasp the handle
        open_gripper = GripperAction(a, extent=1, teleport=teleport)
        detach = DetachObjectAction(a, o)
        new_commands += [detach, open_gripper] + at[::-1]

        # ## for controlled event
        # events = world.get_events(o)
        # if events is not None:
        #     new_commands += world.get_events(o)

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

    elif name in ['pick', 'pick_half', 'pick_from_supporter']:
        if 'from_supporter' in name:
            a, o, rp, o2, p2, g = args[:6]
        else:
            a, o, p, g = args[:4]
        t = get_traj(args[-1])
        close_gripper = GripperAction(a, teleport=teleport)  ## , position=g.grasp_width
        attach = AttachObjectAction(a, g, o, verbose=verbose)
        new_commands = t + [close_gripper, attach]
        if name in ['pick', 'pick_from_supporter']:
            new_commands += t[::-1]

    elif name == 'grasp_handle':
        a, o, p, g, q, aq1, aq2, t = args
        t = get_traj(t)
        close_gripper = GripperAction(a, teleport=teleport)  ## , position=g.grasp_width
        attach = AttachObjectAction(a, g, o, debug_rc2oc=debug_rc2oc)
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
    ##    variates of place
    ## ------------------------------------

    elif name in ['place', 'place_half', 'place_to_supporter', 'arrange']:
        if 'to_supporter' in name:
            a, o, rp, o2, p2, g = args[:6]
        else:
            if name == 'arrange':
                a, o, r, p = args[:4]
            else:
                a, o, p = args[:3]
            o2 = None
            if hasattr(p, 'support'):
                o2 = p.support
        t = get_traj(args[-1])
        open_gripper = GripperAction(a, extent=1, teleport=teleport)
        detach = DetachObjectAction(a, o, supporter=o2, verbose=verbose)
        new_commands = [detach, open_gripper] + t[::-1]
        if name != 'place_half':
            new_commands = t + new_commands

    elif name == 'nudge_door':
        a, o, p1, p2, g = args[:5]
        at, bt = args[-2:]

        at = get_traj(at)
        close_gripper = GripperAction(a, extent=0, teleport=teleport)
        attach = AttachObjectAction(a, g, o)
        bt = get_traj(bt)
        pstn = SetJointPositionAction(o, p2)  ## just in case attachment failed
        detach = DetachObjectAction(a, o)
        open_gripper = GripperAction(a, extent=1, teleport=teleport)

        new_commands = at + [close_gripper, attach] + bt + [detach, open_gripper, pstn] + at[::-1]

    elif name == 'ungrasp_handle':
        a, o, p, g, q, aq1, aq2, t = args
        t = get_traj(t)
        detach = DetachObjectAction(a, o)
        open_gripper = GripperAction(a, extent=1, teleport=teleport)
        pstn = SetJointPositionAction(o, p)  ## just in case attachment failed
        new_commands = [detach, open_gripper, pstn] + t[::-1]

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

    elif name == 'sprinkle':
        a, o1, p1, o2, p2, g, q, t = args
        t = get_traj(t)
        new_commands = t + t[::-1]

    ## ------------------------------------
    ##    symbolic high-level actions
    ## ------------------------------------

    elif 'clean' in name:
        body = args[1] if name == 'just-clean' else args[0]
        new_commands = [JustClean(body)]

    elif 'cook' in name:
        body = args[1] if name == 'just-cook' else args[0]
        new_commands = [JustCook(body)]

    elif 'serve' in name:
        body = args[1] if name == 'just-serve' else args[0]
        new_commands = [JustServe(body)]

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

    if simulate:
        from world_builder.world import State
        filtered_commands = []
        for action in new_commands:
            name = action.__class__.__name__
            ## change GripperAction to GripperCommand
            if isinstance(action, GripperAction):
                action.get_gripper_path(State(world))
                command = GripperCommand(world.robot, action.arm, action.position, teleport=teleport)
                filtered_commands.append(command)
            elif 'ttach' not in name:
                filtered_commands.append(action)
        return filtered_commands

    if name.endswith('handle') and o is not None:
        ## for controlled event
        events = world.get_events(o)
        if events is not None:
            new_commands += world.get_events(o)

    return new_commands


def repair_skeleton(skeleton, domain_pddl):
    from pddlstream.algorithms.constraints import WILD
    from pddl_domains.pddl_utils import load_num_args_from_domain_pddl
    """ 
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
    """
    operators = load_num_args_from_domain_pddl(domain_pddl)
    new_skeleton = []
    for a in skeleton:
        name = a[0]
        args = a[1:]
        if name in operators:
            args = list(args) + [WILD] * (operators[name] - len(args))
            new_skeleton.append(tuple([name, args]))
        else:
            print(f'[actions.repair_skeleton]\t provided skeleton action {a} is not in domain operators')
    return new_skeleton
