from pybullet_tools.utils import enable_preview
from pybullet_tools.camera_utils import get_pose2d
from cogarch_tools.processes.motion_agent import *
from world_builder.actions import *
from robot_builder.robot_utils import BASE_JOINTS, get_joints_by_names, BASE_GROUP


class TeleOpAgent(MotionAgent):
    def __init__(self, world, goals=[], **kwargs):
        super(MotionAgent, self).__init__(world, **kwargs)
        self.goals = list(goals)
        self.plan_step = None
        self.plan = None
        self.command_options = self.initiate_commands()

        self.robot.joints = get_joints_by_names(self.robot, BASE_JOINTS)
        self.robot.base_group = BASE_GROUP

    def initiate_commands(self):

        ## enable debug panel
        enable_preview()
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, True)

        ## robot initial state
        x, y, yaw = get_pose2d(self.robot)
        torso_lift = self.robot.get_joint_position(self.robot.torso_lift_joint)
        head_pan = self.robot.get_joint_position(self.robot.head_pan_joint)
        head_tilt = self.robot.get_joint_position(self.robot.head_tilt_joint)
        self.last_command = [
            x, y, yaw, 0, 0, torso_lift,
            head_pan, head_tilt,
            0, 0, 0, 0, 0, 0,
            0, 0, 0
        ]

        commands = []

        ## test base motion
        commands.append(add_parameter(name='x', lower=-10, upper=10, initial=x))
        commands.append(add_parameter(name='y', lower=-10, upper=10, initial=y))
        commands.append(add_parameter(name='yaw', lower=0, upper=2*PI, initial=yaw))
        commands.append(add_button(name='turn left'))
        commands.append(add_button(name='turn right'))
        commands.append(add_parameter(name='torso_lift_joint', lower=0, upper=0.31, initial=torso_lift))

        ## test head motion
        commands.append(add_parameter(name='head_pan_joint (- is right)', lower=-PI/2, upper=PI/2, initial=head_pan))
        commands.append(add_parameter(name='head_tilt_joint (- is up)', lower=-PI/2, upper=PI/2, initial=head_tilt))

        ## test discrete action
        commands.append(add_button(name='open/close drawer'))
        commands.append(add_button(name='open/close door'))
        commands.append(add_button(name='pick up (left gripper)'))
        commands.append(add_button(name='pick up (right gripper)'))
        commands.append(add_button(name='put down (left gripper)'))
        commands.append(add_button(name='put down (right gripper)'))

        ## test discrete kitchen actions
        commands.append(add_button(name='turn on/off switch'))  ## water faucet, stove
        commands.append(add_button(name='chop'))  ## veggie into two parts
        commands.append(add_button(name='crack'))  ## egg into yellow box and shell

        return commands

    def policy(self, observation):
        observation.assign()

        last_command = self.last_command
        current_command = [p.readUserDebugParameter(self.command_options[i]) for i in range(len(self.command_options))]
        self.last_command = current_command
        for i in range(len(self.command_options)):
            command_option = p.readUserDebugParameter(self.command_options[i])
            command_option_last = last_command[i]
            if abs(command_option - command_option_last) > 0.01:

                ## move base
                if i in [0, 1, 2, 3, 4]:
                    x, y, yaw = get_pose2d(self.robot)
                    if i == 0:
                        x = command_option
                    elif i == 1:
                        y = command_option
                    elif i == 2:
                        yaw = command_option
                    elif i == 3:
                        yaw += PI/2
                    elif i == 4:
                        yaw -= PI/2
                    pose = Pose(point=Point(x=x, y=y), euler=Euler(yaw=yaw))
                    conf = (x, y, yaw)
                    return TeleportAction(conf)

                ## move joints
                elif i in [5, 6, 7]:
                    joint_name = {
                        5: self.robot.torso_lift_joint,
                        6: self.robot.head_pan_joint,
                        7: self.robot.head_tilt_joint
                    }[i]
                    with PoseSaver(self.robot):
                        # old_conf = self.robot.get_joint_positions()
                        self.robot.set_joint_position(joint_name, command_option)
                        conf = self.robot.get_joint_positions()
                        # j = joint_from_name(self.robot, joint_name)
                        # print(f'{joint_name}: old conf {old_conf[j]}, new conf {conf[j]}')
                    return TeleportAction(conf)

                ## change object states
                elif i in range(8, 17):

                    category = {8: 'drawer', 9: 'door', 10: 'movable', 11: 'movable',
                                12: 'surface', 13: 'surface', 14: 'switch',
                                15: 'movable', 16: 'movable'}[i]
                    objects = self.world.OBJECTS_BY_CATEGORY[category]

                    print(f'\nlooking for {category} in range')

                    if i in [8, 9]:
                        collided = check_collision(self.robot, self.world.objects)
                        affected = [(obj.body, obj.joint) for obj in objects if (obj.body, obj.joint) in collided]
                        print(f'closest {category} is:')
                        return OpenJointAction(affected)

                    elif i in [10, 11, 12, 13]:
                        gripper = {10: 'left', 11: 'right', 12: 'left', 13: 'right'}[i]
                        closest = check_in_view(self.robot, objects=objects)
                        print(f'closest {category} include {closest}')
                        if len(closest) > 0:

                            if i in [10, 11]: ## not already in hand
                                return PickUpAction(closest[0], gripper=gripper)

                            elif i in [12, 13]: ## put down on surface
                                return PutDownAction(closest[0], gripper=gripper)

                        elif i in [12, 13]: ## put down on the floor
                            floors = self.world.OBJECTS_BY_CATEGORY['floor']
                            robot_aabb = get_aabb(self.robot)
                            floors = [f for f in floors if aabb_overlap(get_aabb(f), robot_aabb)]
                            return PutDownAction(floors[0], gripper=gripper)

                    elif i == 14:
                        closest = check_in_view(self.robot, objects=objects)
                        print(f'closest {category} include {closest}')
                        if len(closest) > 0:
                            return OpenJointAction(closest)

                    elif i == 15:
                        closest = check_in_view(self.robot, objects=objects)
                        closest = [c for c in closest if c.category in OBJECT_PARTS]
                        print(f'closest {category} include {closest}')
                        if len(closest) > 0 and self.robot.has_object_in_hand('knife'):
                            return ChopAction(closest[0])

                    return InteractAction()

    def init_experiment(self, args, **kwargs):
        args.monitoring = True


def check_collision(robot, objects):

    robot_aabb = get_aabb(robot)

    ## check for all objects
    found = []
    found_objects = []
    for obj in objects:
        aabb = get_aabb(obj.body, obj.joint)
        if aabb_overlap(aabb, robot_aabb):
            found.append(obj.shorter_name)
            found_objects.append((obj.body, obj.joint))

    ## found collision
    if len(found) > 0:
        line = '\n\nColliding with '
        for obj in found:
            line += f'{obj}, '
        # print(line)

        # offset = [2, 0, 2]
        # ROBOT_TO_OBJECT[robot].draw(text=line, offset=offset)

    return found_objects


def check_in_view(robot, objects=None, max_distance=1.5):

    camera = robot.cameras[0]
    cam = camera.get_pose()[0]
    found = {} ## map object in view to distance
    # colors = [RED, GREEN, YELLOW]
    # index = 0
    for obj in objects:
        # if isinstance(obj, Surface):
        #     set_color(obj.body, colors[index], obj.link)
        #     index += 1
        points = [get_com_pose(obj.body, 0)[0]] ## center of mass
        (x1, y1, _), (x2, y2, z) = get_aabb(obj.body, obj.link)
        points.extend([(x1,y1,z),(x1,y2,z),(x2,y1,z),(x2,y2,z)])  ## upper points in aabb
        points.append(((x1+x2)/2, (y1+y2)/2, z)) ## center point on top surface
        print(f'points in {obj}', nice(tuple(points)))
        for pt in points:
            if camera.point_in_view(pt):
                distance = get_point_distance(pt, cam)
                if obj not in found or distance < found[obj]:
                    found[obj] = get_point_distance(pt, cam)
        # closest = get_closest_points(robot.body, obj.body)
        # print(obj.name, closest)
    found = {k: v for k, v in sorted(found.items(), key=lambda item: item[1]) if v < max_distance}
    objects_in_hand = robot.get_objects_in_hands()
    closest = [c for c in found if c not in objects_in_hand]
    print('found objects in view', closest)
    return closest


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
        new_attachments = update_attachments(state=state, obj=obj, attach_distance=0.1)
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
        new_attachments, removed_attachment = remove_attachment(state, obj)
        return state.new_state(attachments=new_attachments)


OBJECT_PARTS = {
    'Veggie': ['VeggieLeaf', 'VeggieStem'],
    'Egg': ['EggFluid', 'EggShell']
} ## object.category are all lower case
OBJECT_PARTS = {k.lower(): v for k, v in OBJECT_PARTS.items()}


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