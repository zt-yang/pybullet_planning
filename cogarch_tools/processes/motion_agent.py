import numpy as np
import math

import pybullet as p
from world_builder.world import Agent
from world_builder.actions import MoveAction, TeleportAction, TurnAction, DriveAction, MoveInSE3Action
from pybullet_tools.bullet_utils import draw_pose2d, get_pose2d, clip_delta, multiply2d, invert2d, MIN_DISTANCE, \
    nice, get_point_distance
from pybullet_tools.utils import get_sample_fn, get_difference_fn, aabb_overlap, set_pose, BodySaver, \
    pairwise_collisions, get_aabb, plan_joint_motion, randomize, waypoints_from_path, plan_nonholonomic_motion, \
    get_nonholonomic_distance_fn, get_distance_fn, PoseSaver, get_extend_fn, get_nonholonomic_extend_fn, \
    PI, add_parameter, add_button, Pose, Point, Euler, adjust_path, get_com_pose
from world_builder.actions import InteractAction, ChopAction, OpenJointAction, PutDownAction, PickUpAction, \
    OpenJointAction, OBJECT_PARTS


APPROX_DIFFERENTIAL = True


class MotionAgent(Agent):
    # TODO: rename nonholonomic to differential drive
    requires_conf = requires_poses = True

    def __init__(self, world, reversible=False, monitor=False, **kwargs):
        super(MotionAgent, self).__init__(world, **kwargs)
        self.reversible = reversible
        self.monitor = monitor
        self.holonomic = not self.world.drive
        self.teleport = self.world.teleport
        self.sample_fn = get_sample_fn(self.robot, self.robot.joints, custom_limits=self.robot.custom_limits)
        self.difference_fn = get_difference_fn(self.robot, self.robot.joints)
        if APPROX_DIFFERENTIAL or self.holonomic:
            #self.distance_fn = lambda *args: np.reciprocal(self.duration_fn(*args))
            self.distance_fn = get_distance_fn(self.robot, self.robot.joints, weights=self.robot.weights)
            self.extend_fn = get_extend_fn(self.robot, self.robot.joints, resolutions=self.robot.resolutions)
        else:
            linear_tol = 1e-3 # 1e-3 | 1e-6
            angular_tol = math.radians(1e-1) # 1e-3 | 1e-6
            self.distance_fn = get_nonholonomic_distance_fn(self.robot, self.robot.joints, reversible=self.reversible,
                                                            linear_tol=linear_tol, angular_tol=angular_tol,
                                                            weights=self.robot.weights)
            self.extend_fn = get_nonholonomic_extend_fn(self.robot, self.robot.joints, reversible=self.reversible,
                                                        linear_tol=linear_tol, angular_tol=angular_tol,
                                                        resolutions=self.robot.resolutions)
        #self.distance_fn = get_duration_fn(self.robot, self.robot.joints, velocities=self.max_velocities)

        self.path = None
        self.handles = []
        # TODO: memory

    def update(self, observation):
        # TODO: check whether the states have changed significantly since last
        observation.assign()
        return set(observation.obstacles) - {self.robot}

    def sample_goal(self, obstacles):
        with BodySaver(self.robot):
            while True:
                target_conf = self.sample_fn()
                self.robot.set_positions(target_conf)
                if not pairwise_collisions(self.robot, obstacles, max_distance=MIN_DISTANCE):
                    return target_conf

    ## ------ moved to world.set_path(path)
    # def set_path(self, path):
    #     remove_handles(self.handles)
    #     self.path = path
    #     if self.path is None:
    #         return self.path
    #     if isinstance(self.path[0], tuple):  ## pr2
    #         #path = adjust_path(self.robot, self.robot.joints, self.path)
    #         self.handles.extend(draw_pose2d_path(self.path[::4], length=0.05))
    #         #wait_if_gui()
    #     else:  ## SE3 gripper
    #         path = [p.values for p in path]
    #         self.handles.extend(draw_pose3d_path(path, length=0.05))
    #     return self.path

    def sample_path(self, obstacles):
        #start_conf = self.robot.get_positions()
        goal_conf = self.sample_goal(obstacles)
        self.handles = draw_pose2d(goal_conf, z=0.)
        #self.robot.set_positions(start_conf)
        with BodySaver(self.robot):
            plan_motion = plan_joint_motion if self.holonomic else plan_nonholonomic_motion
            # TODO: could have some tolerance for the "two boundary problem"
            kwargs = {} if self.holonomic else {'reversible': self.reversible}
            # TODO: plan with decreasing max_distance
            path = plan_motion(self.robot, self.robot.joints, goal_conf,
                               obstacles=obstacles, self_collisions=False,  #max_distance=MIN_DISTANCE/2.,
                               weights=self.robot.weights, resolutions=self.robot.resolutions,
                               disabled_collisions=self.robot.disabled_collisions, **kwargs)
        return self.set_path(path)

    def check_path(self, path, obstacles):
        if path is None:
            return False
        with BodySaver(self.robot):
            for conf in randomize(path):
                self.robot.set_positions(conf)
                if pairwise_collisions(self.robot, obstacles, max_distance=MIN_DISTANCE / 2.): # Make sure no regions
                    return False
            return True
        # TODO: pickup at the back of bot

    # def follow_arm_path(self, current_arm_conf):
    #


    def follow_path(self, current_conf):
        # TODO: also see SS-Replan
        # TODO: discard negligible displacements
        # TODO: maybe only control for x, y positions and don't worry about orientations at all
        # TODO: ensure that we move in the correct direction using the dot product with the average waypoint direction
        # curve = interpolate_path(self.robot, self.robot.joints, path)
        self.robot.set_positions(current_conf)
        if not self.path:
            return None

        if self.teleport:
            action = TeleportAction(self.path[-1])
            self.set_path(path=[])
            return action

        indices = list(range(len(self.path)))
        for i in reversed(indices):
            if self.distance_fn(current_conf, self.path[i]) < 1e-2:
                # TODO: track the error
                nearby_index = i
                self.path = self.path[nearby_index+1:]
                break
        else:
            closest_index = min(indices, key=lambda i: self.distance_fn(current_conf, self.path[i]))
            self.path = self.path[closest_index:]
        if not self.path:
            return None
        new_path = [current_conf] + list(self.extend_fn(current_conf, self.path[0])) + self.path[1:]
        new_path = adjust_path(self.robot, self.robot.joints, new_path)
        waypoints = waypoints_from_path(new_path, tolerance=1e-6)
        if len(waypoints) <= 1:
            return None # TODO: return true
        target_conf = waypoints[1]
        move_action = self.move_action(current_conf, target_conf)
        # print(f'            to {target_conf} by taking {move_action}')
        return move_action

    def move_action(self, current_conf, target_conf, velocity_scale=0.95):
        if len(current_conf) == 6:
            return MoveInSE3Action(target_conf)
        target_local = multiply2d(invert2d(current_conf), target_conf)
        #delta = target_local
        delta = self.difference_fn(target_local, np.zeros(3))
        # TODO: don't include wrap around once using adjust_path
        delta = clip_delta(delta, velocity_scale * self.max_velocities, self.time_step)
        # new_conf = multiply2d(current_conf, target_local)
        # print(new_conf, target_conf)
        # handles = draw_pose2d(target_conf, z=0.) # + draw_pose2d(current_conf, z=0.)
        if np.allclose(delta[1:], np.zeros(2)):
            return DriveAction(delta[0])
        if np.allclose(delta[:2], np.zeros(2)):
            return TurnAction(delta[2])
        return MoveAction(delta=delta)

    def policy(self, observation):
        obstacles = self.update(observation)
        current_conf = np.array(self.robot.get_positions())
        if self.monitor and not self.check_path(self.path, obstacles):
            self.path = None
        if self.path is None:
            self.sample_path(obstacles)
        if (self.path is None) or (len(self.path) == 0):
            return None
        action = self.follow_path(current_conf)
        if action is None:
            self.path = None
        return action


class TeleOpAgent(MotionAgent):
    def __init__(self, world, goals=[], **kwargs):
        super(MotionAgent, self).__init__(world, **kwargs)
        self.goals = list(goals)
        self.plan_step = None
        self.plan = None
        self.command_options = self.initiate_commands()

    def initiate_commands(self):

        ## robot initial state
        x, y, yaw = get_pose2d(self.robot)
        torso_lift = self.robot.get_joint_position('torso_lift_joint')
        head_pan = self.robot.get_joint_position('head_pan_joint')
        head_tilt = self.robot.get_joint_position('head_tilt_joint')
        self.last_command = [x, y, yaw, torso_lift, head_pan, head_tilt, 0, 0, 0, 0]

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
                    joint_name = {5: 'torso_lift_joint', 6: 'head_pan_joint', 7: 'head_tilt_joint'}[i]
                    with PoseSaver(self.robot):
                        # old_conf = self.robot.get_joint_positions()
                        self.robot.set_joint_position(joint_name, command_option)
                        conf = self.robot.get_joint_positions()
                        # j = joint_from_name(self.robot, joint_name)
                        # print(f'{joint_name}: old conf {old_conf[j]}, new conf {conf[j]}')
                    return TeleportAction(conf)

                ## change object states
                elif i in range(8, 17):

                    category = {8: 'drawer', 9: 'door', 10: 'moveable', 11: 'moveable',
                                12: 'surface', 13: 'surface', 14: 'switch',
                                15: 'moveable', 16: 'moveable'}[i]
                    objects = self.world.OBJECTS_BY_CATEGORY[category]

                    print(f'\nlooking for {category} in range')

                    if i in [8, 9]:
                        collided = check_collision(self.robot, self.world.objects)
                        affected = [(obj.body, obj.joint) for obj in objects if (obj.body, obj.joint) in collided]
                        print(f'closest {category} is:')
                        return OpenJointAction(affected)

                    elif i in [10, 11, 12, 13]:
                        gripper = {10:'left', 11:'right', 12:'left', 13:'right'}[i]
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