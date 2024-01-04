import sys
import time
from itertools import product
from collections import defaultdict
import copy
import random
import shutil
from os.path import join, isdir, abspath, basename, isfile
import os
import json
import numpy as np

from pddlstream.language.constants import Equal, AND
from pddlstream.algorithms.downward import set_cost_scale

from pybullet_tools.utils import get_max_velocities, WorldSaver, elapsed_time, get_pose, unit_pose, \
    CameraImage, euler_from_quat, get_link_name, get_joint_position, joint_from_name, add_button, \
    BodySaver, set_pose, INF, add_parameter, irange, wait_for_duration, get_bodies, remove_body, \
    read_parameter, pairwise_collision, str_from_object, get_joint_name, get_name, get_link_pose, \
    get_joints, multiply, invert, is_movable, remove_handles, set_renderer, HideOutput, wait_unlocked, \
    get_movable_joints, apply_alpha, get_all_links, set_color, set_all_color, dump_body, clear_texture, \
    get_link_name, get_aabb, draw_aabb, GREY, GREEN, quat_from_euler, wait_for_user, get_camera_matrix, \
    Euler, PI, get_center_extent, create_box, RED, unit_quat, set_joint_position, get_joint_limits
from pybullet_tools.pr2_streams import Position, get_handle_grasp_gen, pr2_grasp
from pybullet_tools.general_streams import pose_from_attachment, LinkPose, RelPose
from pybullet_tools.bullet_utils import set_zero_world, nice, open_joint, get_pose2d, summarize_joints, get_point_distance, \
    is_placement, is_contained, add_body, close_joint, toggle_joint, ObjAttachment, check_joint_state, \
    set_camera_target_body, xyzyaw_to_pose, nice, LINK_STR, CAMERA_MATRIX, visualize_camera_image, equal, \
    draw_pose2d_path, draw_pose3d_path, sort_body_parts, get_root_links, colorize_world, colorize_link, \
    draw_fitted_box, find_closest_match, get_objs_in_camera_images, multiply_quat, is_joint_open
from pybullet_tools.pr2_primitives import Pose, Conf, get_ik_ir_gen, get_motion_gen, \
    Attach, Detach, Clean, Cook, control_commands, link_from_name, \
    get_gripper_joints, GripperCommand, apply_commands, State, Command

from .entities import Region, Environment, Robot, Surface, ArticulatedObjectPart, Door, Drawer, Knob, \
    Camera, Object, StaticCamera
from world_builder.utils import GRASPABLES
from world_builder.samplers import get_learned_yaw

DEFAULT_CONSTANTS = ['@movable', '@bottle', '@edible', '@medicine']  ## , '@world'


class WorldBase(object):
    """ parent api for working with planning + replanning architecture """
    def __init__(self, time_step=1e-3, teleport=False, drive=True, prevent_collisions=False,
                 constants=DEFAULT_CONSTANTS, segment=False):
        ## for world attributes
        self.scramble = False
        self.time_step = time_step
        self.drive = drive
        self.teleport = teleport
        self.prevent_collisions = prevent_collisions
        self.segment = segment

        ## for planning
        self.constants = constants

        ## for visualization
        self.handles = []

        ## for data generation
        self.note = None
        self.camera = None
        self.img_dir = None
        self.cameras = []

        ## for exposed observation model
        self.exposed_observation_cameras = None
        self.space_markers = None

    def get_name(self, body):
        raise NotImplementedError

    def cat_to_bodies(self, cat, **kwargs):
        raise NotImplementedError

    ###########################################################################

    def save_test_case(self, output_dir, save_rgb=False, save_depth=False, **kwargs):
        if not isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self.save_problem(output_dir, **kwargs)

        for suffix in ['log.txt', 'commands.pkl', 'time.json']:
            if isfile(f"{output_dir}_{suffix}"):
                shutil.move(f"{output_dir}_{suffix}", join(output_dir, suffix))

        ## save the end image
        if save_rgb:
            self.visualize_image(img_dir=output_dir, rgb=True)
        if save_depth:
            self.visualize_image(img_dir=output_dir)

    def save_problem(self, output_dir, **kwargs):
        raise NotImplementedError

    ###########################################################################

    def add_camera(self, pose=unit_pose(), img_dir=join('visualizations', 'camera_images'),
                   width=640, height=480, fx=400, **kwargs):

        # camera_matrix = get_camera_matrix(width=width, height=height, fx=525., fy=525.)
        camera_matrix = get_camera_matrix(width=width, height=height, fx=fx)
        camera = StaticCamera(pose, camera_matrix=camera_matrix, **kwargs)
        self.cameras.append(camera)
        self.camera = camera
        self.img_dir = img_dir
        return camera

    def visualize_image(self, pose=None, img_dir=None, index=None,
                        image=None, segment=False, far=8, segment_links=False,
                        camera_point=None, target_point=None, **kwargs):
        from pybullet_tools.bullet_utils import visualize_camera_image

        if not isinstance(self.camera, StaticCamera):
            self.add_camera()
        if pose is not None:
            self.camera.set_pose(pose)
        if img_dir is not None:
            self.img_dir = img_dir
        if index is None:
            index = self.camera.index
        if image is None:
            image = self.camera.get_image(segment=segment, segment_links=segment_links, far=far,
                                          camera_point=camera_point, target_point=target_point)
        visualize_camera_image(image, index, img_dir=self.img_dir, **kwargs)

    def initiate_exposed_cameras(self):
        if self.exposed_observation_cameras is not None:
            return
        ## add cameras facing the front and front-down
        self.exposed_observation_cameras = []
        quat_front = (0.5, 0.5, -0.5, -0.5)
        quat_right = multiply_quat(quat_front, quat_from_euler(Euler(yaw=0, pitch=PI / 2, roll=0)))
        quat_left = multiply_quat(quat_front, quat_from_euler(Euler(yaw=0, pitch=-PI / 2, roll=0)))
        quat_front_down = multiply_quat(quat_front, quat_from_euler(Euler(yaw=0, pitch=0, roll=-PI / 4)))
        for pose in [((3.9, 7, 1.3), (0.5, 0.5, -0.5, -0.5)),
                     ((2.9, 7, 3.3), quat_front_down)]:
            self.exposed_observation_cameras.append(self.add_camera(pose=pose))

    def initiate_space_markers(self, s=0.03):
        if self.space_markers is not None:
            return
        self.space_markers = {}
        for space in self.cat_to_bodies('space', get_all=True):
            body, _, link = space
            center, extent = get_center_extent(body, link=link)
            marker = create_box(s, s, s, color=RED)
            set_pose(marker, (center, unit_quat()))
            size = extent[0] * extent[1] * extent[2]
            self.space_markers[marker] = {
                'space': space, 'name': self.get_name(space),
                'size': size, 'extent': extent, 'center': center, 'marker': marker
            }
        return self.space_markers

    ###########################################################################

    def remove_handles(self):
        remove_handles(self.handles)

    def add_handles(self, handles):
        self.handles.extend(handles)


#######################################################################################


class World(WorldBase):
    """ api for building world and tamp problems """
    def __init__(self, **kwargs):
        ## conf_noise=None, pose_noise=None, depth_noise=None, action_noise=None,  # TODO: noise model class?
        super().__init__(**kwargs)

        self.robot = None
        self.ROBOT_TO_OBJECT = {}
        self.BODY_TO_OBJECT = {}
        self.OBJECTS_BY_CATEGORY = defaultdict(list)
        self.REMOVED_BODY_TO_OBJECT = {}
        self.REMOVED_OBJECTS_BY_CATEGORY = defaultdict(list)

        self.ATTACHMENTS = {}
        self.sub_categories = {}
        self.sup_categories = {}
        self.SKIP_JOINTS = False
        self.floorplan = None  ## for saving LISDF
        self.init = []
        self.init_del = []
        self.articulated_parts = {k: [] for k in ['door', 'drawer', 'knob', 'button']}
        self.changed_joints = []
        self.non_planning_objects = []
        self.not_stackable = {}
        self.c_ignored_pairs = []
        self.planning_config = {'camera_zoomins': []}

        ## for visualization
        self.path = None
        self.outpath = None
        self.instance_names = {}

        self.clean_object = set()

    def clear_viz(self):
        self.remove_handles()
        self.remove_redundant_bodies()

    def remove_redundant_bodies(self):
        with HideOutput():
            for b in get_bodies():
                if b not in self.BODY_TO_OBJECT and b not in self.ROBOT_TO_OBJECT \
                        and b not in self.non_planning_objects:
                    remove_body(b)
                    print('world.removed redundant body', b)

    def add_not_stackable(self, body, surface):
        if body not in self.not_stackable:
            self.not_stackable[body] = []
        self.not_stackable[body].append(surface)

    def check_not_stackable(self, body, surface):
        return body in self.not_stackable and surface in self.not_stackable[body]

    def make_transparent(self, name, transparency=0.5):
        if isinstance(name, str):
            obj = self.name_to_object(name)
            colorize_link(obj.body, obj.link, transparency=transparency)
        elif isinstance(name, int) or isinstance(name, tuple):
            colorize_link(name, transparency=transparency)
        elif isinstance(name, Object):
            colorize_link(name.body, name.link, transparency=transparency)

    def make_doors_transparent(self, transparency=0.5):
        colorize_world(self.fixed, transparency)

    def set_path(self, path):
        remove_handles(self.handles)
        self.path = path
        if self.path is None:
            return self.path
        if isinstance(self.path[0], tuple):  ## pr2
            #path = adjust_path(self.robot, self.robot.joints, self.path)
            self.handles.extend(draw_pose2d_path(self.path[::4], length=0.05))
            #wait_if_gui()
        else:  ## SE3 gripper
            path = [p.values for p in path]
            self.handles.extend(draw_pose3d_path(path, length=0.05))
        return self.path

    def set_skip_joints(self):
        ## not automatically add doors and drawers, save planning time
        self.SKIP_JOINTS = True

    @property
    def max_delta(self):
        return self.max_velocities * self.time_step

    @property
    def objects(self):
        return [k for k in self.BODY_TO_OBJECT.keys() if k not in self.ROBOT_TO_OBJECT]

    @property
    def all_objects(self):
        return self.objects + [k for k in self.REMOVED_BODY_TO_OBJECT.keys()]

    @property
    def movable(self):  ## include steerables if want to exclude them when doing base motion plannig
        return [self.robot] + self.cat_to_bodies('moveable')  ## + self.cat_to_bodies('steerable')
        # return [obj for obj in self.objects if obj not in self.fixed]

    @property
    def floors(self):
        return self.cat_to_bodies('floor')

    @property
    def fixed(self):
        objs = [obj for obj in self.objects if not isinstance(obj, tuple)]
        objs += [o for o in self.non_planning_objects if isinstance(o, int) and o not in objs]
        objs = [o for o in objs if o not in self.floors and o not in self.movable]
        return objs

    @property
    def ignored_pairs(self):
        found = self.c_ignored_pairs
        if self.floorplan is not None and 'kitchen' in self.floorplan:
            a = self.cat_to_bodies('counter', get_all=True)[0]

            ## not colliding with the oven which is placed inside the counter
            b = self.cat_to_bodies('oven', get_all=True)[0]
            if (a, b) not in found:
                found.extend([(a, b), (b, a)])

            ## find all surfaces and spaces associated with the counter
            counter = self.BODY_TO_OBJECT[a] if a in self.BODY_TO_OBJECT else self.REMOVED_BODY_TO_OBJECT[a]
            link_bodies = [(a, None, lk) for lk in counter.surfaces + counter.spaces]
            for link_body in link_bodies:
                objs = self.BODY_TO_OBJECT[link_body].supported_objects if link_body in self.BODY_TO_OBJECT \
                    else self.REMOVED_BODY_TO_OBJECT[link_body].supported_objects
                for obj in objs:
                    if obj.body not in self.movable and (a, obj.body) not in found:
                        found.extend([(a, obj.body), (obj.body, a)])

        # plate = self.name_to_body('plate')
        # if plate is not None:
        #     found.append([(plate, self.robot.body), (self.robot.body, plate)])
        return found

    def add_ignored_pair(self, pair):
        a, b = pair
        self.c_ignored_pairs.extend([(a, b), (b, a)])

    def get_planning_objects(self):
        return [o.lisdf_name for o in self.BODY_TO_OBJECT.values()]

    def get_attr(self, obj, attr):
        preds = ['left', 'right', 'hand'] + ['joint', 'door', 'drawer', 'knob', 'button']
        if isinstance(obj, str):
            if obj.lower() in preds:
                return obj
            obj = self.name_to_object(obj)

        elif isinstance(obj, Object):
            obj = obj
        elif obj in self.BODY_TO_OBJECT:
            obj = self.BODY_TO_OBJECT[obj]
        elif obj in self.REMOVED_BODY_TO_OBJECT:
            obj = self.REMOVED_BODY_TO_OBJECT[obj]
        else: ## get readable list
            return obj
        if isinstance(obj, str):
            return obj
        if obj is None:
            return obj
        return getattr(obj, attr)

    def get_name(self, body, use_default_link_name=False):
        name = self.get_attr(body, 'name')
        if use_default_link_name and name == body and len(body) == 2:
            name = get_link_name(body[0], body[1])
        return name

    def get_category(self, body):
        return self.get_attr(body, 'category')

    def get_debug_name(self, body):
        """ for viewing pleasure :) """
        return self.get_attr(body, 'debug_name')

    def get_lisdf_name(self, body):
        """ for recording objects in lisdf files generated """
        return self.get_attr(body, 'lisdf_name')

    def get_instance_name(self, body):
        """ for looking up objects in the grasp database """
        if isinstance(body, tuple) and body in self.instance_names:
            return self.instance_names[body]
        return self.get_attr(body, 'instance_name')

    def get_mobility_id(self, body):
        return self.get_attr(body, 'mobility_id')

    def get_mobility_identifier(self, body):
        return self.get_attr(body, 'mobility_identifier')

    def get_events(self, body):
        return self.get_attr(body, 'events')

    def add_box(self, object, pose=None):
        obj = self.add_object(object, pose=pose)
        obj.is_box = True
        return obj

    def add_highlighter(self, body):
        if isinstance(body, Object):
            obj = body
            body = obj.body
        else:
            obj = self.BODY_TO_OBJECT[body]
        if self.get_attr(obj, 'is_box'):
            draw_aabb(get_aabb(body), color=GREEN)
        else:
            draw_fitted_box(body, draw_box=True)
            # set_all_color(body, (1, 0, 0, 1))

    def add_object(self, obj: Object, pose=None) -> Object:

        OBJECTS_BY_CATEGORY = self.OBJECTS_BY_CATEGORY
        BODY_TO_OBJECT = self.BODY_TO_OBJECT
        category = obj.category
        name = obj.name
        body = obj.body
        joint = obj.joint
        link = obj.link
        class_name = obj.__class__.__name__.lower()

        if category not in OBJECTS_BY_CATEGORY:
            OBJECTS_BY_CATEGORY[category] = []

        ## be able to find eggs as moveables
        if class_name != category:
            if category not in self.sup_categories:
                self.sup_categories[category] = class_name
            if class_name not in self.sub_categories:
                self.sub_categories[class_name] = []
            if category not in self.sub_categories[class_name]:
                self.sub_categories[class_name].append(category)

        ## automatically name object
        if name is None:
            next = len([o.name for o in OBJECTS_BY_CATEGORY[category] if '#' in o.name])
            name = '{}#{}'.format(category, next + 1)
        ## TODO: better deal with the same instance of unnamed objects
        name = name.lower()
        if joint is None and link is None:
            count = 1
            while self.name_to_body(name) is not None:
                if '#' in name:
                    name = name[:name.index('#')]
                name = '{}#{}'.format(name, count)
                count += 1
        obj.name = name

        OBJECTS_BY_CATEGORY[category].append(obj)

        ## -------------- different types of object --------------
        ## object parts: doors, drawers, knobs
        if joint is not None:
            BODY_TO_OBJECT[(body, joint)] = obj
            obj.name = f"{BODY_TO_OBJECT[body].name}{LINK_STR}{obj.name}"
            from lisdf_tools.lisdf_loader import PART_INSTANCE_NAME
            n = self.get_instance_name(body)
            part_name = get_link_name(body, obj.handle_link)
            n = PART_INSTANCE_NAME.format(body_instance_name=n, part_name=part_name)
            self.instance_names[(body, obj.handle_link)] = n

        ## object parts: surface, space
        elif link is not None:
            BODY_TO_OBJECT[(body, None, link)] = obj
            obj.name = f"{BODY_TO_OBJECT[body].name}{LINK_STR}{obj.name}"
            if category == 'surface':
                BODY_TO_OBJECT[body].surfaces.append(link)
            if category == 'space':
                BODY_TO_OBJECT[body].spaces.append(link)

        ## object
        elif not isinstance(obj, Robot):
            BODY_TO_OBJECT[body] = obj
            self.get_doors_drawers(obj.body, skippable=True)

        ## robot
        else:
            self.ROBOT_TO_OBJECT[body] = obj

        if link is not None or joint is not None:
            parent = BODY_TO_OBJECT[body]
            obj.path = parent.path
            obj.scale = parent.scale
            obj.mobility_id = parent.mobility_id
            obj.mobility_category = parent.mobility_category
            obj.mobility_identifier = parent.mobility_identifier

        if pose is not None:
            add_body(obj, pose)

        obj.world = self
        return obj

    def get_whole_fact(self, fact, init):
        if fact[0].lower() in ['isopenposition', 'isclosedposition']:
            fact += [f[2] for f in init if f[0].lower() == 'atposition' and f[1] == fact[1]]
            print('world.get_whole_fact | ', fact)
        return fact

    def add_to_init(self, fact):
        self.init.append(fact)

    def del_fr_init(self, fact):
        self.init_del.append(fact)

    def add_to_cat(self, body, cat):
        object = self.get_object(body)
        if cat not in self.OBJECTS_BY_CATEGORY:
            self.OBJECTS_BY_CATEGORY[cat] = []
        self.OBJECTS_BY_CATEGORY[cat].append(object)
        if cat not in object.categories:
            object.categories.append(cat)

    def add_robot(self, robot, name='robot', max_velocities=None):
        self.robot = robot  # TODO: multi-robot
        self.ROBOT_TO_OBJECT[robot.body] = robot
        self.max_velocities = get_max_velocities(robot, robot.joints) if (max_velocities is None) else max_velocities
        robot.world = self

    def add_joint_object(self, body, joint, category=None):
        if category is None:
            category, state = check_joint_state(body, joint)
        joints = {k: [] for k in ['door', 'drawer', 'knob']}
        if 'door' in category:
            joints['door'].append(joint)
            if (body, joint) not in self.BODY_TO_OBJECT:
                door = Door(body, joint=joint)
                self.add_object(door)
        elif 'drawer' in category:
            joints['drawer'].append(joint)
            if (body, joint) not in self.BODY_TO_OBJECT:
                drawer = Drawer(body, joint=joint)
                self.add_object(drawer)
        elif 'knob' in category:
            joints['knob'].append(joint)
            if (body, joint) not in self.BODY_TO_OBJECT:
                knob = Knob(body, joint=joint)
                self.add_object(knob)
                self.add_to_cat(knob, 'joint')
        return joints

    def get_doors_drawers(self, body, skippable=False):
        obj = self.BODY_TO_OBJECT[body]
        if skippable and obj.doors is not None:
            return obj.doors, obj.drawers, obj.knobs

        doors = []
        drawers = []
        knobs = []
        if not skippable or not self.SKIP_JOINTS:
            for joint in get_joints(body):
                joints = self.add_joint_object(body, joint)
                doors.extend(joints['door'])
                drawers.extend(joints['drawer'])
                knobs.extend(joints['knob'])

        obj.doors = doors
        obj.drawers = drawers
        obj.knobs = knobs
        return doors, drawers, knobs

    def add_joints_by_keyword(self, body_name, joint_name=None, category=None):
        body = self.name_to_body(body_name)
        joints = [j for j in get_joints(body) if is_movable(body, j)]
        if joint_name is not None:
            joints = [j for j in joints if joint_name in get_joint_name(body, j)]
        for j in joints:
            self.add_joint_object(body, j, category=category)
        return [(body, j) for j in joints]

    def add_surface_by_keyword(self, body, link_name):
        if isinstance(body, str):
            body = self.name_to_body(body)
        link = link_from_name(body, link_name)
        surface = Surface(body, link=link)
        self.add_object(surface)
        return surface

    def summarize_supporting_surfaces(self):
        from pybullet_tools.logging import myprint as print
        return_dict = {}
        padding = '    '
        print('--------------- summarize_supporting_surfaces --------------')
        print(padding, 'surface', self.cat_to_objects('surface'))
        print(padding, 'supporter', self.cat_to_objects('supporter'))
        for surface in set(self.cat_to_objects('surface') + self.cat_to_objects('supporter')):
            print(padding, surface.name, surface.supported_objects)
            return_dict[surface.name] = [o.name for o in surface.supported_objects]
        print('-------------------------------------------------')
        return return_dict

    def summarize_supported_movables(self):
        from pybullet_tools.logging import myprint as print
        return_dict = {}
        padding = '    '
        print('--------------- summarize_supported_movables --------------')
        print(padding, 'moveable', self.cat_to_objects('moveable'))
        for movable in set(self.cat_to_objects('moveable')):
            print(padding, movable.name, movable.supporting_surface)
            return_dict[movable.name] = movable.supporting_surface.name
        print('-------------------------------------------------')
        return return_dict

    def summarize_all_types(self):
        printout = ''
        for typ in ['moveable', 'surface', 'door', 'drawer']:
            num = len(self.cat_to_bodies(typ))
            if num > 0:
                printout += "{type}({num}), ".format(type=typ, num=num)
        return printout

    def summarize_all_objects(self, print_fn=None):
        if print_fn is None:
            from pybullet_tools.logging import myprint as print_fn

        BODY_TO_OBJECT = self.BODY_TO_OBJECT
        ROBOT_TO_OBJECT = self.ROBOT_TO_OBJECT
        REMOVED_BODY_TO_OBJECT = self.REMOVED_BODY_TO_OBJECT

        # bodies = copy.deepcopy(get_bodies())
        # bodies.sort()
        # print('----------------')
        # print(f'PART I: all pybullet bodies')
        # print('----------------')
        # for body in bodies:
        #     if body in BODY_TO_OBJECT:
        #         object = BODY_TO_OBJECT[body]
        #         n = object.name
        #         pose = nice(get_pose(body))
        #         line = f'{body}  |  Name: {n}, Pose: {pose}'
        #
        #         doors, drawers, knobs = self.get_doors_drawers(body)
        #         doors = [get_joint_name(body, j).replace(f'{n}--', '') for j in doors]
        #         drawers = [get_joint_name(body, j).replace(f'{n}--', '') for j in drawers]
        #
        #         if len(doors) > 0:
        #             line += f'  |  Doors: {doors}'
        #         if len(drawers) > 0:
        #             line += f'  |  Drawers: {drawers}'
        #         print(line)
        #
        #     elif body in ROBOT_TO_OBJECT:
        #         object = ROBOT_TO_OBJECT[body]
        #         n = object.name
        #         pose = nice(object.get_pose())
        #         line = f'{body}  |  Name: {n}, Pose: {pose}'
        #         print(line)
        #
        #     else:
        #         print(f'{body}  |  Name: {get_name(body)}, not in BODY_TO_OBJECT')

        print_fn('----------------')
        print_fn(f'PART I: world objects | {self.summarize_all_types()} | obstacles({len(self.fixed)}) = {self.fixed}')
        print_fn('----------------')
        bodies = [self.robot] + sort_body_parts(BODY_TO_OBJECT.keys())
        bodies += sort_body_parts(REMOVED_BODY_TO_OBJECT.keys(), bodies)
        static_bodies = [b for b in get_bodies() if b not in bodies]
        # bodies += static_bodies
        print_not = print_not_2 = False
        for body in bodies:
            if body in ROBOT_TO_OBJECT:
                object = ROBOT_TO_OBJECT[body]
            elif body in REMOVED_BODY_TO_OBJECT:
                object = REMOVED_BODY_TO_OBJECT[body]
            else:
                object = BODY_TO_OBJECT[body]

            ## ---- print class inheritance, if any
            typ_str = object._type()
            # if object._type().lower() in self.sup_categories:
            #     typ = object._type().lower()
            #     typ_str = ''
            #     while typ in self.sup_categories:
            #         typ_str = f"{self.sup_categories[typ].capitalize()} -> {typ_str}"
            #         typ = self.sup_categories[typ]

            line = f'{body}\t  |  {typ_str}: {object.name}'
            if isinstance(body, tuple) and len(body) == 2:
                b, j = body
                pose = get_joint_position(b, j)
                if hasattr(object, 'handle_link') and object.handle_link is not None:
                    line += f'\t|  Handle: {get_link_name(b, object.handle_link)}'
                line += f'\t|  JointLimit: {get_joint_limits(b, j)}'
            elif isinstance(body, tuple) and len(body) == 3:
                b, _, l = body
                pose = get_link_pose(b, l)
            else:
                pose = get_pose(body)
            line += f"\t|  Pose: {nice(pose)}"

            if body in REMOVED_BODY_TO_OBJECT:
                # if object.category in ['filler']:
                #     continue
                if not print_not:
                    print_fn('----------------')
                    print_not = True
                line += f"\t (excluded from planning)"
            elif body in static_bodies:
                if not print_not_2:
                    print_fn('----------------')
                    print_not_2 = True
                line += f"\t (static world objects)"

            print_fn(line)
        print_fn('----------------')

    def get_all_obj_in_body(self, body):
        if isinstance(body, tuple):
            return [body]
        if body not in self.BODY_TO_OBJECT:
            set_camera_target_body(body)
            set_renderer(True)
            wait_unlocked()
        object = self.BODY_TO_OBJECT[body]
        bodies = [body]
        if len(object.doors) > 0:
            bodies += [(body, j) for j in object.doors]
        if len(object.drawers) > 0:
            bodies += [(body, j) for j in object.drawers]
        bodies += [(body, None, l) for l in object.surfaces + object.spaces]
        bodies += [bb for bb in self.BODY_TO_OBJECT.keys() if
                   isinstance(bb, tuple) and bb[0] == body and bb not in bodies]
        bodies = [b for b in bodies if b in self.BODY_TO_OBJECT]
        return bodies

    def remove_bodies_from_planning(self, goals=[], exceptions=[]):
        print('remove_bodies_from_planning | exceptions =', exceptions)
        bodies = []
        if isinstance(goals, tuple):
            goals = [goals]
        for literal in goals:
            for item in literal:
                if not isinstance(item, str) and str(item) not in bodies:
                    if isinstance(item, Object):
                        item = item.pybullet_name
                    if isinstance(item, tuple):
                        bodies.append(str(item[0]))
                    bodies.append(str(item))

        new_exceptions = []
        for b in exceptions:
            if isinstance(b, Object):
                b = b.pybullet_name
            if isinstance(b, tuple) and str(b[0]) not in new_exceptions:
                new_exceptions.append(str(b[0]))
            new_exceptions.append(str(b))
        exceptions = new_exceptions

        all_bodies = list(self.BODY_TO_OBJECT.keys())
        for body in all_bodies:
            if str(body) not in bodies and str(body) not in exceptions:
                self.remove_body_from_planning(body)

        for cat, objs in self.REMOVED_OBJECTS_BY_CATEGORY.items():
            print(f'\t{cat} ({len(objs)}) \t', [obj.name for obj in objs])

    def remove_body_from_planning(self, body):
        if body is None: return
        bodies = self.get_all_obj_in_body(body)
        for body in bodies:
            if body not in self.BODY_TO_OBJECT:
                continue
            cat = self.BODY_TO_OBJECT[body].category
            obj = self.BODY_TO_OBJECT.pop(body)
            self.REMOVED_BODY_TO_OBJECT[body] = obj
            self.non_planning_objects.append(body)

            ## still need to find all floors
            if cat == 'floor':
                continue
            self.remove_object_from_category(cat, obj)

    def remove_category_from_planning(self, category, exceptions=[]):
        if len(exceptions) > 0 and isinstance(exceptions[0], str):
            new_exceptions = []
            for exception in exceptions:
                if exception == 'braiser_bottom':
                    print('exception', exception)
                new_exceptions.append(self.name_to_body(exception))
            exceptions = new_exceptions
        bodies = self.cat_to_bodies(category)
        for body in bodies:
            if body in exceptions:
                continue
            print('removing from planning', body)
            self.remove_body_from_planning(body)

    def remove_body_attachment(self, body):
        obj = self.BODY_TO_OBJECT[body]
        if obj in self.ATTACHMENTS:
            print('world.remove_body_attachment\t', self.ATTACHMENTS[obj])
            self.ATTACHMENTS.pop(obj)

    def remove_object(self, object):
        object = self.get_object(object)
        body = object.body

        ## remove all objects initiated by the body
        bodies = self.get_all_obj_in_body(body)
        for b in bodies:
            obj = self.BODY_TO_OBJECT.pop(b)

            # so cat_to_bodies('moveable') won't find it
            for cat in obj.categories:
                self.remove_object_from_category(cat, obj)
            if hasattr(obj, 'supporting_surface') and isinstance(obj.supporting_surface, Surface):
                surface = obj.supporting_surface
                surface.supported_objects.remove(obj)

        if object in self.ATTACHMENTS:
            self.ATTACHMENTS.pop(object)
        remove_body(body)

    def remove_object_from_category(self, cat, obj):
        all_removed = [str(k) for k in self.REMOVED_OBJECTS_BY_CATEGORY[cat]]
        self.REMOVED_OBJECTS_BY_CATEGORY[cat] += [
            o for o in self.OBJECTS_BY_CATEGORY[cat] if str(o) not in all_removed and
            (o.body == obj.body and o.link == obj.link and o.joint == obj.joint)
        ]
        self.OBJECTS_BY_CATEGORY[cat] = [
            o for o in self.OBJECTS_BY_CATEGORY[cat] if not
            (o.body == obj.body and o.link == obj.link and o.joint == obj.joint)
        ]

    ##################################################################################

    def body_to_name(self, body):
        if body in self.BODY_TO_OBJECT:
            return self.BODY_TO_OBJECT[body].name
        elif body in self.ROBOT_TO_OBJECT:
            return self.ROBOT_TO_OBJECT[body].name
        return None

    def name_to_body(self, name):
        name = name.lower()
        possible = {}
        for body, obj in self.ROBOT_TO_OBJECT.items():
            if name == obj.name:
                return self.robot.body
        for body, obj in self.BODY_TO_OBJECT.items():
            if name == obj.name:
                return body
            if name in obj.name:
                possible[body] = obj.name
        if len(possible) >= 1:
            return find_closest_match(possible)
        return None

    def name_to_object(self, name):
        if self.name_to_body(name) is None:
            return name  ## None ## object doesn't exist
        return self.BODY_TO_OBJECT[self.name_to_body(name)]

    def cat_to_bodies(self, cat, get_all=False):
        bodies = []
        objects = []
        if cat in self.OBJECTS_BY_CATEGORY:
            objects.extend(self.OBJECTS_BY_CATEGORY[cat])
        if get_all and cat in self.REMOVED_OBJECTS_BY_CATEGORY:
            objects.extend(self.REMOVED_OBJECTS_BY_CATEGORY[cat])
        if cat in self.sub_categories:
            for c in self.sub_categories[cat]:
                objects.extend(self.OBJECTS_BY_CATEGORY[c])
                if get_all:
                    objects.extend(self.REMOVED_OBJECTS_BY_CATEGORY[c])

        for o in objects:
            if o.link is not None:
                bodies.append((o.body, o.joint, o.link))
            elif o.joint is not None:
                bodies.append((o.body, o.joint))
            else:
                bodies.append(o.body)
        filtered_bodies = []
        for b in set(bodies):
            if b in self.BODY_TO_OBJECT or cat == 'floor':
                filtered_bodies += [b]
            elif get_all and b in self.REMOVED_BODY_TO_OBJECT or cat == 'floor':
                filtered_bodies += [b]
            # else:
            #     print(f'   world.cat_to_bodies | category {cat} found {b}')
        return filtered_bodies

    def cat_to_objects(self, cat):
        bodies = self.cat_to_bodies(cat)
        return [self.BODY_TO_OBJECT[b] for b in bodies]

    def get_collision_objects(self):
        saved = list(self.BODY_TO_OBJECT.keys())
        return [n for n in get_bodies() if n in saved and n > 1]

    def assign_attachment(self, body, tag=None):
        title = f'   world.assign_attachment({body}) | '
        if tag is not None:
            title += f'tag = {tag} | '
        for child, attach in self.ATTACHMENTS.items():
            if attach.parent.body == body:
                pose = get_pose(child)
                attach.assign()
                if pose != get_pose(child):  ## attachment made a difference
                    print(title, attach, nice(attach.grasp_pose))

    def get_scene_joints(self):
        c = self.cat_to_bodies
        joints = c('door') + c('drawer') + c('knob')
        joints += [bj for bj in self.changed_joints if bj not in joints]
        return joints

    def _change_joint_state(self, body, joint):
        self.assign_attachment(body)
        if (body, joint) not in self.changed_joints:
            self.changed_joints.append((body, joint))

    def toggle_joint(self, body, joint=None):
        if joint is None and isinstance(body, tuple):
            body, joint = body
        toggle_joint(body, joint)
        self._change_joint_state(body, joint)

    def close_joint(self, body, joint=None):
        if joint is None and isinstance(body, tuple):
            body, joint = body
        close_joint(body, joint)
        self._change_joint_state(body, joint)

    def open_joint(self, body, joint=None, extent=1, pstn=None, random_gen=False, **kwargs):
        if joint is None and isinstance(body, tuple):
            body, joint = body
        if random_gen:
            from pybullet_tools.general_streams import sample_joint_position_list_gen
            funk = sample_joint_position_list_gen()
            pstns = funk((body, joint), Position((body, joint)))
            pstn = random.choice(pstns)[0].value
        open_joint(body, joint, extent=extent, pstn=pstn, **kwargs)
        self._change_joint_state(body, joint)

    def open_doors_drawers(self, body, ADD_JOINT=True, **kwargs):
        doors, drawers, knobs = self.get_doors_drawers(body, skippable=True)
        for joint in doors + drawers:
            if isinstance(joint, tuple):
                body, joint = joint
            self.open_joint(body, joint, **kwargs)
            if not ADD_JOINT:
                self.remove_object(joint)

    def close_doors_drawers(self, body, ADD_JOINT=True):
        doors, drawers, knobs = self.get_doors_drawers(body, skippable=True)
        for b, j in doors + drawers:
            self.close_joint(b, j)
            if not ADD_JOINT:
                self.remove_object(j)

    def close_all_doors_drawers(self):
        doors = [(o.body, o.joint) for o in self.cat_to_objects('door')]
        drawers = [(o.body, o.joint) for o in self.cat_to_objects('drawer')]
        for body, joint in doors + drawers:
            self.close_joint(body, joint)

    def open_all_doors_drawers(self, extent=1):
        doors = [(o.body, o.joint) for o in self.cat_to_objects('door')]
        drawers = [(o.body, o.joint) for o in self.cat_to_objects('drawer')]
        for body, joint in doors + drawers:
            self.open_joint(body, joint, extent=extent)

    def open_joint_by_name(self, name, pstn=None):
        body, joint = self.name_to_body(name)
        self.open_joint(body, joint, pstn=pstn)

    def close_joint_by_name(self, name):
        body, joint = self.name_to_body(name)
        self.close_joint(body, joint)

    def toggle_joint_by_name(self, name):
        body, joint = self.name_to_body(name)
        self.toggle_joint(body, joint)

    def get_object(self, obj):
        if isinstance(obj, Object):
            return obj
        elif isinstance(obj, str):
            obj = self.name_to_object(obj)
        elif obj in self.BODY_TO_OBJECT:
            obj = self.BODY_TO_OBJECT[obj]
        elif obj in self.REMOVED_BODY_TO_OBJECT:
            obj = self.REMOVED_BODY_TO_OBJECT[obj]
        elif obj in self.ROBOT_TO_OBJECT:
            obj = self.ROBOT_TO_OBJECT[obj]
        else:
            obj = None
        return obj

    def put_on_surface(self, obj, surface='hitman_tmp', max_trial=20, OAO=False, **kwargs):
        obj = self.get_object(obj)
        surface_obj = self.get_object(surface)
        surface = surface_obj.name

        surface_obj.place_obj(obj, max_trial=max_trial, **kwargs)

        ## ----------- rules of locate specific objects
        world_to_surface = surface_obj.get_pose()
        point, quat = obj.get_pose()
        x, y, z = point
        if 'faucet_platform' in surface:
            (a, b, c), quat = world_to_surface
            obj.set_pose(((a - 0.2, b, z), quat), **kwargs)
        elif 'hitman_tmp' in surface:
            quat = (0, 0, 1, 0)  ## facing out
            obj.set_pose(((0.4, 6.4, z), quat), **kwargs)
        elif obj.category in ['microwave', 'toaster']:
            quat = (0, 0, 1, 0)  ## facing out
            obj.set_pose((point, quat), **kwargs)
        else:
            ## try learned sampling
            learned_quat = get_learned_yaw(obj.lisdf_name, quat)
            if learned_quat is not None:
                # print('    using learned yaw', learned_quat)
                obj.set_pose((point, learned_quat), **kwargs)

        ## ---------- reachability hacks for PR2
        if hasattr(self, 'robot') and 'pr2' in self.robot.name:

            ## hack to be closer to edge
            if 'shelf' in surface:
                surface_to_obj = ((-0.2, 0, -0.2), (0, 0, 1, 0))
                (a, b, _), _ = multiply(world_to_surface, surface_to_obj)
                obj.set_pose(((a, b, z), quat), **kwargs)
                # obj.set_pose(((1, 4.4, z), quat))
                # obj.set_pose(((1.6, 4.5, z), quat)) ## vertical orientation
            elif 'tmp' in surface: ## egg
                if y > 9: y = 8.9
                obj.set_pose(((0.7, y, z), quat), **kwargs)

        ## ---------- center object
        if 'braiser_bottom' in surface:  ## for testing
            (a, b, c), _ = world_to_surface
            obj.set_pose(((a, b, z), quat), **kwargs)
            # obj.set_pose(((0.55, b, z), (0, 0, 0.36488663206619243, 0.9310519565198234)), **kwargs)
        elif 'braiser' in surface:
            (a, b, c), quat = world_to_surface
            obj.set_pose(((a, b, z), quat), **kwargs)
        elif 'front_' in surface and '_stove' in surface:
            obj.set_pose(((0.55, y, z), quat), **kwargs)

        surface_obj.attach_obj(obj, **kwargs)
        if OAO: ## one and only
            self.remove_body_from_planning(self.name_to_body(surface))

    def put_in_space(self, obj, space='hitman_drawer_top', xyzyaw=None, learned=True):
        container = self.name_to_object(space)
        if learned:
            ## one possible pose put into hitman_drawer_top
            # pose = {'hitman_drawer_top': ((1, 7.5, 0.7), (0, 0, 0.3, 0.95)),
            #         'indigo_drawer_top': ((0.75, 8.9, 0.7), (0, 0, 0.3, 0.95))}[space]
            rel_pose = ((0.2, 0, -0.022), (0, 0, 0, 1))
            link_pose = get_link_pose(container.body, link_from_name(container.body, space))
            pose = multiply(link_pose, rel_pose)
            obj.set_pose(pose)
            container.attach_obj(obj)
        else:
            container.place_obj(obj, xyzyaw, max_trial=1)

    def refine_marker_obstacles(self, marker, obstacles):
        ## for steerables
        parent = self.BODY_TO_OBJECT[marker].grasp_parent
        if parent is not None and parent in obstacles:
            obstacles.remove(parent)
        return obstacles

    # def add_camera(self, pose=unit_pose(), img_dir=join('visualizations', 'camera_images')):
    #     camera = StaticCamera(pose, camera_matrix=CAMERA_MATRIX, max_depth=6)
    #     self.cameras.append(camera)
    #     self.camera = camera
    #     self.img_dir = img_dir
    #     if self.camera:
    #         return self.cameras[-1].get_image(segment=self.segment)
    #     return None
    #
    # def visualize_image(self, pose=None, img_dir=None, far=8, index=None,
    #                     camera_point=None, target_point=None, **kwargs):
    #     if not isinstance(self.camera, StaticCamera):
    #         self.add_camera()
    #     if pose is not None:
    #         self.camera.set_pose(pose)
    #     if index is None:
    #         index = self.camera.index
    #     if img_dir is not None:
    #         self.img_dir = img_dir
    #     image = self.camera.get_image(segment=self.segment, far=far,
    #                                   camera_point=camera_point, target_point=target_point)
    #     visualize_camera_image(image, index, img_dir=self.img_dir, **kwargs)

    def init_link_joint_relations(self, all_joints, all_links, verbose=False):
        all_link_poses = {(body, _, link): get_link_pose(body, link) for (body, _, link) in all_links}
        for (body, joint) in all_joints:
            position = get_joint_position(body, joint)
            toggle_joint(body, joint)
            new_link_poses = {(body2, _, link): get_link_pose(body, link) for (body2, _, link) in all_links if body == body2}
            changed_links = [k for k, v in new_link_poses.items() if v != all_link_poses[k]]
            if verbose:
                print(f'init_link_joint_relations | {get_joint_name(body, joint)} / {(body, joint)}')
            for body_link in changed_links:
                obj = self.BODY_TO_OBJECT[body_link]
                obj.governing_joints.append((body, joint))
                if verbose:
                    print(f'\t {get_link_name(body_link[0], body_link[-1])} / {body_link}')
            set_joint_position(body, joint, position)

    def get_indices(self):
        """ for fastamp project """
        body_to_name = {str(k): v.lisdf_name for k, v in self.BODY_TO_OBJECT.items()}
        body_to_name[str(self.robot.body)] = self.robot.name
        body_to_name = dict(sorted(body_to_name.items(), key=lambda item: item[0]))
        return body_to_name

    def get_world_fluents(self, obj_poses=None, init_facts=[], objects=None, use_rel_pose=False,
                          cat_to_bodies=None, cat_to_objects=None, verbose=False,
                          only_fluents=False):
        """ if only_fluents = Ture: return only AtPose, AtPosition """

        robot = self.robot
        BODY_TO_OBJECT = self.BODY_TO_OBJECT

        if cat_to_bodies is None:
            def cat_to_bodies(cat):
                ans = self.cat_to_bodies(cat)
                if objects is not None:
                    ans = [obj for obj in ans if obj in set(objects)]
                return ans

        if cat_to_objects is None:
            def cat_to_objects(cat):
                ans = self.cat_to_objects(cat)
                if objects is not None:
                    ans = [obj for obj in ans if obj.body in set(objects)]
                return ans

        graspables = [o.body for o in cat_to_objects('object') if o.category in GRASPABLES]
        graspables = list(set(cat_to_bodies('moveable') + graspables))
        surfaces = list(set(cat_to_bodies('supporter') + cat_to_bodies('surface')))
        spaces = list(set(cat_to_bodies('container') + cat_to_bodies('space')))
        all_supporters = surfaces + spaces
        all_links = [l for l in all_supporters if isinstance(l, tuple)]
        knobs = cat_to_bodies('knob')
        all_joints = cat_to_bodies('drawer') + cat_to_bodies('door') + knobs

        self.init_link_joint_relations(all_joints, all_links, verbose=False)

        # if 'feg' in self.robot.name or True:
        #     use_rel_pose = False
        #     if '@world' in self.constants:
        #         self.constants.remove('@world')

        def get_body_pose(body):
            if obj_poses is None:
                pose = Pose(body, get_pose(body))
            else:  ## in observation
                pose = obj_poses[body]

            for fact in init_facts:
                if fact[0] == 'pose' and fact[1] == body and equal(fact[2].value, pose.value):
                    return fact[2]
            return pose

        def get_body_link_pose(obj):
            body = obj.body
            link = obj.link
            if obj_poses is None:
                joint, position = None, None
                if len(obj.governing_joints) > 0:
                    joint = obj.governing_joints[0][1]
                    position = get_joint_position(body, joint)
                pose = LinkPose(body, value=get_link_pose(body, link), joint=joint, position=position)
            else:  ## in observation
                pose = obj_poses[body]

            for fact in init_facts:
                if fact[0] == 'linkpose' and fact[1] == body and equal(fact[2].value, pose.value):
                    return fact[2]
            return pose

        def get_link_position(body):
            position = Position(body)
            for fact in init_facts:
                if fact[0] == 'position' and fact[1] == body and equal(fact[2].value, position.value):
                    return fact[2]
            return position

        def get_grasp(body, attachment):
            grasp = pr2_grasp(body, attachment.grasp_pose)
            for fact in init_facts:
                if fact[0] == 'grasp' and fact[1] == body and equal(fact[2].value, grasp.value):
                    return fact[2]
            return grasp

        init = []

        ## ---- object joint positions ------------- TODO: may need to add to saver
        for body in all_joints:
            if BODY_TO_OBJECT[body].handle_link is None:
                continue
            if ('Joint', body) in init or ('joint', body) in init:
                continue
            ## initial position
            position = get_link_position(body)  ## Position(body)
            init += [('Joint', body), ('UnattachedJoint', body),
                     ('Position', body, position), ('AtPosition', body, position),
                     ('IsOpenedPosition' if is_joint_open(body) else 'IsClosedPosition', body, position),
                     ('IsJointTo', body, body[0])
                     ]
            if body in knobs:
                controlled = BODY_TO_OBJECT[body].controlled
                if controlled is not None:
                    init += [('ControlledBy', controlled, body)]

        ## ---- object poses / grasps ------------------
        for body in graspables:
            init += [('Graspable', body)]
            pose = get_body_pose(body)

            if body in self.ATTACHMENTS and not isinstance(self.ATTACHMENTS[body], ObjAttachment):
                attachment = self.ATTACHMENTS[body]
                grasp = get_grasp(body, attachment)
                arm = 'hand'
                if get_link_name(robot, attachment.parent_link).startswith('r_'):
                    arm = 'left'
                if get_link_name(robot, attachment.parent_link).startswith('l_'):
                    arm = 'left'
                init.remove(('HandEmpty', arm))
                init += [('Grasp', body, grasp), ('AtGrasp', arm, body, grasp)]

            elif use_rel_pose:
                supporter = BODY_TO_OBJECT[body].supporting_surface
                supporter_body_link = supporter.pybullet_name
                if supporter.link is not None:
                    supporter_pose = get_body_link_pose(supporter)
                    if (isinstance(supporter_body_link, tuple) and len(supporter_body_link) == 3
                            and supporter_pose.joint is not None):
                        body_joint = (supporter.body, supporter_pose.joint)
                        init += [('JointAffectLink', body_joint, supporter_body_link)]
                        init.remove(('UnattachedJoint', body_joint))
                else:
                    supporter_pose = get_body_pose(supporter.body)
                if supporter is None or body not in graspables + all_supporters:
                    supporter = '@world'
                    rel_pose = pose
                else:
                    attachment = self.ATTACHMENTS[body]
                    rel_pose = pose_from_attachment(attachment)
                    # rel_pose_2 = multiply(invert(supporter_pose.value), pose.value)

                init += [(k, body, rel_pose, supporter_body_link, supporter_pose) for k in ['RelPose', 'AtRelPose']]

                if ('Pose', supporter_body_link, supporter_pose) not in init:
                    init += [(k, supporter_body_link, supporter_pose) for k in ('Pose', 'AtPose')]

            else:
                init += [('Pose', body, pose), ('AtPose', body, pose)]

            ## potential places to put on
            for surface in surfaces:
                if self.check_not_stackable(body, surface):
                    continue
                init += [('Stackable', body, surface)]
                if is_placement(body, surface, below_epsilon=0.02) or BODY_TO_OBJECT[surface].is_placement(body):
                    # if is_placement(body, surface, below_epsilon=0.02) != BODY_TO_OBJECT[surface].is_placement(body):
                    #     print('   \n different conclusion about placement', body, surface)
                    #     wait_unlocked()
                    init += [('Supported', body, pose, surface)]

            ## potential places to put in ## TODO: check size
            for space in spaces:
                init += [('Containable', body, space)]
                if is_contained(body, space) or BODY_TO_OBJECT[space].is_contained(body):
                    # if is_contained(body, space) != BODY_TO_OBJECT[space].is_contained(body):
                    #     print('   \n different conclusion about containment', body, space)
                    #     wait_unlocked()
                    if verbose: print('   found contained', self.get_debug_name(body), self.get_debug_name(space))
                    init += [('Contained', body, pose, space)]

        if use_rel_pose:
            wp = Pose('@world', unit_pose())
            init += [('Pose', '@world', wp), ('AtPose', '@world', wp)]

        ## ---- cart poses / grasps ------------------
        for body in cat_to_bodies('steerable'):
            pose = get_body_pose(body)
            init += [('Pose', body, pose), ('AtPose', body, pose)]

            obj = BODY_TO_OBJECT[body]
            for marker in obj.grasp_markers:
                init += [('Marked', body, marker.body)]

        if only_fluents:
            fluents_pred = ['AtPose', 'AtPosition']
            init = [i for i in init if i[0] in fluents_pred]
        return init

    def get_facts(self, conf_saver=None, init_facts=[], obj_poses=None, objects=None,
                  verbose=True, use_rel_pose=True):

        def cat_to_bodies(cat):
            ans = self.cat_to_bodies(cat)
            if objects is not None:
                ans = [obj for obj in ans if obj in set(objects)]
            return ans

        def cat_to_objects(cat):
            ans = self.cat_to_objects(cat)
            if objects is not None:
                ans = [obj for obj in ans if obj.body in set(objects)]
            return ans

        BODY_TO_OBJECT = self.BODY_TO_OBJECT

        set_cost_scale(cost_scale=1)
        init = [Equal(('PickCost',), 1), Equal(('PlaceCost',), 1),
                ('CanMove',), ('CanPull',)]

        ## ---- robot conf ------------------
        init += self.robot.get_init(init_facts=init_facts, conf_saver=conf_saver)

        ## ---- poses, positions, grasps ------------------
        init += self.get_world_fluents(obj_poses, init_facts, objects, use_rel_pose=use_rel_pose,
                                       cat_to_bodies=cat_to_bodies, cat_to_objects=cat_to_objects,
                                       verbose=verbose)

        ## ---- object types -------------
        for cat, objects in self.OBJECTS_BY_CATEGORY.items():
            if cat.lower() == 'moveable': continue
            if cat.lower() in ['edible', 'plate', 'cleaningsurface', 'heatingsurface']:
                objects = self.OBJECTS_BY_CATEGORY[cat]
                init += [(cat, obj.pybullet_name) for obj in objects if obj.pybullet_name in BODY_TO_OBJECT]
                # init += [(cat, obj.pybullet_name) for obj in objects if obj.pybullet_name in BODY_TO_OBJECT]

            for obj in objects:
                if cat in ['space', 'surface'] and (cat, obj.pybullet_name) not in init:
                    init += [(cat, obj.pybullet_name)]
                cat2 = f"@{cat}"
                if cat2 in self.constants:
                    init += [('OfType', obj, cat2)]

        ## --- for testing IK
        # lid = self.name_to_body('braiserlid')
        # surface = self.name_to_body('indigo_tmp')
        # pose = Pose(lid, xyzyaw_to_pose((0.694, 8.694, 0.814, 1.277)))
        # init += [('Pose', lid, pose), ('MagicPose', lid, pose), ('Supported', lid, pose, surface)]

        # ## --- for testing containment
        # egg = self.name_to_body('egg')
        # fridge = self.name_to_body('fridge')
        # init += [('MagicalObj1', egg), ('MagicalObj2', fridge)]

        ## ---- for testing attachment
        init += [self.get_whole_fact(f, init) for f in self.init if f not in init]
        for f in self.init_del:
            f = self.get_whole_fact(f, init)
            if f in init:
                init.remove(f)

        # weiyu debug
        ## ---- clean object -------------
        for obj in self.clean_object:
            init.append(("Cleaned", obj))

        return init

    def add_clean_object(self, obj):
        self.clean_object.add(obj)

    def get_planning_config(self):
        import platform
        import os
        config = {
            'base_limits': self.robot.custom_limits,
            'body_to_name': self.get_indices(),
            'system': platform.system(),
            'host': os.uname()[1]
        }
        config.update(self.planning_config)
        if self.camera is not None:
            t, r = self.camera.pose
            if isinstance(t, np.ndarray):
                if isinstance(t[0], np.ndarray):
                    t = tuple(t[0]), t[1]
                t = tuple(t)
            config['obs_camera_pose'] = (t, r)
        return config

    def save_problem_pddl(self, goal, output_dir, world_name='world_name', init=None, **kwargs):
        from world_builder.world_generator import generate_problem_pddl
        if init is None:
            init = self.get_facts(**kwargs)
        generate_problem_pddl(self, init, goal, world_name=world_name,
                              out_path=join(output_dir, 'problem.pddl'))

    def save_lisdf(self, output_dir, verbose=False, **kwargs):
        from world_builder.world_generator import to_lisdf
        to_lisdf(self, join(output_dir, 'scene.lisdf'), verbose=verbose, **kwargs)

    def save_planning_config(self, output_dir, domain=None, stream=None,
                             pddlstream_kwargs=None):
        from world_builder.world_generator import get_config_from_template
        """ planning related files and params are referred to in template directory """
        config = self.get_planning_config()

        if domain is not None:
            config.update({
                'domain': domain,
                'stream': stream,
            })
        if pddlstream_kwargs is not None:
            config['pddlstream_kwargs'] = {k: str(v) for k, v in pddlstream_kwargs.items()}
        with open(join(output_dir, 'planning_config.json'), 'w') as f:
            json.dump(config, f, indent=4)

    def get_type(self, body):
        return [self.BODY_TO_OBJECT[body].category]

    def find_surfaces_for_placement(self, obj, surfaces, verbose=False):
        from pybullet_tools.pr2_streams import get_stable_gen
        if verbose:
            self.summarize_supporting_surfaces()

        def get_area(surface):
            area = surface.ly
            for o in surface.supported_objects:
                area -= o.ly
            return area

        state = State(self)
        funk = get_stable_gen(state)
        possible = []
        for s in surfaces:
            try:
                p = next(funk(obj, copy.deepcopy(s)))[0]
                possible.append(s)
            except Exception:
                pass
        if verbose:
            print(f'   find {len(possible)} out of {len(surfaces)} surfaces for {obj}', possible)
        possible = sorted(possible, key=get_area, reverse=True)
        return possible

    def save_problem(self, output_dir, goal=None, init=None, domain=None, stream=None, problem=None,
                     pddlstream_kwargs=None, **kwargs):
        world_name = f"{problem}_{basename(output_dir)}"
        if self.note is not None:
            world_name += f"_{self.note}"

        self.save_lisdf(output_dir, world_name=world_name, **kwargs)
        self.save_problem_pddl(goal, output_dir, world_name=world_name, init=init)
        self.save_planning_config(output_dir, domain=domain, stream=stream,
                                  pddlstream_kwargs=pddlstream_kwargs)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.robot)


#######################################################

class State(object):
    def __init__(self, world, objects=[], attachments={}, facts=[], variables={},
                 grasp_types=None, gripper=None,
                 unobserved_objs=None, observation_model=None): ##
        self.world = world
        if len(objects) == 0:
            # objects = [o for o in world.objects if isinstance(o, int)]
            objects = get_bodies()
            objects.remove(world.robot)
        if grasp_types is None:
            grasp_types = world.robot.grasp_types
        self.objects = list(objects)

        self.observation_model = observation_model
        assert observation_model in ['gpt4v', 'exposed', None]
        self.unobserved_objs = unobserved_objs
        if self.observation_model == 'exposed' and unobserved_objs is None:
            self.space_markers = self.initiate_space_markers()
            self.unobserved_objs, self.unobserved_spaces = self.initiate_exposed_observation_model()
            self.spaces = {v['space']: v for v in self.space_markers.values()}
            self.assumed_obj_poses = {}

        if len(attachments) == 0:
            attachments = copy.deepcopy(world.ATTACHMENTS)
        self.attachments = dict(attachments) # TODO: relative pose
        self.facts = list(facts) # TODO: make a set?
        self.variables = defaultdict(lambda: None)
        self.variables.update(variables)
        self.assign()
        self.saver = WorldSaver(bodies=self.bodies)

        ## serve as problem for streams
        self.gripper = gripper
        if grasp_types is None:
            grasp_types = world.robot.grasp_types
        self.grasp_types = grasp_types
        ## allowing both types causes trouble when the AConf used for generating IK isn't the same as the one during execution

    def get_gripper(self, arm=None, visual=True):
        if self.gripper is None:
            self.gripper = self.robot.get_gripper(arm=arm, visual=visual)
        return self.gripper

    def remove_gripper(self):
        if self.gripper is not None:
            remove_body(self.gripper)
            self.gripper = None

    @property
    def robot(self):
        return self.world.robot # TODO: use facts instead

    @property
    def robots(self):
        return [self.world.robot]

    @property
    def bodies(self):
        return [self.robot] + self.objects

    @property
    def regions(self):
        return [obj for obj in self.objects if isinstance(obj, Region)]

    @property
    def floors(self):
        return self.world.floors

    @property
    def fixed(self):   ## or the robot will go through tables
        return self.world.fixed
        # objs = [obj for obj in self.objects if obj not in self.movable]
        # if hasattr(self.world, 'BODY_TO_OBJECT'):  ## some objects are not in planning
        #     objs = [o for o in self.world.objects if o in objs and \
        #             self.world.BODY_TO_OBJECT[o].category != 'floor']
        # return objs
        # return [obj for obj in self.objects if isinstance(obj, Region) or isinstance(obj, Environment)]

    @property
    def movable(self): ## include steerables if want to exclude them when doing base motion plannig
        return self.world.movable
        # return [self.robot] + self.world.cat_to_bodies('moveable') ## + self.world.cat_to_bodies('steerable')
        # return [obj for obj in self.objects if obj not in self.fixed]

    @property
    def obstacles(self):
        return {obj for obj in self.objects + self.world.fixed if obj not in self.regions} \
            - set(self.attachments)

    @property
    def ignored_pairs(self):
        return self.world.ignored_pairs

    def restore(self): # TODO: could extend WorldSaver
        self.saver.restore()

    def scramble(self):
        set_zero_world(self.bodies)

    def copy(self): # __copy__
        return self.new_state()

    def new_state(self, objects=None, attachments=None, facts=None, variables=None, unobserved_objs=None):
        # TODO: could also just update the current state
        if objects is None:
            objects = self.objects
        if attachments is None:
            attachments = self.attachments
        if facts is None:
            facts = self.facts
        if variables is None:
            variables = self.variables
        if unobserved_objs is None:
            unobserved_objs = self.unobserved_objs
        return State(self.world, objects=objects, attachments=attachments, facts=facts,
                     variables=variables, unobserved_objs=unobserved_objs)

    def assign(self):
        # TODO: topological sort
        for attachment in self.attachments.values():
            attachment.assign()
        return self

    def filter_facts(self, predicate): # TODO: predicates
        return [fact[1:] for fact in self.facts if fact[0].lower() == predicate.lower()]

    def apply_action(self, action): # Transition model
        if action is None:
            return self
        if isinstance(action, list):
            print('world.apply action')
        # assert isinstance(action, Action)
        # wait_for_user()
        return action.transition(self.copy())

    #########################################################################################################

    def camera_observation(self, include_rgb=False, include_depth=False, include_segment=False, camera=None):
        if not (include_rgb or include_depth or include_segment):
            return None
        if camera is None:
            if len(self.robot.cameras) > 0:
                [camera] = self.robot.cameras
            else:
                [camera] = self.world.cameras
        rgb, depth, seg, pose, matrix = camera.get_image(
            segment=(self.world.segment or include_segment), segment_links=False)
        if not include_rgb:
            rgb = None
        if not include_depth:
            depth = None
        if not include_segment:
            seg = None
        return CameraImage(rgb, depth, seg, pose, matrix)

    ####################################################################################

    def initiate_space_markers(self):
        return self.world.initiate_space_markers()

    def initiate_exposed_observation_model(self):
        self.world.initiate_exposed_cameras()

        ## get_observed and unobserved objects
        objs = self.get_exposed_observation(show=False)
        unobserved_objs = [b for b in set(get_bodies()) - set(objs)]

        ## find unobserved spaces and objects
        unobserved_spaces = [self.space_markers[b]['space'] for b in unobserved_objs if b in self.space_markers]
        unobserved_objs = [b for b in unobserved_objs if b not in self.space_markers]
        for unobserved, name in [(unobserved_objs, 'objs'), (unobserved_spaces, 'spaces')]:
            printout = [self.world.get_name(b) for b in unobserved]
            print(f'\tstate.initiate_exposed_observation_model | {name}', printout)

        ## add assumed poses for unobserved objects
        assumed_poses = {}
        for body in unobserved_objs:
            from world_builder.loaders_nvidia_kitchen import get_nvidia_kitchen_hacky_pose  ## TODO: hacky poses
            space = random.choice(unobserved_spaces)
            obj = self.world.BODY_TO_OBJECT[body] if body in self.world.BODY_TO_OBJECT \
                else self.world.REMOVED_BODY_TO_OBJECT[body]
            pose = get_nvidia_kitchen_hacky_pose(obj, space)
            if pose is not None:
                assumed_poses[body] = pose

        return assumed_poses, unobserved_spaces

    def get_exposed_observation(self, show=False):
        kwargs = dict(include_rgb=True, include_depth=True, include_segment=True)
        camera_images = []
        for camera in self.world.cameras:
            camera_images.append(self.camera_observation(camera=camera, **kwargs))
        objs = get_objs_in_camera_images(camera_images, world=self.world, show=show)
        objs.sort()
        return objs

    def sample_observation(self, include_conf=False, include_poses=False,
                           include_facts=False, include_variables=False, step=None, observe_visual=True, **kwargs): # Observation model
        # TODO: could technically also not require robot, camera_pose, or camera_matrix
        # TODO: make confs and poses state variables
        # robot_conf = self.robot.get_positions() if include_conf else None
        robot_conf = BodySaver(self.robot) if include_conf else None # TODO: unknown base but known arms
        obj_poses = {obj: get_pose(obj) for obj in self.objects if obj in get_bodies()} if include_poses else None
        facts = list(self.facts) if include_facts else None
        variables = dict(self.variables) if include_variables else None

        image = None
        if self.observation_model == 'gpt4v':
            from vlm_tools.vlm_utils import save_rgb_jpg, query_gpt4v
            kwargs['include_rgb'] = True
            image = self.camera_observation(**kwargs)
            jpg_path = join('observations', f'observation_{step}.jpg')
            save_rgb_jpg(image.rgbPixels, jpg_path=jpg_path)
            response = query_gpt4v(image.rgbPixels, jpg_path=jpg_path)
            wait_for_user()

        if self.observation_model == 'exposed' and observe_visual:
            objs = self.get_exposed_observation()
            obj_poses = {obj: get_pose(obj) for obj in objs}

        return Observation(self, robot_conf=robot_conf, obj_poses=obj_poses, unobserved_objs=self.unobserved_objs,
                           facts=facts, variables=variables, image=image)

    #########################################################################################################

    def get_facts(self, **kwargs):
        init = self.world.get_facts(**kwargs)
        if self.observation_model == 'exposed':
            init = self.modify_exposed_facts(init)

        ## ---- those added to state.variables[label, body]
        for k in self.variables:
            init += [(k[0], k[1])]
        return init

    def modify_exposed_facts(self, init):
        ## remove facts about the poses of unobserved objects
        to_remove = []
        to_add = []
        poses = {}
        for fact in init:
            if fact[0].lower() in ['pose', 'atpose', 'contained', 'supported']:
                obj_body = fact[1] if not isinstance(fact[1], str) else eval(fact[1].split('|')[0])
                if obj_body in self.unobserved_objs:
                    to_remove.append(fact)
                    mod_fact = list(copy.deepcopy(fact))
                    if obj_body not in poses:
                        poses[obj_body] = Pose(obj_body, self.unobserved_objs[obj_body])
                    mod_fact[2] = poses[obj_body]
                    to_add.append(mod_fact)
        init = [f for f in init if f not in to_remove] + to_add
        return init

    # def get_unexposed_spaces(self):
    #     objs = self.get_exposed_observation(show=True)
    #     unexposed_spaces = [spaces[i] for i in range(len(spaces)) if markers[i] not in objs]
    #     print('state.unexposed_spaces', [self.world.get_name(s) for s in unexposed_spaces])
    #     return unexposed_spaces

    def get_planning_config(self):
        return self.world.get_planning_config()

    def get_fluents(self, **kwargs):
        return self.world.get_world_fluents(**kwargs)

    def __repr__(self):
        return '{}{}'.format(self.__class__.__name__, self.objects)

#######################################################


class Outcome(object):
    def __init__(self, collisions=[], violations=[]):
        self.collisions = list(collisions)
        self.violations = list(violations)

    def __len__(self):
        return max(map(len, [self.collisions, self.violations]))

    def __repr__(self):
        return '{}{}'.format(self.__class__.__name__, str_from_object(self.__dict__))


def analyze_outcome(state):
    # TODO: possibly move to State and rename inconsistencies
    robots = {state.robot}
    violations = set()
    for body in robots:
        if not body.within_limits():
            violations.add(body)

    movable = robots | set(state.movable)
    bodies = robots | set(state.obstacles) # TODO: attachments
    collisions = set()
    for body1, body2 in product(movable, bodies):
        if (body1 != body2) and pairwise_collision(body1, body2, max_distance=0.):
            collisions.add(frozenset([body1, body2]))

    return Outcome(collisions=collisions, violations=violations)

#######################################################


class Observation(object):
    # TODO: just update a dictionary for everything
    def __init__(self, state, robot_conf=None, obj_poses=None, unobserved_objs=None,
                 image=None, facts=None, variables=None, collision=False):
        self.state = state
        self.robot_conf = robot_conf
        self.obj_poses = obj_poses
        self.unobserved_objs = unobserved_objs
        self.rgb_image = self.depth_image = self.seg_image = self.camera_matrix = self.camera_pose = None
        if image is not None:
            self.rgb_image, self.depth_image, self.seg_image, self.camera_matrix, self.camera_pose = image
        # self.facts = facts
        self.variables = variables
        self.collision = collision
        # TODO: noisy conf, pose, RGB, and depth observations
        # TODO: map observation

    @property
    def facts(self):
        return self.state.get_facts(conf_saver=self.robot_conf.conf_saver,
                                    obj_poses=self.obj_poses)

    @property
    def objects(self):
        if self.obj_poses is None:
            return None
        return sorted(self.obj_poses) # TODO: attachments

    # @property
    # def regions(self):
    #     if self.objects is None:
    #         return None
    #     return [obj for obj in self.objects if isinstance(obj, Region)]

    @property
    def obstacles(self):
        if self.objects is None:
            return None
        return [obj for obj in self.objects if obj in self.state.obstacles]

    def assign(self): # TODO: rename to update_pybullet
        if self.robot_conf is not None:
            self.robot_conf.restore() # TODO: sore all as Savers instead?
            # self.robot.set_positions(observation.robot_conf)
        if self.obj_poses is not None:
            for obj, pose in self.obj_poses.items():
                if self.state.unobserved_objs is not None and obj in self.state.unobserved_objs:
                    continue
                set_pose(obj, pose)
        return self

    def update_unobserved_objs(self):
        objs = self.obj_poses.keys()
        unobserved_objs = self.unobserved_objs
        self.unobserved_objs = {b: v for b, v in unobserved_objs.items() if b not in objs}

        newly_observed = []
        for obj, pose in self.obj_poses.items():
            if obj in self.unobserved_objs and self.unobserved_objs[obj] != pose:
                newly_observed.append(obj)
                print('world.update_unobserved_objs.newly_observed\t', newly_observed, pose)
        return newly_observed

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.robot_conf, sorted(self.obj_poses))

#######################################################


class Process(object):
    def __init__(self, world, name=None, **kwargs):
        self.world = world
        self.name = name
        self.runtimes = []
        self.outcomes = [] # TODO: outcome visiblity

    @property
    def robot(self):
        return self.world.robot

    @property
    def time_step(self):
        return self.world.time_step

    @property
    def max_velocities(self):
        return self.world.max_velocities

    @property
    def max_delta(self):
        return self.world.max_delta

    @property
    def num_steps(self):
        return len(self.runtimes)

    @property
    def current_time(self):
        return self.time_step*self.num_steps

    def initialize(self, state):
        # TODO: move creation of bodies to agents
        self.state = state ## YANG< HPN
        return state # TODO: return or just update?

    def evolve(self, state, ONCE=False, verbose=False, step=None):
        start_time = time.time()
        # new_state = self.wrapped_transition(state)
        if True:
            new_state = self.wrapped_transition(state, ONCE=ONCE, verbose=verbose, step=step)
            if verbose: print(f'  evolve \ finished wrapped_transition inner in {round(time.time() - start_time, 4)} sec')
        if verbose: print(f'  evolve \ finished wrapped_transition 2 in {round(time.time()-start_time, 4)} sec')

        ## --------- added by YANG to stop simulation if action is None ---------
        if ONCE and new_state is None:
            return None
            # new_state = state
        ## ----------------------------------------------------------------------

        outcome = analyze_outcome(new_state) # TODO: include delta from state (i.e. don't penalize ongoing issues)
        self.outcomes.append(outcome)
        if self.world.prevent_collisions and outcome:
            print(outcome)
            new_state = state.assign()
        self.runtimes.append(elapsed_time(start_time))
        return new_state

    def wrapped_transition(self, state, ONCE=False, verbose=False):
        raise NotImplementedError()

#######################################################


class Exogenous(Process):
    def __init__(self, world, **kwargs):
        super(Exogenous, self).__init__(world, **kwargs)
        self.states = []
    def wrapped_transition(self, state, **kwargs):
        #self.states.append(state) # TODO: before, after, or both
        new_state = self.transition(state.copy())
        if new_state is None:
            new_state = state
        assert isinstance(new_state, State)
        self.states.append(state)
        return new_state
    def transition(self, state): # Operates directly on the state
        raise NotImplementedError()

#######################################################


class Agent(Process): # Decision
    # TODO: make these strings
    requires_conf = requires_poses = requires_facts = requires_variables = \
        requires_rgb = requires_depth = requires_segment = False # requires_cloud

    def __init__(self, world, **kwargs):
        super(Agent, self).__init__(world, **kwargs)
        self.world = world
        self.observations = []
        self.actions = []

    def wrapped_transition(self, state, ONCE=False, verbose=False, **kwargs):
        # TODO: move this to another class
        start_time = time.time()
        observe_visual = len(self.actions) == 0 or 'GripperAction' in str(self.actions[-1])
        observation = state.sample_observation(
            include_conf=self.requires_conf, include_poses=self.requires_poses,
            include_facts=self.requires_facts, include_variables=self.requires_variables,
            include_rgb=self.requires_rgb, include_depth=self.requires_depth,
            include_segment=self.requires_segment, observe_visual=observe_visual, **kwargs)  # include_cloud=self.requires_cloud,
        if verbose: print(f'   wrapped_transition \ made observation in {round(time.time() - start_time, 4)} sec')
        start_time = time.time()
        if self.world.scramble:
            # if not self.requires_conf or self.requires_cloud:
            state.scramble()
        action = self.policy(observation)
        if verbose: print(f'   wrapped_transition \ chosen action in {round(time.time() - start_time, 4)} sec')
        start_time = time.time()
        state.restore()
        self.observations.append(observation)
        self.actions.append(action)
        result = state.apply_action(action)
        if verbose: print(f'   wrapped_transition \ applied action in {round(time.time() - start_time, 4)} sec')

        ## --------- added by YANG to stop simulation if action is None ---------
        if ONCE and action is None: result = None
        ## ----------------------------------------------------------------------
        return result

    def policy(self, observation): # Operates indirectly on the state
        raise NotImplementedError()


#######################################################


def evolve_processes(state, processes=[], max_steps=INF, ONCE=False, verbose=False):
    # TODO: explicitly separate into exogenous and agent?
    world = state.world
    time_step = world.time_step
    parameter = add_parameter(name='Real-time / sim-time', lower=0, upper=5, initial=0)
    button = add_button(name='Pause / Play')
    start_time = time.time()
    current_time = 0.
    for agent in processes:
        state = agent.initialize(state)

    facts = state.facts
    for step in irange(max_steps):
        if verbose:
            print('Step: {} | Current time: {:.3f} | Elapsed time: {:.3f}'.format(step, current_time, elapsed_time(start_time)))

        # TODO: sample nearby and then extend
        for agent in processes:
            state = agent.evolve(state, ONCE=ONCE, verbose=verbose, step=step)

            ## stop simulation if action is None
            if ONCE and state is None:
                return None

        # if verbose: print('state add', [f for f in state.facts if f not in facts])
        # if verbose: print('state del', [f for f in facts if f not in state.facts])

        current_time += time_step
        wait_for_duration(time_step)
    return state
