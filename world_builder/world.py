import sys
import time
from itertools import product
from collections import defaultdict
import copy
import shutil
from os.path import join, isdir, abspath, basename, isfile
import os
import json
import numpy as np

from pddlstream.language.constants import Equal, AND
from pddlstream.algorithms.downward import set_cost_scale

from pybullet_tools.utils import get_max_velocities, WorldSaver, elapsed_time, get_pose, unit_pose, \
    CameraImage, euler_from_quat, get_link_name, get_joint_position, joint_from_name, \
    BodySaver, set_pose, INF, add_parameter, irange, wait_for_duration, get_bodies, remove_body, \
    read_parameter, pairwise_collision, str_from_object, get_joint_name, get_name, get_link_pose, \
    get_joints, multiply, invert, is_movable, remove_handles, set_renderer, HideOutput, wait_unlocked, \
    get_movable_joints, apply_alpha, get_all_links, set_color, set_all_color, dump_body, clear_texture, get_link_name
from pybullet_tools.pr2_streams import Position, get_handle_grasp_gen, pr2_grasp
from pybullet_tools.general_streams import pose_from_attachment
from pybullet_tools.bullet_utils import set_zero_world, nice, open_joint, get_pose2d, summarize_joints, get_point_distance, \
    is_placement, is_contained, add_body, close_joint, toggle_joint, ObjAttachment, check_joint_state, \
    set_camera_target_body, xyzyaw_to_pose, nice, LINK_STR, CAMERA_MATRIX, visualize_camera_image, equal, \
    draw_pose2d_path, draw_pose3d_path, sort_body_parts, get_root_links, colorize_world, colorize_link, \
    draw_fitted_box
from pybullet_tools.pr2_primitives import Pose, Conf, get_ik_ir_gen, get_motion_gen, \
    Attach, Detach, Clean, Cook, control_commands, link_from_name, \
    get_gripper_joints, GripperCommand, apply_commands, State, Command

from .entities import Region, Environment, Robot, Surface, ArticulatedObjectPart, Door, Drawer, Knob, \
    Camera, Object, StaticCamera
from world_builder.utils import GRASPABLES


class World(object):
    """ api for building world and tamp problems """
    def __init__(self, time_step=1e-3, prevent_collisions=False, camera=True, segment=False,
                 teleport=False, drive=True,
                 conf_noise=None, pose_noise=None, depth_noise=None, action_noise=None): # TODO: noise model class?
        # self.args = args
        self.time_step = time_step
        self.prevent_collisions = prevent_collisions
        self.scramble = False
        self.camera = camera
        self.segment = segment
        self.teleport = teleport
        self.drive = drive

        self.BODY_TO_OBJECT = {}
        self.ROBOT_TO_OBJECT = {}
        self.OBJECTS_BY_CATEGORY = {}
        self.ATTACHMENTS = {}
        self.sub_categories = {}
        self.sup_categories = {}
        self.SKIP_JOINTS = False
        self.cameras = []
        self.floorplan = None  ## for saving LISDF
        self.init = []
        self.init_del = []
        self.articulated_parts = {k: [] for k in ['door', 'drawer', 'knob', 'button']}
        self.REMOVED_BODY_TO_OBJECT = {}
        self.non_planning_objects = []
        self.not_stackable = {}

        ## for visualization
        self.handles = []
        self.path = None
        self.outpath = None
        self.camera = None
        self.instance_names = {}
        self.constants = ['@movable', '@bottle', '@edible', '@world']

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

    def remove_handles(self):
        remove_handles(self.handles)

    def add_handles(self, handles):
        self.handles.extend(handles)

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
        elif isinstance(name, int):
            colorize_link(name, transparency=transparency)
        elif isinstance(name, tuple):
            colorize_link(name[0], name[-1], transparency=transparency)
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
    def objects(self):
        return [k for k in self.BODY_TO_OBJECT.keys() if k not in self.ROBOT_TO_OBJECT]

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
        objs = [o for o in objs if o not in self.floors and o not in self.movable]
        objs += [o for o in self.non_planning_objects if isinstance(o, int) and o not in objs]
        return objs

    @property
    def ignored_pairs(self):
        found = []
        if self.floorplan is not None and 'kitchen' in self.floorplan:
            a = self.cat_to_bodies('counter')[0]
            b = self.cat_to_bodies('oven')[0]
            found.extend([(a, b), (b, a)])

        # plate = self.name_to_body('plate')
        # if plate is not None:
        #     found.append([(plate, self.robot.body), (self.robot.body, plate)])
        return found

    def get_name(self, body):
        if body in self.BODY_TO_OBJECT:
            return self.BODY_TO_OBJECT[body].name
        return None

    def get_debug_name(self, body):
        """ for viewing pleasure :) """
        if isinstance(body, Object):
            return body.debug_name
        if body in self.BODY_TO_OBJECT:
            return self.BODY_TO_OBJECT[body].debug_name
        return None

    def get_lisdf_name(self, body):
        """ for recording objects in lisdf files generated """
        if body in self.BODY_TO_OBJECT:
            return self.BODY_TO_OBJECT[body].lisdf_name
        return None

    def get_instance_name(self, body):
        """ for looking up objects in the grasp database """
        if isinstance(body, tuple) and body in self.instance_names:
            return self.instance_names[body]
        elif body in self.BODY_TO_OBJECT:
            return self.BODY_TO_OBJECT[body].instance_name
        return None

    def get_events(self, body):
        return self.BODY_TO_OBJECT[body].events

    def add_box(self, object, pose=None):
        obj = self.add_object(object, pose=pose)
        obj.is_box = True
        return obj

    def add_highlighter(self, body):
        draw_fitted_box(body)
        # set_all_color(body, (1, 0, 0, 1))

    def add_object(self, object, pose=None):
        OBJECTS_BY_CATEGORY = self.OBJECTS_BY_CATEGORY
        BODY_TO_OBJECT = self.BODY_TO_OBJECT
        category = object.category
        name = object.name
        body = object.body
        joint = object.joint
        link = object.link
        class_name = object.__class__.__name__.lower()

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
        object.name = name

        OBJECTS_BY_CATEGORY[category].append(object)

        ## -------------- different types of object --------------
        ## object parts: doors, drawers, knobs
        if joint is not None:
            BODY_TO_OBJECT[(body, joint)] = object
            object.name = f"{BODY_TO_OBJECT[body].name}{LINK_STR}{object.name}"
            from lisdf_tools.lisdf_loader import PART_INSTANCE_NAME
            n = self.get_instance_name(body)
            part_name = get_link_name(body, object.handle_link)
            n = PART_INSTANCE_NAME.format(body_instance_name=n, part_name=part_name)
            self.instance_names[(body, object.handle_link)] = n

        ## object parts: surface, space
        elif link is not None:
            BODY_TO_OBJECT[(body, None, link)] = object
            object.name = f"{BODY_TO_OBJECT[body].name}{LINK_STR}{object.name}"
            if category == 'surface':
                BODY_TO_OBJECT[body].surfaces.append(link)
            if category == 'space':
                BODY_TO_OBJECT[body].spaces.append(link)

        ## object
        elif not isinstance(object, Robot):
            BODY_TO_OBJECT[body] = object
            self.get_doors_drawers(object.body)

        ## robot
        else:
            self.ROBOT_TO_OBJECT[body] = object

        if pose is not None:
            add_body(object, pose)

        object.world = self
        return object

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

    def get_doors_drawers(self, body):
        obj = self.BODY_TO_OBJECT[body]
        if obj.doors is not None:
            return obj.doors, obj.drawers, obj.knobs

        doors = []
        drawers = []
        knobs = []
        if not self.SKIP_JOINTS:
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
            elif isinstance(body, tuple) and len(body) == 3:
                b, _, l = body
                pose = get_link_pose(b, l)
            else:
                pose = get_pose(body)
            line += f"\t|  Pose: {nice(pose)}"

            if body in REMOVED_BODY_TO_OBJECT:
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

    def remove_bodies_from_planning(self, goals):
        bodies = []
        for literal in goals:
            for item in literal:
                if not isinstance(item, str) and item not in bodies:
                    if isinstance(item, Object):
                        item = item.pybullet_name
                    bodies.append(item)
                    if isinstance(item, tuple):
                        bodies.append(item[0])
        all_bodies = list(self.BODY_TO_OBJECT.keys())
        for body in all_bodies:
            if body not in bodies:
                self.remove_body_from_planning(body)

    def remove_body_from_planning(self, body):
        if body is None: return
        bodies = self.get_all_obj_in_body(body)
        for body in bodies:
            if body not in self.BODY_TO_OBJECT:
                continue
            category = self.BODY_TO_OBJECT[body].category
            obj = self.BODY_TO_OBJECT.pop(body)
            self.OBJECTS_BY_CATEGORY[category] = [
                o for o in self.OBJECTS_BY_CATEGORY[category] if not
                (o.body == obj.body and o.link == obj.link and o.joint == obj.joint)
            ]
            self.REMOVED_BODY_TO_OBJECT[body] = obj
            self.non_planning_objects.append(body)

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
                self.OBJECTS_BY_CATEGORY[cat] = [
                    o for o in self.OBJECTS_BY_CATEGORY[cat] if not
                    (o.body == obj.body and o.link == obj.link and o.joint == obj.joint)
                ]
            if hasattr(obj, 'supporting_surface') and isinstance(obj.supporting_surface, Surface):
                surface = obj.supporting_surface
                surface.supported_objects.remove(obj)

        if object in self.ATTACHMENTS:
            self.ATTACHMENTS.pop(object)
        remove_body(body)

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
                possible[body] = obj
        if len(possible) >= 1:
            counts = {b: len(o.name) for b, o in possible.items()}
            counts = dict(sorted(counts.items(), key=lambda item: item[1]))
            return list(counts.keys())[0]
            # return possible[0]
        return None

    def name_to_object(self, name):
        if self.name_to_body(name) == None:
            return name  ## object doesn't exist
        return self.BODY_TO_OBJECT[self.name_to_body(name)]

    def cat_to_bodies(self, cat):
        bodies = []
        objects = []
        if cat in self.OBJECTS_BY_CATEGORY:
            objects.extend(self.OBJECTS_BY_CATEGORY[cat])
        if cat in self.sub_categories:
            for c in self.sub_categories[cat]:
                objects.extend(self.OBJECTS_BY_CATEGORY[c])
        for o in objects:
            if o.link != None:
                bodies.append((o.body, o.joint, o.link))
            elif o.joint != None:
                bodies.append((o.body, o.joint))
            else:
                bodies.append(o.body)
        filtered_bodies = []
        for b in set(bodies):
            if b in self.BODY_TO_OBJECT:
                filtered_bodies += [b]
            # else:
            #     print(f'   world.cat_to_bodies | category {cat} found {b}')
        return filtered_bodies

    def cat_to_objects(self, cat):
        bodies = self.cat_to_bodies(cat)
        return [self.BODY_TO_OBJECT[b] for b in bodies]

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

    def toggle_joint(self, body, joint):
        toggle_joint(body, joint)
        self.assign_attachment(body)

    def close_joint(self, body, joint):
        close_joint(body, joint)
        self.assign_attachment(body)

    def open_joint(self, body, joint, extent=1, pstn=None):
        open_joint(body, joint, extent=extent, pstn=pstn)
        self.assign_attachment(body)

    def open_doors_drawers(self, body, ADD_JOINT=True):
        doors, drawers, knobs = self.get_doors_drawers(body, SKIP=False)
        for joint in doors + drawers:
            self.open_joint(body, joint, extent=1)
            if not ADD_JOINT:
                self.remove_object(joint)

    def close_doors_drawers(self, body, ADD_JOINT=True):
        doors, drawers, knobs = self.get_doors_drawers(body, SKIP=False)
        for joint in doors + drawers:
            self.close_joint(body, joint)
            if not ADD_JOINT:
                self.remove_object(joint)

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
        return obj

    def put_on_surface(self, obj, surface='hitman_tmp', max_trial=20, OAO=False):
        obj = self.get_object(obj)
        surface_obj = self.get_object(surface)
        surface = surface_obj.name

        surface_obj.place_obj(obj, max_trial=max_trial)

        ## ----------- rules of locate specific objects
        world_to_surface = surface_obj.get_pose()
        point, quat = obj.get_pose()
        x, y, z = point
        if 'faucet_platform' in surface:
            (a, b, c), quat = world_to_surface
            obj.set_pose(((a - 0.2, b, z), quat))
        elif 'hitman_tmp' in surface:
            quat = (0, 0, 1, 0)  ## facing out
            obj.set_pose(((0.4, 6.4, z), quat))
        elif obj.category in ['microwave', 'toaster']:
            quat = (0, 0, 1, 0)  ## facing out
            obj.set_pose((point, quat))

        ## ---------- reachability hacks for PR2
        if hasattr(self, 'robot') and 'pr2' in self.robot.name:

            ## hack to be closer to edge
            if 'shelf' in surface:
                surface_to_obj = ((-0.2, 0, -0.2), (0, 0, 1, 0))
                (a, b, _), _ = multiply(world_to_surface, surface_to_obj)
                obj.set_pose(((a, b, z), quat))
                # obj.set_pose(((1, 4.4, z), quat))
                # obj.set_pose(((1.6, 4.5, z), quat)) ## vertical orientation
            elif 'tmp' in surface: ## egg
                if y > 9: y = 8.9
                obj.set_pose(((0.7, y, z), quat))

        ## ---------- center object
        if 'braiser_bottom' in surface:  ## for testing
            (a, b, c), _ = world_to_surface
            obj.set_pose(((0.55, b, z), (0, 0, 0.36488663206619243, 0.9310519565198234)))
        elif 'braiser' in surface:
            (a, b, c), quat = world_to_surface
            obj.set_pose(((a, b, z), quat))
        elif 'front_' in surface and '_stove' in surface:
            obj.set_pose(((0.55, y, z), quat))

        surface_obj.attach_obj(obj)
        if OAO: ## one and only
            self.remove_body_from_planning(self.name_to_body(surface))

    def put_in_space(self, obj, space='hitman_drawer_top', xyzyaw=None, learned=True):
        container = self.name_to_object(space)
        if learned:
            ## one possible pose put into hitman_drawer_top
            pose = {'hitman_drawer_top': ((1, 7.5, 0.7), (0, 0, 0.3, 0.95)),
                    'indigo_drawer_top': ((0.75, 8.9, 0.7), (0, 0, 0.3, 0.95))}[space]
            xyzyaw = list(pose[0])
            xyzyaw.append(euler_from_quat(pose[1])[-1])
            ## xyzyaw = (1.093, 7.088, 0.696, 2.8)
        container.place_obj(obj, xyzyaw, max_trial=1)
        container.attach_obj(obj)

    def refine_marker_obstacles(self, marker, obstacles):
        ## for steerables
        parent = self.BODY_TO_OBJECT[marker].grasp_parent
        if parent is not None and parent in obstacles:
            obstacles.remove(parent)
        return obstacles

    def add_camera(self, pose=unit_pose(), img_dir=join('visualizations', 'camera_images')):
        camera = StaticCamera(pose, camera_matrix=CAMERA_MATRIX, max_depth=6)
        self.cameras.append(camera)
        self.camera = camera
        self.img_dir = img_dir
        if self.camera:
            return self.cameras[-1].get_image(segment=self.segment)
        return None

    def visualize_image(self, pose=None, img_dir=None, far=8, index=None,
                        camera_point=None, target_point=None, **kwargs):
        if not isinstance(self.camera, StaticCamera):
            self.add_camera()
        if pose is not None:
            self.camera.set_pose(pose)
        if index is None:
            index = self.camera.index
        if img_dir is not None:
            self.img_dir = img_dir
        image = self.camera.get_image(segment=self.segment, far=far,
                                      camera_point=camera_point, target_point=target_point)
        visualize_camera_image(image, index, img_dir=self.img_dir, **kwargs)

    def get_indices(self):
        """ for fastamp project """
        body_to_name = {str(k): v.lisdf_name for k, v in self.BODY_TO_OBJECT.items()}
        body_to_name[str(self.robot.body)] = self.robot.name
        body_to_name = dict(sorted(body_to_name.items(), key=lambda item: item[0]))
        return body_to_name

    def get_facts(self, init_facts=[], conf_saver=None, obj_poses=None, verbose=True,
                  use_rel_pose=True, objects=None):
        def cat_to_bodies(cat):
            ans = self.cat_to_bodies(cat)
            if objects is not None:
                ans = [obj for obj in ans if obj in objects]
            return ans
        def cat_to_objects(cat):
            ans = self.cat_to_objects(cat)
            if objects is not None:
                ans = [obj for obj in ans if obj.body in objects]
            return ans
        robot = self.robot.body
        name_to_body = self.name_to_body
        BODY_TO_OBJECT = self.BODY_TO_OBJECT

        if 'feg' in self.robot.name or True:
            use_rel_pose = False
            if '@world' in self.constants:
                self.constants.remove('@world')

        def get_body_pose(body):
            if obj_poses == None:
                pose = Pose(body, get_pose(body))
            else:  ## in observation
                pose = obj_poses[body]

            for fact in init_facts:
                if fact[0] == 'pose' and fact[1] == body and equal(fact[2].value, pose.value):
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

        set_cost_scale(cost_scale=1)
        init = [Equal(('PickCost',), 1), Equal(('PlaceCost',), 1),
                ('CanMove',), ('CanPull',)]

        ## ---- robot conf ------------------
        init += self.robot.get_init(init_facts=init_facts, conf_saver=conf_saver)

        ## ---- object poses / grasps ------------------
        graspables = [o.body for o in cat_to_objects('object') if o.category in GRASPABLES]
        graspables = set(cat_to_bodies('moveable') + graspables)
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
                if supporter is None or supporter.body not in graspables:
                    supporter = '@world'
                    rel_pose = pose
                else:
                    attachment = self.ATTACHMENTS[body]
                    rel_pose = pose_from_attachment(attachment)

                init += [('RelPose', body, rel_pose, supporter), ('AtRelPose', body, rel_pose, supporter)]

            else:
                init += [('Pose', body, pose), ('AtPose', body, pose)]

            ## potential places to put on
            for surface in cat_to_bodies('supporter') + cat_to_bodies('surface'):
                if self.check_not_stackable(body, surface):
                    continue
                init += [('Stackable', body, surface)]
                if is_placement(body, surface, below_epsilon=0.02) or \
                        BODY_TO_OBJECT[surface].is_placement(body):
                    # if is_placement(body, surface, below_epsilon=0.02) != BODY_TO_OBJECT[surface].is_placement(body):
                    #     print('   \n different conclusion about placement', body, surface)
                    #     wait_unlocked()
                    init += [('Supported', body, pose, surface)]

            ## potential places to put in ## TODO: check size
            for space in cat_to_bodies('container') + cat_to_bodies('space'):
                init += [('Containable', body, space)]
                if is_contained(body, space) or BODY_TO_OBJECT[space].is_contained(body):
                    # if is_contained(body, space) != BODY_TO_OBJECT[space].is_contained(body):
                    #     print('   \n different conclusion about containment', body, space)
                    #     wait_unlocked()
                    if verbose: print('   found contained', body, space)
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

        ## ---- object joint positions ------------- TODO: may need to add to saver
        knobs = cat_to_bodies('knob')
        for body in cat_to_bodies('drawer') + cat_to_bodies('door') + knobs:
            if BODY_TO_OBJECT[body].handle_link is None:
                continue
            if ('Joint', body) in init or ('joint', body) in init:
                continue
            ## initial position
            position = get_link_position(body)  ## Position(body)
            init += [('Joint', body),
                     ('Position', body, position), ('AtPosition', body, position),
                     ('IsClosedPosition', body, position),
                     ('IsJointTo', body, body[0])
                     ]
            if body in knobs:
                controlled = BODY_TO_OBJECT[body].controlled
                if controlled is not None:
                    init += [('ControlledBy', controlled, body)]

        ## ---- object types -------------
        for cat in self.OBJECTS_BY_CATEGORY:
            if cat.lower() == 'moveable': continue
            if cat in ['CleaningSurface', 'HeatingSurface', 'edible']:
                objects = self.OBJECTS_BY_CATEGORY[cat]
                init += [(cat, obj.pybullet_name) for obj in objects if obj.pybullet_name in BODY_TO_OBJECT]
            else:
                for obj in cat_to_bodies(cat):
                    if (cat, obj) not in init:
                        init += [(cat, obj)]
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

        return init

    def get_planning_config(self):
        import platform
        config = {
            'base_limits': self.robot.custom_limits,
            'body_to_name': self.get_indices(),
            'system': platform.system()
        }
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

    def save_planning_config(self, output_dir, template_dir=None, domain=None, stream=None, problem=None):
        from world_builder.world_generator import get_config_from_template
        """ planning related files and params are referred to in template directory """
        config = self.get_planning_config()

        if template_dir is not None:
            config.update(get_config_from_template(template_dir))
            config.update({
                'domain_full': abspath(join(template_dir, 'domain_full.pddl')),
                'domain': abspath(join(template_dir, 'domain.pddl')),
                'stream': abspath(join(template_dir, 'stream.pddl')),
            })
        elif domain is not None:
            config.update({
                'domain': domain,
                'stream': stream,
            })
        with open(join(output_dir, 'planning_config.json'), 'w') as f:
            json.dump(config, f, indent=4)

    def save_test_case(self, goal, output_dir, template_dir=None, init=None,
                       domain=None, stream=None, problem=None,
                       save_rgb=False, save_depth=False, **kwargs):
        if not isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        world_name = basename(output_dir)

        self.save_lisdf(output_dir, world_name=world_name, **kwargs)
        self.save_problem_pddl(goal, output_dir, world_name=world_name, init=init)
        self.save_planning_config(output_dir, template_dir=template_dir,
                                  domain=domain, stream=stream, problem=problem)

        ## move other log files:
        for suffix in ['log.txt', 'commands.pkl', 'time.json']:
            if isfile(f"{output_dir}_{suffix}"):
                shutil.move(f"{output_dir}_{suffix}", join(output_dir, suffix))

        ## save the end image
        if save_rgb:
            self.visualize_image(img_dir=output_dir, rgb=True)
        if save_depth:
            self.visualize_image(img_dir=output_dir)

    def get_type(self, body):
        return [self.BODY_TO_OBJECT[body].category]

    # def get_scale(self, ):

    @property
    def max_delta(self):
        return self.max_velocities * self.time_step

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.robot)


#######################################################

class State(object):
    def __init__(self, world, objects=[], attachments={}, facts=[],
                 variables={}, grasp_types=['top']):
        self.world = world
        if len(objects) == 0:
            # objects = [o for o in world.objects if isinstance(o, int)]
            objects = get_bodies()
            objects.remove(world.robot)
        self.objects = list(objects)

        if len(attachments) == 0:
            attachments = copy.deepcopy(world.ATTACHMENTS)
        self.attachments = dict(attachments) # TODO: relative pose
        self.facts = list(facts) # TODO: make a set?
        self.variables = defaultdict(lambda: None)
        self.variables.update(variables)
        self.assign()
        self.saver = WorldSaver(bodies=self.bodies)

        ## serve as problem for streams
        self.gripper = None
        self.grasp_types = grasp_types ##, 'side']
        ## allowing both types causes trouble when the AConf used for generating IK isn't the same as the one during execution

    def get_gripper(self, arm='left', visual=True):
        if self.gripper is None:
            self.gripper = self.robot.create_gripper(arm=arm, visual=visual)
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
        objs = [obj for obj in self.objects if obj not in self.movable]
        if hasattr(self.world, 'BODY_TO_OBJECT'):  ## some objects are not in planning
            objs = [o for o in self.world.objects if o in objs and \
                    self.world.BODY_TO_OBJECT[o].category != 'floor']
        return objs
        # return [obj for obj in self.objects if isinstance(obj, Region) or isinstance(obj, Environment)]
    @property
    def movable(self): ## include steerables if want to exclude them when doing base motion plannig
        return self.world.movable
        # return [self.robot] + self.world.cat_to_bodies('moveable') ## + self.world.cat_to_bodies('steerable')
        # return [obj for obj in self.objects if obj not in self.fixed]
    @property
    def obstacles(self):
        return {obj for obj in self.objects if obj not in self.regions} - set(self.attachments)
    @property
    def ignored_pairs(self):
        return self.world.ignored_pairs
    def restore(self): # TODO: could extend WorldSaver
        self.saver.restore()
    def scramble(self):
        set_zero_world(self.bodies)
    def copy(self): # __copy__
        return State(self.world, objects=self.objects, attachments=self.attachments,
                     facts=self.facts, variables=self.variables) # TODO: use instead of new_state
    def new_state(self, objects=None, attachments=None, facts=None, variables=None):
        # TODO: could also just update the current state
        if objects is None:
            objects = self.objects
        if attachments is None:
            attachments = self.attachments
        if facts is None:
            facts = self.facts
        if variables is None:
            variables = self.variables
        return State(self.world, objects=objects, attachments=attachments, facts=facts, variables=variables)
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
        return action.transition(self.copy())
    def camera_observation(self, include_rgb=False, include_depth=False, include_segment=False):
        if not (self.world.amera or include_rgb or include_depth or include_segment):
            return None
        [camera] = self.robot.cameras
        rgb, depth, seg, pose, matrix = camera.get_image(
            segment=(self.world.segment or include_segment), segment_links=False)
        if not include_rgb:
            rgb = None
        if not include_depth:
            depth = None
        if not include_segment:
            seg = None
        return CameraImage(rgb, depth, seg, pose, matrix)
    def sample_observation(self, include_conf=False, include_poses=False,
                           include_facts=False, include_variables=False, **kwargs): # Observation model
        # TODO: could technically also not require robot, camera_pose, or camera_matrix
        # TODO: make confs and poses state variables
        #robot_conf = self.robot.get_positions() if include_conf else None
        robot_conf = BodySaver(self.robot) if include_conf else None # TODO: unknown base but known arms
        obj_poses = {obj: get_pose(obj) for obj in self.objects if obj in get_bodies()} if include_poses else None
        facts = list(self.facts) if include_facts else None
        variables = dict(self.variables) if include_variables else None
        image = None  ##self.camera_observation(**kwargs)
        return Observation(self, robot_conf=robot_conf, obj_poses=obj_poses,
                           facts=facts, variables=variables, image=image)

    def get_facts(self, **kwargs):
        init = self.world.get_facts(**kwargs)

        ## ---- those added to state.variables[label, body]
        for k in self.variables:
            init += [(k[0], k[1])]
        return init

    def get_planning_config(self):
        return self.world.get_planning_config()

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
    def __init__(self, state, robot_conf=None, obj_poses=None, image=None, facts=None, variables=None, collision=False):
        self.state = state
        self.robot_conf = robot_conf
        self.obj_poses = obj_poses
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
            #self.robot.set_positions(observation.robot_conf)
        if self.obj_poses is not None:
            for obj, pose in self.obj_poses.items():
                set_pose(obj, pose)
        return self
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
    def evolve(self, state, ONCE=False, verbose=False):
        start_time = time.time()
        # new_state = self.wrapped_transition(state)
        if True:
            new_state = self.wrapped_transition(state, ONCE=ONCE, verbose=verbose)
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
    def wrapped_transition(self, state, ONCE=False, verbose=False):
        # TODO: move this to another class
        start_time = time.time()
        observation = state.sample_observation(
            include_conf=self.requires_conf, include_poses=self.requires_poses,
            include_facts=self.requires_facts, include_variables=self.requires_variables,
            include_rgb=self.requires_rgb, include_depth=self.requires_depth,
            include_segment=self.requires_segment)  # include_cloud=self.requires_cloud,
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
        if ONCE and action == None: result = None
        ## ----------------------------------------------------------------------
        return result
    def policy(self, observation): # Operates indirectly on the state
        raise NotImplementedError()

#######################################################

def evolve_processes(state, processes=[], max_steps=INF, ONCE=False, verbose=False):
    # TODO: explicitly separate into exogenous and agent?
    world = state.world
    time_step = world.time_step
    # parameter = add_parameter(name='Real-time / sim-time', lower=0, upper=5, initial=0)
    # button = add_button(name='Pause / Play')
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
            state = agent.evolve(state, ONCE=ONCE, verbose=verbose)

            ## --------- added by YANG to stop simulation if action is None ---------
            if ONCE and state == None: return None
            ## ----------------------------------------------------------------------------

        # if verbose: print('state add', [f for f in state.facts if f not in facts])
        # if verbose: print('state del', [f for f in facts if f not in state.facts])

        current_time += time_step
        # wait_for_duration(read_parameter(parameter) * time_step)
        #wait_if_gui()
    return state
