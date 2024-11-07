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
from pprint import pformat, pprint


from pybullet_tools.utils import get_max_velocities, WorldSaver, elapsed_time, get_pose, unit_pose, \
    euler_from_quat, get_link_name, get_joint_position, joint_from_name, add_button, SEPARATOR, \
    BodySaver, set_pose, INF, add_parameter, irange, wait_for_duration, get_bodies, remove_body, \
    read_parameter, pairwise_collision, str_from_object, get_joint_name, get_name, get_link_pose, \
    get_joints, multiply, invert, is_movable, remove_handles, set_renderer, HideOutput, wait_unlocked, \
    get_movable_joints, apply_alpha, get_all_links, set_color, set_all_color, dump_body, clear_texture, \
    get_link_name, get_aabb, draw_aabb, GREY, GREEN, quat_from_euler, wait_for_user, get_camera_matrix, \
    Euler, PI, get_center_extent, create_box, RED, unit_quat, set_joint_position, get_joint_limits, \
    get_camera_pose
from pybullet_tools.pr2_streams import Position, get_handle_grasp_gen, pr2_grasp
from pybullet_tools.general_streams import pose_from_attachment, LinkPose, RelPose
from pybullet_tools.bullet_utils import set_zero_world, nice, open_joint, summarize_joints, get_point_distance, \
    add_body, close_joint, toggle_joint, check_joint_state, \
    nice, LINK_STR, CAMERA_MATRIX, equal, sort_body_parts, get_root_links, colorize_world, colorize_link, \
    draw_fitted_box, find_closest_match, multiply_quat, is_joint_open, get_merged_aabb, tupify
from pybullet_tools.pose_utils import ObjAttachment, draw_pose2d_path, draw_pose3d_path, xyzyaw_to_pose, \
    is_placement, is_contained, get_learned_yaw
from pybullet_tools.camera_utils import get_pose2d, get_camera_image_at_pose, visualize_camera_image, \
    set_camera_target_body, set_camera_target_body
from pybullet_tools.logging_utils import print_dict, myprint, print_debug, print_pink

from pybullet_tools.pr2_primitives import Pose, Conf, get_ik_ir_gen, get_motion_gen, \
    Attach, Detach, Clean, Cook, control_commands, link_from_name, \
    get_gripper_joints, GripperCommand, apply_commands, State, Command

from world_builder.entities import Region, Location, Robot, Surface, ArticulatedObjectPart, Door, Drawer, \
    Knob, Camera, Object, StaticCamera
from world_builder.world_utils import GRASPABLES, get_objs_in_camera_images, make_camera_collage, \
    get_camera_image, sort_body_indices
from world_builder.init_utils import add_joint_status_facts

DEFAULT_CONSTANTS = ['@movable', '@bottle', '@edible', '@medicine']  ## , '@world'


class WorldBase(object):
    """ parent api for working with planning + replanning architecture """
    def __init__(self, time_step=1e-3, teleport=False, drive=True, prevent_collisions=False,
                 constants=DEFAULT_CONSTANTS, segment=False, use_rel_pose=False):
        ## for world attributes
        self.scramble = False
        self.time_step = time_step
        self.drive = drive
        self.teleport = teleport
        self.prevent_collisions = prevent_collisions
        self.segment = segment

        ## for planning
        self.constants = constants
        self.use_rel_pose = use_rel_pose
        self.body_to_english_name = {}
        self.english_name_to_body = {}

        ## for visualization
        self.handles = []

        ## for data generation
        self.note = None
        self.camera = None
        self.img_dir = None
        self.cameras = []
        self.pickled_scene_and_problem = None

        ## for observation models
        self.observation_cameras = None
        self.space_markers = None

    def cat_to_bodies(self, cat, **kwargs):
        raise NotImplementedError

    def get_name(self, body):
        raise NotImplementedError

    def get_english_name(self, body):
        """ rephrasing joint names for llm communication """
        name = self.get_name(body)
        if body in self.body_to_english_name:
            name = self.body_to_english_name[body]
        else:
            if name is None:
                print(f'self.get_name({body}) is None')
            name = ''.join([c for c in name if not c.isdigit()])
            name = name.replace('#', '')
            if '::' in name:
                body_name, part_name = name.split('::')
                if body_name in part_name:
                    name = part_name
            name = name.replace('::', "'s ")
            name = name.replace('_', ' ').replace('-', ' ')
            self.body_to_english_name[body] = name

        if name not in self.english_name_to_body:
            self.english_name_to_body[name] = body
            self.english_name_to_body[name.replace("'s ", " ")] = body
        return name

    def set_english_names(self, names):
        for name, english_name in names.items():
            body = self.name_to_body(name) if callable(self.name_to_body) else self.name_to_body[name]
            self.body_to_english_name[body] = english_name

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
        from pybullet_tools.camera_utils import visualize_camera_image

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

    ####################################################################

    def set_camera_points(self, front_camera_point, downward_camera_point):
        self.front_camera_point = front_camera_point
        self.downward_camera_point = downward_camera_point

    def initiate_observation_cameras(self):
        if self.observation_cameras is not None:
            return
        ## add cameras facing the front and front-down
        self.observation_cameras = []
        quat_front = (0.5, 0.5, -0.5, -0.5)
        quat_right = multiply_quat(quat_front, quat_from_euler(Euler(yaw=0, pitch=PI / 2, roll=0)))
        quat_left = multiply_quat(quat_front, quat_from_euler(Euler(yaw=0, pitch=-PI / 2, roll=0)))
        quat_front_down = multiply_quat(quat_front, quat_from_euler(Euler(yaw=0, pitch=0, roll=-PI / 4)))
        for pose, name in [
            [(self.front_camera_point, quat_front), 'front_camera'],
            [(self.downward_camera_point, quat_front_down), 'downward_camera']
        ]:
            self.observation_cameras.append(self.add_camera(pose=pose, name=name))

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
    """ api for building world and tamp problem_sets """
    def __init__(self, **kwargs):
        ## conf_noise=None, pose_noise=None, depth_noise=None, action_noise=None,  # TODO: noise model class?
        super().__init__(**kwargs)

        self.name = 'full_world'
        self.robot = None
        self.body_to_name = {}

        self.ROBOT_TO_OBJECT = {}
        self.BODY_TO_OBJECT = {}
        self.OBJECTS_BY_CATEGORY = defaultdict(list)
        self.REMOVED_BODY_TO_OBJECT = {}
        self.REMOVED_OBJECTS_BY_CATEGORY = defaultdict(list)

        self.attachments = {}  ## {child_obj: create_attachment(parent_obj, link_name, child_obj)}
        self.sub_categories = {}
        self.sup_categories = {}
        self.SKIP_JOINTS = False
        self.floorplan = None  ## for saving LISDF
        self.init = []
        self.init_del = []
        self.relevant_objects = defaultdict(list)
        self.articulated_parts = {k: [] for k in ['door', 'drawer', 'knob', 'button']}
        self.changed_joints = []
        self.non_planning_objects = []
        self.not_stackable = defaultdict(list)
        self.not_containable = defaultdict(list)
        self.c_ignored_pairs = []
        self.planning_config = {'camera_zoomins': [], 'camera_poses': {}}
        self.inited_link_joint_relations = False

        ## for speeding up planning
        self.learned_bconf_list_gen = None
        self.learned_pose_list_gen = None
        self.learned_position_list_gen = None
        self.learned_bconf_database = None
        self.learned_pose_database = None
        self.learned_position_database = None

        ## for visualization
        self.path = None
        self.outpath = None
        self.instance_names = {}

        self.clean_object = set()

    def clear_viz(self):
        self.remove_handles()
        self.remove_redundant_bodies()

    def remove_redundant_bodies(self, verbose=False):
        with HideOutput():
            for b in get_bodies():
                if b not in self.BODY_TO_OBJECT and b not in self.ROBOT_TO_OBJECT \
                        and b not in self.non_planning_objects:
                    remove_body(b)
                    if verbose:
                        print('world.removed redundant body', b)

    ## --------------------------------------------------------------------------------------

    def add_not_containable(self, body, space, verbose=False):
        self.not_containable[body].append(space)
        if verbose:
            print(f'world.do_not_contain({body}, {space})')

    def check_not_containable(self, body, space):
        return body in self.not_containable and space in self.not_containable[body]

    def add_not_stackable(self, body, surface, verbose=False):
        if surface is None:
            return
        self.not_stackable[body].append(surface)
        if verbose:
            print(f'world.do_not_stack({body}, {surface})')

    def check_not_stackable(self, body, surface):
        return body in self.not_stackable and surface in self.not_stackable[body]

    def summarize_forbidden_placements(self):
        for dic, name1, name2 in [
            (self.not_stackable, 'world.not_stackable', 'cannot_supported'),
            (self.not_containable, 'world.not_containable', 'connot_contain'),
        ]:
            dic_to_print = defaultdict(list)
            dic_to_print_inv = defaultdict(list)
            for k, v in dic.items():
                key = self.get_name_from_body(k)
                values = [self.get_name_from_body(vv) for vv in v]
                dic_to_print[key] = values
                for vv in values:
                    dic_to_print_inv[vv].append(key)
            print_dict(dic_to_print, name1)
            print_dict(dic_to_print_inv, name2)
            print()
        print()

    ## --------------------------------------------------------------------------------------

    def make_transparent(self, name, transparency=0.5):
        if isinstance(name, str):
            obj = self.name_to_object(name)
            colorize_link(obj.body, obj.link, transparency=transparency)
        elif isinstance(name, tuple) and len(name) == 2:
            obj = self.BODY_TO_OBJECT[name]
            obj.make_links_transparent(transparency)
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
        lst = [k for k in self.BODY_TO_OBJECT.keys() if k not in self.ROBOT_TO_OBJECT]
        return sort_body_indices(lst)

    @property
    def all_objects(self):
        lst = [k for k in self.REMOVED_BODY_TO_OBJECT.keys()]
        return self.objects + sort_body_indices(lst)

    @property
    def movable(self):  ## include steerables if want to exclude them when doing base motion plannig
        return [self.robot] + self.cat_to_bodies('movable')  ## + self.cat_to_bodies('steerable')
        # return [obj for obj in self.objects if obj not in self.fixed]

    @property
    def floors(self):
        return self.cat_to_bodies('floor')

    @property
    def fixed(self):
        objs = [obj for obj in self.objects if not isinstance(obj, tuple)]
        objs += [o for o in self.non_planning_objects if isinstance(o, int) and o not in objs]
        objs = [o for o in objs if o not in self.floors and o not in self.movable]
        # ## remove objects that are attached to a movable planning object
        # def get_attached_body(obj):
        #     return obj.body if hasattr(obj, 'body') else obj
        # attached_objects = {get_attached_body(a.child): get_attached_body(a.parent) for a in self.attachments.values()}
        # objs = [o for o in objs if not (o in attached_objects and attached_objects[o] in self.movable)]
        return sort_body_indices(objs)

    ## -----------------------------------------------------------------------

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

    def del_ignored_pair(self, pair):
        a, b = pair
        for pair in [(a, b), (b, a)]:
            if pair in self.c_ignored_pairs:
                self.c_ignored_pairs.remove(pair)

    def print_ignored_pairs(self):
        print_fn = print_pink

        printed = []
        print_fn('---------------- IGNORED COLLISION PAIRS ------------- ')
        for (a, b) in self.c_ignored_pairs:
            if (b, a) in printed:
                continue
            print_fn(f'\t({self.get_debug_name(a)}, {self.get_debug_name(b)})')
            printed.append((a, b))
        print_fn('--------------------------------------------------------')

    ## -----------------------------------------------------------------------

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
        if use_default_link_name and name == body and len(body) == 3:
            name = get_link_name(body[0], body[-1])
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

    def get_world_aabb(self):
        aabbs = []
        for body in self.BODY_TO_OBJECT:
            aabbs.append(get_aabb(body))
        return get_merged_aabb(aabbs)

    def get_objects_by_type(self, objects=None):
        if objects is None:
            objects = list(self.BODY_TO_OBJECT)
        extra_categories = ['food', 'utensil', 'condiment', 'appliance', 'region', 'button', 'knob']
        result = self.summarize_all_types(return_full=True, categories=extra_categories)
        summary = {}
        for cat, bodies in result.items():
            summary[f"<{cat}>"] = [self.get_english_name(body) for body in bodies if body in objects]
        print_dict(summary, 'objects by category')
        return pformat(summary, indent=3)

    ## ---------------------------------------------------------

    def add_box(self, obj, pose=None):
        obj = self.add_object(obj, pose=pose)
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

        ## be able to find eggs as movables
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
            while self.name_to_body(name, verbose=False) is not None:
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

            self.instance_names[(body, None, obj.handle_link)] = n

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

    def load_saved_grasps(self, body):
        obj = body if isinstance(body, Object) else self.BODY_TO_OBJECT[body]
        return obj.grasps

    def get_whole_fact(self, fact, init):
        if fact[0].lower() in ['isopenposition', 'isclosedposition']:
            fact += [f[2] for f in init if f[0].lower() == 'atposition' and f[1] == fact[1]]
            print('world.get_whole_fact | ', fact)
        return fact

    def add_to_relevant_objects(self, o, o2):
        """ used by remove_bodies_from_planning(), in case object reducer removes important objects """
        self.relevant_objects[o].append(o2)

    def add_to_init(self, fact):
        self.init.append(fact)

    def del_fr_init(self, fact):
        self.init_del.append(fact)

    def add_to_cat(self, body, cat):
        obj = self.get_object(body)
        if obj is None:
            return
        if cat not in self.OBJECTS_BY_CATEGORY:
            self.OBJECTS_BY_CATEGORY[cat] = []
        self.OBJECTS_BY_CATEGORY[cat].append(obj)
        if cat not in obj.categories:
            obj.categories.append(cat)

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

    def remove_all_object_labels_in_pybullet(self):
        for obj in self.get_all_objects():
            obj.erase()

    def summarize_supporting_surfaces(self):
        from pybullet_tools.logging_utils import myprint as print
        return_dict = {}
        print('\n================ summarize_supporting_surfaces ================')
        print(f"\tsurface\t{self.cat_to_objects('surface')}")
        print(f"\tsupporter\t{self.cat_to_objects('supporter')}")
        print('--------------- ')
        for surface in set(self.cat_to_objects('surface') + self.cat_to_objects('supporter')):
            print(f'\t{surface.name}\t{surface.supported_objects}')
            return_dict[surface.name] = [o.name for o in surface.supported_objects]
        print('================================================================\n')
        return return_dict

    def summarize_supported_movables(self):
        from pybullet_tools.logging_utils import myprint as print
        return_dict = {}
        print('\n================ summarize_supported_movables ================')
        print(f"\tmovable\t{self.cat_to_objects('movable')}")
        print('--------------- ')
        for movable in set(self.cat_to_objects('movable')):
            print(f"\t{movable.name}\t{movable.supporting_surface}")
            surface = movable.supporting_surface
            return_dict[movable.name] = surface.name if surface is not None else surface
        print('================================================================')
        return return_dict

    def summarize_attachments(self):
        return {k.body: (v.parent.body, v.parent_link, v.grasp_pose) for k, v in self.attachments.items()}

    def summarize_all_types(self, return_full=False, categories=[]):
        summary = {}
        printout = []
        for typ in ['movable', 'surface', 'space', 'joint', 'door', 'drawer']+categories:
            bodies = self.cat_to_bodies(typ)
            if return_full:
                summary[typ] = bodies
            else:
                num = len(bodies)
                if num > 0:
                    printout.append(f"{typ}({num})")
        if return_full:
            return summary
        return ', '.join(printout)

    def summarize_all_objects(self, print_fn=None, draw_object_labels=False):
        if print_fn is None:
            from pybullet_tools.logging_utils import myprint as print_fn

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

        print_fn('-'*80)
        print_fn(f'PART I: world objects | {self.summarize_all_types()} | obstacles({len(self.fixed)}) = {self.fixed}')
        print_fn('-'*80)

        bodies = [self.robot] + sort_body_parts(BODY_TO_OBJECT.keys())
        bodies += sort_body_parts(REMOVED_BODY_TO_OBJECT.keys(), bodies)
        static_bodies = [b for b in get_bodies() if b not in bodies]
        # bodies += static_bodies
        print_not = print_not_2 = False
        for body in bodies:
            line = ''

            if isinstance(body, Robot):
                body = body.body
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

            line += f'{body}\t  |  {typ_str}: {object.name}\t  |  categories: {object.categories}'

            ## partnet mobility objects
            if hasattr(object, 'mobility_identifier'):
                line += f' | asset: {object.mobility_identifier}'

            ## joints
            if isinstance(body, tuple) and len(body) == 2:
                b, j = body
                pose = get_joint_position(b, j)
                if hasattr(object, 'handle_link') and object.handle_link is not None:
                    line += f'\t|  Handle: {get_link_name(b, object.handle_link)}'
                line += f'\t|  JointLimit: {nice(get_joint_limits(b, j))}'

            ## links
            elif isinstance(body, tuple) and len(body) == 3:
                b, _, l = body
                pose = get_link_pose(b, l)

            ## whole objects
            else:
                pose = get_pose(body)

            line += f"\t|  Pose: {nice(pose)}"

            ## bodies not included in planning
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

            elif draw_object_labels:
                object.draw()

            print_fn(line)

        print_dict(self.attachments, '\nPART II: world attachments')

    def summarize_body_indices(self, print_fn=print):
        print_fn(SEPARATOR+f'Robot: {self.robot} | Objects: {self.objects}\n'
                 f'Movable: {self.movable} | Fixed: {self.fixed} | Floor: {self.floors}'+SEPARATOR)

    def summarize_collisions(self, return_verbose_line=False, title='[world.summarize_collisions]'):
        log = self.robot.get_collisions_log()
        data = {f'[cc]{k}|'+self.body_to_name[str(k)]: v for k, v in log.items() if str(k) in self.body_to_name}
        # data = dict(sorted(data.items(), key=lambda d: d[1], reverse=True))
        line = f'\t{title} = {data}'
        if return_verbose_line:
            return line
        myprint(line)
        return log

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

    def remove_bodies_from_planning(self, goals=[], exceptions=[], skeleton=[], subgoals=[], verbose=False):

        title = "[world.remove_bodies_from_planning]\t"

        ## important for keeping related links and joints for planning
        self.init_link_joint_relations()

        ## for logging and replaying before objects are removed
        self.planning_config.update({
            'supporting_surfaces': self.summarize_supporting_surfaces(),
            'supported_movables': self.summarize_supported_movables(),
            'attachments': self.summarize_attachments(),
            'body_to_name': self.get_indices()
        })

        is_test_goal = False
        if isinstance(goals, tuple):
            goals = [goals]
            is_test_goal = True

        ## consider the objects provided in skeleton and subgoals
        all_tuples = goals + skeleton
        if subgoals is not None:
            all_tuples += subgoals

        ## find all relevant objects mentioned in the goal literals
        bodies = []
        for literal in all_tuples:
            if is_test_goal:
                items = literal[1]
                if items in self.BODY_TO_OBJECT:
                    items = [items]
                elif items is None:
                    continue
            else:
                items = literal[1:]
            for item in items:
                if not isinstance(item, str) and str(item) not in bodies:
                    if isinstance(item, Object):
                        item = item.pybullet_name

                    ## the surface that supports it for `pick_from_supporter`
                    if isinstance(item, int):
                        obj = self.BODY_TO_OBJECT[item]
                        surface = obj.supporting_surface
                        if surface is not None:
                            bodies.append(str(surface.pybullet_name))

                    ## links or joints
                    if isinstance(item, tuple):
                        bodies.append(str(item[0]))
                        ## all links that are affected by joints for `pull_handle_with_link`
                        if len(item) == 2:
                            joint = self.BODY_TO_OBJECT[item]
                            bodies.extend([str(link) for link in joint.affected_links])

                    bodies.append(str(item))

        ## find all relevant objects to the given exceptions
        new_exceptions = []
        for b in exceptions:
            if isinstance(b, Object):
                b = b.pybullet_name
            if isinstance(b, tuple) and str(b[0]) not in new_exceptions:
                new_exceptions.append(str(b[0]))
            new_exceptions.append(str(b))

        ## find hacky relevant objects added by world loader
        ## TODO: fix this with smarter object reducer seq
        hacky_exceptions = []
        # if hasattr(self, 'relevant_objects'):
        #     search_preserved_bodies = new_exceptions + [eval(o) for o in bodies if o.isdigit() or '(' in o]
        #     if verbose:
        #         print(f'{title} self.relevant_objects: {dict(self.relevant_objects)}'
        #               f'\t search_preserved_bodies = {search_preserved_bodies}')
        #     for b in search_preserved_bodies:
        #         objs = self.relevant_objects[b]
        #         if len(objs) > 0:
        #             hacky_exceptions.extend(objs)
        #             if verbose:
        #                 print(f'{title} adding relevant bodies {objs} for preserved body {b}')
        exceptions = new_exceptions + hacky_exceptions

        ## remove all other objects
        all_bodies = list(self.BODY_TO_OBJECT.keys())
        for body in all_bodies:
            if str(body) not in bodies and str(body) not in exceptions:
                self.remove_body_from_planning(body)

        if verbose:
            print('\nworld.remove_bodies_from_planning | exceptions =', exceptions)
            for cat, objs in self.REMOVED_OBJECTS_BY_CATEGORY.items():
                print(f'\t{cat} ({len(objs)}) \t', [f"{obj.name}|{obj.pybullet_name}" for obj in objs])
            print()

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

    def remove_object_categories_by_init(self, init):
        movables = [f[1] for f in init if f[0].lower() == 'graspable']
        surfaces = list(set([f[2] for f in init if f[0].lower() == 'stackable']))
        spaces = list(set([f[2] for f in init if f[0].lower() == 'containable']))
        cats = [('movable', movables), ('surface', surfaces), ('space', spaces)]
        for body, obj in self.BODY_TO_OBJECT.items():
            for cat, lst in cats:
                if cat in obj.categories and cat not in lst:
                    obj.categories.remove(cat)
                    print_debug(f'[world.remove_object_categories_by_init]\tremoving {cat} from object categories of {obj.debug_name}')

    def remove_body_attachment(self, body, verbose=False):
        if isinstance(body, Object):
            obj = body
        else:
            obj = self.BODY_TO_OBJECT[body]
        if obj in self.attachments:
            if verbose:
                print('world.remove_body_attachment\t', self.attachments[obj])
            self.attachments.pop(obj)

    def remove_object(self, object, **kwargs):
        object = self.get_object(object)
        body = object.body

        ## remove all objects initiated by the body
        bodies = self.get_all_obj_in_body(body)
        for b in bodies:
            obj = self.BODY_TO_OBJECT.pop(b)

            # so cat_to_bodies('movable') won't find it
            for cat in obj.categories:
                self.remove_object_from_category(cat, obj)
            if hasattr(obj, 'supporting_surface') and isinstance(obj.supporting_surface, Surface):
                surface = obj.supporting_surface
                surface.supported_objects.remove(obj)

        self.remove_body_attachment(object, **kwargs)
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

    def set_learned_bconf_list_gen(self, list_gen_fn):
        """ likely defined in world_builder/loaders_{DOMAIN}.py """
        self.learned_bconf_list_gen = list_gen_fn

    def set_learned_pose_list_gen(self, pose_list_gen):
        """ likely defined in world_builder/loaders_{DOMAIN}.py """
        self.learned_pose_list_gen = pose_list_gen

    def set_learned_position_list_gen(self, list_gen_fn):
        """ likely defined in world_builder/loaders_{DOMAIN}.py """
        self.learned_position_list_gen = list_gen_fn

    def reset_learned_samplers(self):
        self.learned_bconf_list_gen = None
        self.learned_pose_list_gen = None
        self.learned_position_list_gen = None
        self.learned_bconf_database = None
        self.learned_pose_database = None
        self.learned_position_database = None

    def remove_unpickleble_attributes(self):
        cached_attributes = {
            'learned_pose_list_gen': self.learned_pose_list_gen,
            'learned_bconf_list_gen': self.learned_bconf_list_gen,
            'learned_position_list_gen': self.learned_position_list_gen
        }
        self.reset_learned_samplers()
        self.robot.reset_ik_solvers()
        return cached_attributes

    def recover_unpickleble_attributes(self, cached_attributes):
        self.learned_pose_list_gen = cached_attributes['learned_pose_list_gen']
        self.learned_bconf_list_gen = cached_attributes['learned_bconf_list_gen']
        self.learned_position_list_gen = cached_attributes['learned_position_list_gen']

    ##################################################################################

    def get_all_body_objects(self, include_removed=False):
        all_objects = list(self.ROBOT_TO_OBJECT.items()) + list(self.BODY_TO_OBJECT.items())
        if include_removed:
            all_objects += list(self.REMOVED_BODY_TO_OBJECT.items())
        return all_objects

    def get_all_bodies(self, include_removed=False):
        all_objects = self.get_all_body_objects(include_removed=include_removed)
        return [a[0] for a in all_objects]

    def get_all_objects(self, include_removed=False):
        all_objects = self.get_all_body_objects(include_removed=include_removed)
        return [a[1] for a in all_objects]

    def get_collision_objects(self):
        saved = list(self.BODY_TO_OBJECT.keys())
        return [n for n in get_bodies() if n in saved and n > 1]

    ## ---------------------------------------------------------

    def get_name_from_body(self, body):
        obj = self.body_to_object(body)
        if obj is not None:
            return obj.name
        return None

    def name_to_body(self, name, include_removed=False, verbose=False):
        name = name.lower()
        possible = {}
        all_objects = self.get_all_body_objects(include_removed)
        for body, obj in all_objects:
            if name == obj.name:
                return body
            if name in obj.name:
                possible[body] = obj.name
        if len(possible) >= 1:
            return find_closest_match(possible)
        if verbose:
            print('[world.name_to_body] cannot find name', name)
        return None

    def body_to_object(self, body):
        if body in self.BODY_TO_OBJECT:
            return self.BODY_TO_OBJECT[body]
        if body in self.REMOVED_BODY_TO_OBJECT:
            return self.REMOVED_BODY_TO_OBJECT[body]
        elif body in self.ROBOT_TO_OBJECT:
            return self.ROBOT_TO_OBJECT[body]
        # print(f'world.body_to_object | body {body} not found')
        return None  ## object doesn't exist

    def name_to_object(self, name, **kwargs):
        body = self.name_to_body(name, **kwargs)
        return self.body_to_object(body)

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

        if cat == 'joint':
            filtered_bodies += self.cat_to_bodies('door', get_all) + self.cat_to_bodies('drawer', get_all)
            filtered_bodies = set(filtered_bodies)
        return sort_body_indices(filtered_bodies)

    def cat_to_objects(self, cat):
        bodies = self.cat_to_bodies(cat)
        return [self.BODY_TO_OBJECT[b] for b in bodies]

    ## ---------------------------------------------------------

    def assign_attachment(self, body, tag=None, verbose=False):
        title = f'   world.assign_attachment({body}) | '
        if tag is not None:
            title += f'tag = {tag} | '
        for child, attach in self.attachments.items():
            if attach.parent.body == body:
                pose = get_pose(child)
                attach.assign()
                if verbose and pose != get_pose(child):  ## attachment made a difference
                    print(title, attach, nice(attach.grasp_pose))

    def get_scene_joints(self):
        c = self.cat_to_bodies
        joints = c('door') + c('drawer') + c('knob')
        joints += [bj for bj in self.changed_joints if bj not in joints]
        return joints

    def _change_joint_state(self, body, joint, **kwargs):
        self.assign_attachment(body, **kwargs)
        if (body, joint) not in self.changed_joints:
            self.changed_joints.append((body, joint))

    def toggle_joint(self, body, joint=None):
        if joint is None and isinstance(body, tuple):
            body, joint = body
        toggle_joint(body, joint)
        self._change_joint_state(body, joint)

    def close_joint(self, body, joint=None, **kwargs):
        if joint is None and isinstance(body, tuple):
            body, joint = body
        close_joint(body, joint)
        self._change_joint_state(body, joint, **kwargs)

    def open_joint(self, body, joint=None, extent=1, pstn=None, random_gen=False, verbose=True, **kwargs):
        if joint is None and isinstance(body, tuple):
            body, joint = body
        if random_gen:
            from pybullet_tools.general_streams import sample_joint_position_list_gen
            funk = sample_joint_position_list_gen(State(self))
            pstns = funk((body, joint), Position((body, joint)))
            if len(pstns) == 0:
                return
            pstn = random.choice(pstns)[0].value
        pstn = open_joint(body, joint, extent=extent, pstn=pstn, **kwargs)
        if verbose:
            name = self.body_to_object((body, joint)).name
            print(f'[world.open_joint({body}, {joint}, {round(pstn, 3)})]\tobj name = {name}')
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

    def close_all_doors_drawers(self, **kwargs):
        doors = [(o.body, o.joint) for o in self.cat_to_objects('door')]
        drawers = [(o.body, o.joint) for o in self.cat_to_objects('drawer')]
        for body, joint in doors + drawers:
            self.close_joint(body, joint, **kwargs)

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

    def get_object(self, obj, verbose=False):
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
            if verbose:
                print('[world.object] cannot find object', obj)
            obj = None
        return obj

    def put_on_surface(self, obj, surface='hitman_countertop', max_trial=20, OAO=False, verbose=False, **kwargs):
        """ OAO: one and only (I forgot why I added it here ... it sounds pretty lonely actually) """

        obj = self.get_object(obj)
        surface_obj = self.get_object(surface)
        if surface_obj is None or obj is None:
            print(f'[world.put_on_surface]\tskipping obj={obj} or surface_obj={surface_obj} not found')
            return
        surface = surface_obj.name

        surface_obj.place_obj(obj, max_trial=max_trial, verbose=verbose, **kwargs)

        ## ----------- rules of locate specific objects
        world_to_surface = surface_obj.get_pose()
        point, quat = obj.get_pose()
        x, y, z = point
        if 'faucet_platform' in surface:
            # (a, b, c), quat = world_to_surface
            # obj.set_pose(((a-0.2, b, z), quat), **kwargs)
            if 'faucet' in obj.categories:
                quat = quat_from_euler(Euler(PI, PI, 0))
            obj.set_pose((point, quat), **kwargs)
        elif 'hitman_tmp' in surface or 'hitman_countertop' in surface:
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

        return obj

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

    def init_link_joint_relations(self, all_links=[], all_joints=None, verbose=False):
        """ find whether moving certain joints would change the link poses or spaces and surfaces """
        if self.inited_link_joint_relations:
            return
        self._init_link_joint_relations(all_links, all_joints, verbose)
        self.inited_link_joint_relations = True

    def _init_link_joint_relations(self, all_links=[], all_joints=None, verbose=False):
        ## another dictionary that needs only initiated once
        body_to_name = self.get_indices()

        if all_joints is None:
            all_links, all_joints = self.get_typed_objects()[-2:]
        # added_link_poses = {(body, _, link): get_link_pose(body, link) for (body, _, link) in all_links}

        lines = []
        for (body, joint) in all_joints:
            position = get_joint_position(body, joint)
            all_link_poses = {(body, None, link): get_link_pose(body, link) for link in get_all_links(body)}

            toggle_joint(body, joint)
            joint_obj = self.BODY_TO_OBJECT[(body, joint)]
            new_link_poses = {(body2, _, link): get_link_pose(body, link) for (body2, _, link) in all_link_poses}
            changed_links = [k for k, v in new_link_poses.items() if v != all_link_poses[k]]
            lines.append(f'\tjoint = {get_joint_name(body, joint)}|{(body, joint)}')
            for body_link in changed_links:
                if body_link in all_links:
                    obj = self.BODY_TO_OBJECT[body_link]
                    obj.set_governing_joints([(body, joint)])
                    lines.append(f'\t\tlink = {get_link_name(body_link[0], body_link[-1])}|{body_link}')
                joint_obj.all_affected_links.append(body_link[-1])
            lines.append(f'\t\tall links affected = {changed_links}')
            set_joint_position(body, joint, position)

        if verbose and len(lines) > 0:
            print(f'\ninit_link_joint_relations ... started')
            [print(l) for l in lines]

    def get_indices(self):
        """ for fastamp project """
        body_to_name = {str(k): v.lisdf_name for k, v in self.BODY_TO_OBJECT.items()}
        body_to_name[str(self.robot.body)] = self.robot.name
        body_to_name = dict(sorted(body_to_name.items(), key=lambda item: item[0]))
        self.body_to_name = body_to_name
        return body_to_name

    def get_typed_objects(self, cat_to_bodies=None, cat_to_objects=None, objects=None):
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
        graspables = list(set(cat_to_bodies('movable') + graspables))
        surfaces = list(set(cat_to_bodies('supporter') + cat_to_bodies('surface')))
        spaces = list(set(cat_to_bodies('container') + cat_to_bodies('space')))
        all_supporters = surfaces + spaces
        all_links = [l for l in all_supporters if isinstance(l, tuple)]
        knobs = cat_to_bodies('knob')
        all_joints = cat_to_bodies('drawer') + cat_to_bodies('door') + knobs
        return graspables, surfaces, spaces, knobs, all_links, all_joints

    def get_world_fluents(self, obj_poses=None, joint_positions=None, init_facts=[], objects=None, use_rel_pose=False,
                          cat_to_bodies=None, cat_to_objects=None, verbose=False, only_fluents=False):
        """ if only_fluents = Ture: return only AtPose, AtPosition
        tips: to help with planning, run something like loaders_nvidia_kitchen.prevent_funny_placements()
        """

        robot = self.robot
        BODY_TO_OBJECT = self.BODY_TO_OBJECT

        graspables, surfaces, spaces, knobs, all_links, all_joints = self.get_typed_objects(
            cat_to_bodies=cat_to_bodies, cat_to_objects=cat_to_objects, objects=objects
        )

        self.init_link_joint_relations(all_links, all_joints)

        # if 'feg' in self.robot.name or True:
        #     use_rel_pose = False
        #     if '@world' in self.constants:
        #         self.constants.remove('@world')

        def get_body_pose(body):
            if obj_poses is not None:
                if body in obj_poses:
                    pose = obj_poses[body]
                else:
                    return None  ## sometimes the world objects have changed during agent state loading
            elif isinstance(body, int):
                pose = get_pose(body)
            elif isinstance(body, tuple):
                pose = get_link_pose(body[0], link=body[-1])
            pose = Pose(body, pose)

            for fact in init_facts:
                if fact[0] == 'pose' and fact[1] == body and equal(fact[2].value, pose.value):
                    return fact[2]
            return pose

        def get_body_joint_position(body):
            if joint_positions is None:
                position = get_joint_position(body[0], body[1])
            elif body in joint_positions:
                position = joint_positions[body]
            else:
                return None  ## sometimes the world objects have changed during agent state loading
            position = Position(body, position)
            for fact in init_facts:
                if fact[0] == 'position' and fact[1] == body and equal(fact[2].value, position.value):
                    return fact[2]
            return position

        def get_body_link_pose(obj):
            body = obj.body
            link = obj.link
            body_link = obj.pybullet_name
            link_pose = get_link_pose(body, link)
            if obj_poses is not None and body_link in obj_poses:
                link_pose = obj_poses[body_link]

            ## the link pose is caused by joint position
            joint, position = None, None
            if len(obj.governing_joints) > 0:
                joint = obj.governing_joints[0][1]
                position = get_joint_position(body, joint)

            pose = LinkPose(body_link, value=link_pose, joint=joint, position=position)

            for fact in init_facts:
                if fact[0] == 'pose' and fact[1] == body and equal(fact[2].value, pose.value):
                    return fact[2]
            return pose

        def get_grasp(body, attachment):
            grasp = pr2_grasp(body, attachment.grasp_pose)  ## TODO: wrong
            for fact in init_facts:
                if fact[0] == 'grasp' and fact[1] == body and equal(fact[2].value, grasp.value):
                    return fact[2]
            return grasp

        init = []

        ## ---- object joint positions ------------- TODO: may need to add to saver
        for body in all_joints:
            obj = BODY_TO_OBJECT[body]
            if obj.handle_link is None:
                continue
            if ('Joint', body) in init or ('joint', body) in init:
                continue
            ## initial position
            position = get_body_joint_position(body)
            if position is None:
                continue
            init += [('Joint', body), ('UnattachedJoint', body), ('IsJointTo', body, body[0]),
                     ('Position', body, position), ('AtPosition', body, position),
                     # ('IsOpenedPosition' if is_joint_open(body) else 'IsClosedPosition', body, position),
                     ]
            if 'door' in obj.get_categories():
                init += [('Door', body)]
            init += add_joint_status_facts(body, position, verbose=False)

            if body in knobs:
                controlled = obj.controlled
                if controlled is not None:
                    init += [('ControlledBy', controlled, body)]

        ## ---- surfaces & spaces -------------
        supporter_poses = {}
        if use_rel_pose:
            for body_link in all_links:
                obj = BODY_TO_OBJECT[body_link]
                if isinstance(body_link, tuple) and len(obj.governing_joints) > 0:
                    supporter_pose = get_body_link_pose(obj)
                    supporter_poses[body_link] = supporter_pose
                    init += [(k, body_link, supporter_pose) for k in ('Pose', 'AtPose', 'StartPose')]

                    init.append(('MovableLink', body_link))
                    for body_joint in obj.governing_joints:
                        init += [('JointAffectLink', body_joint, body_link)]
                        if ('UnattachedJoint', body_joint) in init:
                            init.remove(('UnattachedJoint', body_joint))
                else:
                    init.append(('StaticLink', body_link))
        else:
            init.extend([('StaticLink', body_link) for body_link in all_links])
        init.extend([('StaticLink', body) for body in surfaces if body not in all_links])

        ## ---- for pouring & sprinkling ------------------
        regions = cat_to_bodies('region')
        for region in regions:
            pose = get_body_pose(region)
            init += [('Pose', region, pose), ('AtPose', region, pose)]

        ## ---- object poses / grasps ------------------
        for body in graspables:
            init += [('Graspable', body)]
            pose = get_body_pose(body)
            if pose is None:
                continue
            supporter_obj = BODY_TO_OBJECT[body].supporting_surface

            if body in self.attachments and not isinstance(self.attachments[body], ObjAttachment):
                attachment = self.attachments[body]
                grasp = get_grasp(body, attachment)
                arm = 'hand'
                if get_link_name(robot, attachment.parent_link).startswith('r_'):
                    arm = 'right'
                if get_link_name(robot, attachment.parent_link).startswith('l_'):
                    arm = 'left'
                # init += [('Grasp', body, grasp), ('AtGrasp', arm, body, grasp)]

            elif use_rel_pose and supporter_obj is not None and hasattr(supporter_obj, 'governing_joints') \
                    and len(supporter_obj.governing_joints) > 0:

                if body not in graspables + surfaces + spaces:
                    supporter = '@world'
                    supporter_pose = unit_pose()
                    rel_pose = pose

                else:
                    supporter = supporter_obj.pybullet_name
                    # if supporter_obj.link is not None:
                    #     supporter_pose = supporter_poses[supporter]
                    # else:
                    #     supporter_pose = get_body_pose(supporter_obj.body)
                    attachment = self.attachments[body]
                    rel_pose = pose_from_attachment(attachment)
                    # rel_pose_2 = multiply(invert(supporter_pose.value), pose.value)

                init += [(k, body, rel_pose, supporter) for k in ['RelPose', 'AtRelPose']]

            ## graspable may also be region
            elif len([f for f in init if f[0].lower() == 'pose' and f[1] == body]) == 0:
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
                if self.check_not_containable(body, space):
                    continue
                init += [('Containable', body, space)]
                if is_contained(body, space) or BODY_TO_OBJECT[space].is_contained(body):
                    # if is_contained(body, space) != BODY_TO_OBJECT[space].is_contained(body):
                    #     print('   \n different conclusion about containment', body, space)
                    #     wait_unlocked()
                    if verbose: print('   found contained', self.get_debug_name(body), self.get_debug_name(space))
                    init += [('Contained', body, pose, space)]

        # if use_rel_pose:
        #     wp = Pose('@world', unit_pose())
        #     init += [('Pose', '@world', wp), ('AtPose', '@world', wp)]

        ## ---- cart poses / grasps ------------------
        for body in cat_to_bodies('steerable'):
            pose = get_body_pose(body)
            init += [('Pose', body, pose), ('AtPose', body, pose)]

            obj = BODY_TO_OBJECT[body]
            for marker in obj.grasp_markers:
                init += [('Marked', body, marker.body)]

        for body in cat_to_bodies('location'):
            init += [('Location', body)]

        if only_fluents:
            fluents_pred = ['AtPose', 'AtPosition']
            init = [i for i in init if i[0] in fluents_pred]

        return init

    def get_facts(self, conf_saver=None, init_facts=[], obj_poses=None, joint_positions=None, objects=None, verbose=False):

        from pddlstream.language.constants import Equal, AND
        from pddlstream.algorithms.downward import set_cost_scale

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
                ('CanMove',), ('CanPick',), ('CanGraspHandle',)] + \
               [('CanPull', arm) for arm in self.robot.arms]  ## , ('CanUngrasp',)

        ## ---- robot conf ------------------
        init += self.robot.get_init(init_facts=init_facts, conf_saver=conf_saver)

        ## ---- poses, positions, grasps ------------------
        init += self.get_world_fluents(obj_poses, joint_positions, init_facts, objects, use_rel_pose=self.use_rel_pose,
                                       cat_to_bodies=cat_to_bodies, cat_to_objects=cat_to_objects, verbose=verbose)

        ## ---- object types -------------
        for cat, objects in self.OBJECTS_BY_CATEGORY.items():
            if cat.lower() == 'movable': continue
            if cat.lower() in ['edible', 'food', 'plate', 'sprinkler', 'cleaningsurface', 'heatingsurface']:
                objects = self.OBJECTS_BY_CATEGORY[cat]
                init += [(cat, obj.pybullet_name) for obj in objects if obj.pybullet_name in BODY_TO_OBJECT]
                # init += [(cat, obj.pybullet_name) for obj in objects if obj.pybullet_name in BODY_TO_OBJECT]

            for obj in objects:
                if cat in ['space', 'surface'] and (cat, obj.pybullet_name) not in init:
                    init += [(cat, obj.pybullet_name)]
                    if ('Region', obj.pybullet_name) not in init:
                        init += [('Region', obj.pybullet_name)]
                cat2 = f"@{cat}"
                if cat2 in self.constants:
                    init += [('OfType', obj.pybullet_name, cat2)]

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

    ##################################################################

    def save_problem(self, output_dir, goal=None, init=None, domain=None, stream=None, problem=None,
                     pddlstream_kwargs=None, **kwargs):
        if callable(problem):
            problem = problem.__name__
        world_name = f"{problem}_{basename(output_dir)}"
        if self.note is not None:
            world_name += f"_{self.note}"

        if self.pickled_scene_and_problem:
            problem_name, scene_name = self.pickled_scene_and_problem
            self.add_to_planning_config('problem_name', problem_name)
            self.add_to_planning_config('scene_name', scene_name)
        else:
            self.save_lisdf(output_dir, world_name=world_name, **kwargs)
        self.save_problem_pddl(goal, output_dir, world_name=world_name, init=init)
        self.save_planning_config(output_dir, domain=domain, stream=stream,
                                  pddlstream_kwargs=pddlstream_kwargs)

    def save_problem_pddl(self, goal, output_dir, world_name='world_name', init=None, **kwargs):
        from world_builder.world_generator import generate_problem_pddl
        if init is None:
            init = self.get_facts(**kwargs)
        generate_problem_pddl(self, init, goal, world_name=world_name,
                              out_path=join(output_dir, 'problem.pddl'))

    def save_lisdf(self, output_dir, verbose=False, **kwargs):
        from world_builder.world_generator import to_lisdf
        lisdf_file = join(output_dir, 'scene.lisdf')
        # if isfile(lisdf_file):
        #     lisdf_file = lisdf_file.replace('.lisdf', '_new.lisdf')
        #     print(f'[world.save_lisdf]\t found existing scene.lisdf thus saving {lisdf_file}')
        to_lisdf(self, lisdf_file, verbose=verbose, **kwargs)

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

    def get_planning_config(self):
        import platform
        import os
        config = {
            'base_limits': self.robot.custom_limits,
            'system': platform.system(),
            'host': os.uname()[1]
        }
        config.update(self.planning_config)

        ## add camera pose
        pose = self.camera.pose if self.camera is not None else get_camera_pose()
        config['obs_camera_pose'] = tupify(pose)

        return config

    def add_to_planning_config(self, key, value):
        self.planning_config[key] = value

    def get_type(self, body):
        obj = self.BODY_TO_OBJECT[body] if body in self.BODY_TO_OBJECT else self.REMOVED_BODY_TO_OBJECT[body]
        return obj.categories
        # return [obj.category]

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

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.robot)


#######################################################

class State(object):
    def __init__(self, world, objects=[], attachments={}, facts=[], variables={},
                 grasp_types=None, gripper=None, unobserved_objs=None, observation_model=None): ##
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
            attachments = copy.deepcopy(world.attachments)
        self.attachments = attachments
        self.facts = list(facts) # TODO: make a set?
        self.variables = defaultdict(lambda: None)
        self.variables.update(variables)
        self.assign()  ## object attachments
        self.saver = WorldSaver(bodies=self.bodies)

        ## serve as problem for streams
        self.gripper = gripper
        if grasp_types is None:
            grasp_types = world.robot.grasp_types
        self.grasp_types = grasp_types
        ## allowing both types causes trouble when the AConf used for generating IK isn't the same as the one during execution

    def get_gripper(self, arm=None, visual=True):
        ## TODO: currently only one cloned gripper from the first arm, no problem so far
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
        # return [obj for obj in self.objects if isinstance(obj, Region) or isinstance(obj, Location)]

    @property
    def movable(self): ## include steerables if want to exclude them when doing base motion plannig
        return self.world.movable
        # return [self.robot] + self.world.cat_to_bodies('movable') ## + self.world.cat_to_bodies('steerable')
        # return [obj for obj in self.objects if obj not in self.fixed]

    @property
    def obstacles(self):
        return ({obj for obj in self.objects + self.world.fixed if obj not in self.regions and isinstance(obj, int)}
                - set(self.attachments))

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
        for i, attachment in enumerate(self.attachments.values()):
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
        include_segment = self.world.segment or include_segment
        return get_camera_image(camera, include_rgb=include_rgb, include_depth=include_depth, include_segment=include_segment)

    ####################################################################################

    def initiate_space_markers(self):
        return self.world.initiate_space_markers()

    def initiate_exposed_observation_model(self):
        self.world.initiate_observation_cameras()

        ## get_observed and unobserved objects
        objs = self.get_exposed_observation(show=False, save=True)
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

    def get_exposed_observation(self, **kwargs):
        camera_kwargs = dict(include_rgb=True, include_depth=True, include_segment=True)
        camera_images = []
        for camera in self.world.cameras:
            camera_images.append(self.camera_observation(camera=camera, **camera_kwargs))
        objs = get_objs_in_camera_images(camera_images, world=self.world, **kwargs)
        objs.sort()
        return objs

    def save_default_observation(self, **kwargs):
        camera_kwargs = dict(include_rgb=True, include_depth=False, include_segment=False)
        camera_images = []
        for camera in self.world.cameras:
            camera_images.append(self.camera_observation(camera=camera, **camera_kwargs))
        make_camera_collage(camera_images, **kwargs)

    def sample_observation(self, include_conf=False, include_poses=False,
                           include_facts=False, include_variables=False, step=None, observe_visual=True, **kwargs):  # Observation model
        # TODO: could technically also not require robot, camera_pose, or camera_matrix
        # TODO: make confs and poses state variables
        # robot_conf = self.robot.get_positions() if include_conf else None
        robot_conf = BodySaver(self.robot) if include_conf else None # TODO: unknown base but known arms
        obj_poses = {obj: get_pose(obj) for obj in self.objects if obj in get_bodies()} if include_poses else None
        joint_positions = {
            obj: get_joint_position(obj[0], obj[1]) for obj in self.objects if isinstance(obj, tuple) and len(obj) == 2
        } if include_poses else None
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
                           joint_positions=joint_positions, facts=facts, variables=variables, image=image)

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
    def __init__(self, state, robot_conf=None, obj_poses=None, joint_positions=None, unobserved_objs=None,
                 image=None, facts=None, variables=None, collision=False):
        self.state = state
        self.robot_conf = robot_conf
        self.obj_poses = obj_poses
        self.joint_positions = joint_positions
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
        return self.state.get_facts(conf_saver=self.robot_conf.conf_saver, obj_poses=self.obj_poses,
                                    joint_positions=self.joint_positions)

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
        if self.joint_positions is not None:
            for obj, position in self.joint_positions.items():
                if self.state.unobserved_objs is not None and obj in self.state.unobserved_objs:
                    continue
                set_joint_position(obj[0], obj[1], position)
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
        self.state = state ## YANG < HPN
        return state # TODO: return or just update?

    def evolve(self, state, once=False, verbose=False, step=None):
        start_time = time.time()

        new_state = self.wrapped_transition(state, once=once, verbose=verbose, step=step)
        if verbose: print(f'  evolve \ finished wrapped_transition in {round(time.time() - start_time, 4)} sec')

        ## --------- added by YANG to stop simulation if action is None ---------
        if once and new_state is None:
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

    def wrapped_transition(self, state, once=False, verbose=False):
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

    def wrapped_transition(self, state, once=False, verbose=False, **kwargs):
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
        if once and action is None: result = None
        ## ----------------------------------------------------------------------
        return result

    def policy(self, observation): # Operates indirectly on the state
        raise NotImplementedError()


#######################################################


def evolve_processes(state, processes=[], max_steps=INF, once=False, verbose=False):
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
            state = agent.evolve(state, once=once, verbose=verbose, step=step)

            ## stop simulation if action is None
            if once and state is None:
                return None

        # if verbose: print('state add', [f for f in state.facts if f not in facts])
        # if verbose: print('state del', [f for f in facts if f not in state.facts])

        current_time += time_step
        wait_for_duration(time_step)
    return state
