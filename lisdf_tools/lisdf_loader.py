import os
import shutil
import sys
from os import listdir
from os.path import join, abspath, dirname, isdir, isfile
sys.path.extend(['lisdf'])
from lisdf.parsing.sdf_j import load_sdf
from lisdf.components.model import URDFInclude
import numpy as np
import json
import copy

import warnings
warnings.filterwarnings('ignore')
import pybullet as p

from pybullet_tools.pr2_problems import create_floor
from pybullet_tools.pr2_utils import set_group_conf, get_group_joints, get_viewcone_base
from pybullet_tools.utils import remove_handles, remove_body, get_bodies, remove_body, get_links, \
    clone_body, get_joint_limits, ConfSaver, load_pybullet, connect, wait_if_gui, HideOutput, invert, \
    disconnect, set_pose, set_joint_position, joint_from_name, quat_from_euler, draw_pose, unit_pose, \
    set_camera_pose, set_camera_pose2, get_pose, get_joint_position, get_link_pose, get_link_name, \
    set_joint_positions, get_links, get_joints, get_joint_name, get_body_name, link_from_name, \
    parent_joint_from_link, set_color, dump_body, RED, YELLOW, GREEN, BLUE, GREY, BLACK, read, get_client, \
    reset_simulation, get_movable_joints, JOINT_TYPES, get_joint_type, is_movable, get_camera_matrix, \
    wait_unlocked
from pybullet_tools.bullet_utils import nice, sort_body_parts, equal, clone_body_link, \
    toggle_joint, get_door_links, set_camera_target_body, colorize_world, colorize_link, find_closest_match, \
    is_box_entity, summarize_facts
from pybullet_tools.pr2_streams import get_handle_link
from pybullet_tools.flying_gripper_utils import set_se3_conf

from pddlstream.language.constants import AND, PDDLProblem

from world_builder.entities import Space
from world_builder.robot_builders import create_pr2_robot, create_gripper_robot
from world_builder.utils import get_instance_name, get_camera_zoom_in, get_lisdf_name, get_mobility_id, \
    get_mobility_category, get_mobility_identifier

from lisdf_tools.lisdf_planning import pddl_to_init_goal

ASSET_PATH = join(dirname(__file__), '..', '..', 'assets')
LINK_COLORS = ['#c0392b', '#d35400', '#f39c12', '#16a085', '#27ae60',
               '#2980b9', '#8e44ad', '#2c3e50', '#95a5a6']
LINK_COLORS = [RED, YELLOW, GREEN, BLUE, GREY, BLACK]

LINK_STR = '::'
PART_INSTANCE_NAME = "{body_instance_name}" + LINK_STR + "{part_name}"


class World():
    def __init__(self, lisdf=None):
        self.lisdf = lisdf
        self.body_to_name = {}
        self.instance_names = {}
        self.paths = {}
        self.mobility_ids = {}
        self.mobility_identifiers = {}
        self.name_to_body = {}
        self.ATTACHMENTS = {}
        self.camera = None
        self.robot = None
        self.movable = None
        self.fixed = []
        self.floors = None
        self.body_types = {}

        ## for visualization
        self.handles = []
        self.cameras = []
        self.colored_links = []
        self.camera_kwargs = {}
        self.run_dir = None

    def clear_viz(self):
        self.remove_handles()
        self.remove_redundant_bodies()

    def remove_redundant_bodies(self):
        for b in get_bodies():
            if b not in self.body_to_name:
                remove_body(b)

    def remove_handles(self):
        pass
        # remove_handles(self.handles)

    def add_handles(self, handles):
        self.handles.extend(handles)

    def make_doors_transparent(self, transparency=0.5):
        if len(self.fixed) == 0:
            self.check_world_obstacles()
        self.colored_links = colorize_world(self.fixed, transparency)

    def add_body(self, body, name, instance_name=None, path=None, verbose=False):
        if body is None:
            print('lisdf_loader.body = None')
        if instance_name is None:
            if isinstance(body, tuple):
                instance_name = self.get_part_instance_name(body)
            elif path is not None:
                instance_name = get_instance_name(path)
        self.name_to_body[name] = body
        self.body_to_name[body] = name
        id = body
        if isinstance(body, tuple) and len(body) == 2:
            id = (body[0], get_handle_link(body))

        ## the id is here is either body or (body, link)
        self.paths[id] = path
        mobility_id = get_mobility_id(path)
        mobility_identifier = get_mobility_identifier(path)
        if mobility_id is None and isinstance(body, tuple):
            mobility_id = self.mobility_ids[id[0]]
            mobility_identifier = self.mobility_identifiers[id[0]]
        self.instance_names[id] = instance_name
        self.mobility_ids[id] = mobility_id
        self.mobility_identifiers[id] = mobility_identifier
        if verbose:
            print(f'lisdf_loader.add_body(name={name}, body={body}, '
                  f'mobility_id={mobility_id}, instance_name={instance_name})', )

    def add_robot(self, robot, name='robot', **kwargs):
        if not isinstance(robot, int):
            name = robot.name
        self.add_body(robot.body, name)
        self.robot = robot

    def add_joints(self, body, body_joints, color_handle=False, **kwargs):
        idx = 0
        for body_joint, joint_name in body_joints.items():
            self.add_body(body_joint, joint_name, **kwargs)
            handle_link = get_handle_link(body_joint)
            color = LINK_COLORS[idx] if color_handle else None
            set_color(body, color=color, link=handle_link)
            idx += 1

    def add_spaces(self, body, body_links, color_link=False, **kwargs):
        idx = 0
        for body_link, link_name in body_links.items():
            self.add_body(body_link, link_name, **kwargs)
            color = LINK_COLORS[idx] if color_link else None
            set_color(body, color=color, link=body_link[-1])
            idx += 1

    def get_part_instance_name(self, id):
        n = self.get_instance_name(id[0])
        if len(id) == 2:
            l = get_handle_link(id)
            part_name = get_link_name(id[0], l)
        else:  ## if len(id) == 3:
            part_name = get_link_name(id[0], id[-1])
        return PART_INSTANCE_NAME.format(body_instance_name=n, part_name=part_name)

    def update_objects(self, objects):
        for o in objects.values():
            if LINK_STR in o.name:
                body = self.name_to_body[o.name.split(LINK_STR)[0]]
                id = find_id(body, o.name)
                self.add_body(id, o.name)
            # if o.name in ['pr2', 'feg']:
            #     self.add_robot(id, o.name)

    # def update_robot(self, domain_name):
    #     if 'pr2' in domain_name:
    #         self.robot = 'pr2'
    #     elif 'fe' in domain_name:
    #         self.robot = 'feg'

    def safely_get_body_from_name(self, name, all_possible=False):
        if not all_possible:
            if name in self.name_to_body:
                return self.name_to_body[name]
            elif name[:-1] in self.name_to_body:  ## 'pr20
                return self.name_to_body[name[:-1]]
        possible = {}
        for n, body in self.name_to_body.items():
            if name in n:
                possible[body] = n
        return find_closest_match(possible, all_possible=all_possible)

    def make_transparent(self, obj, transparency=0.5, verbose=False,
                               remove_upper_furnitures=False):
        def color_obj(obj):
            if isinstance(obj, int):
                links = get_links(obj)
                if len(links) == 0:
                    links = [-1]
            else:
                links = [obj[1]]
                obj = obj[0]
            for l in links:
                if verbose: print('coloring', self.body_to_name[obj], '\t', obj, l)
                colorize_link(obj, l, transparency=transparency)

        if isinstance(obj, str):
            obj = self.safely_get_body_from_name(obj, all_possible=True)

        if obj is not None:
            oo = obj if isinstance(obj, list) else [obj]
            for o in oo:
                if transparency == 0 and remove_upper_furnitures:
                    self.remove_object(o)
                else:
                    color_obj(o)

    def check_world_obstacles(self):
        if self.lisdf is None:
            return
        fixed = []
        movable = []
        floors = []
        ignored_pairs = []
        for model in self.lisdf.models:
            body = self.safely_get_body_from_name(model.name)
            if 'pr2' not in model.name and 'feg' not in model.name:
                if model.static: fixed.append(body)
                else: movable.append(body)
            if model.name in ['cabinettop', 'cabinettop#1']:
                filler_body = self.safely_get_body_from_name(model.name.replace('cabinettop', 'cabinettop_filler'))
                ignored_pairs.extend([(body, filler_body), (filler_body, body)])
            if hasattr(model, 'links'):
                for link in model.links:
                    if link.name == 'box':
                        for collision in link.collisions:
                            if collision.shape.size[-1] < 0.05:
                                floors.append(body)
        self.fixed = [f for f in fixed if f not in floors]
        self.movable = movable
        self.floors = floors
        self.ignored_pairs = ignored_pairs

    def summarize_all_types(self, init=None, return_string=True):
        if init is None: return ''
        printout = ''
        results = {}
        for typ in ['graspable', 'surface', 'space', 'door', 'drawer', 'joint']:
            bodies = self.cat_to_bodies(typ, init)
            results[typ] = bodies
            num = len(bodies)
            if typ == 'graspable':
                typ = 'moveable'
            if num > 0:
                printout += f"{typ}({num}) = {bodies}, "
        if return_string:
            return printout
        return results

    def cat_to_bodies(self, cat, init=[]):
        found = [f[1] for f in init if f[0].lower() == cat]
        if cat == 'surface':
            maybe = set([f[2] for f in init if f[0] == 'stackable'])
            found += [f for f in maybe if f not in found]
        return found

    def summarize_all_objects(self, init=None, print_fn=None):
        """ call this after pddl_to_init_goal() where world.update_objects() happens """
        if print_fn is None:
            from pybullet_tools.logging import myprint as print_fn

        self.check_world_obstacles()
        ob = [n for n in self.fixed if n not in self.floors]

        objects = []
        if init is not None:
            for f in init:
                for elem in f:
                    if elem in self.body_to_name and elem not in objects:
                        objects.append(elem)
                if len(f) == 2 and f[1] in self.body_to_name:
                    typ, body = f
                    name = self.body_to_name[body]
                    if body not in self.body_types:
                        self.body_types[body] = []
                        if 'cabinet::link' in name or 'minifridge::link' in name:
                            self.body_types[body].append('storage')
                    self.body_types[body].append(typ)
        self.planning_objects = objects

        print_fn('----------------------------------------------------------------------------------')
        print_fn(f'PART I: world objects | {self.summarize_all_types(init)} | obstacles({len(ob)}) = {ob}')
        print_fn('----------------------------------------------------------------------------------')

        sorted_body_parts = sort_body_parts(self.body_to_name.keys())
        body_parts = [b for b in sorted_body_parts if b in self.planning_objects]
        body_parts += [b for b in sorted_body_parts if b not in self.planning_objects]

        drew_line = False
        for body in body_parts:
            if body not in self.planning_objects and not drew_line:
                print_fn('----------------------------------------------------------------------------------')
                drew_line = True
            name = self.body_to_name[body]
            line = f'{body}\t  |  {name}'
            if body in self.body_types:
                line += f'  |  Types: {self.body_types[body]}'
            if isinstance(body, tuple) and len(body) == 2:
                body, joint = body
                pose = get_joint_position(body, joint)
            elif isinstance(body, tuple) and len(body) == 3:
                body, _, link = body
                pose = get_link_pose(body, link)
            elif self.get_name(body) in ['pr2']:
                pose = get_group_joints(body, 'base')
            else:
                pose = get_pose(body)
            print_fn(f"{line}\t|  Pose: {nice(pose)}")
        print_fn('----------------')

    def summarize_facts(self, init, **kwargs):
        self.init = init
        summarize_facts(init, world=self, **kwargs)

    def get_planning_objects(self):
        return [self.body_to_name[b] for b in self.planning_objects]

    def get_non_planning_objects(self):
        return [self.body_to_name[b] for b in self.body_to_name if b not in self.planning_objects]

    def get_world_fluents(self, only_fluents=False):
        return [f for f in self.init if f[0] in ['atposition', 'atpose']]

    def get_wconf(self, world_index=None, attachments={}):
        """ similar to to_lisdf in world_generator.py """
        wconf = {}
        bodies = copy.deepcopy(get_bodies())
        bodies.sort()
        for body in bodies:
            name = self.body_to_name[body]
            pose = get_pose(body)

            joint_state = {}
            for joint in get_movable_joints(body):
                joint_name = get_joint_name(body, joint)
                position = get_joint_position(body, joint)
                joint_state[joint_name] = position

            wconf[name] = {
                'pose': pose,
                'joint_state': joint_state,
            }
        wconf = {f"w{world_index}_{k}": v for k, v in wconf.items()}
        return wconf

    def get_type(self, body):
        if body in self.body_types:
            return self.body_types[body]
        return []

    def get_category(self, body):
        return self.get_type(body)

    def get_name(self, body):
        if body in self.body_to_name:
            return self.body_to_name[body]
        return None

    def get_debug_name(self, body):
        return f"{self.get_name(body)}|{body}"

    def get_instance_name(self, body):
        if body in self.instance_names:
            return self.instance_names[body]
        return None

    def get_mobility_id(self, body):
        """ e.g. 10797 """
        if is_box_entity(body):
            self.mobility_ids[body] = 'box'
        if body in self.mobility_ids:
            return self.mobility_ids[body]
        return None

    def get_mobility_category(self, body):
        """ e.g. MiniFridge """
        if body in self.paths and self.paths[body] is not None:
            return get_mobility_category(self.paths[body])
        return None

    def get_mobility_identifier(self, body):
        """ e.g. MiniFridge/10797 """
        if is_box_entity(body):
            self.mobility_identifiers[body] = 'box'
        if body in self.mobility_identifiers:
            return self.mobility_identifiers[body]
        return None

    def get_path(self, body):
        if body in self.paths:
            return self.paths[body]
        return None

    def get_lisdf_name(self, pybullet_name, joint=None, link=None):
        if isinstance(pybullet_name, int):
            return self.get_name(pybullet_name)
        elif len(pybullet_name) == 2:
            body, joint = pybullet_name
        elif len(pybullet_name) == 3:
            body, _, link = pybullet_name
        return get_lisdf_name(body, self.get_name(body), joint=joint, link=link)

    # def get_full_name(self, body_id):
    #     """ concatenated string for links and joints,
    #         e.g. fridge::fridge_door (joint), fridge::door_body (body)
    #     """
    #     if len(body_id) == 2:
    #         body, joint = body_id
    #         return LINK_STR.join([get_body_name(body), get_joint_name(body, joint)])
    #     if len(body_id) == 3:
    #         body, _, link = body_id
    #         return LINK_STR.join([get_body_name(body), get_link_name(body, link)])
    #     return None

    def get_events(self, body):
        pass

    def get_indices(self, **kwargs):
        from mamao_tools.data_utils import get_indices
        return get_indices(self.run_dir, **kwargs)

    def add_camera(self, pose=unit_pose(), img_dir=join('visualizations', 'camera_images'),
                   width=640, height=480, fx=400, **kwargs):
        from world_builder.entities import StaticCamera

        # camera_matrix = get_camera_matrix(width=width, height=height, fx=525., fy=525.)
        camera_matrix = get_camera_matrix(width=width, height=height, fx=fx)
        camera = StaticCamera(pose, camera_matrix=camera_matrix, **kwargs)
        self.cameras.append(camera)
        self.camera = camera
        self.img_dir = img_dir

    def visualize_image(self, pose=None, img_dir=None, index=None,
                        image=None, segment=False, segment_links=False,
                        camera_point=None, target_point=None, **kwargs):
        from pybullet_tools.bullet_utils import visualize_camera_image

        if pose is not None:
            self.camera.set_pose(pose)
        if img_dir is not None:
            self.img_dir = img_dir
        if index is None:
            index = self.camera.index
        if image is None:
            image = self.camera.get_image(segment=segment, segment_links=segment_links,
                                          camera_point=camera_point, target_point=target_point)
        visualize_camera_image(image, index, img_dir=self.img_dir, **kwargs)

    def add_joints_by_keyword(self, body_name, joint_name=None):
        body = self.name_to_body[body_name]
        joints = [j for j in get_joints(body) if is_movable(body, j)]
        if joint_name is not None:
            joints = [j for j in joints if joint_name in get_joint_name(body, j)]
        for j in joints:
            self.add_body((body, j), get_joint_name(body, j))
        return [(body, j) for j in joints]

    def open_all_doors(self):
        for body in self.body_to_name:
            if isinstance(body, tuple) and len(body) == 2:
                body, joint = body
                toggle_joint(body, joint)

    def remove_object(self, body):
        name = self.body_to_name.pop(body)
        self.name_to_body.pop(name)
        remove_body(body)


def find_id(body, full_name):
    name = full_name.split(LINK_STR)[1]
    for joint in get_joints(body):
        if get_joint_name(body, joint) == name:
            return (body, joint)
    for link in get_links(body):
        if get_link_name(body, link) == name:
            return (body, None, link)
    print(f'\n\n\nlisdf_loader.find_id | whats {name} in {full_name} ({body})\n\n')
    return None


def change_world_state(world, test_case):
    title = f'lisdf_loader.change_world_state | '
    lisdf_path = join(test_case, 'scene.lisdf')

    lisdf_world = load_sdf(lisdf_path).worlds[0]
    ## may be changes in joint positions
    model_states = {}
    if len(lisdf_world.states) > 0:
        model_states = lisdf_world.states[0].model_states
        model_states = {s.name: s for s in model_states}

    print('-------------------')
    for model in lisdf_world.models:
        if isinstance(model, URDFInclude):
            category = model.content.name
        else:
            category = model.links[0].name
        body = world.name_to_body(model.name)

        ## set pose of body using PyBullet tools' data structure
        if category not in ['pr2', 'feg']:
            pose = (tuple(model.pose.pos), quat_from_euler(model.pose.rpy))
            old = get_pose(body)
            if not equal(old, pose):
                set_pose(body, pose)
                print(f'{title} change pose of {model.name} from {nice(old)} to {nice(pose)}')
                obj = world.BODY_TO_OBJECT[body]
                if obj in world.ATTACHMENTS:
                    parent = world.ATTACHMENTS[obj].parent
                    if isinstance(parent, Space):
                        parent.include_and_attach(obj)
                    else:
                        parent.include_and_attach(obj)
                    world.assign_attachment(parent)

        if model.name in model_states:
            for js in model_states[model.name].joint_states:
                j = joint_from_name(body, js.name)
                position = js.axis_states[0].value
                old = get_joint_position(body, j)
                if not equal(old, position, epsilon=0.001):
                    set_joint_position(body, j, position)
                    print(f'{title} change {model.name}::{js.name} joint position from {nice(old)} to {nice(position)}')
    print('-------------------')


def get_custom_limits(config_path):
    bl = json.load(open(config_path, 'r'))['base_limits']
    if isinstance(bl, dict):
        custom_limits = {int(k): v for k, v in bl.items()}
    else:
        custom_limits = {i: v for i, v in enumerate(bl)}
    return custom_limits


def load_lisdf_pybullet(lisdf_path, verbose=False, use_gui=True, jointless=False,
                        width=1980, height=1238, transparent=True, larger_world=False):

    ## sometimes another lisdf name is given
    if lisdf_path.endswith('.lisdf'):
        lisdf_dir = dirname(lisdf_path)
    else:
        lisdf_dir = lisdf_path
        lisdf_path = join(lisdf_dir, 'scene.lisdf')

    replay_scene = lisdf_path.replace('.lisdf', '_replay.lisdf')
    if isfile(replay_scene):
        lisdf_path = replay_scene

    ## tmp path for putting sdf files, e.g. floor
    tmp_path = join(lisdf_dir, 'tmp')
    if not isdir(tmp_path): os.mkdir(tmp_path)

    ## get custom base limits for robots
    config_path = join(lisdf_dir, 'planning_config.json')
    custom_limits = {}
    if isfile(config_path):
        custom_limits = get_custom_limits(config_path)

    ## --- the floor and pose will become extra bodies
    connect(use_gui=use_gui, shadows=False, width=width, height=height)
    # draw_pose(unit_pose(), length=1.)
    # create_floor()

    world = load_sdf(lisdf_path).worlds[0]
    bullet_world = World(world)

    ## may be changes in joint positions
    model_states = {}
    if len(world.states) > 0:
        model_states = world.states[0].model_states
        model_states = {s.name: s for s in model_states}

    for model in world.models:
        scale = 1
        if isinstance(model, URDFInclude):
            # uri = model.uri.replace('/home/yang/Documents/cognitive-architectures/bullet/', '../../')
            uri = join(ASSET_PATH, 'models', model.uri)
            scale = model.scale_1d
            category = model.content.name
        else:
            uri = join(tmp_path, f'{model.name}.sdf')
            ## TODO: when solving in parallel causes problems
            # if isfile(uri):
            #     os.remove(uri)
            with open(uri, 'w') as f:
                f.write(make_sdf_world(model.to_sdf()))
            category = model.links[0].name

        if verbose:
            print(f'..... loading {model.name} from {abspath(uri)}') ## , end="\r"
        if not isdir(join(ASSET_PATH, 'scenes')):
            os.mkdir(join(ASSET_PATH, 'scenes'))
        with HideOutput():
            body = load_pybullet(uri, scale=scale)
            if isinstance(body, tuple): body = body[0]

        ## instance names are unique strings used to identify object models
        uri = abspath(uri)
        instance_name = get_instance_name(uri)
        if category == 'box' and instance_name is None:
            size = ','.join([str(n) for n in model.links[0].collisions[0].shape.size.round(4)])
            instance_name = f"box({size})"

        ## set pose of body using PyBullet tools' data structure
        if category in ['pr2', 'feg']:
            pose = model.pose.pos
            if category == 'pr2':
                create_pr2_robot(bullet_world, base_q=pose, custom_limits=custom_limits, robot=body)
            elif category == 'feg':
                robot = create_gripper_robot(bullet_world, custom_limits=custom_limits, robot=body)
        else:
            pose = (tuple(model.pose.pos), quat_from_euler(model.pose.rpy))
            set_pose(body, pose)
            bullet_world.add_body(body, model.name, instance_name=instance_name, path=uri)

        if not jointless and model.name in model_states:
            for js in model_states[model.name].joint_states:
                position = js.axis_states[0].value
                set_joint_position(body, joint_from_name(body, js.name), position)

    ## load objects transparent
    if ('test_full_kitchen' in world.name or 'None_' in world.name) and transparent:
        make_furniture_transparent(bullet_world, lisdf_dir, lower_tpy=0.5, upper_tpy=0.2)

    if world.gui is not None:
        camera_pose = world.gui.camera.pose
        ## when camera pose is not saved for generating training data
        if not np.all(camera_pose.pos == 0):
            set_camera_pose2((camera_pose.pos, camera_pose.quat_xyzw))

    planning_config = join(lisdf_dir, 'planning_config.json')
    if isfile(planning_config):
        config = json.load(open(planning_config, 'r'))

        ## add surfaces, spaces, joints accoring to body_to_name
        body_to_name = config['body_to_name']
        if larger_world and 'body_to_name_new' in config:
            body_to_name = config['body_to_name_new']

        for k, v in body_to_name.items():
            if v not in bullet_world.name_to_body:
                bullet_world.add_body(eval(k), v)
                ## e.g. k=(15, 1), v=minifridge::joint_0

        ## camera
        if 'camera_zoomins' in config:
            camera_zoomins = config['camera_zoomins']
            if len(camera_zoomins) > 0:
                bullet_world.camera_kwargs = [get_camera_kwargs(bullet_world, d) for d in camera_zoomins]
            else:
                fridge = bullet_world.name_to_body['minifridge']
                set_camera_target_body(fridge, dx=2, dy=0, dz=2)

    # wait_unlocked()
    bullet_world.run_dir = lisdf_dir
    return bullet_world


def get_camera_kwargs(bullet_world, camera_zoomin):
    name = camera_zoomin['name']
    if '::' in name:
        name = name.split('::')[0]
    if name not in bullet_world.name_to_body and f"{name}#1" in bullet_world.name_to_body:
        camera_zoomin['name'] = name = f"{name}#1"
    body = bullet_world.name_to_body[name]
    dx, dy, dz = camera_zoomin['d']
    camera_point, target_point = set_camera_target_body(body, dx=dx, dy=dy, dz=dz)
    # print(body, bullet_world.get_mobility_category(body))
    if bullet_world.get_mobility_category(body) in ['CabinetTop', 'MiniFridge']:
        target_point[0] += 0.4
    return {'camera_point': camera_point, 'target_point': target_point}


def make_furniture_transparent(bullet_world, lisdf_dir, lower_tpy=0.5, upper_tpy=0.2, **kwargs):
    transparent = ['pr2']
    upper_furnitures = ['cabinettop', 'cabinetupper', 'shelf']
    camera_zoomin = get_camera_zoom_in(lisdf_dir)
    if camera_zoomin is not None:
        target_body = camera_zoomin['name']
        if 'braiser' in target_body:
            transparent.extend(['braiserlid']+upper_furnitures)
        if 'sink' in target_body:
            transparent.extend(['faucet']+upper_furnitures)
        # if 'minifridge' in target_body or 'cabinettop' in target_body:
        #     transparent.extend(['minifridge::', 'cabinettop::'])
    for b in transparent:
        transparency = lower_tpy if b not in upper_furnitures else upper_tpy
        if transparency < 1:
            bullet_world.make_transparent(b, transparency=transparency, **kwargs)


## will be replaced later will <world><gui><camera> tag
HACK_CAMERA_POSES = { ## scene_name : (camera_point, target_point)
    'kitchen_counter': ([3, 8, 3], [0, 8, 1]),
    'kitchen_basics': ([3, 6, 3], [0, 6, 1])
}


def make_sdf_world(sdf_model):
    return f"""<?xml version="1.0" ?>
<!-- tmp sdf file generated from LISDF -->
<sdf version="1.9">
  <world name="tmp_world">

{sdf_model}

  </world>
</sdf>"""

#######################


def pddl_files_from_dir(exp_dir, replace_pddl=False):
    if replace_pddl:
        root_dir = abspath(join(__file__, *[os.pardir]*4))
        cognitive_dir = join(root_dir, 'cognitive-architectures')
        pddl_dir = join(cognitive_dir, 'bullet', 'assets', 'pddl')
        domain_path = join(pddl_dir, 'domains', 'pr2_mamao.pddl')
        stream_path = join(pddl_dir, 'streams', 'pr2_stream_mamao.pddl')
    else:
        domain_path = join(exp_dir, 'domain_full.pddl')
        stream_path = join(exp_dir, 'stream.pddl')
    config_path = join(exp_dir, 'planning_config.json')
    if not isfile(domain_path):
        planning_config = json.load(open(config_path, 'r'))
        domain_path = planning_config['domain_full']
        stream_path = planning_config['stream']
    return domain_path, stream_path, config_path


def revise_goal(goal, world):
    new_goal = []
    for tup in goal:
        new_tup = []
        for elem in tup:
            if elem in world.name_to_body:
                elem = world.name_to_body[elem]
            new_tup.append(elem)
        new_goal.append(tuple(new_tup))
    return new_goal


def pddlstream_from_dir(problem, exp_dir, replace_pddl=False, collisions=True,
                        teleport=False, goal=None, larger_world=False, **kwargs):
    exp_dir = abspath(exp_dir)

    domain_path, stream_path, config_path = pddl_files_from_dir(exp_dir, replace_pddl)

    domain_pddl = read(domain_path)
    stream_pddl = read(stream_path)

    world = problem.world
    init, g, constant_map = pddl_to_init_goal(exp_dir, world, domain_file=domain_path,
                                              larger_world=larger_world)
    world.summarize_all_objects(init)  ## important to get obstacles

    if goal is not None:
        goal = [AND] + revise_goal(goal, world)
    else:
        goal = [AND] + g

    problem.add_init(init)

    custom_limits = problem.world.robot.custom_limits ## planning_config['base_limits']
    stream_map = world.robot.get_stream_map(problem, collisions, custom_limits, teleport, **kwargs)

    print(f'Experiment: \t{exp_dir}\n'
          f'Domain PDDL: \t{domain_path}\n'
          f'Stream PDDL: \t{stream_path}\n'
          f'Config: \t{config_path}\n',
          f'Custom Limits: \t{custom_limits}')

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

#######################


def get_depth_images(exp_dir, width=1280, height=960,  verbose=False, ## , width=720, height=560)
                     camera_pose=((3.7, 8, 1.3), (0.5, 0.5, -0.5, -0.5)), robot=True,
                     img_dir=join('visualizations', 'camera_images'), **kwargs):

    def get_index(body_index):
        return f"[{body_index}]_{b2n[body_index]}"

    def get_image_and_reset(world, index):
        world.visualize_image(index=index, **kwargs)
        reset_simulation()
        world = load_lisdf_pybullet(exp_dir, width=width, height=height)
        world.add_camera(camera_pose, img_dir)
        return world

    os.makedirs(img_dir, exist_ok=True)
    world = load_lisdf_pybullet(exp_dir, width=width, height=height, verbose=True)
    # print('world.name_to_body', world.name_to_body)
    init = pddl_to_init_goal(exp_dir, world)[0]

    world.add_camera(camera_pose, img_dir)
    if not robot: remove_body(world.robot.body)

    b2n = world.body_to_name
    c2b = world.cat_to_bodies
    bodies = c2b('graspable', init) + [world.robot.body]
    body_links = c2b('surface', init) + c2b('space', init)
    body_joints = c2b('door', init) + c2b('drawer', init)

    ## skip if already all exist
    if len(listdir(world.img_dir)) == 1 + len(bodies+body_links+body_joints):
        return

    # world.visualize_image(index='scene', **kwargs)
    get_image_and_reset(world, index='scene')

    links_to_show = {b: [b[2]] for b in body_links}
    for body_joint in body_joints:
        body, joint = body_joint
        links_to_show[body_joint] = get_door_links(body, joint)

    for body in bodies:
        for b in get_bodies():
            if b != body:
                remove_body(b)
        get_image_and_reset(world, get_index(body))

    for bo in body_links + body_joints:
        index = get_index(bo)
        all_bodies = get_bodies()

        # clone_body(bo[0], links=links_to_show[bo], visual=True, collision=True)
        for l in links_to_show[bo]:
            clone_body_link(bo[0], l, visual=True, collision=True)

        for b in all_bodies:
            remove_body(b)
        get_image_and_reset(world, index)


#######################


if __name__ == "__main__":

    for lisdf_test in ['kitchen_lunch']: ## 'm0m_joint_test', 'kitchen_basics', 'kitchen_counter'
        lisdf_path = join(ASSET_PATH, 'scenes', f'{lisdf_test}.lisdf')
        world = load_lisdf_pybullet(lisdf_path, verbose=True)
        wait_if_gui('load next test scene?')
        reset_simulation()
