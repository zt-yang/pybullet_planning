import os
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

from pybullet_tools.pr2_utils import get_group_joints
from pybullet_tools.utils import get_bodies, remove_body, load_pybullet, connect, wait_if_gui, HideOutput, set_pose, set_joint_position, joint_from_name, quat_from_euler, draw_pose, unit_pose, \
    set_camera_pose2, get_pose, get_joint_position, get_link_pose, get_link_name, \
    get_links, get_joints, get_joint_name, set_color, reset_simulation, get_movable_joints, is_movable
from pybullet_tools.bullet_utils import nice, sort_body_parts, clone_body_link, \
    toggle_joint, get_door_links, colorize_world, colorize_link, find_closest_match, is_box_entity
from pybullet_tools.camera_utils import set_camera_target_body
from pybullet_tools.logging_utils import summarize_facts
from pybullet_tools.pr2_streams import get_handle_link

from robot_builder.robot_builders import create_pr2_robot, create_gripper_robot

from world_builder.world_utils import get_instance_name, get_lisdf_name, get_mobility_id, \
    get_mobility_category, get_mobility_identifier
from world_builder.world import WorldBase

from lisdf_tools.lisdf_planning import pddl_to_init_goal
from lisdf_tools.lisdf_utils import find_id, LINK_COLORS, ASSET_PATH, LINK_STR, PART_INSTANCE_NAME, \
    get_custom_limits, make_sdf_world, make_furniture_transparent, get_camera_kwargs


class World(WorldBase):
    def __init__(self, lisdf=None, **kwargs):
        super().__init__(**kwargs)
        self.lisdf = lisdf
        self.body_to_name = {}
        self.instance_names = {}
        self.paths = {}
        self.mobility_ids = {}
        self.mobility_identifiers = {}
        self.name_to_body = {}
        self.attachments = {}

        self.robot = None
        self.movable = None
        self.fixed = []
        self.floors = None
        self.body_types = {}
        self.ignored_pairs = []

        ## for visualization
        self.colored_links = []
        self.camera_kwargs = {}
        self.run_dir = None

    """ same as World in world_builder, for replaying in run_pr2.py """
    @property
    def objects(self):
        return [k for k in self.body_to_name if k != self.robot.body]

    def save_problem(self, output_dir, **kwargs):
        pass

    def clear_viz(self):
        self.remove_handles()
        self.remove_redundant_bodies()

    def remove_redundant_bodies(self):
        for b in get_bodies():
            if b not in self.body_to_name:
                remove_body(b)

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

    def get_object(self, body):
        return body

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

    def remove_from_gym_world(self, name, gym_world,  pose, exceptions=[]):
        bodies = self.safely_get_body_from_name(name, all_possible=True)
        removed = []
        if bodies is not None:
            for b in bodies:
                name = self.body_to_name[b]
                if name in exceptions or '::' in name:
                    continue
                gym_world.set_pose(gym_world.get_actor(name), pose)
                removed.append(name)
        return removed

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

    def add_ignored_pair(self, pair):
        a, b = pair
        self.ignored_pairs.extend([(a, b), (b, a)])

    def summarize_all_types(self, init=None, return_string=True):
        if init is None: return ''
        printout = ''
        results = {}
        for typ in ['graspable', 'surface', 'space', 'door', 'drawer', 'joint']:
            bodies = self.cat_to_bodies(typ, init)
            results[typ] = bodies
            num = len(bodies)
            if typ == 'graspable':
                typ = 'movable'
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
            from pybullet_tools.logging_utils import myprint as print_fn

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

    def get_collision_objects(self):
        return [n for n in self.body_to_name if isinstance(n, int) and n > 1]

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
        if world_index is not None:
            wconf = {f"w{world_index}_{k}": v for k, v in wconf.items()}
        return wconf

    def get_type(self, body):
        if body in self.body_types:
            return self.body_types[body]
        return []

    def get_category(self, body):
        return self.get_type(body)

    def get_name(self, body, use_default_link_name=True):
        if use_default_link_name and isinstance(body, tuple) and len(body) == 3:
            return get_link_name(body[0], body[-1])
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
        from pigi_tools.data_utils import get_indices
        return get_indices(self.run_dir, **kwargs)

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


def load_lisdf_pybullet(lisdf_path, verbose=False, use_gui=True, jointless=False, width=1980, height=1238,
                        larger_world=False, custom_robot_loader={}, robot_builder_args={}, **kwargs):

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
    if not isdir(tmp_path):
        os.mkdir(tmp_path)

    ## get custom base limits for robots
    config_path = join(lisdf_dir, 'planning_config.json')
    custom_limits = {}
    if isfile(config_path):
        custom_limits = get_custom_limits(config_path)

    ## --- the floor and pose will become extra bodies
    connect(use_gui=use_gui, shadows=False, width=width, height=height)
    draw_pose(unit_pose(), length=1.)
    # create_floor()

    lisdf_world = load_sdf(lisdf_path).worlds[0]
    world = World(lisdf_world, **kwargs)

    ## may be changes in joint positions
    model_states = {}
    if len(lisdf_world.states) > 0:
        model_states = lisdf_world.states[0].model_states
        model_states = {s.name: s for s in model_states}

    for model in lisdf_world.models:
        scale = 1
        if isinstance(model, URDFInclude):
            # uri = model.uri.replace('/home/yang/Documents/cognitive-architectures/bullet/', '../../')
            uri = join(ASSET_PATH, 'models', model.uri)
            scale = model.scale_1d
            category = model.content.name
        else:
            uri = join(tmp_path, f'{model.name}.sdf')
            ## TODO: when solving in parallel causes problem_sets
            # if isfile(uri):
            #     os.remove(uri)
            with open(uri, 'w') as f:
                f.write(make_sdf_world(model.to_sdf()))
            category = model.links[0].name

        if verbose:
            print(f'..... loading {model.name} of category {category} from {abspath(uri)}') ## , end="\r"
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
                create_pr2_robot(world, base_q=pose, custom_limits=custom_limits, robot=body)
            elif category == 'feg':
                create_gripper_robot(world, custom_limits=custom_limits, robot=body)

        elif category in custom_robot_loader:
            custom_robot_loader[category](world, custom_limits=custom_limits, robot=body, **robot_builder_args)

        else:
            pose = (tuple(model.pose.pos), quat_from_euler(model.pose.rpy))
            set_pose(body, pose)
            world.add_body(body, model.name, instance_name=instance_name, path=uri)

        if not jointless and model.name in model_states:
            for js in model_states[model.name].joint_states:
                position = js.axis_states[0].value
                set_joint_position(body, joint_from_name(body, js.name), position)

    if lisdf_world.gui is not None:
        camera_pose = lisdf_world.gui.camera.pose
        ## when camera pose is not saved for generating training data
        if not np.all(camera_pose.pos == 0):
            set_camera_pose2((camera_pose.pos, camera_pose.quat_xyzw))

    if not verbose:
        print(f'lisdf_loader.load_lisdf_pybullet | if failed, come set verbose = True')
    planning_config = join(lisdf_dir, 'planning_config.json')
    if isfile(planning_config):
        config = json.load(open(planning_config, 'r'))

        ## add surfaces, spaces, joints according to body_to_name
        if 'body_to_name' in config:
            body_to_name = config['body_to_name']
            if larger_world and 'body_to_name_new' in config:
                body_to_name = config['body_to_name_new']

            for k, v in body_to_name.items():
                if v not in world.name_to_body:
                    pybullet_id = eval(k)
                    if isinstance(pybullet_id, tuple) and '::' in v:
                        body_name = v[:v.index(':')]
                        body_id = world.name_to_body[body_name]
                        if len(pybullet_id) == 2:
                            pybullet_id = (body_id, pybullet_id[-1])
                        if len(pybullet_id) == 3:
                            pybullet_id = (body_id, None, pybullet_id[-1])
                    if verbose:
                        print(f'load_lisdf_pybullet.planning_config | k -> {pybullet_id} : {v} not in world.name_to_body')
                    world.add_body(pybullet_id, v)
                    ## e.g. k=(15, 1), v=minifridge::joint_0

        ## camera
        if 'camera_zoomins' in config:
            camera_zoomins = config['camera_zoomins']
            if len(camera_zoomins) > 0:
                world.camera_kwargs = [get_camera_kwargs(world, d) for d in camera_zoomins]
            elif 'sink#1' in world.name_to_body:
                # fridge = bullet_world.name_to_body['minifridge']
                # set_camera_target_body(fridge, dx=2, dy=0, dz=2)
                sink = world.name_to_body['sink#1']
                set_camera_target_body(sink, dx=3, dy=1, dz=1)

    # wait_unlocked()
    world.run_dir = lisdf_dir
    return world


def get_depth_images(exp_dir, width=1280, height=960, ## verbose=False, width=720, height=560,
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


if __name__ == "__main__":

    for lisdf_test in ['kitchen_lunch']: ## 'm0m_joint_test', 'kitchen_basics', 'kitchen_counter'
        lisdf_path = join(ASSET_PATH, 'scenes', f'{lisdf_test}.lisdf')
        world = load_lisdf_pybullet(lisdf_path, verbose=True)
        wait_if_gui('load next test scene?')
        reset_simulation()
