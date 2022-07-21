LINK_STR = '::'

import os
import sys
from os.path import join, abspath, dirname, isdir, isfile
sys.path.append('lisdf')
from lisdf.parsing.sdf_j import load_sdf
from lisdf.components.model import URDFInclude
import numpy as np
import json

import warnings
warnings.filterwarnings('ignore')

from pybullet_tools.pr2_problems import create_floor
from pybullet_tools.pr2_utils import set_group_conf, get_group_joints, get_viewcone_base
from pybullet_tools.utils import remove_handles, remove_body, get_bodies, remove_body, get_links, \
    clone_body, get_joint_limits, ConfSaver, load_pybullet, connect, wait_if_gui, HideOutput, invert, \
    disconnect, set_pose, set_joint_position, joint_from_name, quat_from_euler, draw_pose, unit_pose, \
    set_camera_pose, set_camera_pose2, get_pose, get_joint_position, get_link_pose, get_link_name, \
    set_joint_positions, get_links, get_joints, get_joint_name, get_body_name, link_from_name, \
    parent_joint_from_link, set_color, dump_body, RED, YELLOW, GREEN, BLUE, GREY, BLACK, read, get_client, \
    reset_simulation
from pybullet_tools.bullet_utils import nice, sort_body_parts, equal, clone_body_link
from pybullet_tools.pr2_streams import get_handle_link
from pybullet_tools.flying_gripper_utils import set_se3_conf

from pddlstream.language.constants import AND, PDDLProblem

from world_builder.entities import Space
from world_builder.loaders import create_gripper_robot, create_pr2_robot

from lisdf_tools.lisdf_planning import pddl_to_init_goal

ASSET_PATH = join(dirname(__file__), '..', '..', 'assets')
LINK_COLORS = ['#c0392b', '#d35400', '#f39c12', '#16a085', '#27ae60',
               '#2980b9', '#8e44ad', '#2c3e50', '#95a5a6']
LINK_COLORS = [RED, YELLOW, GREEN, BLUE, GREY, BLACK]

class World():
    def __init__(self, lisdf):
        self.lisdf = lisdf
        self.body_to_name = {}
        self.name_to_body = {}
        self.ATTACHMENTS = {}
        self.camera = None
        self.robot = None
        self.movable = None
        self.fixed = None
        self.floors = None

        ## for visualization
        self.handles = []
        self.cameras = []

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

    def add_body(self, body, name):
        if body is None:
            print('lisdf_loader.body = None')
        self.body_to_name[body] = name
        self.name_to_body[name] = body

    def add_robot(self, body, name='robot', **kwargs):
        if not isinstance(body, int):
            name = body.name
        self.add_body(body, name)
        self.robot = body

    def add_joints(self, body, body_joints):
        idx = 0
        for body_joint, joint_name in body_joints.items():
            self.add_body(body_joint, joint_name)
            handle_link = get_handle_link(body_joint)
            set_color(body, color=LINK_COLORS[idx], link=handle_link)
            idx += 1

    def add_spaces(self, body, body_links):
        idx = 0
        for body_link, link_name in body_links.items():
            self.add_body(body_link, link_name)
            set_color(body, color=LINK_COLORS[idx], link=body_link[-1])
            idx += 1

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

    def check_world_obstacles(self):
        fixed = []
        movable = []
        floors = []
        for model in self.lisdf.models:
            body = self.name_to_body[model.name]
            if model.name not in ['pr2', 'feg']:
                if model.static: fixed.append(body)
                else: movable.append(body)
            if hasattr(model, 'links'):
                for link in model.links:
                    if link.name == 'box':
                        for collision in link.collisions:
                            if collision.shape.size[-1] < 0.05:
                                floors.append(body)
        self.fixed = [f for f in fixed if f not in floors]
        self.movable = movable
        self.floors = floors

    def summarize_all_types(self, init=None):
        if init is None: return ''
        printout = ''
        for typ in ['graspable', 'surface', 'door', 'drawer']:
            num = len(self.cat_to_bodies(typ, init))
            if typ == 'graspable':
                typ = 'moveable'
            if num > 0:
                printout += "{type}({num}), ".format(type=typ, num=num)
        return printout

    def cat_to_bodies(self, cat, init):
        return [f[1] for f in init if f[0].lower() == cat]

    def summarize_all_objects(self, init=None):
        """ call this after pddl_to_init_goal() where world.update_objects() happens """
        from pybullet_tools.logging import myprint as print

        self.check_world_obstacles()
        ob = [n for n in self.fixed if n not in self.floors]

        print('----------------')
        print(f'PART I: world objects | {self.summarize_all_types(init)} | obstacles({len(ob)}) = {ob}')
        print('----------------')

        for body in sort_body_parts(self.body_to_name.keys()):
            line = f'{body}\t  |  {self.body_to_name[body]}'
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
            print(f"{line}\t|  Pose: {nice(pose)}")
        print('----------------')

    def get_name(self, body):
        if body in self.body_to_name:
            return self.body_to_name[body]
        return None

    def get_debug_name(self, body):
        return f"{self.get_name(body)}|{body}"

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

    def add_camera(self, pose, img_dir=join('visualizations', 'camera_images')):
        from world_builder.entities import StaticCamera
        from pybullet_tools.bullet_utils import CAMERA_MATRIX

        camera = StaticCamera(pose, camera_matrix=CAMERA_MATRIX, max_depth=6)
        self.cameras.append(camera)
        self.camera = camera
        self.img_dir = img_dir

    def visualize_image(self, pose=None, img_dir=None, index=None,
                        image=None, rgb=False):
        from pybullet_tools.bullet_utils import visualize_camera_image

        if pose is not None:
            self.camera.set_pose(pose)
        if img_dir is not None:
            self.img_dir = img_dir
        if index is None:
            index = self.camera.index
        if image is None:
            image = self.camera.get_image(segment=False)
        visualize_camera_image(image, index, img_dir=self.img_dir, rgb=rgb)


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

def load_lisdf_pybullet(lisdf_path, verbose=True, width=1980, height=1238):
    # scenes_path = dirname(os.path.abspath(lisdf_path))
    tmp_path = join(ASSET_PATH, 'tmp')
    if not isdir(tmp_path): os.mkdir(tmp_path)

    config_path = join(lisdf_path, 'planning_config.json')
    custom_limits = {}
    # body_to_name = None
    if isfile(config_path):
        planning_config = json.load(open(config_path))
        custom_limits = {int(k):v for k,v in planning_config['base_limits'].items()}
        # body_to_name = planning_config['body_to_name']
        lisdf_path = join(lisdf_path, 'scene.lisdf')

    ## --- the floor and pose will become extra bodies
    connect(use_gui=True, shadows=False, width=width, height=height)
    # draw_pose(unit_pose(), length=1.)
    # create_floor()

    # with HideOutput():
        # load_pybullet(join('models', 'Basin', '102379', 'mobility.urdf'))
    # load_pybullet(sdf_path)  ## failed
    # load_pybullet(join(tmp_path, 'table#1_1.sdf'))

    world = load_sdf(lisdf_path).worlds[0]
    bullet_world = World(world)

    # if world.name in HACK_CAMERA_POSES:
    #     cp, tp = HACK_CAMERA_POSES[world.name]
    #     set_camera_pose(camera_point=cp, target_point=tp)

    if world.gui != None:
        camera_pose = world.gui.camera.pose
        set_camera_pose2((camera_pose.pos, camera_pose.quat_xyzw))

    ## may be changes in joint positions
    model_states = {}
    if len(world.states) > 0:
        model_states = world.states[0].model_states
        model_states = {s.name: s for s in model_states}

    for model in world.models:
        scale = 1
        if isinstance(model, URDFInclude):
            uri = join(ASSET_PATH, 'scenes', model.uri)
            scale = model.scale_1d
            category = model.content.name
        else:
            uri = join(tmp_path, f'{model.name}.sdf')
            with open(uri, 'w') as f:
                f.write(make_sdf_world(model.to_sdf()))
            category = model.links[0].name

        if verbose: print(f'..... loading {model.name} from {uri}', end="\r")
        if not isdir(join(ASSET_PATH, 'scenes')):
            os.mkdir(join(ASSET_PATH, 'scenes'))
        with HideOutput():
            body = load_pybullet(uri, scale=scale)
            if isinstance(body, tuple): body = body[0]

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
            bullet_world.add_body(body, model.name)

        if model.name in model_states:
            for js in model_states[model.name].joint_states:
                position = js.axis_states[0].value
                set_joint_position(body, joint_from_name(body, js.name), position)

        if not isinstance(model, URDFInclude):
            os.remove(uri)

        # wait_if_gui('load next model?')
    return bullet_world

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

def pddlstream_from_dir(problem, exp_dir, collisions=True, teleport=False):

    world = problem.world

    domain_pddl = read(join(exp_dir, 'domain_full.pddl'))
    stream_pddl = read(join(exp_dir, 'stream.pddl'))
    planning_config = json.load(open(join(exp_dir, 'planning_config.json')))


    init, goal, constant_map = pddl_to_init_goal(exp_dir, world)
    goal = [AND] + goal
    problem.add_init(init)

    custom_limits = problem.world.robot.custom_limits ## planning_config['base_limits']
    stream_map = world.robot.get_stream_map(problem, collisions, custom_limits, teleport)

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

#######################

def get_depth_images(exp_dir, width=1280, height=960,  verbose=False, ## , width=720, height=560)
                     camera_pose=((3.7, 8, 1.3), (0.5, 0.5, -0.5, -0.5)),
                     img_dir=join('visualizations', 'camera_images')):

    os.makedirs(img_dir, exist_ok=True)
    world = load_lisdf_pybullet(exp_dir, width=width, height=height, verbose=True)
    print('world.name_to_body', world.name_to_body)
    init = pddl_to_init_goal(exp_dir, world)[0]

    world.add_camera(camera_pose, img_dir)
    world.visualize_image(index='scene')

    b2n = world.body_to_name
    c2b = world.cat_to_bodies
    bodies = c2b('graspable', init) + [world.robot.body]
    body_links = c2b('surface', init) + c2b('space', init)
    body_joints = c2b('door', init) + c2b('drawer', init)

    links_to_show = {b: [b[2]] for b in body_links}
    for body_joint in body_joints:
        body, joint = body_joint
        with ConfSaver(body):
            min_pstn, max_pstn = get_joint_limits(body, joint)
            set_joint_position(body, joint, min_pstn)
            lps = [get_link_pose(body, l) for l in get_links(body)]
            set_joint_position(body, joint, max_pstn)
            links_to_show[body_joint] = [i for i in range(len(lps)) if lps[i] != get_link_pose(body, i)]

    def get_index(body_index):
        return f"[{body_index}]_{b2n[body_index]}"

    def get_image_and_reset(world, index):
        world.visualize_image(index=index)
        reset_simulation()
        world = load_lisdf_pybullet(exp_dir, width=width, height=height)
        world.add_camera(camera_pose, img_dir)
        return world

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
