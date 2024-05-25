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
from pybullet_tools.utils import remove_handles, remove_body, get_bodies, remove_body, get_links, \
    clone_body, get_joint_limits, ConfSaver, load_pybullet, connect, wait_if_gui, HideOutput, invert, \
    disconnect, set_pose, set_joint_position, joint_from_name, quat_from_euler, draw_pose, unit_pose, \
    set_camera_pose, set_camera_pose2, get_pose, get_joint_position, get_link_pose, get_link_name, \
    set_joint_positions, get_links, get_joints, get_joint_name, get_body_name, link_from_name, \
    parent_joint_from_link, set_color, dump_body, RED, YELLOW, GREEN, BLUE, GREY, BLACK, read, get_client, \
    reset_simulation, get_movable_joints, JOINT_TYPES, get_joint_type, is_movable, wait_unlocked
from pybullet_tools.bullet_utils import nice, sort_body_parts, equal, clone_body_link, \
    toggle_joint, get_door_links
from pybullet_tools.camera_utils import set_camera_target_body

from pddlstream.language.constants import AND, PDDLProblem

from world_builder.entities import Space, StaticCamera
from world_builder.world_utils import get_camera_zoom_in
from world_builder.paths import PBP_PATH

from lisdf_tools.lisdf_planning import pddl_to_init_goal


ASSET_PATH = join(dirname(__file__), '..', '..', 'assets')

LINK_COLORS = ['#c0392b', '#d35400', '#f39c12', '#16a085', '#27ae60',
               '#2980b9', '#8e44ad', '#2c3e50', '#95a5a6']
LINK_COLORS = [RED, YELLOW, GREEN, BLUE, GREY, BLACK]
LINK_STR = '::'
PART_INSTANCE_NAME = "{body_instance_name}" + LINK_STR + "{part_name}"


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
                if obj in world.attachments:
                    parent = world.attachments[obj].parent
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

    # weiyu debug
    transparent.append("cabinettop")

    for b in transparent:
        transparency = lower_tpy if b not in upper_furnitures else upper_tpy
        if b == "cabinettop":
            transparency = 0.5

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


def pddl_files_from_dir(exp_dir, replace_pddl=False, domain_name='pr2_mamao.pddl',
                        stream_name='pr2_stream_mamao.pddl'):
    if replace_pddl:
        root_dir = abspath(join(__file__, *[os.pardir]*4))
        if 'cognitive-architectures' not in root_dir:
            root_dir = join(root_dir, 'cognitive-architectures')
        pddl_dir = join(root_dir, 'bullet', 'assets', 'pddl')
        domain_path = join(pddl_dir, 'domains', domain_name)
        stream_path = join(pddl_dir, 'streams', stream_name)
        if not isfile(domain_path):
            pddl_dir = join(PBP_PATH, 'pddl')
            domain_path = join(pddl_dir, domain_name)
            stream_path = join(pddl_dir, stream_name)
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
                        teleport=False, goal=None, larger_world=False,
                        domain_name=None, stream_name=None, **kwargs):
    exp_dir = abspath(exp_dir)

    domain_path, stream_path, config_path = pddl_files_from_dir(exp_dir, replace_pddl, domain_name=domain_name,
                                                                stream_name=stream_name)

    domain_pddl = read(domain_path)
    stream_pddl = read(stream_path)

    world = problem.world
    init, g, constant_map = pddl_to_init_goal(exp_dir, world, domain_file=domain_path, larger_world=larger_world)
    world.summarize_all_objects(init)  ## important to get obstacles
    world.summarize_facts(init, name='Initial loaded facts from pddl')

    if goal is not None:
        goal = [AND] + revise_goal(goal, world)
    else:
        goal = [AND] + g

    problem.add_init(init)

    custom_limits = problem.world.robot.custom_limits ## planning_config['base_limits']
    stream_map = world.robot.get_stream_map(problem, collisions, custom_limits, teleport,
                                            domain_pddl, **kwargs)

    print(f'Experiment: \t{exp_dir}\n'
          f'Domain PDDL: \t{domain_path}\n'
          f'Stream PDDL: \t{stream_path}\n'
          f'Config: \t{config_path}\n',
          f'Custom Limits: \t{custom_limits}')

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)
