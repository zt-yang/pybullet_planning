import numpy as np
import os
import sys
import json
import copy
import shutil
import numpy as np
from os.path import join, isdir, isfile, dirname, abspath, basename
from os import listdir

from pybullet_tools.utils import get_bodies, euler_from_quat, get_collision_data, get_joint_name, \
    get_joint_position, get_camera, joint_from_name, get_color, reset_simulation, \
    link_from_name
from pybullet_tools.bullet_utils import LINK_STR, nice, is_box_entity, random
from pybullet_tools.pose_utils import is_placement
from pybullet_tools.logging_utils import get_readable_list
from pybullet_tools.stream_agent import FAILED
from pybullet_tools.general_streams import Position
from pybullet_tools.pr2_primitives import Pose

from world_builder.paths import ASSET_PATH
from world_builder.entities import Robot, LINK_STR
from world_builder.world_utils import read_xml, get_file_by_category, get_model_scale

from lisdf_tools.lisdf_loader import get_depth_images

LISDF_PATH = join('assets', 'scenes')
EXP_PATH = join('test_cases')

# ACTOR_STR = """
#     <include name="pr2">
#       <uri>../models/drake/pr2_description/urdf/pr2_simplified.urdf</uri>
#       {pose_xml}
#     </include>
# """
MODEL_BOX_STR = """
    <model name="{name}">
      <static>{is_static}</static>
      {pose_xml}
      <link name="box">
        <collision name="box_collision">
          <geometry>
            <box>
              <size>{wlh}</size>
            </box>
          </geometry>
        </collision>
        <visual name="box_visual">
          <geometry>
            <box>
              <size>{wlh}</size>
            </box>
          </geometry>
          <material>
            <diffuse>{rgb} 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
"""
MODEL_URDF_STR = """
    <include name="{name}">
      <uri>{file}</uri>
      <static>{is_static}</static>
      <scale>{scale}</scale>
      {pose_xml}
    </include>
"""
MODEL_STATE_STR = """
      <model name="{name}">{joints_xml}
      </model>"""
STATE_JOINTS_STR = """
        <joint name="{name}"><angle>{angle}</angle></joint>"""
STATE_STR = """
    <state world_name="{name}">{state_sdf}
    </state>
"""
WORLD_STR = """<?xml version="1.0" ?>
<!-- sdf file created by Yang's kitchen scene generator -->
<sdf version="1.9">
  <world name="{name}">
{camera_sdf}
{actor_sdf}
{models_sdf}
{state_sdf}
  </world>
</sdf>"""
CAMERA_STR = """
    <gui>
      <camera name="default_camera" definition_type="lookat">
        <xyz>{camera_point}</xyz>
        <point_to>{target_point}</point_to>
      </camera>
    </gui>
"""


def list_to_xml(lst):
    return " ".join([str(round(o, 3)) for o in lst])


def to_pose_xml(pose):
    xyz, quat = pose
    euler = euler_from_quat(quat)
    return f"<pose>{list_to_xml(xyz)} {list_to_xml(euler)}</pose>"


def get_camera_spec():
    import math
    _, _, _, _, _, _, _, _, yaw, pitch, dist, target = get_camera()
    pitch_rad = math.radians(pitch)
    dz = -dist * math.sin(pitch_rad)
    l = abs(dist * math.cos(pitch_rad))
    yaw_rad = math.radians(yaw)
    dx = l * math.sin(yaw_rad)
    dy = -l * math.cos(yaw_rad)
    camera = [target[0]+dx, target[1]+dy, target[2]+dz]
    return camera, list(target)


def get_output_dir(exp_name=None, world_name=None, root_path=None, output_dir=None):
    """ if exp_name != None, will be generated into kitchen-world/test_cases/{exp_name}/scene.lisdf
        if world_name != None, will be generated into kitchen-world/assets/scenes/{world_name}.lisdf
        if out_path != None, will be generated into out_path
    """
    exp_path = EXP_PATH
    lisdf_path = LISDF_PATH
    if root_path is not None:
        exp_path = join(root_path, exp_path)
        lisdf_path = join(root_path, lisdf_path)

    if exp_name is not None:
        output_dir = join(exp_path, exp_name)
        if isdir(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        output_dir = join(exp_path, exp_name, "scene.lisdf")

    elif output_dir is not None:
        output_dir = output_dir

    else:
        output_dir = join(lisdf_path, f"{world_name}.lisdf")
    return output_dir


def to_lisdf(world, output_dir, world_name=None, verbose=True, **kwargs):
    output_dir = get_output_dir(output_dir=output_dir, world_name=world_name, **kwargs)
    if world_name is None:
        world_name = basename(output_dir)

    floorplan = world.floorplan
    objects = []
    SCALING = 1
    if floorplan is not None:
        objects, _, _, SCALING, _, _, _, _ = read_xml(floorplan)
        objects = {k.lower(): v for k, v in objects.items()}
    _, _, _, _, _, _, _, _, yaw, pitch, dist, target = get_camera()

    def get_file_scale(name):
        o = objects[name]
        file = get_file_by_category(o['category'])
        l = o['l'] / SCALING
        w = o['w'] / SCALING
        scale = get_model_scale(file, l, w)
        return file, scale

    actor_sdf = ''
    models_sdf = ''
    state_sdf = ''
    model_joints = {}  ## model_name : joints_xml
    c = world.cat_to_bodies
    movables = c('movable')

    ## first add all actor and models
    bodies = copy.deepcopy(get_bodies())
    bodies.sort()
    for body in bodies:
        obj = world.get_object(body)
        if obj is None:
            continue

        is_static = 'false' if body in movables else 'true'
        pose_xml = to_pose_xml(obj.get_pose())
        # print(obj.name)
        if isinstance(obj, Robot):
            ACTOR_STR = world.robot.get_lisdf_string()
            actor_sdf = ACTOR_STR.format(name=obj.name, pose_xml=pose_xml)

            """ useful but not used """
            # if exp_name != None:
            #     actor_sdf = actor_sdf.replace('../models/', '../../assets/models/')
            # if out_path != None:
            #     actor_sdf = actor_sdf.replace('../models/', '../../models/')

            ## robot joint states
            joints_xml = ''
            all_joints = world.robot.get_all_joints()
            js = [joint_from_name(body, j) for j in all_joints]
            # js = list(get_group_joints(body, 'torso'))
            # js.extend(list(get_arm_joints(body, 'left'))+list(get_arm_joints(body, 'right')))
            for j in js:
                joints_xml += STATE_JOINTS_STR.format(
                    name=get_joint_name(body, j),
                    angle=round(get_joint_position(body, j), 3)
                )
            state_sdf += MODEL_STATE_STR.format(name=obj.name, joints_xml=joints_xml)

        elif obj.is_box: ## and obj.name not in objects:
            if len(get_collision_data(body)) == 0:
                print('world_generator | len(get_collision_data(body)) == 0')
            dim = get_collision_data(body)[0].dimensions
            wlh = " ".join([str(round(o, 3)) for o in dim])
            color = get_color(obj.body)[:3]
            rgb = " ".join([str(round(o, 3)) for o in color])
            models_sdf += MODEL_BOX_STR.format(
                name=obj.name, is_static=is_static,
                pose_xml=pose_xml, wlh=wlh, rgb=rgb
            )

        else:
            if not hasattr(obj, 'path'):
                print('world_generator.file', obj)
            file = join('..', '..', 'assets') + obj.path.replace(ASSET_PATH, '')
            scale = obj.scale

            models_sdf += MODEL_URDF_STR.format(
                name=obj.lisdf_name, file=file,
                is_static=is_static,
                pose_xml=pose_xml, scale=scale
            )

    ## then add joint states of models
    joints = world.get_scene_joints()
    for j in joints:
        body, joint = j
        name = world.get_object(body).name
        joint_name = world.get_object(j).name
        if name not in model_joints:
            model_joints[name] = ""
        model_joints[name] += STATE_JOINTS_STR.format(
            name=joint_name.replace(name+LINK_STR, ''),
            angle=round(get_joint_position(body, joint), 3)
        )
    for name, joints_xml in model_joints.items():
        state_sdf += MODEL_STATE_STR.format(name=name, joints_xml=joints_xml)
    state_sdf = STATE_STR.format(
        name=world_name, state_sdf=state_sdf
    )

    ## finally add camera pose
    cp, tp = get_camera_spec()
    camera_sdf = CAMERA_STR.format(camera_point=list_to_xml(cp), target_point=list_to_xml(tp))

    ## put everything together
    world_sdf = WORLD_STR.format(
        name=world_name, actor_sdf=actor_sdf, models_sdf=models_sdf,
        state_sdf=state_sdf, camera_sdf=camera_sdf
    )

    with open(output_dir, 'w') as f:
        f.write(world_sdf)
    if verbose: print(f'\n\nwritten {output_dir}\n\n')

    return LISDF_PATH


def test_get_camera_spec():
    from pybullet_tools.utils import connect, disconnect, set_camera_pose, get_camera, \
        create_box, set_pose, quat_from_euler, set_renderer, wait_if_gui, unit_pose
    import numpy as np
    import random

    def random_point(lim=9):
        return [random.uniform(-lim, lim) for k in range(3)]

    def equal(a, b, epsilon=0.01):
        return np.linalg.norm(np.asarray(a) - np.asarray(b)) < epsilon

    connect(use_gui=True, shadows=False, width=1980, height=1238)

    set_renderer(True)
    something = create_box(1, 1, 1, mass=1)
    somepose = unit_pose() ## (random_point(3), quat_from_euler((0, 0, 0)))
    set_pose(something, somepose)
    # set_static()

    for i in range(10):
        camera_point = random_point() ## [3, 5, 3] ##
        target_point = random_point() ## [0, 6, 1] ##
        print(f'test {i} | \t camera_point {nice(camera_point)}\t target_point {nice(target_point)}')

        ## somehow it can't set the pose ....
        set_camera_pose(camera_point=camera_point, target_point=target_point)
        camera_point_est, target_point_est = get_camera_spec()
        if not (equal(camera_point, camera_point_est) and equal(target_point, target_point_est)):
            print(f'test {i} | \t camera_point_est {nice(camera_point_est)}\t target_point_est {nice(target_point_est)}')
    disconnect()


def clean_domain_pddl(pddl_str, all_pred_names):
    string = 'xyzijk'
    need_preds = list(all_pred_names)
    pddl_str = pddl_str[:pddl_str.index('(:functions')].lower()
    pddl_str = pddl_str[:pddl_str.rfind(')')]
    started = False
    for line in pddl_str.split('\n'):
        if started and '(' in line and ')' in line:
            line = line[line.index('(')+1:line.index(')')]
            if ' ' in line:
                line = line[:line.index(' ')]
            if line in all_pred_names and line in need_preds:
                need_preds.remove(line)
        if '(:predicates' in line:
            started = True
    pddl_str = pddl_str + '\n'
    for n in need_preds:
        args = f'{n}'
        for i in range(all_pred_names[n]):
            args += f" ?{string[i]}"
        pddl_str += f'    ({args})\n'
    pddl_str += '  )\n)'
    # print(pddl_str)
    return pddl_str


def save_to_outputs_folder(output_path, data_path, data_generation=False, multiple_solutions=False):

    def move(src, des):
        if isfile(src) or isdir(src):
            shutil.move(src, des)

    if data_generation:
        original = 'visualizations'
        if isfile(join(original, 'log.json')):
            for subdir in ['constraint_networks', 'stream_plans']:
                if len(listdir(join(original, subdir))) < 1: continue
                shutil.move(join(original, subdir), join(output_path, subdir))
            for subfile in ['log.json']:
                shutil.move(join(original, subfile), join(output_path, subfile))
        # else:
        #     new_outpath = f"{outpath}_failed"
        #     shutil.move(outpath, new_outpath)
        #     outpath = new_outpath

    """ =========== move to data collection folder =========== """
    # ## 'one_fridge_pick_pr2'
    # data_path = outpath.replace('test_cases', join('outputs', 'one_fridge_pick_pr2'))
    move(output_path, data_path)
    move(join(data_path, 'time.json'), join(data_path, 'plan.json'))

    """ =========== move the log and plan =========== """
    move(f"multiple_solutions.json", join(data_path, 'multiple_solutions.json'))

    if isfile(join(data_path, 'plan.json')):
        with open(join(data_path, 'plan.json'), 'r') as f:
            if json.load(f)[0]['plan'] == FAILED:
                shutil.rmtree(data_path)


def save_to_kitchen_worlds(state, pddlstream_problem, exp_name='test_cases', exit=True,
                           floorplan=None, world_name=None, root_path=None, save_depth_images=False):
    exp_path = EXP_PATH
    if root_path is not None:
        exp_path = join(root_path, exp_path)
    outpath = join(exp_path, exp_name)
    if isdir(outpath):
        shutil.rmtree(outpath)
    os.mkdir(outpath)

    ## --- scene in scene.lisdf
    to_lisdf(state.world, pddlstream_problem.init, exp_name=exp_name,
             world_name=world_name, root_path=root_path)
    state.world.outpath = outpath

    ## --- init and goal in problem.pddl
    all_pred_names = generate_problem_pddl(state.world, pddlstream_problem.init, pddlstream_problem.goal,
                                           world_name=world_name, out_path=join(outpath, 'problem.pddl'))

    ## --- domain and stream copied over  ## shutil.copy()
    with open(join(outpath, 'domain_full.pddl'), 'w') as f:
        f.write(pddlstream_problem.domain_pddl)
    with open(join(outpath, 'domain.pddl'), 'w') as f:
        f.write(clean_domain_pddl(pddlstream_problem.domain_pddl, all_pred_names))
    with open(join(outpath, 'stream.pddl'), 'w') as f:
        f.write(pddlstream_problem.stream_pddl)
    with open(join(outpath, 'planning_config.json'), 'w') as f:
        json.dump(state.get_planning_config(), f)

    if save_depth_images and state.world.camera != None:
        # reset_simulation()
        # get_depth_images(outpath, camera_pose=state.world.camera.pose,
        #                  img_dir=join(outpath, 'depth_maps'))
        reset_simulation()
        get_depth_images(outpath, camera_pose=state.world.camera.pose,
                         img_dir=join(outpath, 'rgb_images'), rgb=True)
        # reset_simulation()
        # get_depth_images(outpath, camera_pose=state.world.camera.pose,
        #                  rgbd=True, robot=False, img_dir=outpath)

    if exit:
        print('exit at the end of save_to_kitchen_worlds')
        sys.exit()


def get_config_from_template(template_path):
    print('\n\nget_config_from_template', get_config_from_template, '\n\n')
    planning_config = json.load(open(join(template_path, 'planning_config.json'), 'r'))
    config = {k: v for k, v in planning_config.items() \
              if k in ['base_limits', 'robot_builder', 'robot_name', 'obs_camera_pose']}
    return config


PDDL_STR = """
(define
  (problem {world_name})
  (:domain {domain_name})

  (:objects
{objects_pddl}
  )

  (:init
{init_pddl}
  )

  (:goal (and
    {goal_pddl}
  ))
)
        """


def generate_problem_pddl(world, facts, goals, world_name='lisdf', domain_name='domain',
                          out_path=None, added_obj=None, added_init=None):

    if world_name is None and hasattr(world, 'lisdf'):
        world_name = world.lisdf.name

    ########################################################

    objects = world.get_planning_objects()
    objects.extend(world.robot.joint_groups)
    objects.extend([str(i[1]) for i in facts if i[0].lower() == 'wconf'])
    objects_pddl = '\t'+'\n\t'.join(sorted(objects))
    if added_obj is not None:
        objects_pddl += '\n\t;; added objects for larger problem' + \
                        '\n\t' + '\n\t'.join(sorted(added_obj))

    ########################################################
    if added_init is not None and 'sink' in out_path:
        new_supported = [i for i in added_init if i[0].lower() == 'supported']
        old_supported = [i for i in facts if i[0].lower() == 'supported']
        for i in old_supported:
            for j in new_supported:
                o = j[1] if not isinstance(j[1], str) else world.name_to_body[j[1]]
                s = j[-1] if not isinstance(j[-1], str) else world.name_to_body[j[-1]]
                if i[1] == o:
                    facts.remove(i)

    init_pddl = generate_init_pddl(world, facts)
    if added_init is not None:
        init_pddl += '\n\n\t;; added facts for larger problem' + \
                     generate_init_pddl(world, added_init)

    ########################################################

    if goals[0] == 'and':
        goals = [list(n) for n in goals[1:]]
    goal_pddl = '\n\t'.join([get_pddl_from_list(g, world) for g in sorted(goals)]).lower()

    problem_pddl = PDDL_STR.format(
        objects_pddl=objects_pddl, init_pddl=init_pddl, goal_pddl=goal_pddl,
        world_name=world_name, domain_name=domain_name
    ).lower()

    ########################################################

    if out_path is not None:
        with open(out_path, 'w') as f:
            f.writelines(problem_pddl)
    else:
        from pybullet_tools.logging_utils import myprint as print
        print(f'----------------{problem_pddl}')

    all_pred_names = {}  # pred: arity
    for fact in list(set(facts)):
        pred = fact[0]
        if pred.lower() not in all_pred_names:
            all_pred_names[pred.lower()] = len(fact[1:])
    return all_pred_names


def get_pddl_from_list(fact, world):
    fact = get_readable_list(fact, world, NAME_ONLY=True, TO_LISDF=True)
    line = ' '.join([str(ele) for ele in fact])
    line = line.replace("'", "")
    return '(' + line + ')'


def generate_init_pddl(world, facts):

    init_pddl = ''
    kinds = {}  # pred: continuous vars
    by_len = {}  # pred: length of fact
    predicates = {}  # pred: [fact]
    for fact in list(set(facts)):
        pred = fact[0]
        if pred in ['=']: continue
        fact = get_pddl_from_list(fact, world)

        if pred not in predicates:
            predicates[pred] = []

            num_con = len([o for o in fact[1:] if not isinstance(o, str)])
            if num_con not in kinds:
                kinds[num_con] = []
            kinds[num_con].append(pred)

            num = len(fact)
            if num not in by_len:
                by_len[num] = []
            by_len[num].append(pred)

        predicates[pred].append(fact)

    kinds = {k: v for k, v in sorted(kinds.items(), key=lambda item: item[0])}
    by_len = {k: v for k, v in sorted(by_len.items(), key=lambda item: item[0])}

    for kind, preds in kinds.items():
        pp = {k: v for k, v in predicates.items() if k in preds}

        if kind == 0:
            init_pddl += '\t;; discrete facts (e.g. types, affordances)'
        else:
            init_pddl += '\t;; facts involving continuous vars'

        for oo, ppds in by_len.items():
            ppp = {k: v for k, v in pp.items() if k in ppds}
            ppp = {k: v for k, v in sorted(ppp.items())}
            for pred in ppp:
                init_pddl += '\n\t' + '\n\t'.join(predicates[pred])
            init_pddl += '\n'

    return init_pddl
