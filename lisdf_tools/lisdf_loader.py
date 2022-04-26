LINK_STR = '::'

import os
import sys
from os.path import join, abspath, dirname, isdir, isfile
sys.path.append('lisdf')
from lisdf.parsing.sdf_j import load_sdf
from lisdf.components.model import URDFInclude

import warnings
warnings.filterwarnings('ignore')

from pybullet_tools.pr2_problems import create_floor
from pybullet_tools.pr2_utils import set_group_conf, get_group_joints
from pybullet_tools.utils import load_pybullet, connect, wait_if_gui, HideOutput, \
    disconnect, set_pose, set_joint_position, joint_from_name, quat_from_euler, draw_pose, unit_pose, \
    set_camera_pose, set_camera_pose2, get_pose, get_joint_position, get_link_pose, get_link_name, \
    set_joint_positions, get_links, get_joints, get_joint_name, get_body_name
from pybullet_tools.bullet_utils import nice

ASSET_PATH = join(dirname(__file__), '..', '..', 'assets')

class World():
    def __init__(self, lisdf):
        self.lisdf = lisdf
        self.body_to_name = {}
        self.name_to_body = {}

    @property
    def robot(self):
        return self.name_to_body['pr2']

    def add_body(self, body, name):
        self.body_to_name[body] = name
        self.name_to_body[name] = body

    def update_objects(self, objects):
        for o in objects.values():
            if LINK_STR in o.name:
                body = self.name_to_body[o.name.split(LINK_STR)[0]]
                id = find_id(body, o.name)
                self.add_body(id, o.name)

    def summarize_all_objects(self):
        """ call this after pddl_to_init_goal() where world.update_objects() happens """
        from pybullet_tools.logging import myprint as print

        print('----------------')
        print(f'PART I: world objects |')
        for body, name in self.body_to_name.items():

            line = f'{body}\t  |  {name}'
            if isinstance(body, tuple) and len(body) == 2:
                body, joint = body
                pose = get_joint_position(body, joint)
            elif isinstance(body, tuple) and len(body) == 3:
                body, _, link = body
                pose = get_link_pose(body, link)
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

def load_lisdf_pybullet(lisdf_path, verbose=True):
    # scenes_path = dirname(os.path.abspath(lisdf_path))
    tmp_path = join(ASSET_PATH, 'tmp')

    connect(use_gui=True, shadows=False, width=1980, height=1238)
    draw_pose(unit_pose(), length=1.)
    create_floor()

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
        else:
            uri = join(tmp_path, f'{model.name}.sdf')
            with open(uri, 'w') as f:
                f.write(make_sdf_world(model.to_sdf()))

        with HideOutput():
            if verbose: print(f'..... loading {model.name} from {uri}', end="\r")
            body = load_pybullet(uri, scale=scale)
            if isinstance(body, tuple): body = body[0]
            bullet_world.add_body(body, model.name)

        ## set pose of body using PyBullet tools' data structure
        if model.name == 'pr2':
            pose = model.pose.pos
            set_group_conf(body, 'base', pose)
        else:
            pose = (model.pose.pos, quat_from_euler(model.pose.rpy))
            set_pose(body, pose)

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


if __name__ == "__main__":

    for lisdf_test in ['kitchen_lunch']: ## 'm0m_joint_test', 'kitchen_basics', 'kitchen_counter'
        lisdf_path = join(ASSET_PATH, 'scenes', f'{lisdf_test}.lisdf')
        world = load_lisdf_pybullet(lisdf_path, verbose=True)
        wait_if_gui('load next test scene?')
        disconnect()
