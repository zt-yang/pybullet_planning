from os.path import join, abspath, dirname, isdir, isfile

import lisdf.components as C
from lisdf.parsing import load_all

from pddlstream.language.constants import Equal, AND

from pybullet_tools.pr2_streams import WConf
from pybullet_tools.pr2_primitives import Pose, Conf
from pybullet_tools.pr2_utils import get_arm_joints, get_group_joints, create_gripper
from pybullet_tools.utils import quat_from_euler, remove_body

class Problem():
    def __init__(self, world):
        self.world = world
        self.robot = world.robot
        self.fixed, self.movable, self.floors = self.init_from_world(world)
        self.grasp_types = ['top']
        self.gripper = None

    def init_from_world(self, world):
        fixed = []
        movable = []
        floors = []
        for model in world.lisdf.models:
            if model.name != 'pr2':
                body = world.name_to_body[model.name]
                if model.static: fixed.append(body)
                else: movable.append(body)
            if hasattr(model, 'links'):
                for link in model.links:
                    if link.name == 'box':
                        for collision in link.collisions:
                            if collision.shape.size[-1] < 0.05:
                                floors.append(model)
        return fixed, movable, floors

    @property
    def obstacles(self):
        return [n for n in self.fixed if n not in self.floors]

    def add_init(self, init):
        pass

    def get_gripper(self, arm='left', visual=True):
        # upper = get_max_limit(problem.robot, get_gripper_joints(problem.robot, 'left')[0])
        # set_configuration(gripper, [0]*4)
        # dump_body(gripper)
        if self.gripper is None:
            self.gripper = create_gripper(self.robot, arm=arm, visual=visual)
        return self.gripper

    def remove_gripper(self):
        if self.gripper is not None:
            remove_body(self.gripper)
            self.gripper = None

def pddl_to_init_goal(exp_dir, world):

    lisdf, domain, problem = load_all(
        join(exp_dir, 'scene.lisdf'),
        join(exp_dir, 'domain.pddl'),
        join(exp_dir, 'problem.pddl'),
    )
    robot = world.robot
    existed = [] ## {k: [] for k in ['q', 'aq', 'p', 'g', 'hg', 'pstn', 'lp']}
    def check_existed(o):
        for e in existed:
            if o.__dict__ == e.__dict__:
                return e
        existed.append(o)
        return o

    def prop_to_list(v):
        args = [v.predicate.name]
        for arg in v.arguments:
            if isinstance(arg, C.PDDLObject):
                elem = arg.name
                if elem in world.name_to_body:
                    elem = world.name_to_body[elem]
            else:
                typ = ''.join([i for i in arg.name if not i.isdigit()])
                index = int(''.join([i for i in arg.name if i.isdigit()]))
                value = list(arg.value.value)
                if typ == 'q':
                    elem = Conf(robot, get_group_joints(robot, 'base'), value, index=index)
                elif typ == 'aq':
                    elem = Conf(robot, get_arm_joints(robot, args[-1]), value, index=index)
                elif typ == 'p':
                    body = args[-1]
                    if len(value) == 4:
                        value = value[:3] + [0, 0] + value[-1:]
                    elem = Pose(body, (tuple(value[:3]), quat_from_euler(value[-3:])), index=index)
                else:
                    print(f'\n\n\n\n\nnot implemented for typ {typ}\n\n\n\n')
                elem = check_existed(elem)
            args.append(elem)
        return args

    goal = [prop_to_list(v) for v in problem.conjunctive_goal]
    init = [prop_to_list(v) for v in problem.init]

    poses = {i[1]: i[2] for i in init if i[0] == 'AtPose'}
    positions = {i[1]: i[2] for i in init if i[0] == 'AtPosition'}
    wconf = WConf(poses, positions)
    init += [('WConf', wconf), ('InWConf', wconf)]

    init += [Equal(('PickCost',), 1), Equal(('PlaceCost',), 1)]

    return init, goal