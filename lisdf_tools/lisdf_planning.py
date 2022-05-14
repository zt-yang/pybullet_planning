from os.path import join, abspath, dirname, isdir, isfile

import lisdf.components as C
from lisdf.parsing import load_all

from pddlstream.language.constants import Equal, AND

from pybullet_tools.pr2_streams import WConf, HandleGrasp, Position, LinkPose ##, MarkerGrasp
from pybullet_tools.pr2_primitives import Pose, Conf, APPROACH_DISTANCE, Grasp, \
    TOP_HOLDING_LEFT_ARM
from pybullet_tools.pr2_utils import get_arm_joints, get_group_joints, create_gripper
from pybullet_tools.utils import quat_from_euler, remove_body, get_unit_vector, unit_quat, \
    multiply
from pybullet_tools.bullet_utils import xyzyaw_to_pose
from pybullet_tools.flying_gripper_utils import get_se3_joints

class Problem():
    def __init__(self, world):
        self.world = world
        self.robot = world.robot
        self.fixed, self.movable, self.floors = self.init_from_world(world)
        # self.grasp_types = ['top']
        self.gripper = None

    def init_from_world(self, world):
        fixed = []
        movable = []
        floors = []
        for model in world.lisdf.models:
            if model.name not in ['pr2', 'feg']:
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
            self.gripper = self.robot.create_gripper(arm=arm, visual=visual)
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
    world.update_objects(problem.objects)
    robot = world.robot
    existed = [] ## {k: [] for k in ['q', 'aq', 'p', 'g', 'hg', 'pstn', 'lp']}
    def check_existed(o, debug=False):
        for e in existed:
            if debug:
                print('check_existed', o, e)
                print(o.__dict__)
                print(e.__dict__)
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
                value = arg.value.value
                robot_body = robot.body
                if isinstance(value, tuple): value = list(value)
                if typ == 'q':
                    if 'pr2' in robot.name:
                        elem = Conf(robot_body, get_group_joints(robot_body, 'base'), value, index=index)
                    elif 'feg' in robot.name:
                        elem = Conf(robot_body, get_se3_joints(robot_body), value, index=index)
                elif typ == 'aq':
                    elem = Conf(robot_body, get_arm_joints(robot_body, args[-1]), value, index=index)
                elif typ == 'p':
                    if len(value) == 4:
                        value = xyzyaw_to_pose(value)
                    elif len(value) == 2:
                        value = (tuple(value[0]), quat_from_euler(value[1]))
                    elem = Pose(args[-1], value, index=index)
                elif typ == 'lp':
                    continue
                elif typ == 'pstn':
                    elem = Position(args[-1], value, index=index)
                elif typ.endswith('g'):
                    body = args[-1]
                    g = value
                    approach_vector = APPROACH_DISTANCE * get_unit_vector([1, 0, 0])
                    app = multiply((approach_vector, unit_quat()), g)
                    a = TOP_HOLDING_LEFT_ARM
                    if typ == 'g':
                        elem = Grasp('top', body, g, app, a, index=index)
                    elif typ == 'hg':
                        elem = HandleGrasp('side', body, g, app, a, index=index)
                else:
                    elem = None
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