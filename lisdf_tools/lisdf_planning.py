import copy
import json
from os.path import join, abspath, dirname, isdir, isfile

import lisdf.components as C
from lisdf.parsing import load_all

from pddlstream.language.constants import Equal, AND

from pybullet_tools.pr2_streams import WConf, HandleGrasp, Position  ##, MarkerGrasp
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
        # self.grasp_types = ['top']
        self.gripper = None

    @property
    def grasp_types(self):
        return self.robot.grasp_types

    @property
    def fixed(self):
        if self.world.fixed is None:
            self.world.check_world_obstacles()
        return self.world.fixed

    @property
    def movable(self):
        if self.world.movable is None:
            self.world.check_world_obstacles()
        return self.world.movable

    @property
    def floors(self):
        if self.world.floors is None:
            self.world.check_world_obstacles()
        return self.world.floors

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

    domain_file = join(exp_dir, 'domain.pddl')
    if not isfile(domain_file):
        config_file = join(exp_dir, 'planning_config.json')
        domain_file = json.loads(open(config_file).read())['domain']

    lisdf, domain, problem = load_all(
        join(exp_dir, 'scene.lisdf'),
        domain_file,
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

    def pose_from_tuple(value):
        if len(value) == 4:
            value = xyzyaw_to_pose(value)
        elif len(value) == 6:
            value = (tuple(value[:3]), quat_from_euler(value[3:]))
        elif len(value) == 2 and len(value[1]) == 3:
            value = (tuple(value[0]), quat_from_euler(value[1]))
        return value

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
                        if len(value) == 3:
                            elem = Conf(robot_body, get_group_joints(robot_body, 'base'), value, index=index)
                        elif len(value) == 4:
                            elem = Conf(robot_body, get_group_joints(robot_body, 'base-torso'), value, index=index)
                    elif 'feg' in robot.name:
                        elem = Conf(robot_body, get_se3_joints(robot_body), value, index=index)
                elif typ == 'aq':
                    elem = Conf(robot_body, get_arm_joints(robot_body, args[-1]), value, index=index)
                elif typ == 'p':
                    elem = Pose(args[-1], pose_from_tuple(value), index=index)
                elif typ == 'lp':
                    continue
                elif typ == 'pstn':
                    elem = Position(args[-1], value, index=index)
                elif typ.endswith('g'):
                    body = args[-1]
                    g = pose_from_tuple(value)
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
        return tuple(args)

    goal = [prop_to_list(v) for v in problem.conjunctive_goal]
    init = [prop_to_list(v) for v in problem.init]

    poses = {}  ## {i[1]: i[2] for i in init if i[0] == 'atpose'}
    positions = {i[1]: i[2] for i in init if i[0] == 'atposition'}

    ## just create a new one, if there aren't any in the (:objects
    inwconf = [i[1] for i in init if i[0].lower() == 'inwconf']
    if inwconf == ['None']: # no more wconf for the new planner
        to_remove = [('wconf', inwconf[0]), ('inwconf', inwconf[0])]
        to_add = [('wconf', None), ('inwconf', None)]
        init = [i for i in init if i not in to_remove] + to_add

    elif len(inwconf) > 0:
        to_remove = []
        inwconf = inwconf[0]
        # import ipdb; ipdb.set_trace()
        index = int(''.join([i for i in inwconf if i.isdigit()]))
        wconf = WConf(poses, positions, index=index)
        init += [('wconf', wconf), ('inwconf', wconf)]
        to_remove += [('wconf', inwconf), ('inwconf', inwconf)]

        newwconfpst = [i for i in init if i[0].lower() == 'newwconfpst']
        for n in newwconfpst:
            index = int(''.join([i for i in n[-1] if i.isdigit()]))
            new_positions = copy.deepcopy(positions)
            new_positions[n[2]] = n[3]
            new_wconf = WConf(poses, new_positions, index=index)
            init += [('wconf', new_wconf), ('newwconfpst', wconf, n[2], n[3], new_wconf)]
            to_remove += [('wconf', n[-1])]
        to_remove += newwconfpst
        init = [i for i in init if i not in to_remove]

    else:
        wconf = WConf(poses, positions)
        init += [('WConf', wconf), ('InWConf', wconf)]

    # ## ----------- debugging
    # new_init = []
    # remove_objects = [(5, 19), (5, 23), 6, 7, 8, 9, 10, 11, (5, 10)]
    # for i in init:
    #     found = False
    #     for elem in i:
    #         if elem in remove_objects:
    #             found = True
    #     if not found:
    #         new_init.append(i)
    # init = new_init
    # remove_objects = [str(o) for o in remove_objects]
    # world.body_to_name = {k: v for k, v in world.body_to_name.items() if str(k) not in remove_objects}

    init += [Equal(('PickCost',), 1), Equal(('PlaceCost',), 1)]
    constants = {k: k for k in domain.constants}

    return init, goal, constants