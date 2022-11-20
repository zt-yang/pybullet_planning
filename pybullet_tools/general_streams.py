from __future__ import print_function

import copy
import random
from itertools import islice, count
import math

import numpy as np

from pybullet_tools.utils import invert, get_all_links, get_name, set_pose, get_link_pose, \
    pairwise_collision, sample_placement, get_pose, Point, Euler, set_joint_position, \
    BASE_LINK, get_joint_position, get_aabb, quat_from_euler, flatten_links, multiply, \
    get_joint_limits, unit_pose, point_from_pose, draw_point, PI, quat_from_pose, angle_between, \
    tform_point
from pybullet_tools.pr2_primitives import Pose

from pybullet_tools.bullet_utils import sample_obj_in_body_link_space, nice, is_contained, \
    visualize_point, collided, sample_pose, xyzyaw_to_pose


class Position(object):
    num = count()
    def __init__(self, body_joint, value=None, index=None):
        self.body, self.joint = body_joint
        if value is None:
            value = get_joint_position(self.body, self.joint)
        elif value == 'max':
            value = self.get_limits()[1]
        elif value == 'min':
            value = self.get_limits()[0]
        self.value = float(value)
        if index == None: index = next(self.num)
        self.index = index
    @property
    def bodies(self):
        return flatten_links(self.body)
    @property
    def extent(self):
        if self.value == self.get_limits()[1]:
            return 'max'
        elif self.value == self.get_limits()[0]:
            return 'min'
        return 'middle'
    def assign(self):
        set_joint_position(self.body, self.joint, self.value)
    def iterate(self):
        yield self
    def get_limits(self):
        return get_joint_limits(self.body, self.joint)
    def __repr__(self):
        index = self.index
        #index = id(self) % 1000
        return 'pstn{}={}'.format(index, nice(self.value))


class HandleGrasp(object):
    def __init__(self, grasp_type, body, value, approach, carry, index=None):
        self.grasp_type = grasp_type
        self.body = body
        self.value = tuple(value) # gripper_from_object
        self.approach = tuple(approach)
        self.carry = tuple(carry)
        if index == None: index = id(self)
        self.index = index
    def get_attachment(self, robot, arm):
        return robot.get_attachment(self, arm)
        # tool_link = link_from_name(robot, PR2_TOOL_FRAMES[arm])
        # return Attachment(robot, tool_link, self.value, self.body)
    def __repr__(self):
        return 'hg{}={}'.format(self.index % 1000, nice(self.value))


""" ==============================================================

            From relative pose to world pose

    ==============================================================
"""


class RelPose(object):
    num = count()
    def __init__(self, body, #link=BASE_LINK,
                 reference_body=None, reference_link=BASE_LINK,
                 confs=[], support=None, init=False, index=None):
        self.body = body
        #self.link = link
        self.reference_body = reference_body
        self.reference_link = reference_link
        # Could also perform recursively
        self.confs = tuple(confs) # Attachment is treated as a conf
        self.support = support
        self.init = init
        self.observations = 0
        if index is None:
            index = next(self.num)
        self.index = index
        self.value = self.get_reference_from_body()
    @property
    def bodies(self):
        bodies = set() # (self.body, None)
        #if self.reference_body is not None:
        #    bodies.update({self.reference_body, frozenset(get_link_subtree(self.body, self.reference_link))})
        for conf in self.confs:
            bodies.update(conf.bodies)
        return bodies
    def assign(self):
        for conf in self.confs: # Assumed to be totally ordered
            conf.assign()
    def get_world_from_reference(self):
        if self.reference_body is None:
            return unit_pose()
        self.assign()
        return get_link_pose(self.reference_body, self.reference_link)
    def get_world_from_body(self):
        self.assign()
        return get_link_pose(self.body, BASE_LINK)
    def get_reference_from_body(self):
        return multiply(invert(self.get_world_from_reference()),
                        self.get_world_from_body())
    def draw(self, **kwargs):
        point_reference = point_from_pose(self.get_reference_from_body())
        if self.reference_body is None:
            return draw_point(point_reference, **kwargs)
        return draw_point(point_reference, parent=self.reference_body,
                          parent_link=self.reference_link, **kwargs)
    def __repr__(self):
        index = self.index  ## id(self) % 1000
        if self.reference_body is None:
            pose = get_pose(self.body)
            return 'wp{}={}'.format(index, nice(pose))
        rel_pose = self.get_reference_from_body()
        return 'rp{}=({},{})'.format(index, (self.reference_body, self.reference_link), nice(rel_pose))


def pose_from_attachment(attachment, **kwargs):
    return RelPose(attachment.child, reference_body=attachment.parent,
                   reference_link=attachment.parent_link, confs=[attachment], **kwargs)


def get_compute_pose_kin():
    def fn(o1, rp, o2, p2):
        if o1 == o2:
            return None
        # p1 = RelPose(o1, support=rp.support, confs=(p2.confs + rp.confs),
        #              init=(rp.init and p2.init))
        p1 = Pose(o1, value=multiply(p2.value, rp.value))
        return (p1,)
    return fn


""" ==============================================================

            Sampling placement ?p

    ==============================================================
"""


def get_stable_gen(problem, collisions=True, num_trials=20, verbose=False,
                   learned_sampling=True, **kwargs):
    from pybullet_tools.pr2_primitives import Pose
    obstacles = problem.fixed if collisions else []
    world = problem.world

    def gen(body, surface):
        if surface is None:
            surfaces = problem.surfaces
        else:
            surfaces = [surface]

        ## --------- Special case for plates -------------
        result = check_plate_placement(body, surfaces, obstacles, num_trials)
        if result is not None:
            return result
        ## ------------------------------------------------

        count = num_trials
        while count > 0: ## True
            count -= 1
            surface = random.choice(surfaces) # TODO: weight by area
            if isinstance(surface, tuple): ## (body, link)
                body_pose = sample_placement(body, surface[0], bottom_link=surface[-1], **kwargs)
            else:
                body_pose = sample_placement(body, surface, **kwargs)
            if body_pose is None:
                break

            ## hack to reduce planning time
            if learned_sampling:
                body_pose = learned_pose_sampler(world, body, surface, body_pose)

            p = Pose(body, body_pose, surface)
            p.assign()
            if not any(pairwise_collision(body, obst) for obst in obstacles if obst not in {body, surface}):
                yield (p,)
    return gen


def learned_pose_sampler(world, body, surface, body_pose):
    ## hack to reduce planning time
    if 'eggblock' in world.get_name(body) and 'braiser_bottom' in world.get_name(surface):
        (x, y, z), quat = body_pose
        x = 0.55
        body_pose = (x, y, z), quat
    return body_pose


def get_stable_list_gen(problem, num_samples=5, collisions=True, **kwargs):
    funk = get_stable_gen(problem, collisions=collisions, **kwargs)

    def gen(body, surface):
        g = funk(body, surface)
        poses = []
        while len(poses) < num_samples:
            try:
                pose = next(g)
                poses.append(pose)
            except StopIteration:
                break
        return poses
    return gen


def check_plate_placement(body, surfaces, obstacles, num_samples, num_trials=30):
    from pybullet_tools.pr2_primitives import Pose
    surface = random.choice(surfaces)
    poses = []
    trials = 0

    if 'plate-fat' in get_name(body):
        while trials < num_trials:
            y = random.uniform(8.58, 9)
            body_pose = ((0.84, y, 0.88), quat_from_euler((0, math.pi / 2, 0)))
            p = Pose(body, body_pose, surface)
            p.assign()
            if not any(pairwise_collision(body, obst) for obst in obstacles if obst not in {body, surface}):
                poses.append(p)
                # for roll in [-math.pi/2, math.pi/2, math.pi]:
                #     body_pose = (p.value[0], quat_from_euler((roll, math.pi / 2, 0)))
                #     poses.append(Pose(body, body_pose, surface))

                if len(poses) >= num_samples:
                    return [(p,) for p in poses]
            trials += 1
        return []

    if isinstance(surface, int) and 'plate-fat' in get_name(surface):
        aabb = get_aabb(surface)
        while trials < num_trials:
            body_pose = xyzyaw_to_pose(sample_pose(body, aabb))
            p = Pose(body, body_pose, surface)
            p.assign()
            if not any(pairwise_collision(body, obst) for obst in obstacles if obst not in {body, surface}):
                poses.append(p)
                if len(poses) >= num_samples:
                    return [(p,) for p in poses]
            trials += 1
        return []

    return None


def get_mod_pose(pose):
    (x, y, z), quat = pose
    return ((x, y, z+0.01), quat)


def get_contain_gen(problem, collisions=True, max_attempts=60, verbose=False, learned_sampling=False, **kwargs):
    from pybullet_tools.pr2_primitives import Pose
    obstacles = problem.fixed if collisions else []
    world = problem.world

    def gen(body, space):
        #set_renderer(verbose)
        title = f"  get_contain_gen({body}, {space}) |"
        if space is None:
            spaces = problem.spaces
        else:
            spaces = [space]
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            space = random.choice(spaces)  # TODO: weight by area

            if isinstance(space, tuple):
                x, y, z, yaw = sample_obj_in_body_link_space(body, body=space[0], link=space[-1],
                                                             PLACEMENT_ONLY=True, verbose=verbose, **kwargs)
                body_pose = ((x, y, z), quat_from_euler(Euler(yaw=yaw)))
            else:
                body_pose = None
            if body_pose is None:
                break

            ## special sampler for data collection
            if 'storage' in world.get_name(space) or 'storage' in world.get_type(space) or learned_sampling:
                from world_builder.loaders import place_in_cabinet
                if verbose:
                    print('use special pose sampler')
                body_pose = place_in_cabinet(space, body, place=False)

            ## there will be collision between body and that link because of how pose is sampled
            p_mod = p = Pose(body, get_mod_pose(body_pose), space)
            p_mod.assign()
            obs = [obst for obst in obstacles if obst not in {body, space}]
            if not collided(body, obs, articulated=False, verbose=True):
                p = Pose(body, body_pose, space)
                yield (p,)
        if verbose:
            print(f'{title} reached max_attempts = {max_attempts}')
        yield None
    return gen


def get_contain_list_gen(problem, collisions=True, max_attempts=60, num_samples=10,
                    verbose=False, learned_sampling=False, **kwargs):
    funk = get_contain_gen(problem, collisions, max_attempts, verbose, learned_sampling, **kwargs)

    def gen(body, space):
        g = funk(body, space)
        poses = []
        while len(poses) < num_samples:
            try:
                pose = next(g)
                poses.append(pose)
            except StopIteration:
                break
        return poses
    return gen


def get_pose_in_space_test():
    def test(o, p, r):
        p.assign()
        answer = is_contained(o, r)
        print(f'general_streams.get_pose_in_space_test({o}, {p}, {r}) = {answer}')
        return answer
    return test


""" ==============================================================

            Sampling joint position ?pstn

    ==============================================================
"""


def get_joint_position_open_gen(problem):
    def fn(o, psn1, fluents=[]):  ## ps1,
        if psn1.extent == 'max':
            psn2 = Position(o, 'min')
        elif psn1.extent == 'min':
            psn2 = Position(o, 'max')
        return (psn2,)
    return fn


def sample_joint_position_open_list_gen(problem, num_samples = 10):
    def fn(o, psn1, fluents=[]):
        psn2 = None
        if psn1.extent == 'max':
            psn2 = Position(o, 'min')
            higher = psn1.value
            lower = psn2.value
        elif psn1.extent == 'min':
            psn2 = Position(o, 'max')
            higher = psn2.value
            lower = psn1.value
        else:
            # return [(psn1, )]
            higher = Position(o, 'max').value
            lower = Position(o, 'min').value
            if lower > higher:
                sometime = lower
                lower = higher
                higher = sometime

        positions = []
        if psn2 == None or abs(psn1.value - psn2.value) > math.pi/2:
            # positions.append((Position(o, lower+math.pi/2), ))
            lower += math.pi/2 ## - math.pi/8
            higher = min(lower + math.pi/6, get_joint_limits(o[0], o[1])[1])
            ptns = [np.random.uniform(lower, higher) for k in range(num_samples)]
            ptns.append(1.77)
            positions.extend([(Position(o, p), ) for p in ptns])
        else:
            positions.append((psn2,))

        for pstn in positions:
            yield pstn
        # return positions
    return fn


# ## discarded
# def get_position_gen(problem, collisions=True, extent=None):
#     obstacles = problem.fixed if collisions else []
#     def fn(o, fluents=[]):  ## ps1,
#         ps2 = Position(o, extent)
#         return (ps2,)
#     return fn
#
#
# ## discarded
# def get_joint_position_test(extent='max'):
#     def test(o, pst):
#         pst_max = Position(o, extent)
#         if pst_max.value == pst.value:
#             return True
#         return False
#     return test


""" ==============================================================

            Sampling grasps ?g

    ==============================================================
"""


def is_top_grasp(robot, arm, body, grasp, pose=unit_pose(), top_grasp_tolerance=PI/4): # None | PI/4 | INF
    if top_grasp_tolerance is None:
        return True
    grasp_pose = robot.get_grasp_pose(pose, grasp.value, arm, body=body)
    grasp_orientation = (Point(), quat_from_pose(grasp_pose))
    grasp_direction = tform_point(grasp_orientation, Point(x=+1))
    return angle_between(grasp_direction, Point(z=-1)) <= top_grasp_tolerance # TODO: direction parameter


def get_grasp_gen(problem, collisions=True, top_grasp_tolerance=None, # None | PI/4 | INF
                  randomize=True, visualize=False, RETAIN_ALL=False):
    robot = problem.robot
    grasp_type = 'hand'
    arm = 'left'

    def fn(body):
        from .bullet_utils import get_hand_grasps
        grasps_O = get_hand_grasps(problem, body, visualize=visualize, RETAIN_ALL=RETAIN_ALL)
        grasps = robot.make_grasps(grasp_type, arm, body, grasps_O, collisions=collisions)
        if top_grasp_tolerance is not None:
            grasps = [grasp for grasp in grasps if is_top_grasp(
                robot, arm, body, grasp, top_grasp_tolerance=top_grasp_tolerance)]
        if randomize:
            random.shuffle(grasps)
        # return [(g,) for g in grasps]
        for g in grasps:
           yield (g,)
    return fn


def get_grasp_list_gen(problem, collisions=True, num_samples=10, **kwargs):
    from pybullet_tools.pr2_primitives import get_grasp_gen as get_box_grasp_gen
    funk = get_grasp_gen(problem, collisions, **kwargs)
    funk2 = get_box_grasp_gen(problem, collisions)

    def gen(body):
        ## use the original grasp generator for box
        if len(get_all_links(body)) == 1:
            return funk2(body)
        g = funk(body)
        grasps = []
        while len(grasps) < num_samples:
            try:
                grasp = next(g)
                grasps.append(grasp)
            except StopIteration:
                break
        return grasps
    return gen


""" ==============================================================

            Sampling handle grasps ?hg

    ==============================================================
"""


def get_handle_link(body_joint):
    from world_builder.entities import ArticulatedObjectPart
    body, joint = body_joint
    j = ArticulatedObjectPart(body, joint)
    return j.handle_link


def get_handle_pose(body_joint):
    from world_builder.entities import ArticulatedObjectPart
    body, joint = body_joint
    j = ArticulatedObjectPart(body, joint)
    return j.get_handle_pose()


def get_handle_width(body_joint):
    from world_builder.entities import ArticulatedObjectPart
    body, joint = body_joint
    j = ArticulatedObjectPart(body, joint)
    return j.handle_width


def get_handle_grasp_list_gen(problem, collisions=True, num_samples=10, **kwargs):
    funk = get_handle_grasp_gen(problem, collisions, **kwargs)

    def gen(body):
        g = funk(body)
        grasps = []
        while len(grasps) < num_samples:
            try:
                grasp = next(g)
                grasps.append(grasp)
            except StopIteration:
                break
        return grasps
    return gen


def get_handle_grasp_gen(problem, collisions=False, max_samples=2,
                         randomize=False, visualize=False, verbose=False):
    collisions = True
    obstacles = problem.fixed if collisions else []
    world = problem.world
    robot = problem.robot
    title = 'pr2_streams.get_handle_grasp_gen |'
    def fn(body_joint):
        body, joint = body_joint
        handle_link = get_handle_link(body_joint)
        # print(f'{title} handle_link of body_joint {body_joint} is {handle_link}')

        g_type = 'top'
        arm = 'hand'
        if robot.name.startswith('pr2'):
            arm = 'left'
        from bullet_utils import get_hand_grasps

        grasps = get_hand_grasps(problem, body, link=handle_link, HANDLE_FILTER=True,
                    visualize=visualize, RETAIN_ALL=False, LENGTH_VARIANTS=True, verbose=verbose)

        if verbose: print(f'\n{title} grasps =', [nice(g) for g in grasps])

        app = robot.get_approach_vector(arm, g_type)
        grasps = [HandleGrasp('side', body_joint, g, robot.get_approach_pose(app, g),
                              robot.get_carry_conf(arm, g_type, g)) for g in grasps]
        for grasp in grasps:
            if robot.name.startswith('feg'):
                body_pose = get_link_pose(body, handle_link)
                if verbose: print(f'{title} get_link_pose({body}, {handle_link})'
                                  f' = {nice(body_pose)} | grasp = {nice(grasp.value)}')
                grasp.grasp_width = robot.compute_grasp_width(arm, body_pose,
                                    grasp.value, body=body_joint, verbose=verbose) if collisions else 0.0
            elif robot.name.startswith('pr2'):
                grasp.grasp_width = get_handle_width(body_joint)

        if randomize:
            random.shuffle(grasps)
        if max_samples is not None and len(grasps) > max_samples:
            random.shuffle(grasps)
            grasps = grasps[:max_samples]
        # return [(g,) for g in grasps]
        for g in grasps:
           yield (g,)
    return fn


def linkpose_from_position(pose):
    pose.assign()
    handle_link = get_handle_link((pose.body, pose.joint))
    # joint = world.BODY_TO_OBJECT[(pose.body, pose.joint)]
    pose_value = get_link_pose(pose.body, handle_link)
    return pose_value ## LinkPose(pose.body, joint, pose_value)


def get_pose_from_attachment(problem):
    from pybullet_tools.pr2_primitives import Pose
    world = problem.world
    def fn(o, w):
        old_pose = get_pose(o)
        w.assign()
        for body in set([b[0] for b in w.positions]):
            world.assign_attachment(body, tag='during pre-processing')

        if old_pose != get_pose(o):
            p = Pose(o, get_pose(o))
            return (p,)
        return None
    return fn


""" ==============================================================

            Checking collisions

    ==============================================================
"""


def get_cfree_approach_pose_test(problem, collisions=True):
    # TODO: apply this before inverse kinematics as well
    arm = 'left'
    obstacles = problem.fixed

    def test(b1, p1, g1, b2, p2):
        if not collisions or (b1 == b2):
            return True
        p2.assign()
        gripper = problem.get_gripper()
        result = False
        for _ in problem.robot.iterate_approach_path(arm, gripper, p1, g1, obstacles=obstacles,  body=b1):
            if pairwise_collision(b1, b2) or pairwise_collision(gripper, b2):
                result = False
                break
            result = True
        return result
    return test



""" ==============================================================

            Dealing with collisions

    ==============================================================
"""


def process_motion_fluents(fluents, robot, verbose=False):
    if verbose:
        print('Fluents:', fluents)
    attachments = []
    for atom in fluents:
        predicate, args = atom[0], atom[1:]
        if predicate == 'atpose':
            o, p = args
            if o not in ['@world']:
                p.assign()
        elif predicate == 'atgrasp':
            a, o, g = args
            attachments.append(g.get_attachment(robot, a))
        elif predicate == 'atposition':
            o, p = args
            p.assign()
        elif predicate == 'ataconf': # TODO: the arm conf isn't being set pre/post moves correctly
            # a, q = args
            # q.assign()
            pass
        else:
            raise NotImplementedError(atom)
    return attachments