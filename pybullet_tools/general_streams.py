from __future__ import print_function

import copy
import random
from itertools import islice, count
import math

import numpy as np
import pybullet

from pybullet_tools.utils import invert, ConfSaver, get_name, set_pose, get_link_pose, \
    pairwise_collision, sample_placement, get_pose, Point, Euler, set_joint_position, \
    BASE_LINK, get_joint_position, get_aabb, quat_from_euler, flatten_links, multiply, \
    get_joint_limits, unit_pose, point_from_pose, draw_point, PI, quat_from_pose, angle_between, \
    tform_point, interpolate_poses, draw_pose, RED, remove_handles, stable_z, wait_unlocked, \
    get_aabb_center, set_renderer, timeout, get_aabb_extent, wait_if_gui, wait_for_duration, \
    get_joint_type, PoseSaver, draw_aabb, LockRenderer, get_unit_vector, unit_quat, get_center_extent
from pybullet_tools.pr2_primitives import Pose, Grasp, APPROACH_DISTANCE, GRASP_LENGTH

from pybullet_tools.bullet_utils import nice, visualize_point, collided, is_box_entity, \
    query_yes_no
from pybullet_tools.pose_utils import sample_obj_in_body_link_space, is_contained, \
    ObjAttachment, sample_obj_on_body_link_surface, has_much_larger_aabb, adjust_sampled_pose
from pybullet_tools.camera_utils import set_camera_target_body
from pybullet_tools.grasp_utils import get_hand_grasps, sample_from_pickled_grasps, is_top_grasp, \
    needs_special_grasp


class LinkPose(Pose):
    num = count()
    def __init__(self, body, value=None, support=None, init=False, index=None,
                 joint=None, position=None):
        if value is None:
            value = get_link_pose(body[0], body[-1])
        super().__init__(body, value=value, support=support, init=init, index=index)
        self.joint = joint
        self.position = position
    @property
    def bodies(self):
        return flatten_links(self.body[0])
    def assign(self):
        if self.joint is not None:
            set_joint_position(self.body[0], self.joint, self.position)
    def __repr__(self):
        index = self.index
        return 'lp{}={}'.format(index, nice(self.value))


class RelPose2(Pose):
    num = count()
    def assign(self):
        if isinstance(self.support, tuple):
            support_value = get_link_pose(self.support[0], self.support[-1])
        else:
            support_value = get_pose(self.support)
        pose = multiply(support_value, self.value)
        set_pose(self.body, pose)
    def __repr__(self):
        index = self.index
        return 'rp{}={}'.format(index, nice(self.value))


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
        return safely_get_joint_limits(self.body, self.joint)
    def is_prismatic(self):
        return get_joint_type(self.body, self.joint) == pybullet.JOINT_PRISMATIC
    def is_revolute(self):
        return get_joint_type(self.body, self.joint) == pybullet.JOINT_REVOLUTE
    def __repr__(self):
        index = self.index
        #index = id(self) % 1000
        return 'pstn{}={}'.format(index, nice(self.value))


def safely_get_joint_limits(body, joint):
    """ some partnet mobility joint has range [-pi/2, 0] where 0 is closed """
    lower, upper = get_joint_limits(body, joint)
    if upper == 0 and lower < 0:
        i = upper
        upper = lower
        lower = i
    return lower, upper


class HandleGrasp(object):
    def __init__(self, grasp_type, body, value, approach, carry, index=None):
        self.grasp_type = grasp_type
        self.body = body
        self.value = tuple(value) # gripper_from_object
        self.approach = tuple(approach)
        self.carry = tuple(carry)
        if index is None: index = id(self)
        self.index = index

    def get_attachment(self, robot, arm, **kwargs):
        return robot.make_attachment(self, arm, **kwargs)
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
        with PoseSaver(self.body):
            with LockRenderer(True):
                self.assign()
                link_pose = get_link_pose(self.reference_body, self.reference_link)
        return link_pose

    def get_world_from_body(self):
        with PoseSaver(self.body):
            with LockRenderer(True):
                self.assign()
                pose = get_link_pose(self.body, BASE_LINK)
        return pose

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
        return 'rp{}={}'.format(index, nice(rel_pose))


def pose_from_attachment(attachment, **kwargs):
    return RelPose(attachment.child.body, reference_body=attachment.parent,
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


def get_compute_pose_rel_kin():
    def fn(o1, p1, o2, p2):
        if o1 == o2:
            return None
        p2.assign()
        p1.assign()
        if isinstance(o2, tuple):  ## TODO: is this possible?
            parent, parent_link = o2
            parent_link_pose = get_link_pose(parent, parent_link)
        else:
            parent = o2
            parent_link = BASE_LINK
            parent_link_pose = p2.value
        rel_pose = multiply(invert(parent_link_pose), p1.value)
        attachment = ObjAttachment(parent, parent_link, rel_pose, o1)
        rp = pose_from_attachment(attachment)
        return (rp,)
    return fn


""" ==============================================================

            Testing configuration ?conf

    ==============================================================
"""


def get_bconf_close_to_surface(problem):
    world = problem.world

    def test(bconf, surface, fluents=[]):
        ## when the surface is a link
        if isinstance(surface, tuple):
            surface = surface[0]
        point1 = np.asarray(bconf.values[:2])
        point2 = np.asarray(get_pose(surface)[0][:2])
        dist = np.linalg.norm(point1 - point2)
        return dist < 1
    return test


""" ==============================================================

            Sampling placement ?p

    ==============================================================
"""


def get_rel_pose(body, surface, body_pose):
    body_pose = multiply(invert(get_link_pose(surface[0], surface[-1])), body_pose)
    p = RelPose2(body, value=body_pose, support=surface)
    return p


def get_stable_gen(problem, collisions=True, num_samples=20, verbose=False, visualize=False,
                   learned_sampling=True, relpose=False, **kwargs):
    from pybullet_tools.pr2_primitives import Pose
    from world_builder.world_utils import smarter_sample_placement
    obstacles = problem.fixed if collisions else []
    world = problem.world
    robot = world.robot

    def gen(body, surface):
        ## , fluents=[] ## RuntimeError: Fluent streams certified facts cannot be domain facts
        # if fluents:
        #     attachments = process_motion_fluents(fluents, robot)

        title = f"  get_stable_gen({body}, {surface}) |"
        original_pose = get_pose(body)
        p0 = Pose(body, original_pose, surface)

        if surface is None:
            surfaces = problem.surfaces
        else:
            surfaces = [surface]
        obs = [obst for obst in obstacles if obst not in {body, surface}]
        count = num_samples

        ## ------------------------------------------------
        if learned_sampling and world.learned_pose_list_gen is not None:

            result = world.learned_pose_list_gen(world, body, surfaces, num_samples=num_samples-5, verbose=verbose)
            if result is not None:
                for body_pose in result:
                    p = Pose(body, value=body_pose, support=surface)
                    p.assign()
                    coo = collided(body, obs, verbose=verbose, visualize=visualize,
                                   tag='stable_gen_database', world=world)
                    if not coo:
                        count -= 1
                        p0.assign()
                        if relpose:
                            p = get_rel_pose(body, surface, body_pose)
                        yield (p,)

            if verbose: print(title, 'sample without learned_pose_list_gen')
        ## ------------------------------------------------

        while count > 0:
            count -= 1
            surface = random.choice(surfaces)  # TODO: weight by area
            if isinstance(surface, tuple):  ## (body, link)
                # body_pose = sample_obj_on_body_link_surface(body, surface[0], surface[-1])
                body_pose = sample_placement(body, surface[0], bottom_link=surface[-1], **kwargs)

                ## return false when the surface aabb is too small
                if body_pose is None:
                    bottom_aabb = get_aabb(surface[0], link=surface[-1])
                    _, extent = get_center_extent(body)
                    x, y, _ = get_aabb_center(bottom_aabb)
                    z = bottom_aabb.upper[2] + extent[2] / 2 + 0.01
                    body_pose = ((x, y, z), quat_from_euler(Euler(yaw=PI/2)))

            elif has_much_larger_aabb(surface, body):
                body_pose = smarter_sample_placement(body, surface, world, **kwargs)

            else:
                body_pose = sample_placement(body, surface, percent=0, **kwargs)  ## ok if one dimension is smaller
                body_pose = (body_pose[0], original_pose[1])

            if body_pose is None:
                break

            ## hack to reduce planning time
            if learned_sampling:
                body_pose = adjust_sampled_pose(world, body, surface, body_pose)

            p = Pose(body, value=body_pose, support=surface)
            p.assign()
            result = collided(body, obs, verbose=verbose, visualize=visualize, tag='stable_gen', world=world)
            if not result: ##  or ('braiser_bottom' in world.BODY_TO_OBJECT[surface].name):
                # if ('braiser_bottom' in world.BODY_TO_OBJECT[surface].name):
                #     print('\n\nallow bad samples inside pots')
                p0.assign()
                if relpose:
                    p = get_rel_pose(body, surface, body_pose)
                yield (p,)
            else:
                if visualize:
                    wait_unlocked()
    return gen


def get_stable_list_gen(problem, num_samples=5, **kwargs):
    funk = get_stable_gen(problem, num_samples=num_samples, **kwargs)

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


def get_mod_pose(pose):
    (x, y, z), quat = pose
    return ((x, y, z+0.01), quat)


def get_contain_gen(problem, collisions=True, num_samples=20, verbose=False, relpose=False,
                    learned_sampling=True, **kwargs):
    from pybullet_tools.pr2_primitives import Pose
    from world_builder.entities import Object
    obstacles = problem.fixed if collisions else []
    world = problem.world

    def gen(body, space):
        title = f"  get_contain_gen({body}, {space}) |"

        p0 = Pose(body, get_pose(body), space)
        attempts = 0
        obs = [obst for obst in obstacles if obst not in {body, space}]

        if space is None:
            spaces = problem.spaces
        else:
            spaces = [space]

        ## ------------------------------------------------
        if learned_sampling and world.learned_pose_list_gen is not None:
            result = world.learned_pose_list_gen(world, body, spaces, num_samples=num_samples-5, verbose=verbose)
            if result is not None:
                for body_pose in result:
                    p = Pose(body, value=body_pose, support=space)
                    p.assign()
                    # coo = collided(body, obs, verbose=verbose,
                    #                tag='contain_gen_database', world=world)
                    # if not coo:
                    attempts += 1
                    if relpose:
                        p = RelPose2(body, value=body_pose, support=space)
                    yield (p,)

            if verbose:
                print(title, 'sample without learned_pose_list_gen')
        ## ------------------------------------------------

        if isinstance(space, int):
            print('trying to sample pose inside body', space)

        while attempts < num_samples:
            attempts += 1
            space = random.choice(spaces)  # TODO: weight by area

            if isinstance(space, Object):
                space = Object.pybullet_name
            if isinstance(space, tuple):
                result = sample_obj_in_body_link_space(body, body=space[0], link=space[-1],
                                                       PLACEMENT_ONLY=True, verbose=verbose, **kwargs)
                if result is None:
                    break
                x, y, z, yaw = result
                body_pose = ((x, y, z), quat_from_euler(Euler(yaw=yaw)))
            else:
                ## e.g. braiser body
                result = sample_obj_in_body_link_space(body, body=space, link=None,
                                                       PLACEMENT_ONLY=True, verbose=verbose, **kwargs)
                if result is None:
                    break
                _, _, z, yaw = result
                x, y, _ = get_aabb_center(get_aabb(space))
                body_pose = ((x, y, z), quat_from_euler(Euler(yaw=yaw)))

            if body_pose is None:
                break

            ## there will be collision between body and that link because of how pose is sampled
            p_mod = p = Pose(body, get_mod_pose(body_pose), space)
            p_mod.assign()
            if not collided(body, obs, articulated=False, verbose=verbose, tag='contain_gen', world=world):
                p = Pose(body, body_pose, space)
                p0.assign()
                if relpose:
                    p = get_rel_pose(body, space, body_pose)
                yield (p,)
        if verbose:
            print(f'{title} reached max_attempts = {num_samples}')
        yield None
    return gen


def get_contain_list_gen(problem, collisions=True, num_samples=50, relpose=False, **kwargs):
    funk = get_contain_gen(problem, collisions, num_samples=num_samples, relpose=relpose, **kwargs)

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

            Sampling pose above object for sprinkling, pouring

    ==============================================================
"""


def get_above_pose_gen(problem, collisions=True, num_samples=2, visualize=False):
    def gen(region, p2, body):
        if isinstance(region, int):  ## otherwise it's static link
            p2.assign()
            aabb = get_aabb(region)
        else:
            aabb = get_aabb(region[0], link=region[-1])
        h = get_aabb_extent(get_aabb(body))[2]
        (x, y, _), quat = p2.value
        z = aabb.upper[2]
        for dz in [0.03, 0.05, 0.1, 0.15]:
            point = x, y, z + h/2 + dz
            # yield (Pose(body, (point, quat)), )
            for _ in range(num_samples):
                yaw = random.uniform(0, 2*math.pi)
                p = Pose(body, (point, quat_from_euler(Euler(yaw=yaw, roll=PI))))
                if visualize:
                    set_renderer(True)
                    p.assign()
                yield (p, )
    return gen


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


def sample_joint_position_list_gen(problem, num_samples=6):
    funk = sample_joint_position_gen(problem, num_samples=6)

    def gen(o, psn1):
        pstn_gen = funk(o, psn1)
        positions = []
        while len(positions) < num_samples:
            try:
                position = next(pstn_gen)
                positions.append(position)
            except StopIteration:
                break
        return positions
    return gen


def sample_joint_position_closed_gen():
    """ generate closed positions """
    def fn(o, pstn1):
        upper = Position(o, 'max').value
        lower = Position(o, 'min').value
        yield (Position(o, lower), )
    return fn


def visualize_sampled_pstns(x_min, x_max, x_points):
    import matplotlib.pyplot as plt

    # Generate a range of x values within the specified range
    x_values = np.linspace(x_min, x_max, num=1000)

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, np.zeros_like(x_values), color='black', linestyle='-', linewidth=1)  # Plot the x-axis

    # Plot the points
    plt.scatter(x_points, np.zeros_like(x_points), color='blue', label='Points')

    # Set plot labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('1D Plot with Points')

    # Add legend
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.show()


def sample_joint_position_gen(problem, num_samples=14, p_max=PI, to_close=False, visualize=False, verbose=True):
    """ generate open positions if closed=False and closed positions if closed=True (deprecated) """
    world = problem.world

    def fn(o, pstn1):

        if world.learned_position_list_gen is not None:
            pstns = world.learned_position_list_gen(world, o, pstn1, num_samples=num_samples)
            if pstns is not None:
                positions = [(Position(o, p),) for p in pstns]
                for pstn in positions:
                    yield pstn

        is_drawer = pstn1.is_prismatic()

        upper = Position(o, 'max').value
        lower = Position(o, 'min').value
        if upper > p_max:
            upper = min([upper, p_max])
        if lower < -p_max:
            lower = max([lower, -p_max])

        x_min = lower
        x_max = upper

        pstns = []
        a_half = (upper - lower) / 2
        a_third = (upper - lower) / 3

        if is_drawer:
            if to_close:
                pstns.extend([np.random.uniform(lower, upper - a_half) for k in range(num_samples)])
            else:
                # pstns.extend([np.random.uniform(lower + 2 * a_third, upper) for k in range(num_samples)])
                pstns.extend([np.random.uniform(upper, upper) for k in range(num_samples)])

        else:
            if to_close:
                if lower < 0 and upper == 0:
                    pstns.append(upper)
                    pstns.extend([np.random.uniform(upper - a_third, upper) for k in range(num_samples)])
                    pstns = [pstn for pstn in pstns if pstn > pstn1.value]
                else:
                    pstns.append(lower)
                    pstns.extend([np.random.uniform(lower, lower + a_third) for k in range(num_samples)])
                    pstns = [pstn for pstn in pstns if pstn < pstn1.value]

            else:

                lower_new, upper_new = None, None
                if lower < 0 and upper == 0:
                    # ## prevent from opening all the way is unreachable
                    # if upper - lower > 3/4 * math.pi:
                    #     lower_new = upper - 3/4 * math.pi
                    # pstns.append(upper - math.pi/2)

                    ## prevent from opening too little
                    if upper - lower > 1/2 * math.pi:
                        upper_new = upper - 1/2 * math.pi
                    else:
                        upper_new = lower + a_third
                else:
                    # ## prevent from opening all the way is unreachable
                    # if upper - lower > 3/4 * math.pi:
                    #     upper_new = lower + 3/4 * math.pi
                    # pstns.append(lower + math.pi*2/3)

                    ## prevent from opening too little
                    if upper - lower > 1/2 * math.pi:
                        lower_new = lower + 1/2 * math.pi
                    else:
                        lower_new = upper - a_third

                lower = lower_new if lower_new else lower
                upper = upper_new if upper_new else upper
                pstns.extend([np.random.uniform(lower, upper) for k in range(num_samples)])
                pstns = [pstn for pstn in pstns if abs(pstn) > abs(pstn1.value) + 0.3]

        pstns = [round(pstn, 3) for pstn in pstns]

        if visualize:
            visualize_sampled_pstns(x_min, x_max, pstns)

        if verbose:
            print(f'\tsample_joint_position_gen({o}, {pstn1.value}, closed={to_close}, p_max={p_max}) '
                  f'choosing from {pstns}, joint limits = [{round(x_min, 3)}, {round(x_max, 3)}]')
        random.shuffle(pstns)
        positions = [(Position(o, p), ) for p in pstns]

        for pstn2 in positions:
            yield pstn2
        # return positions
    return fn


""" ==============================================================

            Sampling grasps ?g

    ==============================================================
"""


def get_grasp_gen(problem, collisions=True, num_samples=20, randomize=True, verbose=True, debug=False,
                  top_grasp_tolerance=None, side_grasp_tolerance=None,  # None | PI/4 | INF
                  test_offset=False, loaded_offset=None, **kwargs):
    robot = problem.robot
    world = problem.world
    grasp_type = 'hand'
    arm = robot.arms[0]

    def fn(body, randomize_here=randomize):
        ## ----- get grasp transformations
        loaded = world.load_saved_grasps(body) if hasattr(world, 'load_saved_grasps') else None
        if loaded is None:
            grasps_O = get_hand_grasps(world, body, verbose=verbose, test_offset=test_offset, **kwargs)
        else:
            grasps_O, handles = sample_from_pickled_grasps(loaded, world, body, pose=get_pose(body), debug=debug,
                                                           offset=loaded_offset, k=num_samples)
            robot.debug_handles += handles
            randomize_here = False

        ## ----- get grasp objects
        grasps = robot.make_grasps(grasp_type, arm, body, grasps_O, collisions=collisions)
        # debug weiyu: assume that we don't need contact
        # grasps = robot.make_grasps(grasp_type, arm, body, grasps_O, collisions=False)

        ## ------ filter grasps
        if not test_offset and not needs_special_grasp(body, world):
            count_old = len(grasps)
            if top_grasp_tolerance is not None or side_grasp_tolerance is not None:
                grasps = [grasp for grasp in grasps if is_top_grasp(
                    robot, arm, body, grasp, top_grasp_tolerance=top_grasp_tolerance,
                    side_grasp_tolerance=side_grasp_tolerance
                )]
            count_new = len(grasps)
            if verbose:
                print(f'\tget_grasp_gen(top_grasp_tolerance={top_grasp_tolerance},'
                      f'side_grasp_tolerance={side_grasp_tolerance})',
                      f' selected {count_new} out of {count_old} grasps')

        if randomize_here:
            random.shuffle(grasps)
        # print(f'get_grasp_gen({body}, {world.get_name(body)}) = {len(grasps)} grasps')
        # return [(g,) for g in grasps]
        i = 0
        for g in grasps:
            i += 1
            yield (g,)

        # if debug:
        #     set_renderer(True)
        #     wait_unlocked()
        #     set_renderer(False)

    return fn


# def get_box_grasp_gen(problem, grasp_length=GRASP_LENGTH, grasp_types=None, collisions=False, randomize=True):
#     from pybullet_tools.pr2_utils import TOP_HOLDING_LEFT_ARM, SIDE_HOLDING_LEFT_ARM, \
#         get_top_grasps, get_side_grasps
#
#     world = problem.world
#     robot = world.robot
#     if grasp_types is None:
#         grasp_types = problem.grasp_types
#
#     def fn(body):
#         grasps = []
#         arm = robot.arms[0]
#
#         if 'top' in grasp_types:
#             # approach_vector = APPROACH_DISTANCE*get_unit_vector([1, 0, 0])
#             # grasps.extend(Grasp('top', body, g, multiply((approach_vector, unit_quat()), g), TOP_HOLDING_LEFT_ARM)
#             #               for g in get_top_grasps(body, grasp_length=grasp_length))
#             app = robot.get_approach_vector(arm, 'top')
#             grasps.extend(Grasp('top', body, g, robot.get_approach_pose(app, g), TOP_HOLDING_LEFT_ARM)
#                           for g in get_top_grasps(body, grasp_length=grasp_length))
#         if 'side' in grasp_types:
#             # approach_vector = APPROACH_DISTANCE*get_unit_vector([2, 0, -1])
#             # grasps.extend(Grasp('side', body, g, multiply((approach_vector, unit_quat()), g), SIDE_HOLDING_LEFT_ARM)
#             #               for g in get_side_grasps(body, grasp_length=grasp_length))
#             app = robot.get_approach_vector(arm, 'side')
#             grasps.extend(Grasp('side', body, g, robot.get_approach_pose(app, g), SIDE_HOLDING_LEFT_ARM)
#                           for g in get_side_grasps(body, grasp_length=grasp_length))
#         if 'hand' in grasp_types:
#             from pybullet_tools.grasp_utils import get_hand_grasps
#             approach_vector = APPROACH_DISTANCE*get_unit_vector([0, 0, -1])
#             grasps.extend(Grasp('hand', body, g, multiply(g, (approach_vector, unit_quat())), g)
#                           for g in get_hand_grasps(world, body))
#         filtered_grasps = []
#         for grasp in grasps:
#             grasp_width = robot.compute_grasp_width(arm, grasp, body=body) if collisions else 0.0
#             if grasp_width is not None:
#                 grasp.grasp_width = grasp_width
#                 filtered_grasps.append(grasp)
#         if randomize:
#             random.shuffle(filtered_grasps)
#         return [(g,) for g in filtered_grasps]
#         #for g in filtered_grasps:
#         #    yield (g,)
#     return fn


def get_grasp_list_gen(problem, collisions=True, num_samples=10, **kwargs):
    funk = get_grasp_gen(problem, collisions, num_samples=num_samples, **kwargs)
    # funk2 = get_box_grasp_gen(problem, collisions)

    def gen(body):
        # ## use the original grasp generator for box
        # if is_box_entity(body):  ## len(get_all_links(body)) == 1:
        #     return funk2(body)
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


def get_handle_link(body_joint, is_knob=False):
    from world_builder.entities import ArticulatedObjectPart, Knob
    body, joint = body_joint
    if is_knob:
        j = Knob(body, joint)
    else:
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
    funk = get_handle_grasp_gen(problem, collisions, max_samples=num_samples, **kwargs)

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
                         randomize=False, visualize=False, retain_all=False, verbose=False):
    collisions = True
    obstacles = problem.fixed if collisions else []
    world = problem.world
    robot = problem.robot
    title = 'general_streams.get_handle_grasp_gen |'

    def fn(body_joint):
        body, joint = body_joint
        is_knob = body_joint in world.cat_to_bodies('knob')
        handle_link = get_handle_link(body_joint, is_knob=is_knob)
        grasp_length = 0.13 if is_knob else 0.1
        # print(f'{title} handle_link of body_joint {body_joint} is {handle_link}')

        grasps = get_hand_grasps(world, body, link=handle_link, handle_filter=True,
                                 length_variants=True, grasp_length=grasp_length,
                                 visualize=visualize, verbose=verbose, retain_all=retain_all)

        if verbose: print(f'\n{title} grasps =', [nice(g) for g in grasps])

        g_type = 'top'
        arm = 'hand'
        if robot.name.startswith('pr2'):
            arm = 'left'
        app = robot.get_approach_vector(arm, g_type, scale=0.5)
        grasps = [HandleGrasp('side', body_joint, g, robot.get_approach_pose(app, g),
                              robot.get_carry_conf(arm, g_type, g)) for g in grasps]
        for grasp in grasps:
            if robot.name.startswith('pr2') and False:
                grasp.grasp_width = get_handle_width(body_joint)
            else:  ## if robot.name.startswith('feg'):
                body_pose = get_link_pose(body, handle_link)
                if verbose: print(f'{title} get_link_pose({body}, {handle_link})'
                                  f' = {nice(body_pose)} | grasp = {nice(grasp.value)}')
                grasp.grasp_width = robot.compute_grasp_width(arm, grasp, body=body_joint,
                                                              verbose=verbose) if collisions else 0.0

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


def get_reachable_test(radius=1.3, verbose=False):
    def test(a, o, p, g, q):
        with PoseSaver(o):
            set_pose(o, p.value)
            c_obj = get_aabb_center(get_aabb(o))[:2]
            c_robot = q.values[:2]
            distance = np.linalg.norm(c_obj - np.asarray(c_robot))
            result = distance < radius
            if verbose:
                print(f'general_streams.get_reachable_test({o}, {p}, {q}) -> {result}')
            return result
    return test


""" ==============================================================

            Nudge Grasp for Doors

    ==============================================================
"""


def get_nudge_grasp_list_gen(problem, collisions=True, num_samples=10, **kwargs):
    funk = get_nudge_grasp_gen(problem, collisions=collisions, max_samples=num_samples, **kwargs)

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


def get_nudge_grasp_gen(problem, collisions=True, nudge_back=False, max_samples=2,
                        randomize=False, visualize=False, retain_all=False, verbose=False):
    obstacles = problem.fixed if collisions else []
    world = problem.world
    robot = problem.robot
    title = f'general_streams.get_nudge_grasp_gen(nudge_back={nudge_back}, max_samples={max_samples}) |'

    def fn(body_joint):
        body, joint = body_joint
        handle_link = get_handle_link(body_joint)

        grasps = get_hand_grasps(world, body, link=handle_link, nudge=True, nudge_back=nudge_back,
                                 visualize=visualize, retain_all=retain_all, verbose=verbose)

        if verbose: print(f'\n{title} grasps =', [nice(g) for g in grasps])

        g_type = 'top'
        arm = 'hand'
        if robot.name.startswith('pr2'):
            arm = 'left'
        app = robot.get_approach_vector(arm, g_type, scale=0.5)
        grasps = [HandleGrasp('side', body_joint, g, robot.get_approach_pose(app, g),
                              robot.get_carry_conf(arm, g_type, g)) for g in grasps]

        if randomize:
            random.shuffle(grasps)
        if max_samples is not None and len(grasps) > max_samples:
            random.shuffle(grasps)
            grasps = grasps[:max_samples]
        # return [(g,) for g in grasps]
        for g in grasps:
           yield (g,)
    return fn


""" ==============================================================

            Checking collisions

    ==============================================================
"""


def get_cfree_obj_approach_pose_test(robot, collisions=True):
    """ KUKA version """
    def test(b1, p1, g1, b2, p2, fluents=[]):
        if not collisions or (b1 == b2) or b2 in ['@world']:
            return True
        if fluents:
            process_motion_fluents(fluents, robot)
        p2.assign()
        grasp_pose = multiply(p1.value, invert(g1.value))
        approach_pose = multiply(p1.value, invert(g1.approach), g1.value)
        for obj_pose in interpolate_poses(grasp_pose, approach_pose):
            set_pose(b1, obj_pose)
            if pairwise_collision(b1, b2):
                return False
        return True
    return test


def get_cfree_approach_pose_test(problem, collisions=True):
    """ PR2 version """
    # TODO: apply this before inverse kinematics as well
    arm = problem.robot.arms[0]
    gripper = problem.get_gripper()

    def test(b1, p1, g1, b2, p2):
        if not collisions or (b1 == b2) or b2 in ['@world']:
            return True
        p2.assign()
        bb2 = b2[0] if isinstance(b2, tuple) else b2
        result = True
        for _ in problem.robot.iterate_approach_path(arm, gripper, p1.value, g1, body=b1):
            if pairwise_collision(b1, bb2) or pairwise_collision(gripper, bb2):
                result = False
                break
        return result
    return test


def get_cfree_pose_pose_test(problem, collisions=True, visualize=False, **kwargs):
    robot = problem.robot
    ignored_pairs = problem.ignored_pairs
    def test(b1, p1, b2, p2, fluents=[]):
        if not collisions or (b1 == b2) or b2 in ['@world'] or (b1, b2) in ignored_pairs:
            return True
        if fluents:
            process_motion_fluents(fluents, robot)
        p1.assign()
        p2.assign()
        bb2 = b2[0] if isinstance(b2, tuple) else b2
        result = not pairwise_collision(b1, bb2, **kwargs)
        if not result and visualize:
            wait_unlocked()
        return result #, max_distance=0.001)
    return test


def get_cfree_rel_pose_pose_test(problem, collisions=True, visualize=False, **kwargs):
    robot = problem.robot
    ignored_pairs = problem.ignored_pairs
    def test(b1, rp1, b2, p2, b3, p3, fluents=[]):
        if not collisions or (b1 == b3) or b3 in ['@world'] or (b1, b2) in ignored_pairs:
            return True
        if fluents:
            process_motion_fluents(fluents, robot)
        p2.assign()
        rp1.assign()
        p3.assign()
        result = not pairwise_collision(b1, b3, **kwargs)
        if not result and visualize:
            wait_unlocked()
        return result #, max_distance=0.001)
    return test


def get_cfree_approach_rel_pose_test(problem, collisions=True):
    """ PR2 version """
    # TODO: apply this before inverse kinematics as well
    arm = problem.robot.arms[0]
    gripper = problem.get_gripper()

    def test(b1, rp1, b2, p2, g, b3, p3):
        if not collisions or (b1 == b3) or b3 in ['@world']:
            return True
        p3.assign()
        pose = multiply(p2.value, rp1.value)
        result = False
        for _ in problem.robot.iterate_approach_path(arm, gripper, pose, g, body=b1):
            if pairwise_collision(b1, b3) or pairwise_collision(gripper, b3):
                result = False
                break
            result = True
        return result
    return test


## ------------------------------------------------------------------------------------------------


def _check_cfree_traj_pose(c, b2, p2, world, collisions=True, verbose=False, visualize=True):
    from pybullet_tools.flying_gripper_utils import pose_from_se3
    # TODO: infer robot from c
    if not collisions:
        return True
    state = c.assign()

    if isinstance(b2, tuple):
        b2 = b2[0]
    if b2 in state.attachments:
        return True
    p2.assign()

    handles = []
    if visualize:
        if len(c.commands[0].path) == 6:
            for conf in c.commands[0].path:
                pose = pose_from_se3(conf.values)
                handles.extend(draw_pose(pose))
        # else:
        #     play_trajectory(c, p=p2, title='get_cfree_traj_pose_test')

    attached_objects = list(state.attachments.keys())
    if len(attached_objects) > 0 and isinstance(attached_objects[0], tuple):
        attached_objects = [b for b, j in attached_objects]
    obstacles = attached_objects + [world.robot]

    count = 0
    length = len(c.commands[0].path)
    result = True
    for _ in c.apply(state):
        count += 1
        if count == length > 10:
            continue
        title = f'[step {count}/{length}]'
        state.assign()
        if collided(b2, obstacles, verbose=verbose, world=world, tag=title):
            result = False
    if visualize:
        remove_handles(handles)
    return result


def get_cfree_traj_pose_test(problem, **kwargs):
    world = problem.world

    def test(c, b2, p2):
        return _check_cfree_traj_pose(c, b2, p2, world, **kwargs)
    return test


def get_cfree_traj_pose_at_bconf_at_joint_position_test(problem, **kwargs):
    world = problem.world

    def test(c, b2, p2, q, o, pstn):
        with ConfSaver(world.robot.body):
            q.assign()
            pstn.assign()
            return _check_cfree_traj_pose(c, b2, p2, world, **kwargs)
    return test


def get_cfree_traj_pose_at_bconf_at_joint_position_at_link_pose_test(problem, **kwargs):
    world = problem.world

    def test(c, b2, p2, q, o, pstn, l, p):
        with ConfSaver(world.robot.body):
            q.assign()
            pstn.assign()
            p.assign()
            return _check_cfree_traj_pose(c, b2, p2, world, **kwargs)
    return test


## ------------------------------------------------------------------------------------------------


def get_cfree_pose_between_test(robot, collisions=True, visualize=False):
    def test(b1, p1, b2, p2, b3, p3, fluents=[]):
        if not collisions or (b1 == b3) or (b2 == b3) or b2 in ['@world']:
            return True
        if fluents:
            process_motion_fluents(fluents, robot)
        p3.assign()
        with PoseSaver(b1):
            (x, y, z_upper), quat = p1.value
            z_lower = p2.value[0][-1]
            if isinstance(b3, tuple):
                b3 = b3[0]
            for z in np.linspace(z_lower, z_upper, num=5, endpoint=True)[1:-1]:
                set_pose(b1, ((x, y, z), quat))
                if pairwise_collision(b1, b3):
                    robot.log_collisions(b3, source='cfree_pose_between_test')
                    if visualize:
                        set_renderer(True)
                    return False
            return True
    return test


""" ==============================================================

            Dealing with collisions

    ==============================================================
"""


def process_motion_fluents(fluents, robot, verbose=False):

    ## sort the fluents so that AtRelPose is assigned after AtPose
    sorted_fluents = [f for f in fluents if f[0].lower() == 'atpose']
    sorted_fluents += [f for f in fluents if f[0].lower() == 'atrelpose']
    sorted_fluents += [f for f in fluents if f not in sorted_fluents]
    if verbose:
        print(f'[process_motion_fluents] len = {len(fluents)}:')
        print('\t'+'\n\t'.join([str(b) for b in sorted_fluents]))

    attachments = []
    for atom in sorted_fluents:
        predicate, args = atom[0].lower(), atom[1:]
        if predicate in ['atpose', 'atrelpose']:
            o, p = args[:2]
            if o not in ['@world'] and not isinstance(o, tuple):
                p.assign()
        elif predicate in ['atgrasp', 'atgrasphalf']:
            a, o, g = args
            attachments.append(g.get_attachment(robot, a))
        elif predicate == 'atposition':
            o, p = args
            p.assign()
        elif predicate == 'ataconf': # TODO: the arm conf isn't being set pre/post moves correctly
            from pddlstream.language.object import UniqueOptValue
            a, q = args
            # if not isinstance(q, UniqueOptValue):
            #     q.assign()
            # elif verbose:
            #     print('Skipping bc UniqueOptValue', atom)
            pass
        elif predicate == 'atseconf':
            [q] = args
            q.assign()
        else:
            raise NotImplementedError(atom)
    return attachments


def play_trajectory(cmd, p=None, attachment=None, obstacles=[], speed=0.02, title='play_trajectory'):
    set_camera_target_body(p.body, distance=2)
    set_renderer(True)

    def play_once(stepping=False):
        wait_if_gui(title + ' | start?')
        if p is not None:
            p.assign()
        wait_for_duration(speed)
        for conf in cmd.commands[0].path:
            conf.assign()
            if attachment is not None:
                attachment.assign()
                if collided(attachment.child, obstacles, verbose=True, tag='collided grasp path'):
                    print(title + '!!! test_trajectory | collided')

            if stepping:
                wait_if_gui('   step?')
            else:
                wait_for_duration(speed)
        wait_if_gui('finished')

    stepping = False
    answer = True
    while answer:
        play_once(stepping)
        stepping = True
        answer = query_yes_no(f"play again with stepping?", default='no')
    set_renderer(False)
