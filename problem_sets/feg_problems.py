from pybullet_tools.general_streams import Position

from world_builder.builders import *
from world_builder.loaders import *
from world_builder.loaders_partnet_kitchen import load_table_stationaries
from world_builder.loaders_nvidia_kitchen import load_feg_kitchen, load_feg_kitchen_dishwasher
from world_builder.world import State
from world_builder.paths import KITCHEN_WORLD

from robot_builder.robot_builders import *
import math

from os.path import join
import sys

from problem_sets.problem_utils import create_world, pddlstream_from_state_goal, problem_template


def change_world_state(world, test_case):
    sys.path.append(join('../..', 'lisdf'))
    from lisdf_tools.lisdf_utils import change_world_state as change_helper
    testcase_path = join(KITCHEN_WORLD, 'test_cases', test_case)
    return change_helper(world, testcase_path)

####################################################


def test_kitchen_domain(args, world_loader_fn, **kwargs):
    robot_builder_args = {
        'robot_name': 'feg',
        'draw_base_limits': True,
    }
    return problem_template(args, robot_builder_fn=build_oven_domain_robot,
                            robot_builder_args=robot_builder_args,
                            world_loader_fn=world_loader_fn, **kwargs)


####################################################


def test_feg_oven(args, **kwargs):
    def loader_fn(world, **world_builder_args):
        world.set_skip_joints()

        pot, lid, turkey, counter = load_gripper_test_scene(world)
        world.remove_category_from_planning('space')
        world.remove_category_from_planning('surface')

        arm = world.robot.arms[0]
        set_camera_target_body(lid, dx=0.8, dy=0, dz=0.4)

        # quick_demo(world)

        goals = ("test_object_grasps", lid)
        goals = [("Holding", arm, lid)]

        return goals
    return test_kitchen_domain(args, loader_fn, **kwargs)


def test_feg_lid(args, **kwargs):
    def loader_fn(world, **world_builder_args):
        world.set_skip_joints()

        pot, lid, turkey, counter = load_gripper_test_scene(world)
        arm = world.robot.arms[0]

        world.make_transparent(lid)

        world.remove_category_from_planning('space')
        # world.remove_category_from_planning('surface')
        # world.remove_category_from_planning('surface', exceptions=[pot])
        world.remove_category_from_planning('surface', exceptions=[pot, counter])

        set_camera_target_body(lid, dx=0.8, dy=0, dz=0.4)

        ## ------- grasping and placing various objects
        goals = [("Holding", 'hand', lid)]
        goals = [("On", lid, counter)]

        ## ------- on
        # world.remove_object(lid)
        goals = [("Holding", 'hand', turkey)]
        goals = [("On", turkey, pot)]

        ## ------- test HPN
        goals = [("On", turkey, pot), ("On", lid, counter)]

        return goals
    return test_kitchen_domain(args, loader_fn, **kwargs)


def test_feg_joints(args, **kwargs):
    def loader_fn(world, **world_builder_args):
        world.set_skip_joints()

        load_gripper_test_scene(world)

        door = 'chewie_door_right_joint'  ## 'baker_joint' ##'indigo_door_left_joint' ##
        door = world.add_joints_by_keyword('counter', door)[0]
        world.remove_category_from_planning('space')
        world.remove_category_from_planning('surface')
        world.remove_category_from_planning('movable')
        # door = name_to_body(door)

        goals = ('test_handle_grasps', door)
        goals = [("HandleGrasped", 'hand', door)]
        goals = ('test_update_wconf_pst', door)
        goals = ('test_door_pull_traj', door)
        goals = [("Debug2",)]
        goals = [("AtPosition", door, Position(door, 'max'))]
        goals = [("OpenedJoint", door)]
        goals = [("GraspedHandle", door)]
        return goals

    return test_kitchen_domain(args, loader_fn, **kwargs)


def test_feg_cabinets_from(args, **kwargs):
    def loader_fn(world, **world_builder_args):
        world.set_skip_joints()

        pot, lid, turkey, counter, oil, vinegar = load_cabinet_test_scene(world, MORE_MOVABLE=True)

        set_camera_target_body(lid, dx=2, dy=0, dz=0.5)
        world.remove_object(world.name_to_object('braiserlid'))
        world.remove_object(world.name_to_object('turkey'))
        world.remove_object(world.name_to_object('cabbage'))

        cabinet = world.name_to_body('dagger')
        door_1 = world.name_to_body('chewie_door_left_joint')
        left_door = door_2 = world.name_to_body('chewie_door_right_joint')
        right_door = door_3 = world.name_to_body('dagger_door_left_joint')
        door_4 = world.name_to_body('dagger_door_right_joint')
        right_cabinet = world.name_to_body('dagger')
        body = oil
        # world.open_all_doors_drawers()

        world.remove_category_from_planning('door', exceptions=[door_1, door_2])
        world.remove_category_from_planning('space', exceptions=[cabinet])
        world.remove_category_from_planning('surface', exceptions=[counter])

        ## ------- grasping and placing various objects
        goals = ("test_object_grasps", body)
        goals = ('test_reachable_pose', turkey)
        goals = ('test_sample_wconf', body)
        goals = ('test_at_reachable_pose', body)
        goals = [('OpenedJoint', left_door)]
        goals = [('OpenedJoint', right_door)]
        goals = [('OpenedJoint', left_door), ('OpenedJoint', right_door)]
        goals = [('On', oil, counter)]  ## start with all doors open
        goals = [('On', oil, counter), ('OpenedJoint', door_2)]  ## harder
        goals = [('On', oil, counter), ('OpenedJoint', door_1)]  ## hard
        goals = [('Holding', 'hand', oil), ('OpenedJoint', door_1), ('OpenedJoint', door_2)]
        goals = [('On', oil, counter), ('OpenedJoint', door_1), ('OpenedJoint', door_2)]
        goals = [("On", oil, counter)]
        # goals = [('On', oil, cabinet), ('OpenedJoint', left_door), ('OpenedJoint', right_door)]

        ## ------- door is partially open
        ### enough gap to get in but not enough to get out → HPN will fail?
        # open_joint(left_door[0], left_door[1], extent=0.3)
        # open_joint(right_door[0], right_door[1], extent=0.3)
        # goals = [("On", oil, counter)]

        return goals

    return test_kitchen_domain(args, loader_fn, **kwargs)


def test_feg_cabinets_to(args, **kwargs):
    def loader_fn(world, **world_builder_args):
        world.set_skip_joints()

        pot, lid, turkey, counter, oil, vinegar = load_cabinet_test_scene(world, MORE_MOVABLE=True)

        set_camera_target_body(lid, dx=2, dy=0, dz=0.5)
        world.remove_object(world.name_to_object('braiserlid'))
        world.remove_object(world.name_to_object('turkey'))
        world.remove_object(world.name_to_object('cabbage'))

        door_1 = world.name_to_body('chewie_door_left_joint')
        right_door = door_2 = world.name_to_body('chewie_door_right_joint')
        door_3 = world.name_to_body('dagger_door_left_joint')
        door_4 = world.name_to_body('dagger_door_right_joint')
        right_cabinet = world.name_to_body('dagger')

        # world.open_all_doors_drawers()
        # open_joint(left_door[0], left_door[1])
        # set_pose(oil, ((0.312, 7.426, 1.242), quat_from_euler((0, 0, 3.108))))

        world.remove_category_from_planning('space', exceptions=[right_cabinet])
        world.remove_category_from_planning('surface', exceptions=[counter])

        ## ------- grasping and placing various objects
        # goals = [("In", turkey, right_cabinet)]
        goals = [("Holding", 'hand', oil)]
        goals = ("test_handle_grasps", right_door)
        goals = ('test_door_pull_traj', right_door)
        goals = [("HandleGrasped", 'hand', right_door)]
        goals = [("GraspedHandle", right_door)]
        goals = [("On", oil, counter), ("GraspedHandle", right_door)]
        goals = [("On", oil, counter)]
        goals = [("In", oil, right_cabinet)]
        goals = [("In", oil, right_cabinet), ("GraspedHandle", door_2), ("GraspedHandle", door_3)]

        return goals

    return test_kitchen_domain(args, loader_fn, **kwargs)


def test_feg_cabinets_rearrange(args, **kwargs):
    def loader_fn(world, **world_builder_args):
        world.set_skip_joints()

        pot, lid, turkey, veggie, counter, oil, vinegar = load_cabinet_rearrange_scene(world)

        chewie_left = door_1 = world.name_to_body('chewie_door_left_joint')
        door_2 = world.name_to_body('chewie_door_right_joint')
        dagger_left = door_3 = world.name_to_body('dagger_door_left_joint')
        door_4 = world.name_to_body('dagger_door_right_joint')
        left_cabinet = world.name_to_body('sektion')
        right_cabinet = world.name_to_body('dagger')

        # world.open_all_doors_drawers()
        set_camera_target_body(lid, dx=3, dy=0, dz=0.5)

        world.remove_category_from_planning('surface')
        world.remove_category_from_planning('edible')
        world.remove_category_from_planning('braiserbody')
        world.remove_category_from_planning('braiserlid')

        ## ------- grasping and placing various objects
        goals = [("Holding", 'hand', turkey)]
        goals = [("In", turkey, right_cabinet), ("In", veggie, right_cabinet)]
        goals = [("In", oil, left_cabinet), ("In", vinegar, left_cabinet)]
        goals = [("StoredInSpace", '@bottle', left_cabinet),
                 ("GraspedHandle", dagger_left), ("GraspedHandle", chewie_left)]
        # goals = [("StoredInSpace", '@bottle', left_cabinet)]
        # goals = [("StoredInSpace", 'edible', right_cabinet)]
        # goals = [("StoredInSpace", 'edible', right_cabinet), ("GraspedHandle", dagger_left)]

        # goals = [("StoredInSpace", '@bottle', left_cabinet), ("GraspedHandle", dagger_left)]

        # goals = [("StoredInSpace", 'bottle', left_cabinet), ("StoredInSpace", 'edible', right_cabinet)]

        # test_case = 'test_feg_cabinets_rearrange'
        # change_world_state(world, test_case)
        # world.close_joint_by_name('dagger_door_left_joint')
        # world.close_joint_by_name('dagger_door_right_joint')
        # test_case = 'test_feg_cabinets_rearrange_1'

        return goals

    return test_kitchen_domain(args, loader_fn, **kwargs)


def test_feg_clean(args, **kwargs):
    def loader_fn(world, **world_builder_args):
        world.set_skip_joints()

        floor = load_floor_plan(world, plan_name='basin.svg')
        world.remove_object(floor)

        faucet, left_knob = load_basin_faucet(world)
        set_camera_target_body(faucet, dx=1.4, dy=1.5, dz=1)
        goals = ('test_handle_grasps', left_knob)
        goals = [("HandleGrasped", 'hand', left_knob)]
        goals = [("GraspedHandle", left_knob)]

        return goals
    return test_kitchen_domain(args, loader_fn, **kwargs)


def test_feg_cook(args, **kwargs):
    def loader_fn(world, **world_builder_args):
        world.set_skip_joints()

        floor = load_floor_plan(world, plan_name='counter.svg')
        cabbage = load_experiment_objects(world, CABBAGE_ONLY=True)
        bottom, lid = load_pot_lid(world)
        world.remove_object(floor)

        name_to_body = world.name_to_body
        name_to_object = world.name_to_object

        oven = name_to_body('oven')
        handles = world.add_joints_by_keyword('oven', 'knob_joint_2', 'knob')
        set_camera_target_body(oven, dx=0.5, dy=-0.3, dz=1.1)

        counter = name_to_body('counter')
        for surface in ['front_left_stove', 'front_right_stove', 'back_left_stove', 'back_right_stove']:
            set_color(counter, GREY, link_from_name(counter, surface))
        world.put_on_surface(cabbage, 'braiser_bottom')

        left_knob = name_to_object('knob_joint_2')
        set_color(left_knob.body, GREY, left_knob.handle_link)

        left_knob = name_to_body('knob_joint_2')
        goals = ('test_handle_grasps', left_knob)  ## for choosing grasps
        goals = [("HandleGrasped", 'left', left_knob)]
        goals = [("GraspedHandle", left_knob)]

        return goals
    return test_kitchen_domain(args, loader_fn, **kwargs)


def test_feg_dishwasher(args, **kwargs):
    def loader_fn(world, **world_builder_args):
        world.set_skip_joints()
        dishwasher_door = load_feg_kitchen_dishwasher(world)
        goals = ('test_handle_grasps', dishwasher_door)  ## for choosing grasps
        goals = [("GraspedHandle", dishwasher_door)]
        goals = ('test_new_wconf', dishwasher_door)  ## for choosing grasps

        food = world.name_to_body('turkey')
        body = world.name_to_body('platefat')
        surface = world.name_to_body('indigo_tmp')

        world.remove_category_from_planning('surface', exceptions=['surface_plate_left'])
        world.remove_category_from_planning('movable', exceptions=['plate'])

        goals = [("Holding", 'hand', world.name_to_body('turkey'))]

        ALREADY_OPEN = False

        ## ----------- while the dishwasher door is open
        if ALREADY_OPEN:
            world.open_joint(dishwasher_door[0], dishwasher_door[1])
            world.add_to_init(['IsOpenPosition', dishwasher_door])
            world.del_fr_init(['IsClosedPosition', dishwasher_door])
            goals = ("test_object_grasps", body)
            goals = [("Holding", 'hand', body)]
            goals = [("On", body, surface)]
            # goals = [("On", body, surface), ("On", food, body)]  ## fail because movable poses need to be in

        ## ----------- while the dishwasher door is closed
        else:
            goals = [("GraspedHandle", dishwasher_door)]
            goals = [("Debug4",)]
            # goals = [("GraspedHandle", dishwasher_door), ("Holding", 'hand', body)]
            # goals = [("GraspedHandle", dishwasher_door), ("On", body, surface)]
            # goals = [("Holding", 'hand', body)]  ## failed
            # goals = [("On", body, surface)]

        return goals
    return test_kitchen_domain(args, loader_fn, **kwargs)


def test_feg_dish(args, **kwargs):
    def loader_fn(world, **world_builder_args):
        world.set_skip_joints()

        cabbage, turkey, lid = load_feg_kitchen(world)
        set_camera_pose(camera_point=[4, 5, 3.5], target_point=[0, 5, 0.5])  ## made video for cook cabbage

        world.close_joint_by_name('fridge_door')
        # set_camera_target_body(cabbage, dx=0.5, dy=-0.5, dz=0.5)  ## made video for cook cabbage

        # world.remove_category_from_planning('braiserbody')
        world.remove_category_from_planning('space')
        world.remove_category_from_planning('knob', exceptions=['joint_faucet_0'])
        # world.remove_category_from_planning('surface')
        world.remove_category_from_planning('surface', exceptions=['basin_bottom', 'braiser_bottom'])

        goals = ("test_object_grasps", cabbage)
        goals = [("GraspedHandle", world.name_to_body('fridge_door'))]
        goals = [("Holding", 'hand', cabbage)]
        goals = [("On", cabbage, world.name_to_body('basin_bottom')),
                 ('GraspedHandle', world.name_to_body('joint_faucet_0'))]
        goals = [("On", cabbage, world.name_to_body('basin_bottom')),
                 ('GraspedHandle', world.name_to_body('joint_faucet_0')),
                 ("Cleaned", cabbage)]
        goals = [("Cleaned", cabbage)]

        goals = [("Cleaned", cabbage), ("On", cabbage, world.name_to_body('braiser_bottom'))]
        # goals = [("On", world.name_to_body('lid'), world.name_to_body('indigo_tmp')),
        #          ("On", cabbage, world.name_to_body('braiser_bottom')),
        #          ('GraspedHandle', world.name_to_body('knob_joint_2'))]
        # goals = [("On", world.name_to_body('lid'), world.name_to_body('indigo_tmp')),
        #          ("Cooked", cabbage)]
        # goals = [("Cooked", cabbage)]
        # goals = [("Cooked", turkey)]
        return goals
    return test_kitchen_domain(args, loader_fn, **kwargs)


# def test_feg_dish(args):
#     world = create_world(args)
#     world.set_skip_joints()
#
#     load_feg_kitchen(world)
#     custom_limits = {0: (0, 4), 1: (3, 12), 2: (0, 2)}
#     robot = create_gripper_robot(world, custom_limits, initial_q=[0.9, 8, 0.7, 0, -math.pi / 2, 0])
#
#     set_camera_pose(camera_point=[4, 5, 3.5], target_point=[0, 5, 0.5])  ## made video for cook cabbage
#     cabbage = world.name_to_body('cabbage')
#     turkey = world.name_to_body('turkey')
#     world.close_joint_by_name('fridge_door')
#     # set_camera_target_body(cabbage, dx=0.5, dy=-0.5, dz=0.5)  ## made video for cook cabbage
#
#     goals = ("test_object_grasps", cabbage)
#     goals = [("Holding", 'hand', cabbage)]
#     goals = [("Cleaned", cabbage)]
#     # goals = [("Cleaned", cabbage), ("On", cabbage, world.name_to_body('braiser_bottom'))]
#     # goals = [("On", cabbage, world.name_to_body('braiser_bottom')),
#     #          ('GraspedHandle', world.name_to_body('knob_joint_2'))]
#     # goals = [("On", world.name_to_body('lid'), world.name_to_body('indigo_tmp')),
#     #          ("On", cabbage, world.name_to_body('braiser_bottom')),
#     #          ('GraspedHandle', world.name_to_body('knob_joint_2'))]
#
#     # goals = [("On", world.name_to_body('lid'), world.name_to_body('indigo_tmp')),
#     #          ("Cooked", cabbage)]
#     # goals = [("Cooked", cabbage)]
#     # goals = [("Cooked", turkey)]
#
#     ## must exist after all objects and categories have been set
#     set_renderer(True)
#     set_all_static()
#     state = State(world, grasp_types=robot.grasp_types)
#     exogenous = []
#
#     pddlstream_problem = pddlstream_from_state_goal(state, goals, custom_limits=custom_limits,
#                                                     domain_pddl=args.domain_pddl, stream_pddl=args.stream_pddl)
#     save_to_kitchen_worlds(state, pddlstream_problem, exp_name='test_feg_clean_after_open',
#                            floorplan='kitchen_v3.svg', world_name='test_feg_clean_after_open', EXIT=False)
#     return state, exogenous, goals, pddlstream_problem


def test_feg_kitchen_demo(args):
    world = create_world(args)
    world.set_skip_joints()

    dishwasher_door = load_feg_kitchen(world)
    custom_limits = {0: (0, 4), 1: (3, 12), 2: (0, 2)}
    robot = create_gripper_robot(world, custom_limits, initial_q=[0.9, 8, 0.7, 0, -math.pi / 2, 0])

    # set_camera_target_body(world.name_to_body('dishwasher'), dx=0.5, dy=-0.5, dz=0.8)

    food = world.name_to_body('turkey')
    body = world.name_to_body('platefat')
    surface = world.name_to_body('indigo_tmp')
    cabbage = world.name_to_body('cabbage')
    turkey = world.name_to_body('turkey')

    ## ------------ for demo video
    world.open_joint(dishwasher_door[0], dishwasher_door[1])
    goals = [("GraspedHandle", dishwasher_door), ("On", body, surface)]
    goals = [("Cooked", cabbage)]
    # goals = [("Cooked", cabbage),
    #          ("GraspedHandle", dishwasher_door), ("On", body, surface)]

    ## ------- must exist after all objects and categories have been set
    set_renderer(True)
    set_all_static()
    state = State(world, grasp_types=robot.grasp_types)
    exogenous = []

    pddlstream_problem = pddlstream_from_state_goal(state, goals, custom_limits=custom_limits,
                                                    domain_pddl=args.domain_pddl, stream_pddl=args.stream_pddl)
    # save_to_kitchen_worlds(state, pddlstream_problem, exp_name='test_feg_pick',
    #                        floorplan='counter.svg', world_name='test_feg_pick', EXIT=True)
    return state, exogenous, goals, pddlstream_problem


######################################################################


def test_feg_tray(args, **kwargs):
    def loader_fn(world, **world_builder_args):
        world.set_skip_joints()

        items, distractors, tray_bottom, counter = load_table_stationaries(world)

        # random.shuffle(items)
        # item = items[0].body
        item = [i.body for i in items if i.category == 'stapler'][0]
        # item = [i.body for i in items if i.category == 'medicine'][0]
        for i in items:
            if i.body == item:
                continue
            world.remove_body_from_planning(i.body)

        world.add_highlighter(item)

        arm = world.robot.arms[0]
        tray_bottom = tray_bottom.pybullet_name
        distractor = distractors[0].body
        goals = [("Holding", arm, distractor)]
        goals = [("On", distractor, counter)]
        goals = [("On", distractor, counter), ("On", item, tray_bottom)]

        ## -------- just move the body in -----------
        # for d in distractors:
        #     world.remove_object(d)
        goals = ("test_object_grasps", item)
        goals = [("Holding", arm, item)]
        goals = [("On", item, tray_bottom)]
        goals = [("On", item, tray_bottom), ("On", distractor, tray_bottom)]
        # pose = ((2.907455991538716, 2.4702250605425533, 1.280301824238015),
        #         (0.0, 0.0, 0.044548410445173214, 0.9990072267640552))
        # goals = [('AtPose', item, Pose(item, pose))]

        ## -------- move the tray -----------
        # tray = tray_bottom.body
        # world.add_to_cat(tray, 'movable')
        # goals = [("Holding", arm, tray)]

        return goals

    kwargs['robot_builder_args']['initial_xy'] = (4, 3)
    return problem_template(args, robot_builder_fn=build_fridge_domain_robot,
                            world_loader_fn=loader_fn, **kwargs)