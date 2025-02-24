from os.path import dirname, join
from pprint import pprint

from pybullet_tools.utils import wait_unlocked, get_movable_joints, get_joint_name, quat_from_euler, \
    get_joint_positions, set_joint_positions, set_pose, joint_from_name, get_joint_position, \
    get_pose, unit_quat, draw_pose, unit_pose
from pybullet_tools.bullet_utils import load_robot_urdf, get_nice_pose, get_nice_joint_positions

from tutorials.test_utils import get_test_world

SHADOWHAND_URDF_PATH = join(dirname(__file__), 'assets/shadowhand_description')
SHADOWHAND_URDFS = {
    'right': join(SHADOWHAND_URDF_PATH, 'shadowhand.urdf'),
    'left': join(SHADOWHAND_URDF_PATH, 'shadowhand_left.urdf'),
}

# ## XML version has both hands
# SHADOWHAND_URDF_PATH = join(dirname(__file__), 'assets/shadowhand_xml')
# SHADOWHAND_URDFS = {
#     'right': join(SHADOWHAND_URDF_PATH, 'right_hand.xml'),
#     'left': join(SHADOWHAND_URDF_PATH, 'left_hand.xml'),
# }

SHADOWHAND_JOINTS = [
    # 'WRJ2', 'WRJ1',  ## skipping for now
    'FFJ4', 'FFJ3', 'FFJ2', 'FFJ1',
    'MFJ4', 'MFJ3', 'MFJ2', 'MFJ1',
    'RFJ4', 'RFJ3', 'RFJ2', 'RFJ1',
    'LFJ5', 'LFJ4', 'LFJ3', 'LFJ2', 'LFJ1',
    'THJ5', 'THJ4', 'THJ3', 'THJ2', 'THJ1'
]
##


def translate_to_xml_joint_name(k):
    return f'robot0:{k[:-1]}{eval(k[-1])-1}'


def test_shadowhand_urdf(left=False):
    robot = load_robot_urdf(SHADOWHAND_URDFS['left' if left else 'right'])
    ## ['WRJ2', 'WRJ1', 'FFJ4', 'FFJ3', 'FFJ2', 'FFJ1', 'MFJ4', 'MFJ3', 'MFJ2', 'MFJ1', 'RFJ4', 'RFJ3', 'RFJ2', 'RFJ1', 'LFJ5', 'LFJ4', 'LFJ3', 'LFJ2', 'LFJ1', 'THJ5', 'THJ4', 'THJ3', 'THJ2', 'THJ1']
    # print([get_joint_name(robot, j) for j in get_movable_joints(robot)])
    return robot


def set_shadowhand_pose_conf(robot, grasp_sample, verbose=False):
    from dexgraspnet_tools.dexgraspnet_utils import get_grasp_pose
    qpos = grasp_sample['qpos']

    trans, euler = get_grasp_pose(qpos)
    pose = trans, quat_from_euler(euler)
    if verbose: pprint(qpos)
    if verbose: print(f"before:\t {get_nice_pose(robot)}")
    set_pose(robot, pose)
    if verbose: print(f"after:\t {get_nice_pose(robot)}")

    # conf = [0, 0] + [qpos[translate_to_xml_joint_name(k)] for k in SHADOWHAND_JOINTS[2:]]
    conf = [qpos[translate_to_xml_joint_name(k)] for k in SHADOWHAND_JOINTS]
    joints = [joint_from_name(robot, k) for k in SHADOWHAND_JOINTS]
    if verbose: pprint(dict(zip(SHADOWHAND_JOINTS, conf)))
    if verbose: print(f"before:\t {get_nice_joint_positions(robot, joints)}")
    set_joint_positions(robot, joints, conf)
    if verbose: print(f"after:\t {get_nice_joint_positions(robot, joints)}")


## --------------------------------------------------------------------------


def test_load_robot_object_grasp(grasp_object='sem-Bottle-af3dda1cfe61d0fc9403b0d0536a04af'):
    from dexgraspnet_tools.dexgraspnet_utils import load_grasp_data, load_object_in_pybullet
    world = get_test_world(robot=None)
    robot = test_shadowhand_urdf()

    grasp_data = load_grasp_data(grasp_object, filtered=True)
    grasp_sample = grasp_data[1]

    load_object_in_pybullet(grasp_object, float(grasp_sample["scale"]))

    set_shadowhand_pose_conf(robot, grasp_sample)  ## convert for .urdf file
    wait_unlocked()


def test_load_two_hands():
    world = get_test_world(robot=None)
    draw_pose(unit_pose())

    right = test_shadowhand_urdf(left=False)
    set_pose(right, ((0, 0.15, 0), unit_quat()))

    left = test_shadowhand_urdf(left=True)
    set_pose(left, ((0, -0.15, 0), unit_quat()))

    wait_unlocked()


if __name__ == '__main__':
    """
    pip install transforms3d
    """
    # test_load_robot_object_grasp()
    test_load_two_hands()
