from pybullet_tools.utils import aabb_contains_point, get_aabb
from pddlstream.language.generator import from_test
from pybullet_tools.flying_gripper_agent import get_stream_map as get_stream_map_base


def get_stream_map(p, c, l, t, **kwargs):
    stream_map = get_stream_map_base(p, c, l, t, **kwargs)
    stream_map.update({
        'test-plate-in-cabinet': from_test(get_pose_in_cabinet_test(p)),
        # 'test-plate-in-cabinet': from_test(get_pose_in_cabinet_test(p)),
    })
    return stream_map


def get_pose_in_cabinet_test(problem):
    world = problem.world

    def test(o, p):
        cabinet = world.name_to_body('cabinettop')
        in_cabinet = aabb_contains_point(p.value[0], get_aabb(cabinet))
        return in_cabinet
    return test