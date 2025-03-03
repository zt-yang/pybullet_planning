from pybullet_tools.utils import create_box, unit_quat, set_pose, Euler, unit_pose, \
    draw_pose, invert, INF, GREY, TAN, create_cylinder, Attachment, multiply, \
    get_aabb, aabb_union, draw_aabb, get_pose, get_aabb_center, get_aabb_extent

from world_builder.entities import Object


class ComposedObject(Object):
    def __init__(self, body, attachments, **kwargs):
        super(ComposedObject, self).__init__(body, **kwargs)
        self.attachments = attachments
        self._body_to_object = self._compute_transform()

    def _assign_attachments(self):
        for attachment in self.attachments:
            attachment.assign()

    def get_aabb(self):
        aabbs = [get_aabb(b) for b in [self.body] + [a.child for a in self.attachments]]
        return aabb_union(aabbs)

    def draw_aabb(self, **kwargs):
        draw_aabb(self.get_aabb(), **kwargs)

    def get_aabb_extent(self):
        get_aabb_extent(self.get_aabb())

    def _compute_transform(self):
        pose = get_pose(self.body)
        center = get_aabb_center(self.get_aabb())
        return multiply(invert(pose), (center, pose[1]))

    def get_pose(self):
        return multiply(get_pose(self.body), self._body_to_object)

    def set_pose(self, pose):
        set_pose(self.body, multiply(pose, invert(self._body_to_object)))
        self._assign_attachments()


## -------------------------------------------------------------------------------------------------------------


def get_corner_points(w=0.7, l=2.0, x=1.0, y=0.0, z=1.0, gap=0.05):
    xys = [(x + w / 2 - gap, y - l / 2 + gap), (x + w / 2 - gap, y + l / 2 - gap),
           (x - w / 2 + gap, y - l / 2 + gap), (x - w / 2 + gap, y + l / 2 - gap)]
    return [list(xy) + [z] for xy in xys]


def create_table_entity(w=0.7, l=2, h=0.5, x=1, y=0, thickness=0.05, radius=0.03, gap=0.05):
    counter = create_box(w, l, thickness, color=GREY)
    counter_pose = ((x, y, h), unit_quat())
    set_pose(counter, counter_pose)

    ## leg poses
    two_legs = (w - (radius * 2 + gap * 2) * 2) < 0
    if two_legs:
        corner_points = [(x, y - l / 2 + gap, h / 2), (x, y + l / 2 - gap, h / 2)]
    else:
        corner_points = get_corner_points(w, l, x, y, h/2, gap)

    attachments = []
    for point in corner_points:
        pillar = create_cylinder(radius, h, color=GREY)
        pillar_pose = (point, unit_quat())
        transform = multiply(invert(counter_pose), pillar_pose)
        set_pose(pillar, pillar_pose)
        attachments.append(Attachment(counter, -1, transform, pillar))
    return ComposedObject(counter, attachments, category='table', name='table')


# def create_shelf_entity(w=0.7, l=2, x=1, y=0, h_shelf=0.2, n_shelves=4, h_lowest=0.2, thickness=0.05, gap=0.05):
#     counter = create_box(w, l, thickness, color=GREY)
#     counter_pose = ((x, y, h), unit_quat())
#     set_pose(counter, counter_pose)
#     attachments = []
#     for point in get_corner_points(w, l, x, y, h / 2, gap):
#         pillar = create_cylinder(thickness * 0.7, h, color=GREY)
#         pillar_pose = (point, unit_quat())
#         transform = multiply(invert(counter_pose), pillar_pose)
#         set_pose(pillar, pillar_pose)
#         attachments.append(Attachment(counter, -1, transform, pillar))
#     return ComposedObject(counter, attachments, category='table', name='table')
