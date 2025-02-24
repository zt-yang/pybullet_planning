import time
from pybullet_tools.pr2_utils import attach_viewcone, draw_viewcone, get_viewcone_base, set_group_conf, get_arm_joints
from pybullet_tools.utils import get_joint_name, get_joint_position, get_link_name, get_link_pose, get_pose, set_pose, \
    joint_from_name, get_movable_joints, get_joint_positions, set_joint_position, set_joint_positions, link_from_name, \
    get_all_links, BodySaver, get_collision_data, tform_oobb, oobb_from_data, aabb_from_oobb, get_aabb, \
    remove_handles, remove_body, get_custom_limits, get_subtree_aabb, get_image_at_pose, draw_pose, multiply, Pose, RED, \
    all_between, get_name, dump_link, dump_joint, dump_body, PoseSaver, get_color, GREEN, unit_pose, \
    add_text, AABB, Point, Euler, PI, add_line, YELLOW, BLACK, remove_handles, get_com_pose, Pose, invert, \
    stable_z, get_joint_descendants, get_link_children, get_joint_info, get_links, link_from_name, set_renderer, \
    get_min_limit, get_max_limit, get_link_parent, LockRenderer, HideOutput, pairwise_collisions, get_bodies, \
    remove_debug, child_link_from_joint, unit_point, tform_point, buffer_aabb, get_aabb_center, get_aabb_extent, \
    quat_from_euler, get_link_subtree
from pybullet_tools.bullet_utils import BASE_LINK, is_box_entity, collided, nice, BASE_RESOLUTIONS, \
    CAMERA_MATRIX, colorize_link
from pybullet_tools.pose_utils import sample_obj_in_body_link_space, sample_obj_on_body_link_surface, \
    create_attachment, change_pose_interactive
from pybullet_tools.camera_utils import get_camera_image_at_pose, set_camera_target_body
from pybullet_tools.logging_utils import print_pink

from world_builder.world_utils import get_mobility_id, get_mobility_category, get_mobility_identifier, \
    get_instance_name, get_lisdf_name, load_asset, draw_body_label

from robot_builder.robot_utils import BASE_GROUP

import numpy as np

LINK_STR = '::'  ## same as in pybullet_planning/lisdf_tools/lisdf_loader


class Index(object):
    def __int__(self):
        raise NotImplementedError()
    def __eq__(self, other):
        if other is None:
            print('\n entities | __eq__(self, other): if other is None \n')
        if self is other:
            return True
        try:
            return int(self) == int(other)
        except ValueError:
            return False
    def __ne__(self, other):
        return not self == other
    def __lt__(self, other): # For heapq on python3
        return int(self) < int(other)
    def __hash__(self):
        return hash(int(self))


class Joint(Index):
    def __init__(self, body, index):
        self.body = body
        self.index = index
    @property
    def name(self):
        return get_joint_name(self.body, self.index)
    def __int__(self):
        return self.index
    def get_position(self):
        return get_joint_position(self.body, self.index)
    def __repr__(self):
        return self.name


class Link(Index):
    def __init__(self, body, index):
        self.body = body
        self.index = index
    @property
    def name(self):
        return get_link_name(self.body, self.index)
    def __int__(self):
        return self.index
    def get_pose(self):
        return get_link_pose(self.body, self.index)
    def __repr__(self):
        return self.name

#######################################################


class Object(Index):
    def __init__(self, body, joint=None, link=None, category=None, name=None,
                 collision=True, grasps=None, verbose=False):

        if isinstance(body, Object):
            body = body.body

        self.is_box = False
        if isinstance(body, tuple) and isinstance(body[1], str):
            body, path, scale = body[:3]
            self.path = path
            self.scale = scale
            self.mobility_id = get_mobility_id(path)
            self.mobility_category = get_mobility_category(path)
            self.mobility_identifier = get_mobility_identifier(path)
            if name is None and self.mobility_id is not None and not self.mobility_id.isdigit():
                name = self.mobility_id
            self.instance_name = get_instance_name(path)
        elif body in get_bodies() and is_box_entity(body):
            self.is_box = True
            self.instance_name = None
            self.mobility_id = 'box'
            self.mobility_category = 'box'
            self.mobility_identifier = 'box'

        self.body = body
        self.joint = joint
        self.link = link

        ## automatically categorize object by class
        if category is None:
            category = self.__class__.__name__.lower()
        category = category.lower()
        self.category = category
        self.name = name
        self.verbose = verbose  ## whether to omit all debug messages

        self.collision = collision
        self.handles = []
        self.text_handle = None
        self.text = ''
        # self.draw()

        ## in order to move object with objects attached in it
        self.world = None
        self.grasp_markers = []
        self.grasp_parent = None
        self.doors = None
        self.drawers = None
        self.surfaces = []
        self.spaces = []
        self.supporting_surface = None
        self.supported_objects = []
        self.events = []
        self.categories = [category] ## for added categories like movable
        self.grasps = grasps

    def get_categories(self):
        categories = self.categories
        class_cat = self.__class__.__name__.lower()
        if class_cat not in self.categories:
            categories = [class_cat] + categories
        return categories

    def add_grasps(self, grasps):
        self.grasps = grasps

    ## =============== put other object on top of object =============
    ##
    def is_placement(self, body):
        for o in self.supported_objects:
            if o.pybullet_name == body:
                return True
        return False

    def is_contained(self, body):
        for o in self.supported_objects:
            if o.body == body:
                return True
        return False

    def support_obj(self, obj, verbose=False):
        from pybullet_tools.logging_utils import myprint as print
        if verbose: print(f'ADDED {self} supporting_surface ({obj})')
        obj.supporting_surface = self
        if obj not in self.supported_objects:
            self.supported_objects.append(obj)

    def attach_obj(self, obj):
        link = self.link if self.link is not None else -1
        self.world.attachments[obj] = create_attachment(self, link, obj, OBJ=True)
        obj.change_supporting_surface(self)

    def place_new_obj(self, obj_name, category=None, name=None, max_trial=8, verbose=False, **kwargs):

        if category is None:
            category = obj_name

        obj = self.world.add_object(
            Object(load_asset(obj_name.lower(), **kwargs), category=category, name=name)
        )
        self.world.put_on_surface(obj, surface=self.name, max_trial=max_trial, verbose=verbose)
        self.support_obj(obj)
        # set_renderer(True)
        return obj

    def place_obj(self, obj, max_trial=8, timeout=1.5, obstacles=None,
                  visualize=False, interactive=False, verbose=False):
        """ place object on Surface or in Space """
        from world_builder.loaders_partnet_kitchen import check_kitchen_placement

        if isinstance(obj, str):
            raise NotImplementedError('place_obj: obj is str')
            # obj = self.place_new_obj(obj, max_trial=max_trial)
        world = self.world
        if obstacles is None:
            obstacles = [o for o in get_bodies() if o not in [obj, self.body]]

        # if visualize:
        #     set_renderer(True)
        #     set_camera_target_body(obj.body)

        done = False
        if world.learned_pose_list_gen is not None:
            results = world.learned_pose_list_gen(world, obj.body, [self.pybullet_name], num_samples=14)
            if results is not None:
                for body_pose in results:
                    set_pose(obj, body_pose) ## obj.set_pose(body_pose)  ## includes setting attachments
                    coo = collided(obj, obstacles, tag='place_obj_database', world=world, verbose=False)
                    if not coo:
                        done = True
                        break
                    # if visualize:
                    #     wait_unlocked()

        start_time = time.time()
        place_fn = sample_obj_in_body_link_space if isinstance(self, Space) else sample_obj_on_body_link_surface
        while not done:
            x, y, z, yaw = place_fn(obj, self.body, self.link, PLACEMENT_ONLY=True, max_trial=max_trial, verbose=verbose)
            body_pose = Pose(point=Point(x=x, y=y, z=z), euler=Euler(yaw=yaw))
            set_pose(obj, body_pose)
            coo = collided(obj, obstacles, tag='place_obj', world=world, verbose=False)
            if not coo:
                done = True
            if time.time() - start_time > timeout:
                return obj

        ## adjust with arrows
        if interactive:
            obj.change_pose_interactive()

        if self.verbose:
            if isinstance(self, Space):
                supporter_name = f"in Space {self.name}"
            else:
                supporter_name = f"on {self.__class__.__name__.capitalize()} {self.name}"
            print(f'entities.place_obj.placed {obj.name} {supporter_name} at pose {nice(body_pose)}')

        self.attach_obj(obj)
        return obj

    def remove_supporting_surface(self, verbose=False):
        from pybullet_tools.logging_utils import myprint as print
        if self.supporting_surface is not None:
            if verbose:
                print(f'REMOVED {self} supporting_surface ({self.supporting_surface})')
            if self in self.supporting_surface.supported_objects:
                self.supporting_surface.supported_objects.remove(self)
            self.supporting_surface = None

    def change_supporting_surface(self, obj):
        self.remove_supporting_surface()
        obj.support_obj(self)

    def change_pose_interactive(self):
        change_pose_interactive(self.body, self.shorter_name, se3=False)

    ## ====================================================================

    def __int__(self):
        if not hasattr(self, 'body'):
            print(self, 'doesnt have attribute body thus cant be deepcopied')
        if not hasattr(self, 'body') or self.body is None:
            return id(self)  # TODO: hack
        if self.joint is not None:
            return (self.body, self.joint)
        # if self.link is not None:
        #     return (self.body, None, self.link)
        return self.body

    def __repr__(self):
        if hasattr(self, 'name'):
            return self.name
        return 'object.name'

    def _type(self):
        return self.__class__.__name__

    def get_pose(self):
        if self.link is None:
            return get_pose(self.body)
        else:
            return get_link_pose(self.body, self.link)

    def aabb(self):
        if self.link is not None:
            return get_aabb(self.body, link=self.link)
        elif self.joint is not None:
            return get_aabb(self.body, link=self.handle_link)
        return get_aabb(self.body)

    @property
    def lx(self):
        return get_aabb_extent(self.aabb())[0]

    @property
    def ly(self):
        return get_aabb_extent(self.aabb())[1]

    @property
    def ly(self):
        return get_aabb_extent(self.aabb())[1]

    @property
    def x2xmin(self):
        return self.get_pose()[0][0] - self.aabb().lower[0]

    @property
    def xmax2x(self):
        return self.aabb().upper[0] - self.get_pose()[0][0]

    @property
    def y2ymin(self):
        return self.get_pose()[0][1] - self.aabb().lower[1]

    @property
    def z2zmin(self):
        return self.get_pose()[0][2] - self.aabb().lower[2]

    @property
    def height(self):
        return get_aabb_extent(self.aabb())[2]

    def adjust_next_to(self, other, direction='+y', align='+x', dz=0, hinge_gap=0.07):
        cabinet_categories = ['cabinetlower', 'cabinettall', 'minifridge', 'minifridgebase',
                              'dishwasherbox', 'dishwasher']
        gap = {}
        cx, cy, cz = self.get_pose()[0]
        x = y = z = None
        if align == '+x':
            # x = other.aabb().upper[0] - self.lx / 2
            x = other.aabb().upper[0] - (self.aabb().upper[0] - cx)
        if direction == '+y':
            # y = other.aabb().upper[1] + self.ly / 2
            y = other.aabb().upper[1] + (cy - self.aabb().lower[1])
            if self.category in cabinet_categories or other.category in cabinet_categories:
                y += hinge_gap
                gap[other.aabb().upper[1]] = other.aabb().upper[1] + hinge_gap
        elif direction == '-y':
            # y = other.aabb().lower[1] - self.ly / 2
            y = other.aabb().lower[1] - (self.aabb().upper[1] - cy)
            if self.category in cabinet_categories or other.category in cabinet_categories:
                y -= hinge_gap
                gap[other.aabb().lower[1]] = other.aabb().lower[1] - hinge_gap
        elif direction == '+z':
            # z = other.aabb().upper[2] + self.height / 2 + dz
            # y = other.get_pose()[0][1]
            z = other.aabb().upper[2] + (cz - self.aabb().lower[2]) + dz
            y = other.aabb().lower[1] + (cy - self.aabb().lower[1])
        self.adjust_pose(x=x, y=y, z=z)
        # print('adjust_next_to | other', other.aabb().upper[0], 'self', self.aabb().upper[0])
        return gap

    def adjust_pose(self, x=None, y=None, z=None, dx=None, dy=None, dz=None, yaw=None):
        (cx, cy, cz), r = self.get_pose()
        if dx is not None:
            cx += dx
        elif x is not None:
            cx = x
        if dy is not None:
            cy += dy
        elif y is not None:
            cy = y
        if dz is not None:
            cz += dz
        elif z is not None:
            cz = z
        if yaw is not None:
            r = quat_from_euler(Euler(yaw=yaw))
        self.set_pose(((cx, cy, cz), r))

    def set_pose(self, conf):
        links = [get_link_name(self.body, l) for l in get_links(self.body)]
        if 'base' in links:  ## is robot
            set_group_conf(self.body, BASE_GROUP, conf)
        else:
            if len(conf) == 6:
                conf = (conf[:3], quat_from_euler(conf[3:]))
            set_pose(self.body, conf)
        if self.supporting_surface is not None:
            self.supporting_surface.attach_obj(self)

    def get_joint(self, joint):  # int | str
        try:
            return int(joint)
        except ValueError:
            return joint_from_name(self.body, joint)

    def get_joints(self, joints=None):
        if joints is None:
            return get_movable_joints(self.body)  ## get_joints | get_movable_joints
        return tuple(map(self.get_joint, joints))

    def get_joint_position(self, *args, **kwargs):
        return get_joint_positions(self.body, [self.get_joint(*args, **kwargs)])[0]

    def get_joint_positions(self, *args, **kwargs):
        return get_joint_positions(self.body, self.get_joints(*args, **kwargs))

    def set_joint_position(self, joint, positions):
        ans = set_joint_position(self.body, self.get_joint(joint), positions)

        ## when joints move, objects on child link are generated again
        BODY_TO_OBJECT = self.world.BODY_TO_OBJECT
        child_links = get_joint_descendants(self.body, joint)
        for link in child_links:
            if (self.body, None, link) in BODY_TO_OBJECT:
                space = BODY_TO_OBJECT[(self.body, None, link)]
                for obj in space.objects_inside:
                    space.place_obj(obj)
        return ans

    def set_joint_positions(self, joints, positions):
        return set_joint_positions(self.body, self.get_joints(joints), positions)

    def get_link(self, link): # int | str
        try:
            return int(link)
        except ValueError:
            return link_from_name(self.body, link)
    #link_from_name = get_link

    def get_links(self, links=None):
        if links is None:
            return get_all_links(self.body)
        return tuple(map(self.get_joint, links))

    def get_link_pose(self, link=BASE_LINK):
        return get_link_pose(self.body, self.get_link(link))

    def create_saver(self, **kwargs):
        # TODO: inherit from saver
        return BodySaver(self.body, **kwargs)

    def get_link_oobb(self, link, index=0):
        # TODO: get_trimesh_oobb
        link = self.get_link(link)
        surface_data = get_collision_data(self.body, link=link)[index]
        pose = get_link_pose(self, link) # TODO: combine for multiple links
        surface_oobb = tform_oobb(pose, oobb_from_data(surface_data))
        # draw_oobb(surface_oobb, color=RED)
        return surface_oobb

    def get_link_aabb(self, *args, **kwargs):
        return aabb_from_oobb(self.get_link_oobb(*args, **kwargs))

    def get_aabb(self, *args, **kwargs):
        # TODO: is there an easier way to to bind all of these methods?
        return get_aabb(self.body, *args, **kwargs)

    def draw_joints(self, buffer=5e-2):
        handles = []
        if isinstance(self, Robot):
            return handles
        # add_body_name | draw_link_name
        for joint in get_movable_joints(self.body):
            joint_name = get_joint_name(self.body, joint)
            child_link = child_link_from_joint(joint)
            label = f'{joint_name}:{joint}'
            # label = f'{self.name}-{self.body}:{joint_name}-{joint}'
            child_aabb = buffer_aabb(get_aabb(self.body, child_link), buffer=buffer)  # TODO: union of children
            child_lower, child_upper = child_aabb
            # position = unit_point()
            # position = child_upper
            position = get_aabb_center(child_aabb)
            position[0] = child_upper[0]
            child_pose = self.get_link_pose(child_link)
            position = tform_point(invert(child_pose), position)
            handles.append(add_text(label, position=position, parent=self.body, parent_link=child_link))
        self.handles.extend(handles)
        return handles

    def draw(self, text=None, **kwargs):
        if text is None:
            text = f':{self.pybullet_name}'
        link = self.handle_link if hasattr(self, 'handle_link') else self.link
        with LockRenderer(True):
            self.erase()
            if self.name is not None:
                h = draw_body_label(self.body, text=self.name+text, link=link, **kwargs)
                self.handles.append(h)
            # # self.handles.extend(draw_pose(Pose(), parent=self.body, **kwargs))
            # if not isinstance(self, Robot):
            #     self.draw_joints()
        return self.handles

    def erase(self):
        remove_handles(self.handles)
        self.handles = []

    def add_text(self, text, **kwargs):
        if self.text_handle is not None:
            # p.removeUserDebugItem(self.text_handle)
            remove_debug(self.text_handle)
            self.text += '_'
        self.text += text
        self.text_handle = draw_body_label(self.body, self.text, link=self.link, offset=(0, 0.15, 0.15), **kwargs)

    def is_active(self):
        return self.body is not None

    def remove(self):
        self.erase()
        if self.is_active():
            remove_body(self.body)
            self.body = None

    def add_grasp_marker(self, object):
        if object not in self.grasp_markers:
            self.grasp_markers.append(object)
        self.world.BODY_TO_OBJECT[object].grasp_parent = self.body

    def add_events(self, events):
        self.events.extend(events)

    def add_event(self, event):
        self.events.append(event)

    @property
    def pybullet_name(self):
        if self.joint is None and self.link is not None:
            return (self.body, self.joint, self.link)
        elif self.joint is not None and self.link is None:
            return (self.body, self.joint)
        else:
            return self.body

    @property
    def lisdf_name(self):
        return get_lisdf_name(self.body, self.name, joint=self.joint, link=self.link)

    @property
    def shorter_name(self):
        name = self.name.replace('counter#1--', '')
        name = ''.join(char for char in name if not (char == '#' or char.isdigit()))
        return name

    @property
    def debug_name(self):
        return f'{self.pybullet_name}|{self.name}'
        # return f'{self.name}|{self.pybullet_name}'


class Movable(Object):
    def __init__(self, body, **kwargs):
        super(Movable, self).__init__(body, collision=False, **kwargs)


class Steerable(Object):
    def __init__(self, body, **kwargs):
        super(Steerable, self).__init__(body, collision=False, **kwargs)


class Supporter(Object):
    def __init__(self, body, **kwargs):
        super(Supporter, self).__init__(body, collision=False, **kwargs)
        self.supported_objects = []


class Region(Object):
    def __init__(self, body, governing_joints=[], **kwargs):
        super(Region, self).__init__(body, collision=False, **kwargs)
        self.governing_joints = governing_joints

    def set_governing_joints(self, governing_joints):
        """ (body, joint) pairs that change the pose of (body, link) """
        self.governing_joints = governing_joints
        for j in self.governing_joints:
            joint = self.world.BODY_TO_OBJECT[j]
            if self.pybullet_name not in joint.affected_links:
                joint.affected_links.append(self.pybullet_name)


class Stove(Region):
    def __init__(self, body, **kwargs):
        super(Stove, self).__init__(body, **kwargs)


class Floor(Region):
    def __init__(self, body, **kwargs):
        super(Floor, self).__init__(body, **kwargs)
        # self.category = 'floor'
        # self.name = 'floor1'
        self.is_box = True


class Location(Region):
    def __init__(self, body, **kwargs):
        super(Location, self).__init__(body, **kwargs)

#######################################################


class Surface(Region):
    """ to support objects on top, like kitchentop and fridge shelves """
    def __init__(self, body, link, **kwargs):
        super().__init__(body, link=link, **kwargs)
        if self.name is None:
            self.name = get_link_name(body, link)


class Space(Region):
    """ to support object inside, like cabinets and drawers """
    def __init__(self, body, link, **kwargs):
        super().__init__(body, link=link, **kwargs)
        if self.name is None:
            self.name = get_link_name(body, link)

    def place_new_obj(self, obj_name, category=None, max_trial=8, verbose=False,
                      scale=1, random_instance=True, **kwargs):

        if category is None:
            category = obj_name
        # self.world.open_doors_drawers(self.body)

        obj = self.world.add_object(
            Movable(load_asset(obj_name.lower(), random_instance=random_instance, scale=scale),
                     category=category)
        )
        self.place_obj(obj, max_trial=max_trial, visualize=False, **kwargs)

        # world.close_doors_drawers(self.body)
        return obj


#######################################################


class ArticulatedObjectPart(Object):
    def __init__(self, body, joint, min_limit=None, max_limit=None, **kwargs):
        super(ArticulatedObjectPart, self).__init__(body, joint, collision=True, **kwargs)
        self.name = get_joint_name(body, joint)
        if min_limit is None:
            min_limit = get_min_limit(body, joint)
            max_limit = get_max_limit(body, joint)
        self.min_limit = min_limit
        self.max_limit = max_limit

        self.handle_link = self.find_handle_link(body, joint)
        self.handle_horizontal, self.handle_width = self.get_handle_orientation(body)

        self.affected_links = []  ## that are planning objects
        self.all_affected_links = []  ## all links in the asset

    def find_handle_link(self, body, joint, debug=True):
        link = get_joint_info(body, joint).linkName.decode("utf-8")
        children_link = get_link_children(body, link_from_name(body, link))

        ## the only handle, for those carefully engineered names
        links = [l for l in get_links(body) if 'handle' in get_link_name(body, l)]

        if len(links) == 1:
            return links[0]

        ## if there's only one in the link's children
        elif len(links) > 1:
            also_in_children = [l for l in links if l in children_link]
            if len(also_in_children) == 1:
                return also_in_children[0]

        ## when the substring matches, for those carefully engineered names, e.g. counter
        if '_link' in link:
            name = link[:link.index('_link')]
            links = [l for l in get_links(body) if name in get_link_name(body, l)]
            links = [l for l in links if 'handle' in get_link_name(body, l) or 'knob' in get_link_name(body, l)]
            if len(links) == 1:
                return links[0]

        ## try to find in children links
        if len(links) == 0:
            links += children_link

        ## sort links by similarity in names
        words = self.name.split('_')
        if len(links) > 0:
            counts = {links[i]: sum([w in words for w in get_link_name(body, links[i]).split('_')]) for i in range(len(links))}
            counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
            return list(counts.keys())[0]

        ## default to the joint's parent link
        return link_from_name(body, link)

    def get_handle_orientation(self, body):
        aabb = get_aabb(body, self.handle_link)
        x,y,z = [aabb.upper[i] - aabb.lower[i] for i in range(3)]
        if y > z or x > z:
            return True, z
        return False, np.sqrt(x**2 + y**2)

    def get_handle_pose(self):
        return get_link_pose(self.body, self.handle_link)

    def get_pose(self):
        return self.get_handle_pose()

    def make_links_transparent(self, transparency=0.5):
        links = get_link_subtree(self.body, self.joint)
        for link in links:
            colorize_link(self.body, link=link, transparency=transparency)

class Door(ArticulatedObjectPart):
    def __init__(self, body, joint, **kwargs):
        super(Door, self).__init__(body, joint, **kwargs)


class Drawer(ArticulatedObjectPart):
    def __init__(self, body, joint, **kwargs):
        super(Drawer, self).__init__(body, joint, **kwargs)


class Knob(ArticulatedObjectPart):
    def __init__(self, body, joint, **kwargs):
        super(Knob, self).__init__(body, joint, **kwargs)
        self.controlled = None

    # def find_handle_link(self, body, joint):
    #     link = get_joint_info(body, joint).linkName.decode("utf-8")
    #     return get_link_parent(body, link_from_name(body, link))

    def add_controlled(self, body):
        self.controlled = body

#######################################################


class Robot(Object):
    def __init__(self, body, base_link=BASE_LINK, joints=None,
                 custom_limits={}, disabled_collisions={},
                 resolutions=BASE_RESOLUTIONS, weights=None, cameras=[], **kwargs):
        name = get_name(body)
        super(Robot, self).__init__(body, name=name, **kwargs)
        self.base_link = self.get_link(base_link)
        self.joints = self.get_joints(get_movable_joints(self.body) if joints is None else joints)
        self.custom_limits = dict(custom_limits)
        self.disabled_collisions = dict(disabled_collisions)
        self.resolutions = resolutions # TODO: default values if None
        if weights is None:
            with np.errstate(divide='ignore'):
                weights = np.reciprocal(resolutions)
        self.weights = weights
        self.cameras = list(cameras)
        self.objects_in_hand = {'left': -1, 'right': -1}  ## problem with equal
    #@property
    #def joints(self):
    #    return self.active_joints

    def get_pose(self):
        return self.get_link_pose(self.base_link)

    # def get_positions(self, joint_group='base', roundto=None):
    #     if joint_group == 'base':
    #         joints = self.joints
    #     else: ## if joint_group == 'left':
    #         joints = get_arm_joints(self.body, joint_group)
    #     positions = self.get_joint_positions(joints)
    #     if roundto == None:
    #         return positions
    #     return tuple([round(n, roundto) for n in positions])

    def set_base_positions(self, xytheta):
        set_group_conf(self.body, 'base', xytheta)

    def set_positions(self, positions, joints=None):
        if joints is None:
            joints = self.joints
        self.set_joint_positions(joints, positions)

    def get_limits(self, joints=None):
        if joints is None:
            joints = self.joints
        return get_custom_limits(self.body, joints, self.custom_limits)

    def get_aabb(self, *args, **kwargs):
        return get_subtree_aabb(self.body, self.base_link) # Computes the robot's axis-aligned bounding box (AABB)

    def within_limits(self, positions=None):
        if positions is None:
            positions = self.get_positions()
        lower_limits, upper_limits = self.get_limits()
        return all_between(lower_limits, positions, upper_limits)

    def get_objects_in_hands(self):
        objects = []
        for gripper in ['left', 'right']:
            if self.objects_in_hand[gripper] != -1:
                objects.append(self.objects_in_hand[gripper])
        return objects

    def has_object_in_hand(self, obj):
        return obj in [o.category.lower() for o in self.get_objects_in_hands()]
    #def draw(self, *args, **kwargs):
    #    super(Robot, self).draw(*args, **kwargs)
    #    # TODO: add text to base_link


class Camera(object):
    def __init__(self, body, camera_frame, camera_matrix, max_depth=2., name=None,
                 draw_frame=None, rel_pose=unit_pose(), **kwargs):
        self.body = body if isinstance(body, int) else body.body
        self.camera_link = link_from_name(self.body, camera_frame) # optical_frame
        self.camera_matrix = camera_matrix
        self.max_depth = max_depth
        self.name = camera_frame if name is None else name
        self.draw_link = link_from_name(self.body, draw_frame if draw_frame is None else draw_frame)
        self.rel_pose = rel_pose
        self.kwargs = dict(kwargs)
        #self.__dict__.update(**kwargs)
        # self.handles = []

        # self.handles = self.draw()
        # self.get_boundaries()

    def get_pose(self):
        pose = get_link_pose(self.body, self.camera_link)
        pose = multiply(pose, Pose(point=Point(z=0.05)))  ## so that PR2's eyeball won't get in the way
        return pose

    def get_image(self, segment=True, segment_links=False, **kwargs):
        # TODO: apply maximum depth
        #image = get_image(self.get_pose(), target_pos=[0, 0, 1])
        return get_image_at_pose(self.get_pose(), self.camera_matrix,
                                 tiny=False, segment=segment, segment_links=segment_links, **kwargs)

    def draw(self):
        handles = []
        robot = self.body
        eyes_from_camera = multiply(Pose(euler=Euler(yaw=PI / 2)), Pose(euler=Euler(roll=PI / 2)))
        handles.extend(draw_viewcone(eyes_from_camera, depth=self.max_depth, camera_matrix=self.camera_matrix,
                                     parent=self.body, parent_link=self.draw_link))
        # handles.extend(draw_pose(self.get_pose(), length=1))  ## draw frame of camera
        # handles.extend(draw_pose(unit_pose(), length=1, parent=robot, parent_link=robot.base_link))  ## draw robot base frame
        return handles

    def get_boundaries(self):
        """ return the normal vectors of four faces of the viewcone """
        normals = []
        cone_base = get_viewcone_base(depth=self.max_depth, camera_matrix=self.camera_matrix)
        self.cone_base = cone_base
        pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for A, B in pairs:
            A = cone_base[A]
            B = cone_base[B]
            C = np.asarray([0, 0, 0])
            dir = np.cross((B - A), (C - A))
            N = dir / np.linalg.norm(dir)
            normals.append(N)
        self.boundaries = normals

    def point_in_camera_frame(self, point):
        p_world_point = Pose(point=point)
        X_world_eye = self.get_pose()
        p_eye_point = multiply(invert(X_world_eye), p_world_point)
        return p_eye_point[0]

    def point_in_view(self, point_in_world):
        p = self.point_in_camera_frame(point_in_world)
        outside = False
        for normal in self.boundaries:
            if np.dot(normal, p) < 0:
                outside = True
        in_view = not outside
        # from pybullet_tools.bullet_utils import nice
        # print(f'  point in world {point_in_world}, in camera {nice(p)}, in view? {in_view}')
        return in_view

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.name)


class StaticCamera(object):
    def __init__(self, pose, camera_matrix, max_depth=2., name=None, draw_frame=None,
                 camera_point=None, target_point=None):
        self.pose = pose
        self.camera_matrix = camera_matrix
        self.max_depth = max_depth
        self.name = 'unknown_camera' if name is None else name
        # self.kwargs = dict(kwargs)
        # self.__dict__.update(**kwargs)
        # self.handles = []
        # self.handles = self.draw()
        self.get_boundaries()
        self.index = 0
        self.camera_point = camera_point
        self.target_point = target_point

    def get_pose(self):
        return self.pose

    def set_pose(self, pose):
        self.pose = pose

    def get_image(self, segment=True, segment_links=False, far=8,
                  camera_point=None, target_point=None, camera_pose=None, **kwargs):
        # TODO: apply maximum depth
        self.index += 1
        if self.camera_point is not None and self.target_point is not None:
            camera_point = self.camera_point
            target_point = self.target_point
        if camera_point is not None and target_point is not None:
            self.camera_point = camera_point
            self.target_point = target_point
            return get_camera_image_at_pose(camera_point, target_point, self.camera_matrix, far=far,
                                 tiny=False, segment=segment, segment_links=segment_links, **kwargs)
        if camera_pose is not None:
            self.set_pose(camera_pose)
        return get_image_at_pose(self.get_pose(), self.camera_matrix, far=far,
                                 tiny=False, segment=segment, segment_links=segment_links, **kwargs)

    def draw(self):
        return []

    def get_boundaries(self):
        """ return the normal vectors of four faces of the viewcone """
        normals = []
        cone_base = get_viewcone_base(depth=self.max_depth, camera_matrix=self.camera_matrix)
        self.cone_base = cone_base
        pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for A, B in pairs:
            A = cone_base[A]
            B = cone_base[B]
            C = np.asarray([0, 0, 0])
            dir = np.cross((B - A), (C - A))
            N = dir / np.linalg.norm(dir)
            normals.append(N)
        self.boundaries = normals

    def point_in_camera_frame(self, point):
        p_world_point = Pose(point=point)
        X_world_eye = self.get_pose()
        p_eye_point = multiply(invert(X_world_eye), p_world_point)
        return p_eye_point[0]

    def point_in_view(self, point_in_world):
        p = self.point_in_camera_frame(point_in_world)
        outside = False
        for normal in self.boundaries:
            if np.dot(normal, p) < 0:
                outside = True
        in_view = not outside
        # from pybullet_tools.bullet_utils import nice
        # print(f'  point in world {point_in_world}, in camera {nice(p)}, in view? {in_view}')
        return in_view

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.name)


def add_robot_cameras(robot, camera_frames, max_depth=2.5, camera_matrix=CAMERA_MATRIX, draw=False,
                      verbose=False, **kwargs):
    """ for both world_builder.world.World class and lisdf_tools.lisdf_loader.World class """
    cameras = {}
    for camera_frame, draw_frame, rel_pose, camera_name in camera_frames:
        if verbose:
            print_pink(f"adding camera {camera_name} at link{camera_frame}")
        camera = Camera(robot, camera_frame=camera_frame, camera_matrix=camera_matrix,
                        max_depth=max_depth, draw_frame=draw_frame, rel_pose=rel_pose, **kwargs)
        if draw:
            camera.draw()
        cameras[camera_name] = camera
    return cameras
