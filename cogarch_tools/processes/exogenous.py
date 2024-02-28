import random
import colorsys

from collections import defaultdict

from pybullet_tools.bullet_utils import get_color, place_body, sample_safe_placement, check_placement, is_above, is_on
from world_builder.entities import Object
from world_builder.world import Exogenous
from pybullet_tools.utils import add_button, read_counter, \
    set_all_color, convex_combination, RED, create_cylinder, GREEN, get_aabb, BLACK, apply_alpha, TRANSPARENT

def add_facts(state, facts):
    new_facts = list(state.facts)
    for fact in facts:
        if fact not in new_facts:
            new_facts.append(fact)
    return state.new_state(facts=new_facts)

class TeleportingObject(Exogenous):
    def __init__(self, world, obj, region, teleports_per_sec=0., **kwargs):
        super(TeleportingObject, self).__init__(world, **kwargs)
        self.obj = obj
        self.region = region
        self.p_teleport = teleports_per_sec*world.time_step
        self.num_teleports = 0
        self.counter = add_button(name='Teleport {}'.format(self.obj))
        self.last_counter = read_counter(self.counter)

    def transition(self, state): # Exponential, Poisson?
        if (self.last_counter == read_counter(self.counter)) and (random.random() > self.p_teleport):
            return state
        self.last_counter = read_counter(self.counter)
        sample_safe_placement(self.obj, self.region, obstacles=state.bodies)
        self.num_teleports += 1
        return state.new_state()


class MovableObject(Exogenous):
    def __init__(self, world, obj, region, **kwargs):
        super(MovableObject, self).__init__(world, **kwargs)
        self.obj = obj
        self.region = region

    def transition(self, state):
        obj_aabb = get_aabb(self.obj)
        new_facts = list(state.facts)
        # on_facts = state.filter_facts(predicate='On')
        for region in state.regions:
            fact = ('On', self.obj, region)
            if is_on(obj_aabb, get_aabb(region)):
                if fact not in new_facts:
                    new_facts.append(fact)
            else:
                if fact in new_facts:
                    new_facts.remove(fact)
        return state.new_state(facts=new_facts)


class Door(Exogenous):
    def __init__(self, world, obj, space, switch=None, **kwargs):
        super(Door, self).__init__(world, **kwargs)
        self.obj = obj
        self.space = space
        self.switch = switch
        self.color = get_color(obj)

    def transition(self, state):
        if (self.switch is not None) and not state.variables['Pressed', self.switch]:
            set_all_color(self.obj, self.color)
            return None
        set_all_color(self.obj, TRANSPARENT)
        return add_facts(state, [('Opened', self.space)])


class Stove(Exogenous):
    def __init__(self, world, region, switch=None, duration=1., **kwargs):
        super(Stove, self).__init__(world, **kwargs)
        self.region = region
        self.switch = switch
        self.duration = duration
        self.cooked = defaultdict(float)
        self.initial_colors = {}
    def transition(self, state):
        # TODO: store in state
        if (self.switch is not None) and not state.variables['Pressed', self.switch]:
            set_all_color(self.region, BLACK)
            return None
        set_all_color(self.region, RED)
        new_facts = list(state.facts)
        for obj in state.movable:
            if obj.category == 'door': continue
            if (obj not in state.attachments) and check_placement(obj, self.region):
                if obj not in self.initial_colors:
                    self.initial_colors[obj] = get_color(obj)
                self.cooked[obj.name] += self.time_step
                fraction = min(self.cooked[obj.name] / self.duration, 1.)
                set_all_color(obj, convex_combination(self.initial_colors[obj], RED, w=fraction))
                # TODO: burn if it exceeds duration
                if fraction == 1.:
                    fact = ('Cooked', obj)
                    if fact not in new_facts:
                        new_facts.append(fact)
        return state.new_state(facts=new_facts)


class Chute(Exogenous):
    def __init__(self, world, region, button=None, **kwargs):
        super(Chute, self).__init__(world, **kwargs)
        self.region = region
        #if button is None:
        #    button = region
        self.button = button
        # TODO: set to black and have the object drop
    def transition(self, state):
        # TODO: region event parent class
        if (self.button is not None) and not state.variables['Pressed', self.button]:
            # and all(not is_above(robot, get_aabb(self.button)) for robot in state.robots):
            return state
        discarded = set()
        for obj in set(state.movable):
            if (obj not in state.attachments) and check_placement(obj, self.region):
                discarded.add(obj)
        objects = set(state.objects) - discarded
        new_facts = [fact for fact in state.facts if not set(fact[1:]) & discarded]
        for obj in discarded:
            obj.remove()
        state.variables['Pressed', self.button] = False
        return state.new_state(objects=objects, facts=new_facts)


class Spawner(Exogenous):
    def __init__(self, world, region, *args, **kwargs):
        super(Spawner, self).__init__(world, *args, **kwargs)
        self.region = region
        self.num_objects = 0
    def initialize(self, state):
        #self.num_objects = len(state.objects)
        return state
    def transition(self, state):
        #if self.num_objects <= len(state.objects):
        if state.movable:
            return state
        self.num_objects += 1
        hue = random.uniform(60, 240) / 360.
        color = apply_alpha(colorsys.hsv_to_rgb(h=hue, s=1., v=1.), alpha=1.)
        #color = GREEN
        obj = place_body(Object(create_cylinder(radius=0.1, height=0.1, color=color),
                                name='spawned@{}'.format(self.num_objects)))
        sample_safe_placement(obj, self.region, obstacles=state.bodies, min_distance=5e-2)
        new_objects = state.objects + [obj]
        new_facts = list(state.facts) + [
            ('Movable', obj),
        ]
        return state.new_state(objects=new_objects, facts=new_facts)
