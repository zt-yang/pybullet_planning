import numpy as np

from pybullet_tools.utils import get_aabb, get_aabb_extent, get_aabb_center, aabb_overlap, AABB


def get_surface_aabb(surface):
    return get_aabb(surface[0], surface[-1]) if isinstance(surface, tuple) else get_aabb(surface)


def get_surface_area(surface):
    w, l, _ = get_aabb_extent(get_surface_aabb(surface))
    return w * l


def get_distance_to_aabb(surface, region_aabb):
    region_center = get_aabb_center(region_aabb)
    surface_aabb = get_surface_aabb(surface)
    surface_aabb_center = get_aabb_center(surface_aabb)
    return np.linalg.norm(np.asarray(surface_aabb_center) - np.asarray(region_center))


def get_distance_to_aabb_fn(region_aabb):
    def funk(surface):
        return get_distance_to_aabb(surface, region_aabb)
    return funk


def find_big_surfaces(other_surfaces, top_k=1, title='find_big_surfaces\t'):
    ## sort other_surfaces by aabb area
    print(f'\n{title}other_surfaces: {other_surfaces}')
    big_surface = sorted(other_surfaces, key=get_surface_area)[:top_k]
    print(f'{title}surface:\t{big_surface}')
    return big_surface


def find_surfaces_close_to_region(other_surfaces, region_aabb, top_k=2, title='find_surfaces_close_to_region\t'):
    print(f'\n{title}other_surfaces: {other_surfaces}')
    closest_surface = sorted(other_surfaces, key=get_distance_to_aabb_fn(region_aabb))[:top_k]
    print(f'{title}closest surface:\t{closest_surface}')
    return closest_surface


def find_movables_close_to_region(all_movables, region_aabb, title='find_movables_close_to_region\t',
                                  aabb_expansion=0.5):
    found = []
    for o in all_movables:
        mov_aabb = get_aabb(o)
        mov_aabb_expanded = AABB(lower=np.asarray(mov_aabb.lower) - aabb_expansion,
                                 upper=np.asarray(mov_aabb.upper) + aabb_expansion)
        if aabb_overlap(region_aabb, mov_aabb_expanded):
            found.append(o)
            print(f'{title}found close by obstacle:\t {o}')
    return found


def add_surfaces_given_obstacles(world, obstacles, title='add_surfaces_given_obstacles\t'):
    other_surfaces = world.cat_to_bodies('surface')
    add_surfaces = find_big_surfaces(other_surfaces, top_k=1)
    other_surfaces = [s for s in other_surfaces if s not in add_surfaces]
    for o in obstacles:
        region_aabb = get_surface_aabb(o)
        placed = None
        if world.BODY_TO_OBJECT[o].supporting_surface is not None:
            placed = world.BODY_TO_OBJECT[o].supporting_surface.pybullet_name
        add = find_surfaces_close_to_region([s for s in other_surfaces if s != placed], region_aabb, top_k=2)
        add_surfaces.extend(add)
    if len(add_surfaces) > 0:
        print(f'\n{title} for obstacles {obstacles} add surfaces {add_surfaces}\n')
    return add_surfaces
