## written by Caelan Garrett at NVIDIA, for internal use only
import os
import numpy as np
import pybullet as p
from collections import defaultdict, namedtuple
import trimesh
import copy
import traceback
import time
import scipy

from pybullet_tools.utils import read_obj, write, INF, AABB, get_aabb_extent, get_aabb_volume, \
    timeout, ensure_dir, HideOutput, elapsed_time

OBJ = '.obj'
TEMP_URDF_DIR = 'temp_urdfs/'
VHACD_SUFFIX = '_vhacd'
VHACD_DIR = 'vhacd/'

def load_kinematics(urdf_path, build_scene=False):
    #import urdfpy
    import yourdfpy
    return yourdfpy.URDF.load(
        urdf_path,
        build_scene_graph=build_scene,
        build_collision_scene_graph=build_scene,
        load_meshes=False,
        load_collision_meshes=False,
        # filename_handler=None,
        # mesh_dir='',
        force_mesh=False,
        force_collision_mesh=False,
    )


def set_default_inertial(urdf_model, default_mass=1.):
    import yourdfpy
    for link in urdf_model.robot.links:
        if (link.inertial is None):
            # and (link.name != urdf_model._determine_base_link()):
            # TODO: [ WARN] [1648770518.826799422]: The root link _base_link_x has an inertia specified in the URDF,
            #  but KDL does not support a root link with an inertia.
            #  As a workaround, you can add an extra dummy link to your URDF.
            # and (link.collisions or link.visuals)
            link.inertial = yourdfpy.Inertial(
                origin=np.eye(4),
                mass=default_mass,
                inertia=np.eye(3),
            )

def remove_model_visuals(urdf_model):
    for link in urdf_model.robot.links:
        #link.collisions.clear()
        link.visuals.clear()
        #for visual in link.visuals:
        #    visual.material = None

def merge_rigid_leaves(urdf_model, verbose=False):
    # TODO: only merge if too many links
    parent_from_link = {joint.child: joint.name for joint in urdf_model.robot.joints}
    children_from_link = defaultdict(set)
    for joint in urdf_model.robot.joints:
        children_from_link[joint.parent].add(joint.name)

    def recurse_descendants(link_name):
        descendants = set()
        for joint_name in children_from_link[link_name]:
            joint = urdf_model.joint_map[joint_name]
            child_name = joint.child
            descendants.add(child_name)
            descendants.update(recurse_descendants(child_name))
        return descendants

    def recurse_ancestors(link_name, terminal_links=[]):
        if (link_name not in parent_from_link) or (link_name in terminal_links):
            return [link_name]
        joint = urdf_model.joint_map[parent_from_link[link_name]]
        # if joint.type != 'fixed':
        #    return [link_name]
        return recurse_ancestors(joint.parent, terminal_links) + [link_name]

    # TODO: collapse non-rigid ancestors
    # TODO: keep some links reference in kwargs
    merged_links = set()
    for joint in urdf_model.robot.joints:
        subtree = {joint.child} | recurse_descendants(joint.child)
        if all(urdf_model.joint_map[parent_from_link[descendant]].type == 'fixed' for descendant in subtree):
            merged_links.update(subtree)
    kept_links = set(urdf_model.link_map) - merged_links
    if verbose:
        print('Retained ({}): {} | Merged ({}): {}'.format(
            len(kept_links), sorted(kept_links), len(merged_links), sorted(merged_links)))

    for current_link in merged_links:
        ancestor_link = recurse_ancestors(current_link, kept_links)[0]
        assert ancestor_link not in merged_links
        relative_pose = urdf_model.get_transform(frame_to=current_link, frame_from=ancestor_link,
                                                 collision_geometry=False)
        # relative_pose = joint.origin
        for visual in urdf_model.link_map[current_link].visuals:
            visual.origin = np.dot(relative_pose, visual.origin)
            urdf_model.link_map[ancestor_link].visuals.append(visual)
        # relative_pose = urdf_model.get_transform(frame_to=current_link, frame_from=ancestor_link, collision_geometry=True)
        for collision in urdf_model.link_map[current_link].collisions:
            collision.origin = np.dot(relative_pose, collision.origin)
            urdf_model.link_map[ancestor_link].collisions.append(collision)

    urdf_model.robot.links = [link for link in urdf_model.robot.links if link.name not in merged_links]
    urdf_model.robot.joints = [joint for joint in urdf_model.robot.joints if joint.child not in merged_links]
    assert len(urdf_model.robot.links) == len(urdf_model.robot.joints) + 1


def decompose_obj(obj_path, verbose=True):
    assert obj_path.endswith(OBJ)
    root, ext = os.path.splitext(obj_path)
    meshes = read_obj(obj_path, decompose=True)
    meshes = [trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces) for mesh in meshes.values()]
    # TODO: connected components
    # scene = trimesh.load(obj_path, split_object=True, force='scene') # mesh | scene
    # # from srl_stream.visii_world import meshes_from_scene
    # meshes = scene.geometry.values() # TODO: split_object doesn't work correctly

    #with open(obj_path, 'r') as f:
    #    data = trimesh.exchange.obj.load_obj(f, split_object=True, skip_materials=False)
    #if 'geometry' in data:
    #    print(data['geometry'])
    #if len(meshes) == 1:
    #    return [obj_path]
    if verbose:
        # print('Decomposing {} into {} meshes'.format(obj_path, len(meshes)))
        assert len(meshes) > 0
    decomposed_paths = []
    for i, mesh in enumerate(meshes):
        decomposed_path = '{}_part-{}{}'.format(root, i, ext)
        mesh_str = trimesh.exchange.obj.export_obj(mesh)
        write(decomposed_path, mesh_str)
        if verbose:
            print('Saved {}'.format(decomposed_path), end='\r')
        decomposed_paths.append(decomposed_path)
    return decomposed_paths


def remove_suffix(s, suffix):
    # https://stackoverflow.com/questions/1038824/how-do-i-remove-a-substring-from-the-end-of-a-string
    # TODO: os.path.relpath
    if not s.endswith(suffix):
        return s
    return s[:-len(suffix)]



def decompose_model(urdf_model, **kwargs):
    # TODO: connected components when loading
    decomposed_from_original = {}
    for link_data in urdf_model.robot.links:
        new_collisions = []
        for collision_data in link_data.collisions:
            mesh_data = collision_data.geometry.mesh
            if mesh_data is None:
                new_collisions.append(collision_data)
                continue
            obj_filename = mesh_data.filename
            if not obj_filename.endswith(OBJ):
                new_collisions.append(collision_data)
                continue
            obj_path = os.path.abspath(urdf_model._filename_handler(obj_filename))
            if not os.path.exists(obj_path):
                new_collisions.append(collision_data)
                continue

            decomposed_paths = decompose_obj(obj_path, **kwargs)
            if len(decomposed_paths) == 1:
                new_collisions.append(collision_data)
                continue

            dir_path = remove_suffix(obj_path, obj_filename)
            decomposed_from_original[obj_path] = decomposed_paths
            for i, decomposed_path in enumerate(decomposed_paths):
                new_collision_data = copy.deepcopy(collision_data)
                if collision_data.name is not None:
                    new_collision_data.name = '{}_part-{}'.format(collision_data.name, i)
                new_mesh_data = new_collision_data.geometry.mesh
                new_filename = os.path.relpath(decomposed_path, dir_path)
                new_mesh_data.filename = new_filename
                new_collisions.append(new_collision_data)
        link_data.collisions[:] = new_collisions
    return decomposed_from_original




def join_paths(*paths):
    return os.path.abspath(os.path.join(*paths))

def create_vhacd(input_path, output_path=None, output_dir=VHACD_DIR, cache=True, verbose=False, **kwargs):
    # https://github.com/bulletphysics/bullet3/blob/afa4fb54505fd071103b8e2e8793c38fd40f6fb6/examples/pybullet/examples/vhacd.py
    input_path = os.path.abspath(input_path)
    if output_path is None:
        #output_path = join_paths(TEMP_DIR, 'vhacd_{}.obj'.format(next(VHACD_CNT)))
        # https://stackoverflow.com/questions/10501247/best-way-to-generate-random-file-names-in-python
        # import uuid
        # directory = TEMP_DIR
        # filename = '{}.obj'.format(uuid.uuid4())

        # https://stackoverflow.com/questions/27954892/deterministic-hashing-in-python-3
        import zlib
        ensure_dir(output_dir)
        filename, _ = os.path.splitext(os.path.basename(input_path))
        unique = os.path.abspath(input_path)
        #unique = read(input_path) # TODO: not written deterministically in the same order
        identity = zlib.adler32(unique.encode('utf-8')) # TODO: kwargs
        filename = '{}_vhacd_{}.obj'.format(filename, identity)
        output_path = join_paths(output_dir, filename)
        if cache and os.path.exists(output_path):
            return output_path

    start_time = time.time()
    print('Starting V-HACD of {}'.format(input_path))
    log_path = join_paths(VHACD_DIR, 'vhacd_log.txt')
    # TODO: use kwargs to update the default args
    vhacd_kwargs = {
        'concavity': 0.0025,  # Maximum allowed concavity (default=0.0025, range=0.0-1.0)
        'alpha': 0.04,  # Controls the bias toward clipping along symmetry planes (default=0.05, range=0.0-1.0)
        'beta': 0.05,  # Controls the bias toward clipping along revolution axes (default=0.05, range=0.0-1.0)
        'gamma': 0.00125,  # Controls the maximum allowed concavity during the merge stage (default=0.00125, range=0.0-1.0)
        'minVolumePerCH': 0.0001,  # Controls the adaptive sampling of the generated convex-hulls (default=0.0001, range=0.0-0.01)
        'resolution': 100000,  # Maximum number of voxels generated during the voxelization stage (default=100,000, range=10,000-16,000,000)
        'maxNumVerticesPerCH': 64,  # Controls the maximum number of triangles per convex-hull (default=64, range=4-1024)
        'depth': 20,  # Maximum number of clipping stages. During each split stage, parts with a concavity higher than the user defined threshold are clipped according the best clipping plane (default=20, range=1-32)
        'planeDownsampling': 4,  # Controls the granularity of the search for the \"best\" clipping plane (default=4, range=1-16)
        'convexhullDownsampling': 4,  # Controls the precision of the convex-hull generation process during the clipping plane selection stage (default=4, range=1-16)
        'pca': 0,  # Enable/disable normalizing the mesh before applying the convex decomposition (default=0, range={0,1})
        'mode': 0,  # 0: voxel-based approximate convex decomposition, 1: tetrahedron-based approximate convex decomposition (default=0,range={0,1})
        'convexhullApproximation': 1,  # Enable/disable approximation when computing convex-hulls (default=1, range={0,1})
    }
    vhacd_kwargs.update(kwargs)
    with HideOutput(enable=not verbose):
        p.vhacd(input_path, output_path, log_path, **vhacd_kwargs)
    print('Finished V-HACD: {} ({:.3f} sec)'.format(output_path, elapsed_time(start_time)))

    return output_path
    #return create_obj(output_path, **kwargs)


def safe_vhacd(filename, min_length=0.01, min_volume=None, max_time=INF, **kwargs):
    if min_volume is None:
        min_volume = min_length**3
    result = None
    if (min_volume > 0.) or (min_length > 0.): # TODO: scale might vary (pass as argument)
        # TODO: vertices check
        # TODO: return if file already exists
        # https://trimsh.org/trimesh.interfaces.vhacd.html?highlight=vhacd#trimesh.interfaces.vhacd.convex_decomposition
        mesh = trimesh.load(filename, force='mesh') # mesh | scene # TODO: multiple objects
        if mesh.is_convex:
            return result

        # mesh.fill_holes()
        # print(mesh, mesh.is_watertight, mesh.volume)
        # mesh.show()
        aabb = AABB(*mesh.bounds)
        if (get_aabb_volume(aabb) < min_volume) or any(side < min_length for side in get_aabb_extent(aabb)):
            #print('Filename: {} | AABB Volume: {:.3e}'.format(filename, get_aabb_volume(aabb)))
            return result

        try:
            tform, extent = trimesh.bounds.oriented_bounds(mesh, angle_digits=1, ordered=True, normal=None)
            aabb = AABB(-extent/2, +extent/2)
            if (get_aabb_volume(aabb) < min_volume) or any(side < min_length for side in get_aabb_extent(aabb)):
                return result

            hull = mesh.convex_hull
            volume = hull.volume
            extent = get_aabb_extent(aabb)
        except scipy.spatial.qhull.QhullError:
            traceback.print_exc()
            return result

        #print('Filename: {} | Volume: {:.3e} | Extent: {}'.format(filename, volume, np.round(extent, 3)))
        if volume < min_volume:
            return result

    # TODO: hash kwargs
    assert filename.endswith(OBJ)
    with timeout(duration=max_time):
        result = create_vhacd(
            filename,
            # concavity=0.0025,
            # resolution=100000,
            # mode=0,
            **kwargs)
    return result



def vhacd_model(urdf_model, visual=True, in_place=False, cache=True, **kwargs):
    vhacd_from_original = {}
    for link_data in urdf_model.robot.links:
        #if not visual: # TODO: makes loading much less efficient
        #    link_data.visuals.clear()
        data = link_data.collisions
        if not visual:
            data += link_data.visuals

        for collision_data in data:
            mesh_data = collision_data.geometry.mesh
            if mesh_data is None:
                continue
            old_filename = mesh_data.filename
            if not old_filename.endswith(OBJ):
                continue
            old_path = os.path.abspath(urdf_model._filename_handler(old_filename))
            if not os.path.exists(old_path):
                # return None
                continue
            mesh_path = remove_suffix(old_path, old_filename)
            old_root, old_ext = os.path.splitext(old_filename)
            if in_place:
                new_filename = old_root + VHACD_SUFFIX + old_ext
                # new_filename = old_filename
                # new_path = os.path.abspath(urdf_model._filename_handler(new_filename))
                new_path = os.path.join(mesh_path, new_filename)
            else:
                new_path = None

            output_path = safe_vhacd(old_path, output_path=new_path, cache=cache, **kwargs)
            if output_path is None:
                #print('Skipping V-HACD of degenerate {}'.format(old_path))
                continue
            vhacd_from_original[old_path] = output_path
            mesh_data.filename = new_filename if in_place else output_path
    return vhacd_from_original


def process_urdf(urdf_path, urdf_dir=TEMP_URDF_DIR, inertial=True, decompose=True,
                 visual=True, merge=True, vhacd=True, verbose=False, **kwargs):
    # TODO: could set some robot joints to be fixed
    urdf_model = load_kinematics(urdf_path, build_scene=True)

    if not visual:
        remove_model_visuals(urdf_model)
    if merge:
        merge_rigid_leaves(urdf_model)
    if vhacd:
        vhacd_model(urdf_model, visual=visual, **kwargs) # verbose=verbose,
    if decompose:
        decompose_model(urdf_model)
    if inertial:
        set_default_inertial(urdf_model, default_mass=1.)

    ensure_dir(urdf_dir)
    file_name = os.path.basename(urdf_path)
    new_urdf_path = os.path.abspath(os.path.join(urdf_dir, file_name))
    # print(urdf_model.write_xml_string())
    urdf_model.write_xml_file(new_urdf_path)
    if verbose:
        print('Old URDF: {} | New URDF: {}'.format(urdf_path, new_urdf_path))

    return new_urdf_path
