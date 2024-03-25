import os
import shutil
import json
import h5py
import trimesh

import numpy as np
from collections import defaultdict

from pybullet_tools.utils import ensure_dir, Pose, Point, Euler
from pybullet_tools.bullet_utils import get_grasp_db_file, add_grasp_in_db
from nsplan_tools.utils.acronym import create_gripper_marker
import nsplan_tools.utils.transformations as tra


def main():

    structformer_model_dir = "/home/weiyu/data_drive/structformer_assets/acronym_handpicked_v4_textured_acronym_scale"
    acronym_grasp_dir = "/home/weiyu/data_drive/shapenet/acronym_meshes/grasps"
    acronym_mesh_dir = "/home/weiyu/data_drive/shapenet/acronym_meshes/meshes"

    kitchen_models_dir = "/home/weiyu/Research/nsplan/original/kitchen-worlds/assets/models"
    grasps_db_file = "/home/weiyu/Research/nsplan/original/kitchen-worlds/pybullet_planning/databases/hand_grasps_FEGripper.json"

    write_mesh = True
    write_grasp = True
    debug_grasps = False

    # note that there should be no indentation for the second line because there is some hardcoded parsing rule
    # in get_instance_name()
    urdf_template = """<?xml version="1.0"?>
<robot name="{obj_id}">
  <link name="base"/>
  <link name="link_0">
  <visual>
    <origin xyz="0.0 0.0 0.0"/>
    <geometry>
      <mesh filename="{visual_mesh_name}" scale="1.0 1.0 1.0"/>
    </geometry>
  </visual>
  <collision>
    <origin xyz="0.0 0.0 0.0"/>
    <geometry>
      <mesh filename="{collision_mesh_name}" scale="1.0 1.0 1.0"/>
    </geometry>
  </collision>
  </link>
  <joint name="joint_0" type="fixed">
    <origin rpy="0 0 0" xyz="0 0 0"/>
    <child link="link_0"/>
    <parent link="base"/>
  </joint>
</robot>"""

    object_classes = ["Bowl", "Cup", "Mug", "Pan"]  # "WineBottle", "Bottle", "SodaCan", "Teapot"

    objects_dir = os.path.join(structformer_model_dir, "objects")
    visual_meshes_dir = os.path.join(structformer_model_dir, "visual")
    collision_meshes_dir = os.path.join(structformer_model_dir, "meshes")

    if os.path.exists(grasps_db_file):
        grasps_db = json.load(open(grasps_db_file, 'r'))
    else:
        grasps_db = {}

    # this is used in 3 places, partnet_scale, grasp_db, and models instance dir name
    kitchen_world_obj_num = 7000
    model_scales = defaultdict(dict)
    for obj in sorted(os.listdir(objects_dir)):
        obj_id = os.path.splitext(obj)[0]
        obj_cls = obj.split("_")[0]
        if obj_cls not in object_classes:
            continue

        obj_num_str = "{:04d}".format(kitchen_world_obj_num)

        model_scales[obj_cls][obj_num_str] = 1

        if write_mesh:
            # visual
            visual_mesh_name = f"{obj_id}.obj"
            visual_mesh_filename = os.path.join(visual_meshes_dir, f"{obj_id}.obj")
            mat_filename = os.path.join(visual_meshes_dir, f"{obj_id}.mtl")

            # collision
            # rename collision mesh because it has the same name as the visual name
            collision_mesh_name = "collision.obj"
            collision_mesh_filename = os.path.join(collision_meshes_dir, f"{obj_id}.obj")

            # urdf
            urdf = urdf_template.format(obj_id=obj_id,
                                        visual_mesh_name=visual_mesh_name,
                                        collision_mesh_name=collision_mesh_name)

            # copy files
            new_obj_dir = os.path.join(kitchen_models_dir, obj_cls, obj_num_str)
            if not os.path.exists(new_obj_dir):
                os.makedirs(new_obj_dir)
            shutil.copy(visual_mesh_filename, new_obj_dir)
            if os.path.exists(mat_filename):
                shutil.copy(mat_filename, new_obj_dir)
            shutil.copy(collision_mesh_filename, os.path.join(new_obj_dir, collision_mesh_name))
            with open(os.path.join(new_obj_dir, "mobility.urdf"), "w") as fh:
                fh.write(urdf + "\n")

        if write_grasp:

            obj_data = json.load(open(os.path.join(objects_dir, obj), 'r'))

            # retrieve grasps
            grasp_filename = "{}_{}_{}.h5".format(obj_data["class"], obj_data["shapenet_id"], obj_data["scale"])
            grasp_filename = os.path.join(acronym_grasp_dir, grasp_filename)

            if debug_grasps and not write_mesh:
                visual_mesh_filename = os.path.join(visual_meshes_dir, f"{obj_id}.obj")
            else:
                visual_mesh_filename = None

            # important: we can apply mesh_T and acronym scale to go from acronym mesh to the processed object mesh
            #            mesh_T == acronym_scale @ obj_data["canonical_T"] @ obj_data["scale"] @ tra.translation_matrix(-1 * acronym_mesh.centroid)
            #            acronym_scale is f["object/scale"][()] stored in the acronym grasp h5.
            #            we can apply mesh_T to go from acronym grasps to grasps for the processed object mesh
            mesh_T = tra.quaternion_matrix(obj_data["T"]["rotation"])
            mesh_T[:3, 3] = obj_data["T"]["translation"]

            original_acronym_mesh_filename = os.path.join(acronym_mesh_dir, "{}/{}.obj".format(obj_data["class"], obj_data["shapenet_id"]))
            grasp_poses = load_acronym_grasps(grasp_filename, visual_mesh_filename,
                                         original_acronym_mesh_filename, obj_data["scale"],
                                         mesh_T,
                                         debug=debug_grasps)

            grasp_obj_name = "{}#{}".format(obj_cls.lower(), obj_num_str)
            add_grasp_in_db(grasps_db, grasps_db_file, obj_id, grasp_poses, name=grasp_obj_name,
                            length_variants=False, scale=1)

        kitchen_world_obj_num += 1

    print("Please manually update /pybullet_planning/world_builder/asset_constants.py")
    #     'Mug': {
    #         'Mug_159e56c18906830278d8f8c02c47cde0_M': 1,
    #         'Mug_159e56c18906830278d8f8c02c47cde0_S': 1,
    #         'Mug_ba10400c108e5c3f54e1b6f41fdd78a_M': 1,
    #         'Mug_ba10400c108e5c3f54e1b6f41fdd78a_S': 1,
    #     },
    model_scales = dict(model_scales)
    print(model_scales)
    print("Use /pybullet_planning/nsplan_tools/test_skills.py to test grasps")


def load_acronym_grasps(grasp_filename, model_filename,
                        original_acronym_mesh_filename, scale,
                        mesh_T,
                        num_grasps=30, debug=False):

    data = h5py.File(grasp_filename, "r")
    # mesh_fname = data["object/file"][()].decode('utf-8')
    # mesh_scale = data["object/scale"][()]
    # print(mesh_scale)

    T = np.array(data["grasps/transforms"])
    success = np.array(data["grasps/qualities/flex/object_in_gripper"])
    successful_grasps = T[np.random.choice(np.where(success == 1)[0], num_grasps)]

    if debug:
        obj_mesh = trimesh.load(model_filename)

        # visualize original acronym mesh and grasps
        original_acronym_mesh = trimesh.load(original_acronym_mesh_filename)
        original_acronym_mesh.apply_scale(scale)
        successful_grasps_vis = [
            create_gripper_marker(color=[0, 255, 0]).apply_transform(t)
            for t in successful_grasps
        ]
        trimesh.Scene([original_acronym_mesh] + [successful_grasps_vis] + [obj_mesh]).show()

        # visualize transformed mesh and grasps
        for g in successful_grasps_vis:
            g.apply_transform(mesh_T)
        original_acronym_mesh.apply_transform(mesh_T)
        trimesh.Scene([original_acronym_mesh] + [successful_grasps_vis] + [obj_mesh]).show()

    grasp_poses = []
    for grasp in successful_grasps:

        # important: transform grasp for acronym mesh to processed (centered, scaled, reoriented) object mesh
        grasp = mesh_T @ grasp

        # https://sites.google.com/nvidia.com/graspdataset
        # gripper orientation convention is different for kitchen-worlds and acronym
        offset_grasp = grasp @ tra.euler_matrix(0, 0, np.pi / 2)

        # optionally adjust grasp depth
        # approach_offset = np.eye(4)
        # approach_offset[2, 3] = 0.02
        # offset_grasp = grasp @ tra.euler_matrix(0, 0, np.pi / 2) @ approach_offset

        if debug:
            pos = grasp[:3, 3]
            rot = tra.euler_from_matrix(grasp)
            print("acronym grasp pose", pos.tolist() + [*rot])

        pos = offset_grasp[:3, 3].tolist()
        rot = [*tra.euler_from_matrix(offset_grasp)]
        if debug:
            print("kitchen world grasp pose", pos + rot)
            # print("offset grasp pose quat", pos.tolist(), tra.quaternion_from_matrix(offset_grasp))

        # if debug:
        #     vis_grasp = create_gripper_marker(color=[0, 255, 0]).apply_transform(grasp)
        #     vis_pose = trimesh.creation.axis(transform=grasp, origin_size=0.01)
        #     vis_pose_offset = trimesh.creation.axis(transform=offset_grasp, origin_size=0.01)
        #     trimesh.Scene([obj_mesh, vis_grasp, vis_pose, vis_pose_offset]).show()

        grasp_pose = Pose(Point(pos[0], pos[1], pos[2]), Euler(rot[0], rot[1], rot[2]))
        grasp_poses.append(grasp_pose)

    return grasp_poses



if __name__ == "__main__":
    main()

    # TODO: use argparse