#!/usr/bin/env python3
"""
The MIT License (MIT)

Copyright (c) 2020 NVIDIA Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import sys
import json
import trimesh
import argparse
import numpy as np
import os
import h5py
import glob
import pandas as pd
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib.pyplot as plt

from main.utils.acronym import load_mesh, load_grasps, create_gripper_marker

from nltk.corpus import wordnet as wn
from collections import defaultdict

from tqdm import tqdm
import io
from PIL import Image
import main.utils.transformations as tra

import torch
import clip
import json
import pickle


def make_parser():
    parser = argparse.ArgumentParser(
        description="Visualize grasps from the dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", nargs="+", help="HDF5 or JSON Grasp file(s).")
    parser.add_argument(
        "--num_grasps", type=int, default=20, help="Number of grasps to show."
    )
    parser.add_argument(
        "--mesh_root", default=".", help="Directory used for loading meshes."
    )
    return parser


def main(argv=sys.argv[1:]):
    parser = make_parser()
    args = parser.parse_args(argv)

    for f in args.input:
        # load object mesh
        obj_mesh = load_mesh(f, mesh_root_dir=args.mesh_root)

        # get transformations and quality of all simulated grasps
        T, success = load_grasps(f)

        # create visual markers for grasps
        successful_grasps = [
            create_gripper_marker(color=[0, 255, 0]).apply_transform(t)
            for t in T[np.random.choice(np.where(success == 1)[0], args.num_grasps)]
        ]
        failed_grasps = [
            create_gripper_marker(color=[255, 0, 0]).apply_transform(t)
            for t in T[np.random.choice(np.where(success == 0)[0], args.num_grasps)]
        ]

        trimesh.Scene([obj_mesh] + successful_grasps + failed_grasps).show()


def find_shape_description(meta_data_file, shapenet_id):
    df = pd.read_csv(meta_data_file)
    idxs = df.index[df["fullId"] == "wss.{}".format(shapenet_id)].tolist()
    if idxs:
        return df.iloc[idxs[0]].to_dict()
    else:
        return None


def get_obj_types(synset):

    all_hypernyms = []
    while True:
        hypernyms = synset.hypernyms()
        if not hypernyms:
            break
        synset = hypernyms[0]
        all_hypernyms.append(synset.name())

    return all_hypernyms


def construct_isa_graph():
    grasp_dir = "/home/weiyu/data_drive/shapenet/acronym_meshes/grasps"
    shapenet_sem_meta_file = "/home/weiyu/data_drive/shapenet/shapenetsem/models-OBJ/metadata.csv"

    add_wn_synset = True
    add_object_nodes = False
    add_wordnet_hierarchy = True

    graph = nx.DiGraph()

    for grasp_file in tqdm(os.listdir(grasp_dir)):

        if "Mug" not in grasp_file:
            continue

        print("-" * 20)

        grasp_file = os.path.join(grasp_dir, grasp_file)

        data = h5py.File(grasp_file, "r")
        mesh_fname = data["object/file"][()].decode('utf-8')
        mesh_scale = data["object/scale"][()]

        shapenet_id = mesh_fname.split("/")[-1].split(".")[0]

        # add shapenet node
        graph.add_node(shapenet_id, color="red", type="object_id")

        description_dict = find_shape_description(shapenet_sem_meta_file, shapenet_id)
        print(description_dict)
        if pd.isna(description_dict["wnsynset"]):
            continue

        wn_pos, wn_offset = description_dict["wnsynset"][0], int(description_dict["wnsynset"][1:])
        synset = wn.synset_from_pos_and_offset(wn_pos, wn_offset)
        print("wordnet synset", synset)

        hypens = synset.hypernym_paths()

        if add_wn_synset:
            graph.add_node(synset.name(), color="blue", type="wordnet_synset")
            if add_object_nodes:
                graph.add_edge(shapenet_id, synset.name(), relation='HasWordNetSynset')
        if add_wordnet_hierarchy:
            for hypen in hypens:
                if not add_wn_synset:
                    # print(obj_id, hypen[-1].name(), hypen[-2].name())
                    assert add_object_nodes
                    graph.add_edge(shapenet_id, hypen[-2].name(), relation='IsA')
                    graph.add_node(hypen[-2].name(), color="orange", type="extracted_wordnet_synset")
                    for i in range(0, len(hypen) - 2):
                        parent = hypen[i].name()
                        child = hypen[i + 1].name()
                        graph.add_node(hypen[i].name(), color="orange", type="extracted_wordnet_synset")
                        graph.add_node(hypen[i + 1].name(), color="orange", type="extracted_wordnet_synset")
                        graph.add_edge(child, parent, relation='IsA')
                else:
                    for i in range(0, len(hypen) - 1):
                        parent = hypen[i].name()
                        child = hypen[i + 1].name()
                        if parent not in graph.nodes:
                            graph.add_node(parent, color="orange", type="extracted_wordnet_synset")
                        if child not in graph.nodes:
                            graph.add_node(child, color="orange", type="extracted_wordnet_synset")
                        graph.add_edge(child, parent, relation='IsA')

    draw_nx_graph(graph, save_path="/home/weiyu/Research/acronym/scripts")


def draw_nx_graph(graph, save_path, method="graphviz", title="knowledge_graph"):
    """
    This function helps to visualize a network DiGraph
    :param method: which visualization method to use
    :param title: name of the graph
    :param save_path: save directory
    :return:
    """
    if method == "graphviz":
        # There are two ways to visualize networkx graph
        # 1. write dot file to use with graphviz
        # run "dot -Tpng test.dot > test.png"
        dot_path = os.path.join(save_path, '{}.dot'.format(title))
        png_path = os.path.join(save_path, '{}.png'.format(title))
        write_dot(graph, dot_path)
        cmd = 'dot -Tpng {} > {}'.format(dot_path, png_path)
        os.system(cmd)
    elif method == "matplotlib":
        # 2. same layout using matplotlib with no labels
        # Not so good
        plt.title(title)
        pos = graphviz_layout(graph, prog='dot')
        nx.draw(graph, pos, with_labels=False, arrows=True)
        plt.savefig(os.path.join(save_path, 'nx_test.png'))


def visualize_model_and_grasp():

    grasp_dir = "/home/weiyu/data_drive/shapenet/acronym_meshes/grasps"
    mesh_dir = "/home/weiyu/data_drive/shapenet/acronym_meshes"
    shapenet_sem_dir = "/home/weiyu/data_drive/shapenet/shapenetsem/models-OBJ/models"
    shapenet_core_dir = "/home/weiyu/data_drive/shapenet/ShapeNetCore.v2"
    shapenet_sem_meta_file = "/home/weiyu/data_drive/shapenet/shapenetsem/models-OBJ/metadata.csv"

    num_grasps = 20

    total_count = 0
    visual_count = 0
    for grasp_file in os.listdir(grasp_dir):

        if "Mug" not in grasp_file:
            continue

        print("-" * 20)

        grasp_file = os.path.join(grasp_dir, grasp_file)

        data = h5py.File(grasp_file, "r")
        mesh_fname = data["object/file"][()] #.decode('utf-8')
        mesh_scale = data["object/scale"][()]

        shapenet_id = mesh_fname.split("/")[-1].split(".")[0]

        description_dict = find_shape_description(shapenet_sem_meta_file, shapenet_id)
        wn_pos, wn_offset = description_dict["wnsynset"][0], int(description_dict["wnsynset"][1:])
        wn_synset = wn.synset_from_pos_and_offset(wn_pos, wn_offset)
        print("wordnet synset", wn_synset)

        print(mesh_fname)

        obj_mesh = trimesh.load(os.path.join(mesh_dir, mesh_fname))
        obj_mesh = obj_mesh.apply_scale(mesh_scale)

        data = h5py.File(grasp_file, "r")
        T = np.array(data["grasps/transforms"])
        success = np.array(data["grasps/qualities/flex/object_in_gripper"])

        successful_grasps = [
            create_gripper_marker(color=[0, 255, 0]).apply_transform(t)
            for t in T[np.random.choice(np.where(success == 1)[0], num_grasps)]
        ]

        table_mesh = trimesh.creation.box([1, 1, 0])

        print("is mesh watertight?", obj_mesh.is_watertight)
        print("mesh bounding box", obj_mesh.bounds)

        trimesh.Scene([obj_mesh] + successful_grasps + [table_mesh]).show()

        # find object color mesh in shapenet
        visual_mesh_file = os.path.join(shapenet_sem_dir, "{}.obj".format(shapenet_id))
        print(visual_mesh_file)

        total_count += 1

        if os.path.exists(visual_mesh_file):
            print("visual mesh exist?", os.path.exists(visual_mesh_file))
            visual_mesh = trimesh.load(visual_mesh_file)
            print(visual_mesh)
            # visual_mesh.apply_scale(mesh_scale)

            print(visual_mesh.bounds)
            # visual_mesh.show()

            scale_transformation = np.eye(4)
            scale_transformation[:3, :3] *= mesh_scale
            print(scale_transformation)
            visual_mesh.apply_transform(scale_transformation)

            print(mesh_scale)

            print(visual_mesh.bounds)
            trimesh.Scene([visual_mesh] + [table_mesh] + successful_grasps).show()

            visual_count += 1



        # else:
        #     input("here")
        #     file_template = shapenet_core_dir + "/**/{}/models/model_normalized.obj".format(shapenet_id.split(".")[0])
        #     print(file_template)
        #     visual_mesh_file = glob.glob(file_template, recursive=True)
        #     if visual_mesh_file:
        #         visual_mesh_file = visual_mesh_file[0]
        #
        #         visual_mesh = trimesh.load(visual_mesh_file)
        #         print(visual_mesh)
        #         # visual_mesh.apply_scale(mesh_scale)
        #
        #         print(visual_mesh.bounds)
        #         # visual_mesh.show()
        #
        #         scale_transformation = np.eye(4)
        #         scale_transformation[:3, :3] *= mesh_scale
        #         print(scale_transformation)
        #         visual_mesh.apply_transform(scale_transformation)
        #
        #         print(mesh_scale)
        #
        #         print(visual_mesh.bounds)
        #         trimesh.Scene([visual_mesh] + [table_mesh] + successful_grasps).show()
        #     else:
        #         print("not found")

        # ToDo: use https://github.com/adithyamurali/TaskGrasp/blob/master/geometry_utils.py to sample more diverse grasps

    print(total_count)
    print(visual_count)

class ObjectModelLoader:

    def __init__(self):
        self.grasp_dir = "/home/weiyu/data_drive/shapenet/acronym_meshes/grasps"
        self.mesh_dir = "/home/weiyu/data_drive/shapenet/acronym_meshes"
        self.shapenet_sem_dir = "/home/weiyu/data_drive/shapenet/shapenetsem/models-OBJ/models"
        self.shapenet_core_dir = "/home/weiyu/data_drive/shapenet/ShapeNetCore.v2"
        self.shapenet_sem_meta_file = "/home/weiyu/data_drive/shapenet/shapenetsem/models-OBJ/metadata.csv"
        self.num_grasps = 20

        self.models = []
        self.cls_to_models = defaultdict(list)
        for grasp_file in os.listdir(self.grasp_dir):
            obj_cls = grasp_file.split("_")[0]
            self.models.append(grasp_file)
            self.cls_to_models[obj_cls].append(grasp_file)

    def __len__(self):
        return len(self.models)

    def random_cls_model(self, cls, debug=False):

        grasp_file = np.random.choice(self.cls_to_models[cls])
        grasp_file = os.path.join(self.grasp_dir, grasp_file)

        # get meta data
        data = h5py.File(grasp_file, "r")
        mesh_fname = data["object/file"][()] #.decode('utf-8')
        mesh_scale = data["object/scale"][()]
        shapenet_id = mesh_fname.split("/")[-1].split(".")[0]
        description_dict = find_shape_description(self.shapenet_sem_meta_file, shapenet_id)
        wn_pos = description_dict["wnsynset"][0]
        wn_offset = int(description_dict["wnsynset"][1:])
        wn_synset = wn.synset_from_pos_and_offset(wn_pos, wn_offset)
        if debug:
            print("wordnet synset", wn_synset)
            print(mesh_fname)

        # get mesh
        obj_mesh = trimesh.load(os.path.join(self.mesh_dir, mesh_fname))
        obj_mesh = obj_mesh.apply_scale(mesh_scale)
        if debug:
            print("is mesh watertight?", obj_mesh.is_watertight)
            print("mesh bounding box", obj_mesh.bounds)

        # get grasps
        data = h5py.File(grasp_file, "r")
        T = np.array(data["grasps/transforms"])
        success = np.array(data["grasps/qualities/flex/object_in_gripper"])
        successful_grasps = [
            create_gripper_marker(color=[0, 255, 0]).apply_transform(t)
            for t in T[np.random.choice(np.where(success == 1)[0], self.num_grasps)]
        ]

        table_mesh = trimesh.creation.box([1, 1, 0])

        if debug:
            trimesh.Scene([obj_mesh] + successful_grasps + [table_mesh]).show()

        # find object color mesh in shapenet
        visual_mesh_file = os.path.join(self.shapenet_sem_dir, "{}.obj".format(shapenet_id))
        visual_mesh = None
        # print(visual_mesh_file)
        if os.path.exists(visual_mesh_file):
            visual_mesh = trimesh.load(visual_mesh_file)
            # print(visual_mesh)
            # visual_mesh.apply_scale(mesh_scale)
            # print(visual_mesh.bounds)
            # visual_mesh.show()

            scale_transformation = np.eye(4)
            scale_transformation[:3, :3] *= mesh_scale
            # print(scale_transformation)
            visual_mesh.apply_transform(scale_transformation)
            # print(mesh_scale)
            # print(visual_mesh.bounds)
            if debug:
                trimesh.Scene([visual_mesh] + [table_mesh] + successful_grasps).show()

        return wn_synset, obj_mesh, successful_grasps, visual_mesh


def try_clip_binary_spatial():

    oml = ObjectModelLoader()
    classes = oml.cls_to_models.keys()

    table_mesh = trimesh.creation.box([0.5, 0.5, 0])

    clip_model = ClipModel()

    # try cup example
    _, cup_obj_mesh, _, cup_vis_mesh = oml.random_cls_model("Bowl")

    # put the object in the center of the table
    T = np.eye(4)
    T[:2, 3] = -1 * cup_vis_mesh.centroid[:2]
    cup_vis_mesh.apply_transform(T)

    # rotate the object
    # centroid = cup_vis_mesh.centroid
    # T = np.eye(4)
    # T[:3, 3] = -1 * centroid
    # cup_vis_mesh.apply_transform(T)
    # cup_vis_mesh.apply_transform(tra.euler_matrix(np.pi, 0, 0))
    # T[:3, 3] = centroid
    # cup_vis_mesh.apply_transform(T)

    # trimesh.Scene([cup_vis_mesh]).show()
    table_scene = trimesh.Scene([cup_vis_mesh, table_mesh])
    # table_scene = trimesh.Scene([cup_vis_mesh])

    RT = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=-1)
    RT_4x4 = np.concatenate([RT, np.array([0., 0., 0., 1.])[None, :]], 0)
    RT_4x4 = np.linalg.inv(RT_4x4)
    RT_4x4 = RT_4x4 @ np.diag([1, -1, -1, 1])
    # table_scene.camera_transform = RT_4x4

    table_scene.camera_transform = table_scene.camera.look_at([[0., 0., 0]], rotation=tra.euler_matrix(np.pi/4, 0, 0), distance=0.5)

    data = table_scene.save_image()
    pil_img = Image.open(io.BytesIO(data))
    image = np.array(pil_img)

    plt.imshow(image)
    plt.show()

    probs, logits_per_image, _ = clip_model.get_img_texts_compatibility(pil_img, ["a bowl that is upside down", "a bowl that is upright", "a cylinder", "a bowl that is not upright"])
    print(probs)
    print(logits_per_image)


class ClipModel:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def get_img_texts_compatibility(self, img, text):
        """
        :param img: PIL image
        :param text: a str or a list of str
        :return:
        """
        image = self.preprocess(img).unsqueeze(0).to(self.device)
        text = clip.tokenize(text).to(self.device)

        with torch.no_grad():
            # image_features = self.model.encode_image(image)
            # text_features = self.model.encode_text(text)

            logits_per_image, logits_per_text = self.model(image, text)
            image_probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
        return image_probs, logits_per_image, logits_per_text


def run_clip():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]


if __name__ == "__main__":
    # visualize_model_and_grasp()
    # construct_isa_graph()
    random_seed = 2
    np.random.seed(random_seed)
    # torch.manual_seed(cfg.random_seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(cfg.random_seed)
    #     torch.cuda.manual_seed_all(cfg.random_seed)
    #     torch.backends.cudnn.deterministic = True

    try_clip_binary_spatial()

    tra.quaternion_matrix()



