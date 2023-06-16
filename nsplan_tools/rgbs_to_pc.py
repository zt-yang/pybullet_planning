import torch
import numpy as np
import trimesh


def parse_intrinsics(intrinsics):
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    return fx, fy, cx, cy


def expand_as(x, y):
    if len(x.shape) == len(y.shape):
        return x

    for i in range(len(y.shape) - len(x.shape)):
        x = x.unsqueeze(-1)

    return x


def lift(x, y, z, intrinsics, homogeneous=False):
    '''

    :param self:
    :param x: Shape (batch_size, num_points)
    :param y:
    :param z:
    :param intrinsics:
    :return:
    '''
    fx, fy, cx, cy = parse_intrinsics(intrinsics)

    x_lift = (x - expand_as(cx, x)) / expand_as(fx, x) * z
    y_lift = (y - expand_as(cy, y)) / expand_as(fy, y) * z

    if homogeneous:
        return torch.stack((x_lift, y_lift, z, torch.ones_like(z).cuda()), dim=-1)
    else:
        return torch.stack((x_lift, y_lift, z), dim=-1)


def load_multicam_data(render_results, visualize=False, num_pts=1024, dino_feature_key="dino_vit_features",
                       overwrite_n_cams=None, upsample_mode="nearest", return_rays=False, debug_rays=False,
                       glip_mask_key=None, glip_labels=None, visualize_gilp=False, early_return_per_view_data=False,
                       return_camera_index=False):

    cam_poses = render_results["cam_poses"]
    cam_intrinsics = render_results["cam_intrinsics"]
    objs_poses_camera = render_results["objs_poses_camera"]
    objs_poses_world = render_results["objs_poses_world"]
    rgbs = render_results["rgbs"]
    depths = render_results["depths"]
    segs = render_results["segs"]
    obj_ids = render_results["obj_ids"]

    if glip_mask_key is not None and glip_labels is not None:
        glip_masks = render_results[glip_mask_key]
        H, W, glip_labels_dim = glip_masks[0].shape
        print(f"glip dim: {(H, W, glip_labels_dim)}")
        assert len(glip_labels) == glip_labels_dim
        use_glip = True
    else:
        use_glip = False

    # debug
    dinos = render_results[dino_feature_key]  # H, W, C
    H, W, dino_dim = dinos[0].shape
    print(f"dino dim: {(H, W, dino_dim)}")

    if overwrite_n_cams is None:
        n_cams = len(cam_poses)
    else:
        n_cams = overwrite_n_cams

    # augmentation
    # if self.depth_aug:
    #     depth = depth + np.random.randn(*depth.shape) * 0.1

    # change these values depending on the intrinsic parameters of camera used to collect the data. These are what we used in pybullet
    y, x = torch.meshgrid(torch.arange(480), torch.arange(640))

    # old code from ndf
    # # Compute native intrinsic matrix
    # sensor_half_width = 320
    # sensor_half_height = 240
    # vert_fov = 60 * np.pi / 180
    # vert_f = sensor_half_height / np.tan(vert_fov / 2)
    # hor_f = sensor_half_width / (np.tan(vert_fov / 2) * 320 / 240)
    # intrinsics = np.array(
    #     [[hor_f, 0., sensor_half_width, 0.],
    #      [0., vert_f, sensor_half_height, 0.],
    #      [0., 0., 1., 0.]]
    # )
    # # Rescale to new sidelength
    # intrinsics = torch.from_numpy(intrinsics)

    # build depth images from data
    obj_xyzrgbss = [[] for _ in range(len(obj_ids))]
    obj_glip_maskss = [[] for _ in range(len(obj_ids))]
    scene_xyzrgbs = []
    for ci in range(n_cams):
        intrinsics = torch.from_numpy(cam_intrinsics[ci])
        depth = torch.from_numpy(depths[ci].flatten())
        print("[MultiCams] segemented ids for this cam:", np.unique(segs[ci]))

        for oi, obj_id in enumerate(obj_ids):
            # TODO: make this more efficient
            seg_mask = np.where(segs[ci].flatten() == obj_id)
            obj_xyz = lift(x.flatten()[seg_mask], y.flatten()[seg_mask], depth[seg_mask], intrinsics[None, :, :])
            obj_rgb = rgbs[ci].reshape(-1, 3)[seg_mask]

            if use_glip:
                obj_glip_masks = glip_masks[ci].reshape(-1, glip_labels_dim)[seg_mask]
                obj_glip_maskss[oi].append(obj_glip_masks)

            # transform to world coordinate
            # trimesh.PointCloud(obj_xyz, np.concatenate([obj_rgb, np.ones([obj_rgb.shape[0], 1]) * 255.0], axis=-1)).show()
            cam_pose = cam_poses[ci]
            obj_xyz = trimesh.transform_points(obj_xyz, cam_pose)
            # trimesh.PointCloud(obj_xyz, np.concatenate([obj_rgb, np.ones([obj_rgb.shape[0], 1]) * 255.0], axis=-1)).show()
            obj_xyzrgbss[oi].append(np.concatenate([obj_xyz, obj_rgb, obj_dinos], axis=-1))

        # scene_xyz = geometry.lift(x.flatten(), y.flatten(), depth,intrinsics[None, :, :])
        # scene_rgb = rgbs[ci].reshape(-1, 3)
        # cam_pose = cam_poses[ci]
        # scene_xyz = trimesh.transform_points(scene_xyz, cam_pose)
        # scene_xyzrgbs.append(util.crop_pcd(np.concatenate([scene_xyz, scene_rgb], axis=-1), x=[0.0, 0.7], y=[-0.4, 0.4], z=[0.9, 1.5]))

    if early_return_per_view_data:
        return obj_xyzrgbss[0], obj_glip_maskss[0]

    obj_xyzrgbs = []
    obj_glip_masks = []
    for oi, _ in enumerate(obj_ids):
        obj_xyzrgb = np.concatenate(obj_xyzrgbss[oi], axis=0)
        print(f"total pts {obj_xyzrgb.shape}")

        # TODO: subsample view direction as well
        if num_pts is not None and len(obj_xyzrgb) > num_pts:
            # subsample
            rix = torch.randperm(obj_xyzrgb.shape[0])
            obj_xyzrgb = obj_xyzrgb[rix[:num_pts]]
        obj_xyzrgbs.append(obj_xyzrgb)

        if visualize:
            trimesh.PointCloud(obj_xyzrgb[:, :3], np.concatenate([obj_xyzrgb[:, 3:6], np.ones([obj_xyzrgb.shape[0], 1]) * 255.0], axis=-1)).show()

        if use_glip:
            obj_glip_mask = np.concatenate(obj_glip_maskss[oi], axis=0)
            obj_glip_masks.append(obj_glip_mask)

            if visualize_gilp:
                for li, label in enumerate(glip_labels):
                    print(label)
                    print(obj_glip_mask[:, li].shape)
                    print(obj_xyzrgb.shape)

                    colors = glip_score_to_rgba(obj_glip_mask[:, li])
                    # print(np.max(obj_glip_mask[:, li]))
                    # colors = np.concatenate([np.ones([obj_xyzrgb.shape[0], 3]) * 255, obj_glip_mask[:, li].reshape(-1, 1) / np.max(obj_glip_mask[:, li]) * 255], axis=-1)

                    trimesh.PointCloud(obj_xyzrgb[:, :3], colors).show()

    # scene_xyzrgb = np.concatenate(scene_xyzrgbs, axis=0)
    # rix = torch.randperm(scene_xyzrgb.shape[0])
    # scene_xyzrgb = scene_xyzrgb[rix[:num_pts]]
    # if visualize:
    #     trimesh.PointCloud(scene_xyzrgb[:, :3], np.concatenate([scene_xyzrgb[:, 3:], np.ones([scene_xyzrgb.shape[0], 1]) * 255.0], axis=-1)).show()

    to_return = [obj_xyzrgbs]
    if use_glip:
        to_return.append(obj_glip_masks)

    if len(to_return) == 1:
        return to_return[0]
    else:
        return to_return

    # if return_rays:
    #     if use_glip:
    #         return obj_xyzrgbs, obj_view_dirs, obj_glip_masks
    #     else:
    #         return obj_xyzrgbs, obj_view_dirs
    #
    # if use_glip:
    #     return obj_xyzrgbs, obj_glip_masks
    # else:
    #     return obj_xyzrgbs


def glip_score_to_rgba(scores):
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    cmap = cm.jet
    norm = Normalize(vmin=0, vmax=np.max(scores))
    scores = norm(scores)
    colors = cmap(scores) * 255
    return np.array(colors).astype(np.int)