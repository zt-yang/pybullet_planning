import copy
import os
import PIL.Image
import numpy as np
import json
import shutil
from os import listdir
from os.path import join, isdir, isfile, dirname, getmtime
import time

from pybullet_tools.utils import quat_from_euler, reset_simulation, remove_body, get_joint_name, get_link_name, \
    euler_from_quat, set_color, apply_alpha, WHITE, unit_pose
from pybullet_tools.camera_utils import get_segmask, adjust_segmask, adjust_camera_pose, \
    get_obj_keys_for_segmentation
from pybullet_tools.bullet_utils import get_door_links

from lisdf_tools.lisdf_loader import load_lisdf_pybullet, get_depth_images, make_furniture_transparent, get_camera_kwargs
from lisdf_tools.image_utils import crop_image, save_seg_image_given_obj_keys

# from utils import load_lisdf_synthesizer
from pigi_tools.data_utils import get_indices, get_init_tuples, \
    get_body_map, get_world_center, add_to_planning_config
from data_generator.run_utils import copy_dir_for_process, get_data_processing_parser
from world_builder.paths import OUTPUT_PATH


def get_camera_poses(viz_dir):
    config = json.load(open(join(viz_dir, 'planning_config.json')))
    camera_zoomins = [] if 'camera_zoomins' not in config else config['camera_zoomins']
    camera_names = []
    camera_poses = []
    camera_kwargs = []

    ## custom written poses
    if "camera_poses" in config and len(camera_poses) > 0:
        for name, camera_pose in config["camera_poses"].items():
            camera_names.append(name)
            if isinstance(camera_pose, list):
                camera_kwargs.append(dict())
                camera_poses.append(camera_pose)
            elif 'camera_point' in camera_pose:
                camera_kwargs.append(camera_pose)
                camera_poses.append(unit_pose())
            elif 'name' in camera_pose:  ## TODO: not tested
                camera_zoomins = [camera_pose] + camera_zoomins

    ## automatically generating an array of poses
    else:
        ## put one camera in the center (in y axis) and a few in an array, all facing the world
        cx, cy, lx, ly = get_world_center(viz_dir)
        camera_kwargs = [
            {'camera_point': (cx + 5, cy, 2.5), 'target_point': (0, cy, 1)},
        ]
        positions = [cy - 3 * ly / 8, cy - ly / 8, cy + ly / 8, cy + 3 * ly / 8]  ## [cy-ly/3, cy, cy+ly/3]:
        for y in positions:
            camera_kwargs.append(
                # {'camera_point': (cx+1, y, 2.2), 'target_point': (0, y, 0.5)}
                {'camera_point': (cx + 1.8, y, 2.8), 'target_point': (0, y, 0.5)}
            )
        camera_names.extend(['center'] + [f'array_{i}' for i in range(len(positions))])
        camera_poses = [unit_pose()] * len(camera_kwargs)
        add_to_planning_config(viz_dir, {'camera_kwargs': camera_kwargs})

    # ## poses used during scene generation
    # if "obs_camera_pose" in config:
    #     camera_pose = adjust_camera_pose(config["obs_camera_pose"])
    #     camera_poses.append(camera_pose)
    #     camera_names.append('obs_camera_pose')
    #     camera_kwargs.append(dict())
    #     add_to_planning_config(viz_dir, {'img_camera_pose': camera_pose})

    camera_names += ['zoomin_'+c['name'] for c in camera_zoomins]
    return camera_poses, camera_kwargs, camera_zoomins, camera_names


#################################################################


def create_doorless_lisdf(test_dir):
    lisdf_file = join(test_dir, 'scene.lisdf')
    text = open(lisdf_file).read().replace('MiniFridge', 'MiniFridgeDoorless')
    doorless_lisdf = join(test_dir, 'scene_dooless.lisdf')
    with open(doorless_lisdf, 'w') as f:
        f.write(text)
    return doorless_lisdf


def render_transparent_doors(test_dir, viz_dir, camera_pose):
    world = load_lisdf_pybullet(test_dir, width=720, height=560)

    paths = {}
    for m in world.lisdf.models:
        if m.name in ['minifridge', 'cabinet']:
            path = m.uri.replace('../../', '').replace('/mobility.urdf', '')
            paths[m.name] = path

    count = 0
    bodies = copy.deepcopy(world.body_to_name)
    for b, name in bodies.items():
        if name in['minifridge', 'cabinet']:
            doors = world.add_joints_by_keyword(name)
            for _, d in doors:
                for l in get_door_links(b, d):
                    set_color(b, link=l, color=apply_alpha(WHITE, alpha=0.2))
                    count += 1
    print(f'changed {count} doors to transparent')
    world.add_camera(camera_pose, viz_dir)
    world.visualize_image(index='trans', rgb=True)


def render_rgb_image(test_dir, viz_dir, camera_pose):
    world = load_lisdf_pybullet(test_dir, width=720, height=560)
    world.add_camera(camera_pose, viz_dir)
    world.visualize_image(index='scene', rgb=True)


def render_segmented_rgb_images(test_dir, viz_dir, camera_pose, robot=False):
    get_depth_images(test_dir, camera_pose=camera_pose, rgb=True, robot=robot,
                     img_dir=join(viz_dir, 'rgb_images'))


def render_segmented_rgbd_images(test_dir, viz_dir, camera_pose, robot=False):
    get_depth_images(test_dir, camera_pose=camera_pose, rgbd=True, robot=robot,
                     img_dir=join(viz_dir))


def fix_planning_config(viz_dir):
    config_file = join(viz_dir, 'planning_config.json')
    config = json.load(open(config_file, 'r'))
    if 'body_to_name' in config:
        body_to_name = config['body_to_name']
        new_body_to_name = {}
        changed = False
        for k, v in body_to_name.items():
            k = eval(k)
            if isinstance(k, tuple) and not ('link' in v or 'joint' in v):
                name = body_to_name[str(k[0])] + '::'
                if len(k) == 2:
                    name += get_joint_name(k[0], k[-1])
                elif len(k) == 3:
                    name += get_link_name(k[0], k[-1])
                v = name
                changed = True
            new_body_to_name[str(k)] = v
        if changed:
            config['body_to_name'] = new_body_to_name
            # tmp_config_file = join(viz_dir, 'planning_config_tmp.json')
            # shutil.move(config_file, tmp_config_file)
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=3)


def adjust_indices_for_full_kitchen(indices):
    return {k: v for k, v in indices.items() if not 'pr2' in v}


def render_images(test_dir, viz_dir, camera_poses, camera_kwargs, camera_zoomins=[], camera_names=[],
                  crop=False, segment=True, transparent=False, width=1280, height=960, fx=800,
                  done=None, pairs=None, larger_world=False, crop_px=224):
    ## width = 1960, height = 1470, fx = 800
    world = load_lisdf_pybullet(test_dir, width=width, height=height, verbose=False,
                                larger_world=larger_world)
    remove_body(world.robot.body)
    doorless_lisdf = None
    if transparent:
        world.make_doors_transparent()
        doorless_lisdf = create_doorless_lisdf(test_dir)

    """ find the door links """
    body_map = get_body_map(viz_dir, world) if 'mm_' in viz_dir else None
    indices = get_indices(viz_dir, larger=True, body_map=body_map)
    indices = adjust_indices_for_full_kitchen(indices)

    ## pointing at goal regions: initial and final
    camera_poses.extend([unit_pose()] * len(camera_zoomins))
    camera_kwargs.extend([get_camera_kwargs(world, d) for d in camera_zoomins])

    common = dict(img_dir=viz_dir, width=width//2, height=height//2, fx=fx//2)
    crop_kwargs = dict(crop=crop, center=crop, width=width//2, height=height//2, N_PX=crop_px)

    ## a fix for previous wrong lisdf names in planning_config[name_to_body]
    # fix_planning_config(viz_dir)

    if not segment:
        os.makedirs(join(viz_dir, 'images'), exist_ok=True)

    for i in range(len(camera_poses)):
        if segment and done is not None and done[i]:
            continue

        # ---------- make furniture disappear ---------
        if not transparent and len(camera_poses) > 1 and i == len(camera_poses) - 1:
            make_furniture_transparent(world, viz_dir, lower_tpy=1, upper_tpy=0,
                                       remove_upper_furnitures=True)

        world.add_camera(camera_poses[i], **common, **camera_kwargs[i])

        if not crop:
            if 0 < i < len(camera_poses)-len(camera_zoomins):
                crop_kwargs = dict(crop=True, center=False, width=width//2, height=height//2,
                                   N_PX=height//2, align_vertical='top', keep_ratio=True)
            elif i >= len(camera_poses)-len(camera_zoomins):
                ## more zoomed-in on braiser
                n_px = int(height//2 * 0.6) if (i == len(camera_poses)-1) else int(height//2)
                crop_kwargs = dict(crop=True, center=False, width=width//2, height=height//2,
                                   N_PX=n_px, align_vertical='center', keep_ratio=True)

        ## get the scene image
        imgs = world.camera.get_image(segment=True, segment_links=True)
        rgb = imgs.rgbPixels[:, :, :3].astype(np.uint8)
        im = PIL.Image.fromarray(rgb)
        if not crop and crop_kwargs['crop']:
            im = crop_image(im, **{k: v for k, v in crop_kwargs.items() if k not in ['crop', 'center']})

        if not segment:
            im.save(join(viz_dir, 'images', f'{camera_names[i]}.png'))

        else:
            new_key = 'seg_image' if not crop else 'crop_image'
            new_key = 'transp_image' if transparent else new_key
            new_key = f"{new_key}s"
            if len(camera_poses) > 1:
                new_key = f"{new_key}_{i}"
            rgb_dir = join(viz_dir, new_key)
            os.makedirs(rgb_dir, exist_ok=True)
            # print(f'    ..... generating in {new_key}')

            """ save the scene image """
            im.save(join(rgb_dir, f'{new_key}_scene.png'))
            im_name = new_key + "_[{index}]_{name}.png"

            """ get segmask with opaque doors """
            seg = imgs.segmentationMaskBuffer
            # seg = imgs.segmentationMaskBuffer[:, :, 0].astype('int32')
            unique = get_segmask(seg)
            obj_keys = get_obj_keys_for_segmentation(indices, unique)

            """ get segmask with transparent doors """
            if transparent:
                reset_simulation()
                world = load_lisdf_pybullet(doorless_lisdf, width=width, height=height,
                                            verbose=False, jointless=True)
                remove_body(world.robot.body)
                # world.add_camera(camera_pose, viz_dir, width=width, height=height, fx=fx)
                unique = adjust_segmask(unique, world)

            """ get pairs of objects to render """
            if pairs is not None:
                inv_indices = {v: k for k, v in indices.items()}
                indices.update({'+'.join([str(inv_indices[n]) for n in p]): p for p in pairs})

            """ render cropped images """
            files = []
            for k, v in indices.items():
                ## single object/part
                if '+' not in k:
                    keys = obj_keys[v]

                ## pairs of objects/parts
                else:
                    keys = []
                    for vv in v:
                        keys.extend(obj_keys[vv])
                    v = '+'.join([n for n in v])

                ## skip generation if already exists
                file_name = join(rgb_dir, im_name.format(index=str(k), name=v))
                if isfile(file_name): continue
                files.append(file_name)

                ## save the cropped image
                save_seg_image_given_obj_keys(rgb, keys, unique, file_name, **crop_kwargs)

            files = [f for f in files if 'braiserbody' in f]
            if len(files) == 2:
                bottom_file = [f for f in files if 'braiser_bottom' in f][0]
                braiser_file = [f for f in files if f not in bottom_file][0]
                shutil.copy(braiser_file, bottom_file)


def add_key(viz_dir, new_key):
    config_file = join(viz_dir, 'planning_config.json')
    config = json.load(open(config_file, 'r'))
    if 'version_key' not in config or config['version_key'] != new_key:
        config['version_key'] = new_key
        tmp_config_file = join(viz_dir, 'planning_config_tmp.json')
        if isfile(tmp_config_file):
            os.remove(tmp_config_file)
        shutil.move(config_file, tmp_config_file)
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=3)
        os.remove(tmp_config_file)


def check_key_same(viz_dir, accepted_keys):
    config_file = join(viz_dir, 'planning_config.json')
    config = json.load(open(config_file, 'r'))
    if 'version_key' not in config:
        crop_dir = join(viz_dir, 'crop_images')
        if isdir(crop_dir):
            imgs = [join(crop_dir, f) for f in listdir(crop_dir) if 'png' in f]
            if len(imgs) > 0:
                image_time = getmtime(imgs[0])
                now = time.time()
                since_generated = now - image_time
                print('found recently generated images')
                return since_generated < 6000
            return False
        return False
    return config['version_key'] in accepted_keys


def get_num_images(viz_dir, pairwise=False):
    indices = get_indices(viz_dir)
    indices = adjust_indices_for_full_kitchen(indices)
    objs = list(indices.values())
    num_images = len(indices) + 1
    pairs = []
    if pairwise:
        init = get_init_tuples(viz_dir)
        for f in init:
            oo = [i for i in f if i in objs]
            if len(oo) >= 2:
                # print(f)
                pairs.append(oo)
    num_images += len(pairs)
    return num_images, pairs


def check_if_skip(viz_dir):

    # if not exist_instance(viz_dir, '100501'):
    #     return True

    # A = join(viz_dir, 'seg_images_5')
    # B = join(viz_dir, 'seg_images_6')
    # if isdir(B):
    #     shutil.rmtree(B)
    # shutil.copytree(A, B)
    # files = [join(B, f) for f in listdir(B) if 'png' in f]
    # for f in files:
    #     target_dir = f.replace('/seg_images_5', '/seg_images_6')
    #     shutil.move(f, target_dir)
    #
    # return True

    ##################################################

    # A = join(viz_dir, 'seg_images_5')
    # B = join(viz_dir, 'seg_images_6')
    # C = join(viz_dir, 'seg_images_9')
    # shutil.move(A, C)
    # shutil.move(B, A)
    # shutil.move(C, B)
    # # return True

    # ## -----------------------------------------------

    # files = [join(A, f) for f in listdir(A) if 'png' in f]
    # for f in files:
    #     shutil.move(f, f.replace('/seg_images_6', '/seg_images_5'))
    # files = [join(B, f) for f in listdir(B) if 'png' in f]
    # for f in files:
    #     shutil.move(f, f.replace('/seg_images_5', '/seg_images_6'))
    # return True

    ## ---------------------------------------------------------------------

    # file = join(viz_dir, 'seg_images_0', 'seg_images_0_[25]_braiserbody#1.png')
    # if not isfile(file):
    #     print(file)
    # return True

    # run_num = eval(viz_dir.split('/')[-1])
    # if run_num < 80:
    #     return True

    ## ---------------------------------------------------------------------

    # check_file = join(viz_dir, 'seg_images_6', 'seg_images_6_scene.png')
    # if isfile(check_file) and os.path.getmtime(check_file) > MODIFIED_TIME:
    #     return True

    return False


def rearrange_directories(viz_dir, seg_dirs, accepted_keys, redo):

    """ other types of image """
    if not check_key_same(viz_dir, accepted_keys=accepted_keys) or redo:
        # if isdir(rgb_dir):
        #     shutil.rmtree(rgb_dir)
        # for crop_dir in crop_dirs:
        #     if isdir(crop_dir):
        #         shutil.rmtree(crop_dir)
        for seg_dir in seg_dirs:
            if isdir(seg_dir) and '/seg_images_6' in seg_dir: ## and ('/seg_images_5' in seg_dir or 'images_6' in seg_dir): ##
                shutil.rmtree(seg_dir)
        # for transp_dir in transp_dirs:
        #     if isdir(transp_dir):
        #         shutil.rmtree(transp_dir)

    ## ----------------------------------------------------

    # for seg_dir in seg_dirs:
    #     files = [join(seg_dir, f) for f in listdir(seg_dir) if 'braiserbody' in f]
    #     if len(files) == 2:
    #         bottom_file = [f for f in files if 'braiser_bottom' in f][0]
    #         braiser_file = [f for f in files if f not in bottom_file][0]
    #         shutil.copy(braiser_file, bottom_file)

    ## ----------------------------------------------------

    # if exist_instance(viz_dir, '100015') and isdir(seg_dirs[0]):
    #     braiser_files = [join(seg_dirs[0], f) for f in listdir(seg_dirs[0]) \
    #                      if f.endswith('.png') and 'braiser' in f]
    #     if len(braiser_files) > 0:
    #         if 1674437043 > os.path.getmtime(braiser_files[0]) > 1674351569:
    #             print('braiser problem', viz_dir)
    #             for f in braiser_files:
    #                 os.remove(f)


############################################################################


def generate_segmented_images(inputs):
    viz_dir, args = inputs
    redo = args.redo

    viz_dir = join(OUTPUT_PATH, viz_dir)
    print(f"\nviz_dir = {viz_dir}\n")

    skip_folder = check_if_skip(viz_dir)
    if skip_folder:
        return

    ## ---------------------------------------------------------------------

    num_dirs = 6  ## len(camera_kwargs) + len(camera_zoomins)

    rgb_dir = join(viz_dir, 'rgb_images')
    seg_dirs = [join(viz_dir, f'seg_images_{i}') for i in range(num_dirs)]
    crop_dirs = [join(viz_dir, f'crop_images_{i}') for i in range(num_dirs)]
    transp_dirs = [join(viz_dir, f'transp_images_{i}') for i in range(num_dirs)]

    ## choose configuration
    name, dirs, kwargs = 'segmenting', seg_dirs, dict()
    # name, dirs, kwargs = 'cropping', crop_dirs, dict(crop=True)
    # name, dirs, kwargs = 'transparent doors', transp_dirs, dict(crop=True, transparent=True)

    rearrange_directories(viz_dir, seg_dirs, args.accepted_keys, redo)

    ## ---------------------------------------------------------------------

    """ get the camera poses """
    camera_poses, camera_kwargs, camera_zoomins, camera_names = get_camera_poses(viz_dir)

    ## ------------- visualization function to test -------------------

    # if not isdir(rgb_dir):
    #     print(viz_dir, 'rgbing ...')
    #     render_segmented_rgb_images(test_dir, viz_dir, camera_pose, robot=False)
    #     reset_simulation()

    ## Pybullet segmentation mask
    num_imgs, pairs = get_num_images(viz_dir, pairwise=False)

    ## skip if all done
    done = []
    for img_dir in dirs:
        done.append(isdir(img_dir) and len(listdir(img_dir)) >= num_imgs)

        # if isdir(img_dir):
        #     files = [join(img_dir, f) for f in listdir(img_dir) if 'braiserbody' in f]
        #     if len(files) == 2:
        #         bottom_file = [f for f in files if 'braiser_bottom' in f][0]
        #         braiser_file = [f for f in files if f not in bottom_file][0]
        #         shutil.copy(braiser_file, bottom_file)

    if (False in done) or redo:
        print(viz_dir, f'{name} ...')
        test_dir = copy_dir_for_process(viz_dir)
        render_images(test_dir, viz_dir, camera_poses, camera_kwargs, camera_zoomins, camera_names,
                      segment=args.seg, done=done, larger_world=args.larger_world, crop_px=args.crop_px, **kwargs)
        reset_simulation()
        shutil.rmtree(test_dir)
        add_key(viz_dir, args.new_key)
    else:
        print('skipping', viz_dir, name)
