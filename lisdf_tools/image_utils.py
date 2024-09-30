import os
import random
import PIL.Image
import numpy as np
import time
import sys
from os.path import join, isdir, abspath, dirname, isfile
sys.path.append(abspath(join(dirname(__file__), '..')))


from pybullet_tools.utils import AABB
from pybullet_tools.camera_utils import get_segmask


def hex_to_rgba(color):
    """
    Turn a string hex color to a (4,) RGBA color.
    Parameters
    -----------
    color: str, hex color
    Returns
    -----------
    rgba: (4,) np.uint8, RGBA color
    """
    value = str(color).lstrip('#').strip()
    if len(value) == 6:
        rgb = [int(value[i:i + 2], 16) for i in (0, 2, 4)]
        rgba = np.append(rgb, 255).astype(np.uint8) / 255
    else:
        raise ValueError('Only RGB supported')

    return rgba


RED = hex_to_rgba('#e74c3c')
ORANGE = hex_to_rgba('#e67e22')
BLUE = hex_to_rgba('#3498db')
GREEN = hex_to_rgba('#2ecc71')
YELLOW = hex_to_rgba('#f1c40f')
PURPLE = hex_to_rgba('#9b59b6')
GREY = hex_to_rgba('#95a5a6')
CLOUD = hex_to_rgba('#ecf0f1')
MIDNIGHT = hex_to_rgba('#34495e')
WHITE = hex_to_rgba('#ffffff')
BLACK = hex_to_rgba('#000000')

DARKER_RED = hex_to_rgba('#c0392b')
DARKER_ORANGE = hex_to_rgba('#d35400')
DARKER_BLUE = hex_to_rgba('#2980b9')
DARKER_GREEN = hex_to_rgba('#27ae60')
DARKER_YELLOW = hex_to_rgba('#f39c12')
DARKER_PURPLE = hex_to_rgba('#8e44ad')
DARKER_GREY = hex_to_rgba('#7f8c8d')
DARKER_MIDNIGHT = hex_to_rgba('#2c3e50')
DARKER_CLOUD = hex_to_rgba('#bdc3c7')

RAINBOW_COLORS = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE, MIDNIGHT, GREY]
DARKER_COLORS = [DARKER_RED, DARKER_ORANGE, DARKER_YELLOW, DARKER_GREEN,
                 DARKER_BLUE, DARKER_PURPLE, DARKER_MIDNIGHT, DARKER_GREY]

mp4_dir = '/home/yang/Documents/jupyter-worlds/dev/gym_images/'


def draw_bb(im, bb):
    from PIL import ImageOps
    im2 = np.array(ImageOps.grayscale(im))
    for j in range(bb.lower[0], bb.upper[0]+1):
        for i in [bb.lower[1], bb.upper[1]]:
            im2[i, j] = 255
    for i in range(bb.lower[1], bb.upper[1]+1):
        for j in [bb.lower[0], bb.upper[0]]:
            im2[i, j] = 255
    im.show()
    PIL.Image.fromarray(im2).show()


def crop_image(im, bb=None, width=1280, height=960, N_PX=224,
               align_vertical='center', keep_ratio=False):
    from pybullet_tools.utils import get_aabb_extent, get_aabb_center, AABB
    from pybullet_tools.camera_utils import get_segmask

    if bb is None:
        # crop the center of the blank image
        left = int((width - N_PX) / 2)
        if align_vertical == 'center':
            top = int((height - N_PX) / 2)
        elif align_vertical == 'top':
            top = 0
        right = left + N_PX
        if keep_ratio:
            bottom = int(top + N_PX * height / width)
        else:
            bottom = top + N_PX
        cp = (left, top, right, bottom)
        im = im.crop(cp)
        return im

    # draw_bb(im, bb)
    need_resizing = False
    size = N_PX
    padding = 30
    dx, dy = get_aabb_extent(bb)
    cx, cy = get_aabb_center(bb)
    dmax = max(dx, dy)
    if dmax > N_PX:
        dmax += padding * 2
        if dmax > height:
            dmax = height
            cy = height / 2
        need_resizing = True
        size = dmax
    left = max(0, int(cx - size / 2))
    top = max(0, int(cy - size / 2))
    right = left + size
    bottom = top + size
    if right > width:
        right = width
        left = width - size
    if bottom > height:
        bottom = height
        top = height - size
    cp = (left, top, right, bottom)

    im = im.crop(cp)
    if need_resizing:
        im = im.resize((N_PX, N_PX))
    return im


def get_mask_bb(mask):
    if np.all(mask == 0):
        return None
    col = np.max(mask, axis=0)  ## 1280
    row = np.max(mask, axis=1)  ## 960
    col = np.where(col == 1)[0]
    row = np.where(row == 1)[0]
    return AABB(lower=(col[0], row[0]), upper=(col[-1], row[-1]))


def expand_mask(mask):
    y = np.expand_dims(mask, axis=2)
    return np.concatenate((y, y, y), axis=2)


def make_image_background(old_arr):
    new_arr = np.ones_like(old_arr)
    new_arr[:, :, 0] = 178
    new_arr[:, :, 1] = 178
    new_arr[:, :, 2] = 204
    return new_arr


def get_seg_foreground_given_obj_keys(rgb, keys, unique):
    mask = np.zeros_like(rgb[:, :, 0])
    for k in keys:
        if k in unique:
            c, r = zip(*unique[k])
            mask[(np.asarray(c), np.asarray(r))] = 1
    return mask


def save_seg_image_given_obj_keys(rgb, keys, unique, file_name, crop=False, center=False, **kwargs):
    background = make_image_background(rgb)
    mask = get_seg_foreground_given_obj_keys(rgb, keys, unique)
    foreground = rgb * expand_mask(mask)
    background[np.where(mask != 0)] = 0
    new_image = foreground + background
    im = PIL.Image.fromarray(new_image)

    ## crop image with object bounding box centered
    if crop:
        if center:
            bb = get_mask_bb(mask)
        else:
            bb = None
        # if bb is not None:
        #     draw_bb(new_image, bb)
        im = crop_image(im, bb, **kwargs)

    # im.show()
    im.save(file_name)


##############################################################################


def save_seg_mask(camera_image, obj_keys, verbose=False):
    rgb = camera_image.rgbPixels[:, :, :3]
    seg = camera_image.segmentationMaskBuffer
    unique = get_segmask(seg)
    mask = np.zeros_like(rgb[:, :, 0])
    for k in obj_keys:
        if k in unique:
            c, r = zip(*unique[k])
            mask[(np.asarray(c), np.asarray(r))] = 1
    h, w, _ = rgb.shape
    sum1 = mask.sum()
    sum2 = h * w
    if verbose:
        print(f'\t{round(sum1 / sum2 * 100, 2)}%\t | masked {sum1} out of {sum2} pixels ')
    return (rgb, mask)


def make_composed_image(seg_masks, background, is_movable=False, verbose=False):
    """ input is a list of (rgb, mask) pairs """
    composed_rgb = np.zeros_like(background).astype(np.float64)
    composed_mask = np.zeros_like(seg_masks[0][1])
    for _, seg_mask in seg_masks:
        composed_mask = np.logical_or(composed_mask, seg_mask)
    composed_mask = composed_mask.astype(np.float64)

    composed_mask = expand_mask(composed_mask)
    composed_rgb += np.copy(background) * (1 - composed_mask)

    n = len(seg_masks)
    weights = np.arange(n/2, 3*n/2) / (n*(n+1)/2 + n*n/2)
    weights_second_last = weights[-2]
    weights[1:-1] = weights[:-2]
    weights[0] = weights_second_last
    if verbose:
        print(np.round(weights, decimals=3))
    foregrounds = np.zeros_like(background).astype(np.float64)
    for i, (rgb, mask) in enumerate(seg_masks):
        if is_movable:
            mask_here = expand_mask(mask)
            masked_region = np.copy(rgb) * weights[i] + np.copy(background) * (1 - weights[i])
            foregrounds = mask_here * masked_region + (1 - mask_here) * foregrounds
        else:
            composed_rgb += np.copy(rgb) * composed_mask * weights[i]
    if is_movable:
        composed_rgb += composed_mask * foregrounds

    return composed_rgb


def make_composed_image_multiple_episodes(episodes, image_name, verbose=False, crop=None, **kwargs):
    """ finally crop the image and save it """
    background = episodes[0][0][0][0]
    for seg_masks, is_movable in episodes:
        background = make_composed_image(seg_masks, background, is_movable=is_movable, **kwargs)

    composed_rgb = background.astype(np.uint8)

    h, w, _ = composed_rgb.shape
    im = PIL.Image.fromarray(composed_rgb)
    if crop is not None:
        im = im.crop(crop)

    im.save(image_name)
    if verbose:
        print(f'saved composed image {image_name} from {len(episodes)} episodes')


def merge_center_crops_of_images(image_paths, output_path, crop=None, spacing=20, show=False):
    """ used by vlm-tamp """
    images = [PIL.Image.open(path) for path in image_paths]

    if crop is None:
        right, bottom = images[0].size
        left = 0
        top = int(bottom // 4 - spacing // 4)
        bottom -= int(bottom // 4 + spacing // 4)
        crop = left, top, right, bottom
    else:
        left, top, right, bottom = crop

    width = right - left
    height = bottom - top
    n = len(images)
    composed = np.ones((n*height + (n-1)*spacing, width, 4), dtype=np.uint8) * 255
    for i, image in enumerate(images):
        image = image.crop(crop)
        start = i * (height + spacing)
        composed[start:start+height, :, :] = image
        os.remove(image_paths[i])

    compose_im = PIL.Image.fromarray(composed)
    if show:
        compose_im.show()
    else:
        compose_im.save(output_path)
    return composed


#################################################################################


def images_to_gif(img_dir, gif_path, filenames, crop=None):
    import imageio
    start = time.time()
    # print(f'saving to {abspath(gif_file)} with {len(filenames)} frames')
    with imageio.get_writer(gif_path, mode='I') as writer:
        for filename in filenames:
            # image = imageio.imread(filename)
            if crop is not None:
                left, top, right, bottom = crop
                filename = filename[top:bottom, left:right]
            writer.append_data(filename)

    print(f'saved to {abspath(gif_path)} with {len(filenames)} frames in {round(time.time() - start, 2)} seconds')
    return gif_path


def images_to_mp4(images=[], img_dir='images', mp4_path='video.mp4'):
    import cv2
    import os

    def np_arr_to_cv2(image):
        image = image[:, :, :3]
        return image[..., [2, 1, 0]].copy()  ## RGB to BGR for cv2

    def read_image(img_name):
        path = join(img_dir, img_name)
        if isfile(path):
            if img_name.endswith('.png'):
                return cv2.imread(path)
            if img_name.endswith('.npy'):
                arr = np.load(open(path, 'rb'))
                return np_arr_to_cv2(arr)

    fps = 30
    if isinstance(images[0], str):
        images = [img for img in images if img.endswith(".png") or img.endswith(".npy")]  ## os.listdir(img_dir)
        frame = read_image(images[0])
    else:
        frame = images[0]
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') ## cv2.VideoWriter_fourcc(*'XVID') ## cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))

    for image in images:
        if isinstance(images[0], str):
            image = read_image(image)
        elif isinstance(images[0], np.ndarray) and image.shape[-1] == 4:
            image = np_arr_to_cv2(image)
        video.write(image)

    cv2.destroyAllWindows()
    video.release()


def stack_mp4_horizontal(mp4s, size=None, mp4_name='stack_horizontal.mp4'):
    import cv2
    import numpy as np
    import skvideo.io
    from tqdm import tqdm

    fps = 30
    gap = 5
    frames = []
    top_bottom = [[0.3, 0.7], [0.15, 0.75]]
    for i, mp4 in enumerate(mp4s):
        videodata = skvideo.io.vread(mp4)
        num_frames, h, w, c = videodata.shape
        top, bottom = top_bottom[i]
        frames.append(videodata[:, int(h*top):int(h*bottom)-gap, :, :])

    if size is None:
        size = (w, h)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  ## cv2.VideoWriter_fourcc(*'XVID') ## cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(mp4_name, fourcc, fps, size)

    for i in tqdm(range(num_frames), desc=f'merging two videos'):
        imgs = []
        for j, videodata in enumerate(frames):
            img = videodata[i]
            img = img[..., [2, 1, 0]].copy()  ## RGB to BGR for cv2
            imgs.append(img)
        imgs = [imgs[0], 255*np.ones((gap*2, w, 3), dtype=np.uint8), imgs[1]]
        frame = np.vstack(imgs)
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()


def make_collage_mp4(mp4s, num_cols, num_rows, size=None, mp4_name='collage.mp4'):
    import cv2
    import numpy as np
    import skvideo.io
    from tqdm import tqdm

    fps = 20
    max_frames = -1
    frames = []
    for mp4 in tqdm(mp4s, desc=f'reading videos'):
        videodata = skvideo.io.vread(mp4)
        frames.append(videodata[:, ::2, ::2, :])  ## np.swapaxes(videodata, 1, 2)
        num_frames, h, w, c = videodata.shape

        if num_frames > max_frames:
            max_frames = num_frames

    if size is None:
        size = (h, w)
        clip_size = (h // num_rows, w // num_cols)
    else:
        clip_size = (size[0] // num_rows, size[1] // num_cols)
    size = size[::-1]
    clip_size = clip_size[::-1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  ## cv2.VideoWriter_fourcc(*'XVID') ## cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(mp4_name, fourcc, fps, size)

    for i in tqdm(range(max_frames), desc=f'generating collage of {len(mp4s)} videos'):
        rows = []
        this_row = []
        for j, videodata in enumerate(frames):
            img = videodata[i] if i < videodata.shape[0] else videodata[-1]
            img = cv2.resize(img, clip_size)
            img = img[..., [2, 1, 0]].copy()  ## RGB to BGR for cv2
            col = j % num_cols
            this_row.append(img)
            if col == num_cols - 1:
                rows.append(np.hstack(this_row))
                this_row = []
        frame = np.vstack(rows)
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()


def test_make_collage_mp4(mp4s=None, num_cols=2, num_rows=2, mp4_name='collage_4by4.mp4', **kwargs):
    if mp4s is None:
        mp4 = join(mp4_dir, 'gym_replay_batch_gym_0101_16:57.mp4')
        mp4s = [mp4] * (num_cols * num_rows)
    make_collage_mp4(mp4s, num_cols, num_rows, mp4_name=join(mp4_dir, mp4_name), **kwargs)


def make_collage_img(imgs, num_cols, num_rows, size=None, img_name='collage.png'):
    import cv2
    import numpy as np
    from tqdm import tqdm

    images = []
    for img in tqdm(imgs, desc=f'reading imgs'):
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        h, w, c = img.shape
        images.append(img)

    if size is None:
        clip_size = (h // num_rows, w // num_cols)
    else:
        clip_size = (size[0] // num_rows, size[1] // num_cols)
    clip_size = clip_size[::-1]

    rows = []
    this_row = []
    for j, img in enumerate(images):
        img = cv2.resize(img, clip_size)
        # img = img[..., [2, 1, 0]].copy()  ## RGB to BGR for cv2
        col = j % num_cols
        this_row.append(img)
        if col == num_cols - 1:
            rows.append(np.hstack(this_row))
            this_row = []
    frame = np.vstack(rows)

    cv2.imwrite(img_name, frame)


def test_make_collage_img(imgs=None, num_cols=2, num_rows=2, size=(1960, 1470)):
    if imgs is None:
        img = '/home/yang/Documents/fastamp-data-rss/mm_braiser/1066/zoomin/rgb_image_initial.png'
        imgs = [img] * (num_cols * num_rows)

    make_collage_img(imgs, num_cols, num_rows, size=size, img_name=join(mp4_dir, 'collage_4by4.png'))


def get_mp4s_from_dir(task_name, count=16):
    import glob
    mp4s = glob.glob(join(mp4_dir, task_name, '*.mp4'))
    random.shuffle(mp4s)
    if len(mp4s) >= count:
        mp4s = mp4s[:count]
    else:
        assert len(mp4s) > count
    return mp4s


def resize_mp4(input_mp4, output_mp4, size=(1280, 720)):
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(input_mp4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_mp4, fourcc, 5, size)

    while True:
        ret, frame = cap.read()
        if ret == True:
            b = cv2.resize(frame, size, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            out.write(b)
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def resize_mp4s_for_collage(new_dir_name, mp4s, size=(1280, 720)):
    import os
    from tqdm import tqdm

    new_dir_name = join(mp4_dir, new_dir_name)
    if not os.path.exists(new_dir_name):
        os.makedirs(new_dir_name)
    new_mp4s = []
    for mp4 in tqdm(mp4s, desc=f'resizing mp4s for collage'):
        new_mp4 = join(new_dir_name, os.path.basename(mp4))
        if not isfile(new_mp4):
            resize_mp4(mp4, new_mp4, size=size)
        new_mp4s.append(new_mp4)
    return new_mp4s


def make_collage_mp4_rss():
    random.seed(3)  ## storage
    random.seed(0)  ## braiser
    # random.seed(1)  ## sink
    # mp4s = get_mp4s_from_dir('mm_storage', count=4)
    # test_make_collage_mp4(mp4s, num_cols=2, num_rows=2)
    mp4s = get_mp4s_from_dir('mm_braiser', count=25)
    mp4s = resize_mp4s_for_collage('mm_braiser_small', mp4s, size=(3840 // 2, 2160 // 2))
    test_make_collage_mp4(mp4s, num_cols=5, num_rows=5, size=(2160 // 2 * 4, 3840 // 2 * 4))
    # test_make_collage_mp4()


def test_merge_mp4_multiview():
    task_name = 'tt_sink_to_storage_14'
    video_names = {
        'tt_storage_18': 'food-in-fridge-door',
        'tt_storage_32': 'food-in-cabinet-door',
        'tt_storage_44': 'food-in-cabinet',
        'tt_braiser_4': 'zucchini-in-pot',
        'tt_braiser_35': 'potato-in-pot',
        'tt_sink_9': 'zucchini-in-sink',
        'tt_sink_23': 'potato-in-sink-harder',
        'tt_sink_28': 'sweetpotato-in-sink',
        'tt_sink_to_storage_14': 'food-in-fridge',
        'tt_braiser_to_storage_6': 'food-in-fridge-pot',
    }
    mp4s = [join(mp4_dir, f'{task_name}_{view}.mp4') for view in ['top', 'side']]
    mp4_name = join(mp4_dir, video_names[task_name]+'.mp4') ## f'{task_name}_merged.mp4')
    stack_mp4_horizontal(mp4s, mp4_name=mp4_name)


if __name__ == "__main__":
    # test_make_collage_img()
    # make_collage_mp4_rss()
    test_merge_mp4_multiview()
