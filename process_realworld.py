import utils.projector_utils as pu
import utils.plotting_utils as plt
import utils.file_utils as fu
import utils.image_utils as iu
import utils.capture_utils as cu

import os
import cv2
import numpy as np
import multiprocessing as mp
import time
from skimage.filters import gaussian


def saveImage(image, dir, name, convert_rgb_2_bgr=True):
    if not os.path.exists(dir):
        os.mkdir(dir)
    pth = os.path.join(dir, f'{name}.png')
    if convert_rgb_2_bgr:
        r, g, b = cv2.split(image)
        image = np.dstack((b, g, r))
    cv2.imwrite(pth, image)


def find_gt_realworld(dir):
    dir_name = f'{dir}/both'
    dir_name_left = f'{dir}/ambient'
    dir_name_right = f'{dir}/direct'
    images = os.listdir(dir_name+'/debayered_tiff')
    images = list(filter(lambda x: str(x).lower().endswith('.png'), images))
    images = sorted(images, key=lambda x: int(x[:-4]))

    gray_pos = np.loadtxt(f'{dir}/pos.txt').astype(int)

    gts_left, gts_right = [], []
    for idx, img in enumerate(images):
        image = fu.load_png(img, dir_name, 'debayered_tiff', mask_cube=False)
        image = iu.process_image(image, depth=14, blacklevel=0, scale=True)
        image_r = fu.load_png(img, dir_name_right, 'debayered_tiff',
                              mask_cube=False)
        image_l = fu.load_png(img, dir_name_left, 'debayered_tiff',
                              mask_cube=False)
        # image_rc = image_r.copy()  # fu.load_png(img, dir_name_right+'_cube', 'debayered', mask_cube = False)
        # image_lc = image_l.copy()
        # fu.load_png(img, dir_name_left+'_cube', 'debayered', mask_cube = False)
        image_r = iu.process_image(image_r, depth=14, blacklevel=0, scale=True)
        image_l = iu.process_image(image_l, depth=14, blacklevel=0, scale=True)

        x1, y1, x2, y2 = gray_pos[idx]
        gt_left = np.clip(image_l[y1, x1], 0.01, 1)
        gt_right = np.clip(image_r[y2, x2], 0.01, 1)
        gts_left.append(gt_left)
        gts_right.append(gt_right)
        saveImage((image * 65535).astype(np.uint16), f'{dir}/both/images', idx + 1, True)
        saveImage((image_l * 65535).astype(np.uint16), f'{dir}/ambient/images', idx + 1, True)
        saveImage((image_r * 65535).astype(np.uint16), f'{dir}/direct/images', idx + 1, True)
    np.savetxt(f'{dir}/gt_ambient.txt', np.array(gts_left, dtype=np.float32))
    np.savetxt(f'{dir}/gt_direct.txt', np.array(gts_right, dtype=np.float32))


def debayer_rw(img, dir_name):
    if img.endswith('tiff'):
        image_tiff = cv2.imread(f'{dir_name}/{img}', cv2.IMREAD_UNCHANGED)
        imageRaw = cv2.cvtColor(image_tiff, cv2.COLOR_BAYER_BG2RGB)
    else:
        imageRaw = fu.load_cr2(img, path=dir_name, directory='', mask_cube=False)
    rgb = cv2.cvtColor(imageRaw, cv2.COLOR_RGB2BGR)
    return rgb


def par_create(data):
    idx, img, gts_left, gts_right, dir, tresh = data
    dir_name = f'{dir}/both/images'
    dir_name_left = f'{dir}/ambient/images'
    dir_name_right = f'{dir}/direct/images'

    image = fu.load_png(img, dir_name, '', mask_cube=False) / 255
    image_left = fu.load_png(img, dir_name_left, '', mask_cube=False) / 255
    image_right = fu.load_png(img, dir_name_right, '', mask_cube=False) / 255
    gt_left = np.clip(gts_left[idx], 0, 1)
    gt_right = np.clip(gts_right[idx], 0, 1)
    gt_mask, ir, il, r = pu.create_gt_mask(image, image_right, image_left, gt_right, gt_left, thresh=tresh)
    ggt_mask = gt_mask
    saveImage(ggt_mask.astype(np.uint8), f'{dir}/gt_mask/', idx + 1, True)


def create_gt_mask(dir, thresh):
    dir_name = f'{dir}/both/images'
    images = os.listdir(dir_name)
    images = list(filter(lambda x: str(x).lower().endswith('.png'), images))
    gts_left = np.loadtxt(f'{dir}/gt_ambient.txt')
    gts_right = np.loadtxt(f'{dir}/gt_direct.txt')
    data = list(map(lambda x: (int(x[:-4])-1, x, gts_left, gts_right, dir, thresh), images))

    num_proc = 1

    if num_proc > 1:
        with mp.Pool(num_proc) as p:
            p.map(par_create, data)
            print('done')
        return

    for d in data:
        par_create(d)


if __name__ == '__main__':
    folds = ['both', 'ambient', 'direct']#, 'both_cube', 'ambient_cube', 'direct_cube']
    path = 'G:/fax/diplomski/Datasets/realworld/'

    # for fold in folds:
    #     images = os.listdir(path+fold)
    #     images = list(filter(lambda x: x.endswith('.tiff'), images))
    #     images = sorted(images, key=lambda x: int(x[4:-5]))
    #     for idx, image in enumerate(images):
    #         deb_img = debayer_rw(image, path+fold)
    #         saveImage(deb_img, path+fold+'/debayered_tiff', f'{idx+1}', False)
    #         print(image)
        # images = os.listdir(path + fold)
        # images_tiff = list(filter(lambda x: x.endswith('.tiff'), images))
        # images_tiff = sorted(images_tiff, key=lambda x: int(x[4:-5]))
        # for idx, image in enumerate(images_tiff):
        #     deb_img = debayer_rw(image, path+fold)
        #     saveImage(deb_img, path+fold+'/debayered_tiff', f'{idx+1}', False)


    find_gt_realworld(path)
    create_gt_mask(path, 0)
    exit(0)