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
    images = os.listdir(dir_name+'/debayered')
    images = list(filter(lambda x: str(x).lower().endswith('.png'), images))
    images = sorted(images, key=lambda x: int(x[:-4]))

    gray_pos = np.loadtxt(f'{dir}/pos.txt').astype(int)

    gts_left, gts_right = [], []
    for idx, img in enumerate(images):
        image = fu.load_png(img, dir_name, 'debayered', mask_cube=False)
        image = iu.process_image(image, depth=16, blacklevel=2048)
        image_r = fu.load_png(img, dir_name_right, 'debayered',
                              mask_cube=False)
        image_l = fu.load_png(img, dir_name_left, 'debayered',
                              mask_cube=False)
        image_rc = image_r.copy()  # fu.load_png(img, dir_name_right+'_cube', 'debayered', mask_cube = False)
        image_lc = image_l.copy()
        # fu.load_png(img, dir_name_left+'_cube', 'debayered', mask_cube = False)
        image_r = iu.process_image(image_r, depth=16, blacklevel=2048)
        image_l = iu.process_image(image_l, depth=16, blacklevel=2048)

        x1, y1 = gray_pos[idx]
        gt_left = np.clip(image_lc[y1, x1] / 255, 1, 255)
        gt_right = np.clip(image_rc[y1, x1] / 255, 1, 255)
        gts_left.append(gt_left)
        gts_right.append(gt_right)
        saveImage(image, f'{dir}/both/images', idx + 1, True)
        saveImage(image_l, f'{dir}/ambient/images', idx + 1, True)
        saveImage(image_r, f'{dir}/direct/images', idx + 1, True)
    np.savetxt(f'{dir}/gt_ambient.txt', np.array(gts_left, dtype=np.uint8), fmt='%d')
    np.savetxt(f'{dir}/gt_direct.txt', np.array(gts_right, dtype=np.uint8), fmt='%d')


def debayer_rw(img, dir_name):
    if img.endswith('tiff'):
        image_tiff = cv2.imread(f'{dir_name}/{img}', cv2.IMREAD_UNCHANGED)
        imageRaw = cv2.cvtColor(image_tiff, cv2.COLOR_BAYER_RG2BGR)
    else:
        imageRaw = fu.load_cr2(img, path=dir_name, directory='', mask_cube=False)
    rgb = cv2.cvtColor(imageRaw, cv2.COLOR_RGB2BGR)
    return rgb


def par_create(data):
    idx, img, gts_left, gts_right, dir, tresh = data
    dir_name = f'{dir}/both/images'
    dir_name_left = f'{dir}/ambient/images'
    dir_name_right = f'{dir}/direct/images'

    image = fu.load_png(img, dir_name, '', mask_cube=False)
    image_left = fu.load_png(img, dir_name_left, '', mask_cube=False)
    image_right = fu.load_png(img, dir_name_right, '', mask_cube=False)
    gt_left = np.clip(gts_left[idx], 1, 255) / 255
    gt_right = np.clip(gts_right[idx], 1, 255) / 255
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

    num_proc = 8

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
    #     images = list(filter(lambda x: x.endswith('.NEF'), images))
    #     images = sorted(images, key=lambda x: int(x[4:-4]))
    #     for idx, image in enumerate(images):
    #         deb_img = debayer_rw(image, path+fold)
    #         saveImage(deb_img, path+fold+'/debayered', f'{idx+1}', False)
        # images = os.listdir(path + fold)
        # images_tiff = list(filter(lambda x: x.endswith('.tiff'), images))
        # images_tiff = sorted(images_tiff, key=lambda x: int(x[4:-5]))
        # for idx, image in enumerate(images_tiff):
        #     deb_img = debayer_rw(image, path+fold)
        #     saveImage(deb_img, path+fold+'/debayered_tiff', f'{idx+1}', False)


    find_gt_realworld(path)
    create_gt_mask(path, 0)
    exit(0)