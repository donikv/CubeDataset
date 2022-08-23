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
import copy


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
    dir_name_ambient = f'{dir}/ambient'
    images = os.listdir(dir_name)
    images = list(filter(lambda x: str(x).lower().endswith('.png') and not x.startswith('.'), images))
    images = sorted(images, key=lambda x: int(x[:-4]))

    gray_pos = np.loadtxt(f'{dir}/pos.txt').astype(int)
    gray_pos = gray_pos // 2

    gts = []
    for idx, img in enumerate(images):
        image = fu.load_png(img, dir_name, '', mask_cube=False)
        image = iu.process_image(image, depth=14, blacklevel=0, scale=True)
        image_a = fu.load_png(img, dir_name_ambient, '',
                              mask_cube=False)
        # image_rc = image_r.copy()  # fu.load_png(img, dir_name_right+'_cube', 'debayered', mask_cube = False)
        # image_lc = image_l.copy()
        # fu.load_png(img, dir_name_left+'_cube', 'debayered', mask_cube = False)
        image_a = iu.process_image(image_a, depth=14, blacklevel=0, scale=True)

        x1, y1, x2, y2 = gray_pos[idx]
        gt_ambient = np.clip(image_a[y1, x1], 0.001, 1)
        gt_both = np.clip(image[y2, x2], 0.001, 1)
        gt_flash, image_flash_l, r = pu.get_one_ill_from_combination(gt_ambient, gt_both, image_a, image, [y1, x1], offset=15)
        # gt_flash = np.array([0.78907544, 1., 0.27116983])
        gt = np.concatenate((gt_ambient, gt_flash), axis=-1)
        gts.append(gt)
        data = (idx, image, image_a, gt_ambient, gt_flash, [x1, y1, x2, y2], dir, 0, r)
        par_create(data)




    np.savetxt(f'{dir}/gt.txt', np.array(gts, dtype=np.float32))


def debayer_rw(img, dir_name):
    if img.endswith('tiff'):
        image_tiff = cv2.imread(f'{dir_name}/{img}', cv2.IMREAD_UNCHANGED)
        imageRaw = iu.debayer(image_tiff).astype(np.uint16)
        # imageRaw = cv2.cvtColor(image_tiff, cv2.COLOR_BAYER_BG2RGB)
    else:
        imageRaw = fu.load_cr2(img, path=dir_name, directory='', mask_cube=False)
    rgb = cv2.cvtColor(imageRaw, cv2.COLOR_RGB2BGR)
    return rgb


def par_create(data):
    idx, img, img_l, gt_left, gt_right, pos, dir, tresh, r = data
    dir_name = f'{dir}/both/images'
    dir_name_left = f'{dir}/ambient/images'
    dir_name_right = f'{dir}/right/images'

    gt_left = gt_left / gt_left.max()
    gt_right = gt_right / gt_right.max()
    image = img * 2**14
    image_original = copy.copy(image)
    r = 1-r
    gt_mask, ir, il, r = pu.create_gt_mask(image * 255 * 4, img_l, img_l, gt_right, gt_left, thresh=tresh, r=r)
    ggt_mask = gt_mask
    r = r * 255
    imcl = iu.color_correct_single_16(copy.copy(image), u_ill=gt_left, c_ill=1 / 1.713)
    imcf = iu.color_correct_single_16(copy.copy(image), u_ill=gt_right, c_ill=1 / 1.713)
    # imcr = iu.color_correct_single_16(image, u_ill=gt_right, c_ill=1 / 1.713)

    savedir = f'{dir}/organised/{idx+1}'
    os.makedirs(savedir, exist_ok=True)

    saveImage(ggt_mask.astype(np.uint8), f'{savedir}/', 'gt', True)

    saveImage(imcl.astype(np.uint16), f'{savedir}/', "img_corrected_1", True)
    saveImage(imcf.astype(np.uint16), f'{savedir}/', "img_corrected_flash", True)
    saveImage(image_original.astype(np.uint16), f'{savedir}/', "img", True)
    saveImage(r.astype(np.uint8), f'{savedir}/', "gt_mask", True)
    np.savetxt(f'{savedir}/gt.txt', np.concatenate([gt_left, gt_right], axis=-1).reshape((1, -1)))
    np.savetxt(f'{savedir}/cube.txt', np.array(pos).reshape((1, -1)), fmt="%d")

    # saveImage(imcr.astype(np.uint16), f'{dir}/img_corrected_1/', f"{idx + 1}r", True)
    # saveImage((255 - r).astype(np.uint8), f'{dir}/gt_mask/', f"{idx + 1}r", True)



def create_gt_mask(dir, thresh):
    dir_name = f'{dir}/both/images'
    images = os.listdir(dir_name)
    images = list(filter(lambda x: str(x).lower().endswith('.png'), images))
    gts = np.loadtxt(f'{dir}/gt.txt').reshape((-1, 2, 3))
    gray_pos = np.loadtxt(f'{dir}/pos.txt').astype(int) // 2
    gts_left = gts[:, 0, :]
    gts_right = gts[:, 1, :]
    data = list(map(lambda x: (int(x[:-4])-1, x, gts_left, gts_right, gray_pos, dir, thresh), images))

    num_proc = 8

    if num_proc > 1:
        with mp.Pool(num_proc) as p:
            p.map(par_create, data)
            print('done')
        return

    for d in data:
        par_create(d)


if __name__ == '__main__':
    folds = ['both', 'ambient']#, 'both_cube', 'ambient_cube', 'direct_cube']
    path = '/Volumes/Jolteon/fax/images_flash/'

    for fold in folds:
        images = os.listdir(path+fold)
        images = list(filter(lambda x: x.endswith('.tiff') and not x.startswith('.'), images))
        images = sorted(images, key=lambda x: int(x[:-5]))
        for idx, image in enumerate(images):
            deb_img = debayer_rw(image, path+fold)
            saveImage(deb_img, path+fold, f'{idx+1}', False)
            print(image)


    find_gt_realworld(path)
    # create_gt_mask(path, 0)
    # exit(0)