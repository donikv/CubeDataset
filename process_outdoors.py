import cv2
import numpy as np
import os
from skimage.filters import gaussian

import utils.image_utils as iu
import utils.file_utils as fu
import utils.relighting_utils as ru
import utils.projector_utils as pu
import utils.plotting_utils as plt


def load_and_get_gt(path, idx, tiff):
    name = str(idx+1)
    if not tiff:
        im = fu.load_cr2(name+'.NEF', path, directory='', mask_cube=False)
    else:
        im = load_tiff(name+'.tiff', path, directory='')

    im = iu.process_image(im, depth=14, scale=True, blacklevel=0)

    x1, y1, x2, y2 = np.loadtxt(path+'/pos.txt').astype(int)[idx]
    gt1 = im[y1-3:y1+3, x1-3:x1+3].mean(axis=1).mean(axis=0)
    gt1 = np.clip(gt1, 0.001, 1)
    gt2 = im[y2-3:y2+3, x2-3:x2+3].mean(axis=1).mean(axis=0)
    gt2 = np.clip(gt2, 0.001, 1)

    return im, gt1, gt2


def color_mask(path, idx, gts=None):
    name = str(idx+1) + 'm'
    mask = fu.load_png(name+'.png', path, '', mask_cube=False)
    if gts is None:
        gts = np.loadtxt(path + '/gt.txt')[idx].reshape(2,3)

    cm = (np.where(mask == 0, gts[1], gts[0]) * 255).astype(np.uint8)
    return cm

def load_tiff(img, path, directory):
    image_tiff = cv2.imread(f'{path}/{directory}/{img}', cv2.IMREAD_UNCHANGED)
    imageRaw = cv2.cvtColor(image_tiff, cv2.COLOR_BAYER_RG2BGR)
    return imageRaw


def correct_with_mask(path, idx):
    img = cv2.imread(path + '/images/'+str(idx+1) + '.png', cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 2**16
    mask = cv2.imread(path + '/gt/' + str(idx + 1) + '.png', cv2.IMREAD_UNCHANGED)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) / 2**8
    mask = gaussian(mask, 15)

    imgc = np.multiply(img, 1/(1.713*mask))
    return imgc, img, mask

if __name__ == '__main__':
    path = '../Datasets/outdoor/test_two_ill'
    idx = 3

    for idx in range(0, 34):
        im, gt1, gt2 = load_and_get_gt(path, idx, tiff=True)
        gt1 = gt1 / gt1.sum()
        gt2 = gt2 / gt2.sum()

        # im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        # gt = color_mask(path, idx, (gt1, gt2))
        # gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(path + f'/images/{idx + 1}.png', (im * 65535).astype(np.uint16))
        # cv2.imwrite(path + f'/gt/{idx + 1}.png', gt)

        f = open(path+'/gt.txt', 'a+')
        f.write(f'{gt1[0]} {gt1[1]} {gt1[2]} {gt2[0]} {gt2[1]} {gt2[2]}\n')
        f.close()

    gts = np.loadtxt(path + '/gt.txt').reshape(-1, 2, 3)

    for idx in range(0, 34):
        if idx == 7:
            continue
        gt = color_mask(path, idx, None)
        gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path + f'/gt/{idx + 1}.png', gt)

    # for idx in range(0, 34):
        imgc, img, mask = correct_with_mask(path, idx)
        imgc1 = iu.color_correct_single_f32(img, gts[idx][0], c_ill=1/3)
        plt.visualize([img, imgc, mask, imgc1])
        imgc = imgc.astype(np.float32)
        imgc = cv2.cvtColor(imgc, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path + f'/corrected/{idx + 1}.png', imgc * 255)

        gt_mask = cv2.imread(path + f'/{idx+1}m.png', cv2.IMREAD_UNCHANGED)
        cv2.imwrite(path + f'/img_corrected_1/{idx + 1}.png', (imgc1*65535).astype(np.uint16))
        cv2.imwrite(path + f'/gt_mask/{idx + 1}.png', gt_mask)
