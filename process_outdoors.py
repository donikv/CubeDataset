import cv2
import numpy as np
import os

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

    x1, y1, x2, y2 = np.loadtxt(path+'/pos.txt').astype(int)[idx]
    gt1 = im[y1-10:y1+10, x1-10:x1+10].mean(axis=1).mean(axis=0)
    gt1 = np.clip(gt1, 0, 255 * 255) / 255 / 255
    gt2 = im[y2-10:y2+10, x2-10:x2+10].mean(axis=1).mean(axis=0)
    gt2 = np.clip(gt2, 0, 255 * 255) / 255 / 255

    im = iu.process_image(im, depth=14, blacklevel=0)

    return im, gt1, gt2


def color_mask(path, idx, gts=None):
    name = str(idx+1) + 'm'
    mask = fu.load_png(name+'.png', path, '', mask_cube=False)
    if gts is None:
        gts = np.loadtxt(path + '/gt.txt')[idx].reshape(2,3)

    cm = (np.where(mask == 0, gts[0], gts[1]) * 255).astype(np.uint8)
    return cm

def load_tiff(img, path, directory):
    image_tiff = cv2.imread(f'{path}/{directory}/{img}', cv2.IMREAD_UNCHANGED)
    imageRaw = cv2.cvtColor(image_tiff, cv2.COLOR_BAYER_RG2BGR)
    return imageRaw


if __name__ == '__main__':
    path = '../Datasets/outdoor'
    idx = 3

    # gt = color_mask(path, idx, None)
    # gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(path + f'/gt/{idx + 1}.png', gt)
    for idx in range(0, 3):
        im, gt1, gt2 = load_and_get_gt(path, idx, tiff=True)
        gt1 = gt1 / gt1.sum()
        gt2 = gt2 / gt2.sum()

        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        gt = color_mask(path, idx, (gt1, gt2))
        gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path + f'/images/{idx + 1}.png', im)
        cv2.imwrite(path + f'/gt/{idx + 1}.png', gt)

        f = open(path+'/gt.txt', 'a+')
        f.write(f'{gt1[0]} {gt1[1]} {gt1[2]} {gt2[0]} {gt2[1]} {gt2[2]}\n')
        f.close()