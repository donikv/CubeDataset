import cv2
import numpy as np
import os
from itertools import combinations

import utils.image_utils as iu
import utils.file_utils as fu
import utils.relighting_utils as ru
import utils.projector_utils as pu
import utils.plotting_utils as plt


def debayer(rggb):
    red = rggb[:-1:2, :-1:2]
    green = (rggb[:-1:2, 1::2] + rggb[1::2, 0:-1:2]) / 2.0
    blue = rggb[1::2, 1::2]
    img = np.zeros((len(red), len(red[0]), 3))
    img[:, :, 0] = red
    img[:, :, 1] = green
    img[:, :, 2] = blue

    return img


def combine(im1, im2, gt1, gt2, gto=None, offset=0):
    a = np.random.uniform(-1.5, 1.5, 1)
    b = np.random.uniform(0, 2 * im1.shape[1] / 3, 1) + offset

    imi = np.indices(im1.shape[:-1])
    imc, imr = imi[0], imi[1]

    mask = np.expand_dims(a * imc - imr + b > 0, axis=2)
    imc = np.where(mask, im1, im2)
    if gto is None:
        gt = (np.where(mask, gt1, gt2) * 255).astype(np.uint8)
    else:
        gt = (np.where(mask, gto, gt2 * 255)).astype(np.uint8)

    return imc, gt

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
    gt2 = im[y2-10:y2+10, x2-10:x2+10].mean(axis=1).mean(axis=0)
    gt2 = np.clip(gt2, 0.001, 1)

    # im[y1 - 100:y1 + 100, x1 - 100:x1 + 100] = np.zeros((200,200,3))
    # im[y2 - 100:y2 + 100, x2 - 100:x2 + 100] = np.zeros((200,200,3))

    return im, gt1, gt2


def correct(path, idx, gts=None):
    name = str(idx+1)
    image = fu.load_png(name+'.png', path, 'images', mask_cube=False) / 2**16
    if gts is None:
        gts = np.loadtxt(path + '/gt.txt')[idx].reshape(2,3)

    ci = iu.color_correct_single_f32(image, gts[1] + gts[0], c_ill=1/3)
    return ci


def load_tiff(img, path, directory):
    image_tiff = cv2.imread(f'{path}/{directory}/{img}', cv2.IMREAD_UNCHANGED)
    imageRaw = cv2.cvtColor(image_tiff, cv2.COLOR_BAYER_BG2RGB)
    return imageRaw


if __name__ == '__main__':
    path = '../Datasets/collage'
    idx = 0

    gts = np.loadtxt(path + '/gt.txt').reshape(-1, 2, 3)
    for i, j, k in combinations(range(0,4), 3):
        name1 = str(i + 1)
        im1 = load_tiff(name1 + '.tiff', path, directory='')
        name2 = str(j + 1)
        im2 = load_tiff(name2 + '.tiff', path, directory='')
        name3 = str(k + 1)
        im3 = load_tiff(name3 + '.tiff', path, directory='')
        gt1 = gts[i][0]
        gt2 = gts[j][0]
        gt3 = gts[k][0]

        imc, gt = combine(im1, im2, gt1, gt2)
        imc, gt = combine(imc, im3, gt1, gt3, gt, offset=100)

        imc = cv2.cvtColor(imc, cv2.COLOR_RGB2BGR)
        gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
        if not os.path.exists(path + f'/combined_gt/{name1}-{name2}-{name3}.png'):
            cv2.imwrite(path + f'/combined/{name1}-{name2}-{name3}.png', imc)
            cv2.imwrite(path + f'/combined_gt/{name1}-{name2}-{name3}.png', gt)
        # gt1 = gts[i][0]
        # gt2 = gts[j][0]
        # print(f'ang({i}, {j}) = {ru.angular_distance(gt1, gt2)}')
    exit(0)
    # for idx in range(0,1):
    #     im, gt1, gt2 = load_and_get_gt(path, idx, tiff=True)
    #     gt1 = gt1 / gt1.sum()
    #     gt2 = gt2 / gt2.sum()
    #
    #     im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(path + f'/images/{idx + 1}.png', (im * 65535).astype(np.uint16))
    #
    #     f = open(path+'/gt.txt', 'a+')
    #     f.write(f'{gt1[0]} {gt1[1]} {gt1[2]} {gt2[0]} {gt2[1]} {gt2[2]}\n')
    #     f.close()
    #
    #     ic = correct(path, idx)
    #     ic = cv2.cvtColor(ic, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(path + f'/corrected/{idx + 1}.png', (ic * 2**16).astype(np.uint16))