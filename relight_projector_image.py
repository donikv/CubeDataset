import cv2
import numpy as np
import os

import utils.image_utils as iu
import utils.file_utils as fu
import utils.relighting_utils as ru
import utils.projector_utils as pu
import utils.plotting_utils as plt


def load_and_correct(path, idx, tiff):
    def make_white(im, t=10):
        im = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)
        im[:,:,1] = np.where(im[:,:,1] > t , 1, 0)
        im = cv2.cvtColor(im, cv2.COLOR_HLS2RGB)
        return im

    name = str(idx+1)
    if not tiff:
        iml = fu.load_cr2(name+'.NEF', path, directory='left', mask_cube=False)
        imr = fu.load_cr2(name+'.NEF', path, directory='right', mask_cube=False)
    else:
        iml = load_tiff(name+'.tiff', path, directory='left')
        imr = load_tiff(name+'.tiff', path, directory='right')

    x1, y1 = np.loadtxt(path+'/pos.txt').astype(int)[idx]
    gt = iml[y1-10:y1+10, x1-10:x1+10].mean(axis=1).mean(axis=0)
    # gt = np.array([19,159,11])
    gt = np.clip(gt, 0, 255 * 255) / 255 / 255

    iml = iu.color_correct_single_16(np.clip(iml, 0, 255*255), gt, 2)
    imr = iu.color_correct_single_16(np.clip(imr, 0, 255*255), gt, 2)

    iml = iu.process_image(iml, depth=16)
    imr = iu.process_image(imr, depth=16)

    return iml, imr


def combine(imlc, imrc, colors):
    def blackout_shadows(im, t=2):
        im = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)
        im[:,:,1] = np.where(im[:,:,1] < t , 0, im[:,:,1])
        im = cv2.cvtColor(im, cv2.COLOR_HLS2RGB)
        return im

    t=2
    imlc = blackout_shadows(imlc, t)
    imrc = blackout_shadows(imrc, t)

    imlc = iu.color_correct_single(imlc, 1/colors[0], 1, relight=True)
    imrc = iu.color_correct_single(imrc, 1/colors[1], 1, relight=True)

    imc = imlc + imrc
    return imc, imlc, imrc


def load_tiff(img, path, directory):
    image_tiff = cv2.imread(f'{path}/{directory}/{img}', cv2.IMREAD_UNCHANGED)
    imageRaw = cv2.cvtColor(image_tiff, cv2.COLOR_BAYER_RG2BGR)
    return imageRaw


if __name__ == '__main__':
    path = 'G:/fax/diplomski/Datasets/projector_relighted'
    idx = 1
    iml, imr = load_and_correct(path, idx, tiff=True)
    c = ru.random_colors(desaturate=False)
    im, imlc, imrc = combine(iml, imr, c)
    gt = pu.create_gt_mask(im, imr, iml, c[1], c[0])[0]
    plt.visualize([iml, imr, imlc, imrc, im, gt])

    cv2.imwrite(path + f'/images/{idx + 1}.png', im)
    cv2.imwrite(path + f'/gt/{idx + 1}.png', gt)

    f = open(path+'/gt.txt', 'a+')
    f.write(f'{c[0][0]} {c[0][1]} {c[0][2]} {c[1][0]} {c[1][1]} {c[1][2]}\n')
    f.close()