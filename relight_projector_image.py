import cv2
import numpy as np
import os

import utils.image_utils as iu
import utils.file_utils as fu
import utils.relighting_utils as ru
import utils.projector_utils as pu
import utils.plotting_utils as plt

PNG = 'png'
TIFF = 'tiff'
NEF = 'nef'


def load_and_correct(path, idx, type):
    def make_white(im, t=10):
        im = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)
        im[:,:,1] = np.where(im[:,:,1] > t , 1, 0)
        im = cv2.cvtColor(im, cv2.COLOR_HLS2RGB)
        return im

    name = str(idx+1)
    if type == NEF:
        iml = fu.load_cr2(name+'.NEF', path, directory='ambient', mask_cube=False)
        imr = fu.load_cr2(name+'.NEF', path, directory='direct', mask_cube=False)
    elif type == TIFF:
        iml = load_tiff(name+'.tiff', path, directory='left')
        imr = load_tiff(name+'.tiff', path, directory='right')
    else:
        iml = fu.load_png(name+'.png', path, directory='ambient/debayered', mask_cube=False)
        imr = fu.load_png(name+'.png', path, directory='direct/debayered', mask_cube=False)

    iml = iu.process_image(iml, depth=14, blacklevel=2048, scale=True)
    imr = iu.process_image(imr, depth=14, blacklevel=2048, scale=True)

    x1, y1, x2, y2 = np.loadtxt(path+'/pos.txt').astype(int)[idx]
    gt1 = iml[y1-10:y1+10, x1-10:x1+10].mean(axis=1).mean(axis=0)
    gt1 = np.clip(gt1, 0.001, 1)
    gt2 = imr[y2-10:y2+10, x2-10:x2+10].mean(axis=1).mean(axis=0)
    gt2 = np.clip(gt2, 0.001, 1)

    iml = iu.color_correct_single_f32(np.clip(iml, 0, 255*255), gt1, 1)
    imr = iu.color_correct_single_f32(np.clip(imr, 0, 255*255), gt2, 1)

    return iml, imr


def combine(imlc, imrc, colors):
    def blackout_shadows(im, t=15):
        im = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)
        im[:,:,1] = np.where(im[:,:,1] < t , 0, im[:,:,1])
        im = cv2.cvtColor(im, cv2.COLOR_HLS2RGB)
        return im

    t= 2 / (2**14)
    imlc = blackout_shadows(imlc, t)
    imrc = blackout_shadows(imrc, t)

    imlc = iu.color_correct_single_f32(imlc, 1/colors[0], 1, relight=True)
    imrc = iu.color_correct_single_f32(imrc, 1/colors[1], 1, relight=True)

    imc = pu.combine_two_images(imrc, imlc)
    return imc, imlc, imrc


def load_tiff(img, path, directory):
    image_tiff = cv2.imread(f'{path}/{directory}/{img}', cv2.IMREAD_UNCHANGED)
    imageRaw = cv2.cvtColor(image_tiff, cv2.COLOR_BAYER_RG2BGR)
    return imageRaw


if __name__ == '__main__':
    path = 'G:/fax/diplomski/Datasets/third/realworld_tiff'
    for idx in range(0,10):
        iml, imr = load_and_correct(path, idx, type=PNG)
        c = ru.random_colors(desaturate=False)
        im, imlc, imrc = combine(iml, imr, c)
        gt = pu.create_gt_mask(im * 255, imr * 255, iml * 255, c[1], c[0])[0]
        gt = iu.blackout(gt, im)

        plt.visualize([iml, imr, imlc, imrc, im, gt])

        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
        im = (im * 2**16).astype(np.uint16)
        cv2.imwrite(path + f'/relighted/images/{idx + 1}.png', im)
        cv2.imwrite(path + f'/relighted/gt/{idx + 1}.png', gt)

        f = open(path+'/gt.txt', 'a+')
        f.write(f'{c[0][0]} {c[0][1]} {c[0][2]} {c[1][0]} {c[1][1]} {c[1][2]}\n')
        f.close()