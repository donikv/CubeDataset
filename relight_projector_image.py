import cv2
import numpy as np
import os

import utils.image_utils as iu
import utils.file_utils as fu
import utils.relighting_utils as ru
import utils.projector_utils as pu
import utils.plotting_utils as plt

PNG_RW = 'png'
PNG_LAB = 'png_lab'
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
    elif type == PNG_LAB:
        iml = fu.load_png(name+'.png', path, directory='left/images', mask_cube=False)
        imr = fu.load_png(name+'.png', path, directory='right/images', mask_cube=False)
    else:
        iml = fu.load_png(name+'.png', path, directory='ambient/debayered', mask_cube=False)
        imr = fu.load_png(name+'.png', path, directory='direct/debayered', mask_cube=False)


    if type == PNG_LAB:
        iml = iml / 2 ** 16
        imr = imr / 2 ** 16
    else:
        iml = iu.process_image(iml, depth=14, blacklevel=0, scale=True)
        imr = iu.process_image(imr, depth=14, blacklevel=0, scale=True)

    x1, y1, x2, y2 = np.loadtxt(path+'/cube.txt').astype(int)[idx]
    gt1 = np.loadtxt(path + '/gt_right.txt')[idx]
    gt2 = np.loadtxt(path + '/gt_left.txt')[idx]
    # gt1 = imr[y1-10:y1+10, x1-10:x1+10].mean(axis=1).mean(axis=0)
    gt1 = np.clip(gt1, 0.001, 1)
    gt1 /= gt1.max()
    # gt2 = iml[y2-10:y2+10, x2-10:x2+10].mean(axis=1).mean(axis=0)
    gt2 = np.clip(gt2, 0.001, 1)
    gt2 /= gt2.max()
    gt = np.concatenate([gt1, gt2], 0)

    imrc = iu.color_correct_single_f32(np.clip(imr, 0, 255*255), gt1, 1/3)
    imlc = iu.color_correct_single_f32(np.clip(iml, 0, 255*255), gt2, 1/3)

    return imlc, imrc, iml, imr, gt


def combine(imlc, imrc, colors=None):
    def blackout_shadows(im, t=15):
        im = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)
        im[:,:,1] = np.where(im[:,:,1] < t , 0, im[:,:,1])
        im = cv2.cvtColor(im, cv2.COLOR_HLS2RGB)
        return im

    t= 2 / (2**14)
    imlc = blackout_shadows(imlc, t)
    imrc = blackout_shadows(imrc, t)

    if colors is not None:
        imlc = iu.color_correct_single_f32(imlc, 1/colors[0], 1/1.713, relight=True)
        imrc = iu.color_correct_single_f32(imrc, 1/colors[1], 1/1.713, relight=True)

    imc = pu.combine_two_images(imrc, imlc)
    return imc, imlc, imrc


def load_tiff(img, path, directory):
    image_tiff = cv2.imread(f'{path}/{directory}/{img}', cv2.IMREAD_UNCHANGED)
    imageRaw = cv2.cvtColor(image_tiff, cv2.COLOR_BAYER_RG2BGR)
    return imageRaw


if __name__ == '__main__':
    path = 'G:\\fax\\diplomski\\Datasets\\third\\ambient5_tiff'
    images = os.listdir(path + '/both/images')
    # current_relighted_images = os.listdir(path + '/relighted/images/')
    # current_relighted_idxs = list(map(lambda x: int(x[:-4]), current_relighted_images))
    # last_idx = sorted(current_relighted_idxs)[-1] if len(current_relighted_idxs) > 0 else 0
    gts = []
    for img_idx, image in enumerate(images):
        a = image.rfind(".")
        idx = int(image[:a]) - 1

        iml, imr, imlnc, imrnc, c = load_and_correct(path, idx, type=PNG_LAB)
        gts.append(c)
        # c = ru.random_colors(desaturate=False)
        im, imlc, imrc = combine(iml, imr, None)
        imc1 = pu.combine_two_images(imr, imlnc.astype(np.float32) * 1.713)
        gt, _, _, r = pu.create_gt_mask(im * 255, imr * 255, iml * 255, c[1], c[0])
        gt = iu.blackout(gt, im)

        plt.visualize([imr, imlnc, imc1, np.round(r)])

        # im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        # imc1 = cv2.cvtColor(imc1, cv2.COLOR_RGB2BGR)
        # gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
        # im = (im * 2**16).astype(np.uint16)
        imc1 = (imc1 * 2 ** 16).astype(np.uint16)
        r = (r * 255).astype(np.uint8)

        img_idx = img_idx
        # cv2.imwrite(path + f'/relighted/images/{img_idx + 1}.png', im)
        # cv2.imwrite(path + f'/relighted/gt/{img_idx + 1}.png', gt)
        cv2.imwrite(path + f'/gt_mask/{idx + 1}.png', r)
        cv2.imwrite(path + f'/img_corrected_1/{idx + 1}.png', imc1)

        # f = open(path+'/gt.txt', 'a+')
        # f.write(f'{c[0][0]} {c[0][1]} {c[0][2]} {c[1][0]} {c[1][1]} {c[1][2]}\n')
        # f.close()
    np.savetxt(path+'/gt.txt', np.array(gts))