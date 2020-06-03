import cv2
import os

from skimage.filters import gaussian

from utils import file_utils as fu
import utils.image_utils as iu
import utils.plotting_utils as pu
import utils.relighting_utils as ru
from utils.groundtruth_utils import GroundtruthLoader
import numpy as np
import multiprocessing as mp


def process_and_visualize(image, masks, gts, idx, title=None, draw=True,):
    ttls = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)']
    gt1, gt2 = gts[0]
    image = cv2.resize(image, (0, 0), fx=1 / 5, fy=1 / 5)
    corrected2 = iu.color_correct_single(image, gt1, c_ill=1 / np.sqrt(3))

    out_imgs = [image, corrected2]
    for i, mask in enumerate(masks):
        gt1, gt2 = gts[i]
        # rgb = img

        mask = gaussian(mask, 5)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x/255))

        mask = cv2.resize(mask, (0, 0), fx=1 / 5, fy=1 / 5)
        corrected = iu.color_correct(image, mask, gt1, gt2, c_ill=1/np.sqrt(3))
        out_imgs.append(mask)
        out_imgs.append(corrected)


    if draw:
        pu.visualize(out_imgs, titles=list(map(lambda x: ttls[x], range(len(out_imgs)))), in_line=False, out_file=f'./images/res{idx}-{i}.png')
    return corrected


def main_process(data):
    img, gt1, gt2 = data
    image_path = './data/custom'
    masks_path = f'./data/masks/{img}'
    masks = os.listdir(masks_path)
    gt1 = [0.83529411765, 0.67450980392, 0.36470588235]
    gt2 = [0.58431372549, 0.65882352941, 0.64705882353]
    gts = [(gt1, gt2), (gt2, gt1), (gt2, gt1)]
    # gts = [(gt2[img - 1], gt1[img - 1]),(gt2[img - 1], gt1[img - 1]),(gt2[img - 1], gt1[img - 1])]

    image = fu.load_png(str(img) + '.jpg', path=image_path, directory='', mask_cube=False)
    mask = list(map(lambda x: fu.load_png(x, path=masks_path, directory='', mask_cube=False), masks))
    cor = process_and_visualize(image, mask, gts, img)


if __name__ == '__main__':
    gt1 = GroundtruthLoader('custom_gt.txt', path='./data')
    gt2 = GroundtruthLoader('custom_gt2.txt', path='./data')
    image_path = './data/custom'
    mask_path = './data/custom_mask'
    image_names = os.listdir(image_path)
    images = range(1, len(image_names) + 1)
    images = [23]
    images = list(map(lambda x: (x, gt1, gt2), images))

    num_proc = 1

    if num_proc < 2:
        for data in images:
            main_process(data)
    else:
        with mp.Pool(8) as pool:
            pool.map(main_process, images)
    exit(0)
