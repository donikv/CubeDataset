import cv2
import os

from utlis import file_utils as fu
import utlis.image_utils as iu
import utlis.plotting_utils as pu
import utlis.relighting_utils as ru
from utlis.groundtruth_utils import GroundtruthLoader
import numpy as np
import multiprocessing as mp


def process_and_visualize(image, idx, gts1, gts2, mask, title=None, draw=True, ):
    # rgb = img
    gt1 = gts1[idx - 1]

    gt2 = gts2[idx - 1]

    corrected = iu.color_correct(image, mask, gt2, gt1)

    if draw:
        pu.visualize([image, corrected, mask], title=title)
    return corrected


def main_process(data):
    image_path = './data/custom'
    mask_path = './data/custom_mask'
    # images = [17, ]
    img, gt1, gt2 = data

    image = fu.load_png(str(img) + '.jpg', path=image_path, directory='', mask_cube=False)
    mask = fu.load_png(str(img) + '.png', path=mask_path, directory='', mask_cube=False)
    cor = process_and_visualize(image, img, gt1, gt2, mask, title=img)


if __name__ == '__main__':
    gt1 = GroundtruthLoader('custom_gt.txt', path='./data')
    gt2 = GroundtruthLoader('custom_gt2.txt', path='./data')
    image_path = './data/custom'
    mask_path = './data/custom_mask'
    image_names = os.listdir(image_path)
    images = range(1, len(image_names) + 1)
    images = list(map(lambda x: (x, gt1, gt2), images))

    num_proc = 1

    if num_proc < 2:
        for data in images:
            main_process(data)
    else:
        with mp.Pool(8) as pool:
            pool.map(main_process, images)
    exit(0)
