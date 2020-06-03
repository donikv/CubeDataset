import cv2
import os

from utils import file_utils as fu
import utils.image_utils as iu
import utils.plotting_utils as pu
import utils.relighting_utils as ru
from utils.groundtruth_utils import GroundtruthLoader
import numpy as np
import multiprocessing as mp


def process_and_visualize(image, idx, gts, title=None, draw=True):
    # rgb = img
    gt = gts[idx - 1]
    gt = gt / np.linalg.norm(gt)
    corrected = iu.color_correct_single(image, gt, 1/np.sqrt(3))

    if draw:
        pu.visualize([image, corrected, ], title=title)
    return corrected


if __name__ == '__main__':
    gt = GroundtruthLoader('custom_gt.txt', path='./data')
    image_path = './data/custom'
    image_names = os.listdir(image_path)
    images = range(1, len(image_names) + 1)
    images = [23]

    for img in images:
        image = fu.load_png(str(img) + '.jpg', path=image_path, directory='', mask_cube=False)
        cor = process_and_visualize(image, img, gt, title=img)
        cv2.imwrite(f'./data/custom_relighted/{str(img)}.png', cv2.cvtColor(cor, cv2.COLOR_RGB2BGR))
