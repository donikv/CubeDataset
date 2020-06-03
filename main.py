import pickle

import rawpy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.file_utils import load_image, load_png, load
from utils.image_utils import color_correct, cv2_contours, color_correct_single, process_image, \
    adjust_gamma
from utils.plotting_utils import visualize, plot_counturs
from utils.groundtruth_utils import GroundtruthLoader
from utils.relighting_utils import random_colors, angular_distance

gtLoader = GroundtruthLoader('cube+_gt.txt')
gts = gtLoader.gt


def get_ill_diffs():
    gtLoaderL = GroundtruthLoader('cube+_left_gt.txt')
    gtLoaderR = GroundtruthLoader('cube+_right_gt.txt')

    gtsl = gtLoaderL.gt
    gtsr = gtLoaderR.gt
    gt_diff = np.array(list(map(lambda x: angular_distance(x[0], x[1]), zip(gtsl, gtsr))))
    gt_diff_filter = np.array(list(
        map(lambda x: x[0] + 1,
            filter(lambda x: x[1] < 3,
                   enumerate(gt_diff)
                   )
            )
    ))
    return gt_diff_filter

use_raw = False
folder_step = 200 if not use_raw else 100

def get_diff_in_ill(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    image = image[:, :, 0]
    diff = image.max() - image.min()
    return diff


def process_and_visualize(image, idx, title=None, process_raw=True, draw=True, method=1, invert=False):
    height, width, _ = image.shape
    image = cv2.resize(image, (int(width / 10), int(height / 10)))
    original = image.copy()
    if process_raw:
        image = process_image(image)
    else:
        image = adjust_gamma(image, gamma=1)
    # rgb = img
    gt = gts[idx - 1]
    corrected = color_correct_single(image, gt, 1)
    foreground, mask, ident = cv2_contours(corrected, upper=np.array([160, 255, 255]), method=method, invert=invert)

    ill1, ill2 = random_colors()
    relighted = color_correct(image, mask=mask, ill1=1 / ill1, ill2=1 / ill2,
                              c_ill=1)
    colored_mask = np.array(
        [[ill1 * pixel + ill2 * (1 - pixel) for pixel in row] for row in mask])
    if draw:
        visualize([image, corrected, relighted, colored_mask], title=title)
    return relighted, colored_mask, ill1, ill2

save = False
if __name__ == '__main__':
    diffs = get_ill_diffs()

    i = 0
    for img in diffs:
        image = load(img, folder_step, depth=14)
        for j in range(2):
            relighted, mask, ill1, ill2 = process_and_visualize(image, img, process_raw= not use_raw, title=img, draw=True, method=1, invert=bool(j%2))
            if save:
                inv = ''
                if j%2 == 1:
                    inv = '-inv'
                cv2.imwrite(f'./data/relighted/images/{img}{inv}-gray.png', cv2.cvtColor(relighted, cv2.COLOR_RGB2BGR))
                cv2.imwrite(f'./data/relighted/gt/{img}{inv}-gray.png', cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


