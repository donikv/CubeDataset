import cv2
import os

from skimage.filters import gaussian

from utlis import file_utils as fu
import utlis.image_utils as iu
import utlis.plotting_utils as pu
import utlis.relighting_utils as ru
from utlis.groundtruth_utils import GroundtruthLoader
import numpy as np
import multiprocessing as mp


def process_and_visualize(image, idx, gts1, gts2, mask, title=None, draw=True, use_estimation=False):
    image = cv2.resize(image, (0, 0), fx=1 / 5, fy=1 / 5)
    mask = cv2.resize(mask, (0, 0), fx=1 / 5, fy=1 / 5)
    mask = gaussian(mask, 5)
    if use_estimation:
        image1 = iu.mask_image(image, mask)
        image2 = iu.mask_image(image, 1-mask)
        gt2 = ru.gray_world_estimation(image1) / 255
        gt1 = ru.gray_world_estimation(image2) / 255
        # corrected1 = ru.white_balance(image1, gt1)
        # corrected2 = ru.white_balance(image2, gt2)
        # corrected = iu.combine_images_with_mask(corrected1, corrected2, mask)
    else:
        gt1 = gts1[idx - 1]

        gt2 = gts2[idx - 1]



        def sigmoid(x):
            return 1 / (1 + np.exp(-x/255))

        # sig_mask = sigmoid(mask)

        # gt = gt2 / gt1
    corrected = iu.color_correct(image, mask, gt2, gt1, c_ill=1/3)
    corrected2 = iu.color_correct_single(image, gt2, c_ill=1/3)
    # corrected = iu.adjust_gamma(corrected, 0.9)
    # corrected_sig = iu.color_correct(image, sig_mask, gt, np.ones(3), c_ill=1)

    if draw:
        pu.visualize([image, mask, corrected, corrected2],
                     titles=['a)', 'b)', 'c)', 'd)'],
                     in_line=True,
                     out_file=None, #f'./images/model_corrected{idx}.png',
                     custom_transform=lambda x: cv2.flip(x.transpose(1, 0, 2), 1),
                     title=title
                     )
    return corrected


def main_process(data):
    use_corrected_masks = False
    image_path = './data/custom'
    mask_path = './data/custom_mask' if use_corrected_masks else './data/custom_mask_nocor'
    ext = '.png' if use_corrected_masks else '.jpg'
    img, gt1, gt2 = data

    image = fu.load_png(str(img) + '.jpg', path=image_path, directory='', mask_cube=False)
    mask = fu.load_png(str(img) + ext, path=mask_path, directory='', mask_cube=False)
    cor = process_and_visualize(image, img, gt1, gt2, mask, title=img, use_estimation=True)


if __name__ == '__main__':
    gt1 = GroundtruthLoader('custom_gt.txt', path='./data')
    gt2 = GroundtruthLoader('custom_gt2.txt', path='./data')
    image_path = './data/custom'
    mask_path = './data/custom_mask'
    image_names = os.listdir(image_path)
    images = range(1, len(image_names) + 1)
    # images = [17]
    images = list(map(lambda x: (x, gt1, gt2), images))

    num_proc = 1

    if num_proc < 2:
        for data in images:
            main_process(data)
    else:
        with mp.Pool(8) as pool:
            pool.map(main_process, images)
    exit(0)
