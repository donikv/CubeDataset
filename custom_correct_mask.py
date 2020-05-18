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
    # image = cv2.resize(image, (0, 0), fx=1 / 5, fy=1 / 5)
    # mask = cv2.resize(mask, (0, 0), fx=1 / 5, fy=1 / 5)
    mask = gaussian(mask, 5)
    if use_estimation:
        image1 = iu.mask_image(image, mask)
        image2 = iu.mask_image(image, 1-mask)
        gt2 = ru.gray_edge_estimation(image1, mask) / 255
        gt1 = ru.gray_edge_estimation(image2, 1-mask) / 255
        # gt2 = ru.white_patch_estimation(image1, mask) / 255
        # gt1 = ru.white_patch_estimation(image2, 1-mask) / 255
        corrected1 = ru.white_balance(image1, gt2, mask)
        corrected2 = ru.white_balance(image2, gt1, 1-mask)
        corrected1 = np.where(image1 == [0, 0, 0], (gt2 * 255).astype(np.uint8), corrected1)
        corrected2 = np.where(image2 == [0, 0, 0], (gt1 * 255).astype(np.uint8), corrected2)
        corrected = iu.combine_images_with_mask(corrected1, corrected2, mask)
        # corrected1 = ru.white_balance(image, gt1)
        # corrected2 = ru.white_balance(image, gt2)
    else:
        gt1 = gts1[idx - 1]

        gt2 = gts2[idx - 1]



        def sigmoid(x):
            return 1 / (1 + np.exp(-x/255))

        # sig_mask = sigmoid(mask)

        # gt = gt2 / gt1
        corrected = iu.color_correct(image, mask, gt2, gt1, c_ill=1/3)
        corrected1 = iu.color_correct_single(image, gt2, c_ill=1/3)
        corrected2 = iu.color_correct_single(image, gt1, c_ill=1/3)
    # corrected = iu.adjust_gamma(corrected, 0.9)
    # corrected_sig = iu.color_correct(image, sig_mask, gt, np.ones(3), c_ill=1)

    if draw:
        colored_mask = np.where(mask > 0.5, (gt2 * 255).astype(np.uint8), (gt1 * 255).astype(np.uint8))
        pu.visualize([mask, image, corrected, corrected2, corrected1, colored_mask],
                     titles=['a)', 'b)', 'c)', 'd)', 'e)', 'f)'],
                     in_line=False,
                     out_file=None, #f'./images/model_corrected{idx}.png',
                     # custom_transform=lambda x: cv2.flip(x.transpose(1, 0, 2), 1),
                     # title=title
                     )
    return corrected


def main_process(data):
    use_corrected_masks = True
    image_path = '../MultiIlluminant-Utils/data/test/whatsapp/img_corrected_1'
    mask_path = '../MultiIlluminant-Utils/data/test/whatsapp/pmasks' #if use_corrected_masks else './data/custom_mask_nocor'
    ext = '.png' if use_corrected_masks else '.jpg'
    img, gt1, gt2 = data

    image = fu.load_png(img, path=image_path, directory='', mask_cube=False)
    mask = fu.load_png(img, path=mask_path, directory='', mask_cube=False)
    cor = process_and_visualize(image, img, gt1, gt2, mask, title=img, use_estimation=True)


if __name__ == '__main__':
    try:
        gt1 = GroundtruthLoader('custom_gt.txt', path='./data')
        gt2 = GroundtruthLoader('custom_gt2.txt', path='./data')
    except OSError:
        gt1 = None
        gt2 = None
    image_path = '../MultiIlluminant-Utils/data/test/whatsapp/img_corrected_1'
    mask_path = './data/custom_mask'
    image_names = os.listdir(image_path)
    images = range(1, len(image_names) + 1)
    # images = [23]
    images = list(map(lambda x: (x, gt1, gt2), image_names))

    num_proc = 1

    if num_proc < 2:
        for data in images:
            main_process(data)
    else:
        with mp.Pool(8) as pool:
            pool.map(main_process, images)
    exit(0)
