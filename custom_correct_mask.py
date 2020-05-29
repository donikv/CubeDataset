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


def process_with_real_and_predicted(image, idx, mask, gt_mask, gt):
    mask = gaussian(mask, 3)
    image1 = iu.mask_image(image, mask)
    image2 = iu.mask_image(image, 1 - mask)
    gt2 = ru.gray_world_estimation(image1, mask) / 255
    gt1 = ru.gray_world_estimation(image2, 1 - mask) / 255
    corrected1 = ru.white_balance(image1, gt2, mask)
    corrected2 = ru.white_balance(image2, gt1, 1 - mask)
    # corrected1 = np.where(image1 == [0, 0, 0], (gt2 * 255).astype(np.uint8), corrected1)
    # corrected2 = np.where(image2 == [0, 0, 0], (gt1 * 255).astype(np.uint8), corrected2)
    corrected = iu.combine_images_with_mask(corrected1, corrected2, mask)
    # gt1 = ru.gray_world_estimation(image)
    # gt2 = gt1
    colored_mask = np.where(mask > 0.5, (gt2 * 255).astype(np.uint8), (gt1 * 255).astype(np.uint8))

    gt_mask = gt_mask / 255
    gti2 = ru.gray_world_estimation(gt, gt_mask, mask_hsv=False) / 255
    gti1 = ru.gray_world_estimation(gt, 1-gt_mask, mask_hsv=False) / 255

    def pp_angular_difference(cmask, gt):
        gt, cmask = gt.clip(1, 254), cmask.clip(1, 254)
        cmask, gt = cmask/255, gt/255
        dis = np.array([ru.angular_distance(cmask[i, j, :], gt[i, j, :]) for i in range(cmask.shape[0]) for j in range(cmask.shape[1])])
        return dis.mean()

    def pc_angular_difference(gt1, gt2, gti1, gti2):
        d1 = (ru.angular_distance(gt1, gti1) + ru.angular_distance(gt2, gti2)) / 2
        d2 = (ru.angular_distance(gt2, gti1) + ru.angular_distance(gt1, gti2)) / 2
        return np.minimum(d1, d2)
    print(f'{idx}, {pp_angular_difference(colored_mask, gt)}, {pc_angular_difference(gt1, gt2, gti1, gti2)}')
    colored_mask2 = np.where(gt_mask > 0.5, (gti2 * 255).astype(np.uint8), (gti1 * 255).astype(np.uint8))


    # path = 'images/model_corrected_custom_cube6/'
    # if not os.path.exists(path):
    #     os.mkdir(path)
    pu.visualize([image, mask, colored_mask, colored_mask2, corrected, ],
                 titles=['a)', 'b)', 'c)', 'd)', 'e)'],
                 in_line=True,
                 # out_file=f'{path}{idx}',
                 # custom_transform=lambda x: cv2.flip(x.transpose(1, 0, 2), 1),
                 # title=title
                 )
    return

    image1 = iu.mask_image(image, gt_mask)
    image2 = iu.mask_image(image, 1 - gt_mask)
    gt2 = ru.gray_world_estimation(gt, gt_mask) / 255
    gt1 = ru.gray_world_estimation(gt, 1 - gt_mask) / 255
    gt2 = ru.white_patch_estimation(gt, gt_mask) / 255
    gt1 = ru.white_patch_estimation(gt, 1-gt_mask) / 255
    # gt1 /= 2
    # gt2 /= 2
    # corrected1 = ru.white_balance(image1, gt2, gt_mask)
    # corrected2 = ru.white_balance(image2, gt1, 1 - gt_mask)
    # corrected1 = iu.color_correct_single(image1, gt2, c_ill=1/3)
    # corrected2 = iu.color_correct_single(image2, gt1, c_ill=1/3)
    # corrected1 = np.where(image1 == [0, 0, 0], (gt2 * 255).astype(np.uint8), corrected1)
    # corrected2 = np.where(image2 == [0, 0, 0], (gt1 * 255).astype(np.uint8), corrected2)
    corrected_real = iu.combine_images_with_mask(corrected1, corrected2, gt_mask)
    colored_mask2 = np.where(gt_mask > 0.5, (gt2 * 255).astype(np.uint8), (gt1 * 255).astype(np.uint8))

    # pu.visualize([image, gt, mask, gt_mask, colored_mask, colored_mask2, corrected, corrected_real],
    #              titles=['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)'],
    #              in_line=False,
    #              out_file=None,  # f'./images/model_corrected{idx}.png',
    #              # custom_transform=lambda x: cv2.flip(x.transpose(1, 0, 2), 1),
    #              # title=title
    #              )



def process_and_visualize(image, idx, gts1, gts2, mask, title=None, draw=True, use_estimation=False):
    # image = cv2.resize(image, (0, 0), fx=1 / 5, fy=1 / 5)
    # mask = cv2.resize(mask, (0, 0), fx=1 / 5, fy=1 / 5)
    mask = gaussian(mask, 3)
    if use_estimation:
        image1 = iu.mask_image(image, mask)
        image2 = iu.mask_image(image, 1-mask)
        gt2 = ru.gray_world_estimation(image1, mask) / 255
        gt1 = ru.gray_world_estimation(image2, 1-mask) / 255
        corrected1 = ru.white_balance(image1, gt2, mask)
        corrected2 = ru.white_balance(image2, gt1, 1-mask)
        # corrected1 = np.where(image1 == [0, 0, 0], (gt2 * 255).astype(np.uint8), corrected1)
        # corrected2 = np.where(image2 == [0, 0, 0], (gt1 * 255).astype(np.uint8), corrected2)
        corrected = iu.combine_images_with_mask(corrected1, corrected2, mask)
        gt = ru.gray_world_estimation(image) / 255
        corrected1 = ru.white_balance(image, gt)
        corrected2 = ru.white_balance(image, gt2)
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
        path = 'images/model_corrected_unet_cube_comb/'
        if not os.path.exists(path):
            os.mkdir(path)
        pu.visualize([image, mask, colored_mask, corrected1, corrected],
                     titles=['a)', 'b)', 'c)', 'd)', 'e)'],
                     in_line=True,
                     out_file=f'{path}{idx}',
                     # custom_transform=lambda x: cv2.flip(x.transpose(1, 0, 2), 1),
                     # title=title
                     )
    return corrected


def main_process2(data):
    use_corrected_masks = True
    # image_path = '../MultiIlluminant-Utils/data/test/whatsapp/images'
    # mask_path = '../MultiIlluminant-Utils/data/test/whatsapp/pmasks'
    image_path = '../MultiIlluminant-Utils/data/dataset_crf/realworld/srgb8bit'
    mask_path = '../MultiIlluminant-Utils/data/dataset_crf/realworld/pmasks6' #if use_corrected_masks else './data/custom_mask_nocor'
    gt_mask_path = '../MultiIlluminant-Utils/data/dataset_crf/realworld/gt_mask'  # if use_corrected_masks else './data/custom_mask_nocor'
    gt_path = '../MultiIlluminant-Utils/data/dataset_crf/realworld/groundtruth'
    ext = '.png' if use_corrected_masks else '.jpg'
    img, gt1, gt2 = data

    image = fu.load_png(img, path=image_path, directory='', mask_cube=False)
    mask = fu.load_png(img, path=mask_path, directory='', mask_cube=False)
    gt_mask = fu.load_png(img, path=gt_mask_path, directory='', mask_cube=False)
    gt = fu.load_png(img, path=gt_path, directory='', mask_cube=False)
    process_with_real_and_predicted(image, img, mask, gt_mask, gt)


def main_process(data):
    use_corrected_masks = True
    image_path = '../MultiIlluminant-Utils/data/test/whatsapp/images'
    mask_path = '../MultiIlluminant-Utils/data/test/whatsapp/pmasks'
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
    # image_path = '../MultiIlluminant-Utils/data/test/whatsapp/images'
    image_path = '../MultiIlluminant-Utils/data/dataset_crf/realworld/srgb8bit'
    mask_path = './data/custom_mask'
    image_names = os.listdir(image_path)
    images = range(1, len(image_names) + 1)
    # images = [23]
    images = list(map(lambda x: (x, gt1, gt2), image_names))

    num_proc = 1

    if num_proc < 2:
        for data in images:
            main_process2(data)
    else:
        with mp.Pool(8) as pool:
            pool.map(main_process, images)
    exit(0)
