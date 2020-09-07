import cv2
import os

from utils import file_utils as fu
import utils.image_utils as iu
import utils.plotting_utils as pu
import utils.relighting_utils as ru
from utils.groundtruth_utils import GroundtruthLoader
import numpy as np
import multiprocessing as mp

def get_ill_diffs():
    """
    Returns only images from the cube+ dataset that have low angular distance between two illuminants.

    :return: Image indices with small angular distance.
    """
    gtLoaderL = GroundtruthLoader('cube+_left_gt.txt')
    gtLoaderR = GroundtruthLoader('cube+_right_gt.txt')

    gtsl = gtLoaderL.gt
    gtsr = gtLoaderR.gt
    gt_diff = np.array(list(map(lambda x: ru.angular_distance(x[0], x[1]), zip(gtsl, gtsr))))
    gt_diff_filter = np.array(list(
        map(lambda x: x[0] + 1,
            filter(lambda x: x[1] < 3,
                   enumerate(gt_diff)
                   )
            )
    ))
    return gt_diff_filter


def process_with_edges(image, gt, use_edges, use_grad, desaturate, planckian, single):
    """
    Synthetically relights images using a given segmentation method.

    :param img: Image to be processed.
    :param gt: Illumination of the input image.
    :param use_edges: Indicates whether edge detection is used in contour detection.
    :param use_grad: Indicates whether a gradient fill will be used on contours.
    :param desaturate: Indicates whether the input colors should be desaturated.
    :param planckian: Indicates whether colors should be sampled from the planckian locus or a random sample
    :param single: Indicates whether the whole image should be relighted or only the insides of the found contours
    :return: (Success indicator, corrected image, colored mask, relighted image, partially relighted image, contour mask, first illuminant, second illuminant)
    """


    image_cor = iu.color_correct_single(image, c_ill=1, u_ill=gt)
    if use_edges:
        edges, closing = iu.find_edges(image_cor, 10, 10)
        contours, mask, identation_index = iu.cv2_contours(closing, method=-1, upper=np.array([10, 255, 255]), use_grad=use_grad)
    else:
        contours, mask, identation_index = iu.cv2_contours(image_cor, method=1, upper=np.array([10, 255, 255]), invert=True, use_grad=use_grad)

    if identation_index < 10:
        print(identation_index)
        return False, image, None, None, None, mask

    ill1, ill2 = ru.random_colors(desaturate=desaturate, planckian=planckian)
    if single:
        if np.random.uniform(0,1,1) > 0.5:
            ill2 = np.ones(3)
        else:
            ill1 = np.ones(3)
    relighted = iu.color_correct(image_cor, mask=mask, ill1=1 / ill1, ill2=1 / ill2,
                                 c_ill=1/3 if desaturate else 1/5)
    p = 0
    i1, i2 = (ill2/ill1, np.ones(3)) if p > 0.5 else (np.ones(3), ill1/ill2)
    relighted1 = iu.color_correct(image_cor, mask=mask, ill1=i1, ill2=i2,
                                 c_ill=1/3)
    colored_mask = np.array(
        [[ill1 * pixel + ill2 * (1 - pixel) for pixel in row] for row in mask])

    return True, image_cor, colored_mask, relighted, relighted1, mask, ill1, ill2


def main_process(data):
    img = data
    folder_step = 200
    draw = False
    save = True
    use_edges = True
    use_grad = False
    desaturate = True
    planckian = True
    single=False
    image = "/media/donik/Disk/intel_tau" + img + '.tiff'
    gt = "/media/donik/Disk/intel_tau" + img + '.wp'
    image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    height, width, _ = image.shape
    # image = cv2.resize(image, (int(width / 5), int(height / 5)))
    image = iu.process_image(image, 14, blacklevel=0)
    gt = np.loadtxt(gt, delimiter=',')
    # print(image)
    # print(gt)
    # print(image.shape)
    # exit(0)

    succ, image, colored_mask, relighted, relighted1, mask, i1, i2 = process_with_edges(image, gt, use_edges, use_grad, desaturate, planckian, single)
    if succ:
        if draw:
            pu.visualize([image, relighted, colored_mask, mask], title=img)
        if save:
            name = f'{img}-9{"-sing" if single else ""}{"-rand" if not planckian else ""}{"-sat" if not desaturate else ""}{"-grad" if use_grad else ""}{"-edg" if use_edges else ""}.png'
            cv2.imwrite(f'./data/relighted/images/{name}', cv2.cvtColor(relighted, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f'./data/relighted/gt/{name}', cv2.cvtColor((colored_mask * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(f'./data/relighted/gt_mask/{name}',
                        cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(f'./data/relighted/img_corrected_1/{name}',
                        cv2.cvtColor(relighted1, cv2.COLOR_RGB2BGR))
            np.savetxt(f'./data/relighted/ill/{name[:-4]}.txt', np.expand_dims(np.concatenate([i1, i2]), axis=0), fmt='%.7f', newline="")
            print(f'Saved {img}')
    else:
        if draw:
            pu.visualize([image, mask], title=img)

if __name__ == '__main__':
    path = "/media/donik/Disk/intel_tau/paths_field.txt"

    imgs = np.loadtxt(path, dtype="str")
    img_folder = '/home/donik/tau_relighted/images/'
    gt_folder = "/home/donik/tau_relighted/gt/"
    gt_mask_folder = "/home/donik/tau_relighted/gt_mask/"
    img_cor_folder = "/home/donik/tau_relighted/img_corrected_1/"
    ill_folder = "/home/donik/tau_relighted/ill/"
    folders = [img_folder, gt_folder, gt_mask_folder, img_cor_folder, ill_folder]
    for img_folder in folders:
        if not os.path.exists(img_folder):
            os.mkdir(img_folder)
    num_threads = 1
    with mp.Pool(num_threads) as p:
        p.map(main_process, imgs)
        print('done')