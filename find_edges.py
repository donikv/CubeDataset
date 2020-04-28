import cv2

from utlis import file_utils as fu
import utlis.image_utils as iu
import utlis.plotting_utils as pu
import utlis.relighting_utils as ru
from utlis.groundtruth_utils import GroundtruthLoader
import numpy as np
import multiprocessing as mp

def get_ill_diffs():
    gtLoaderL = GroundtruthLoader('cube+_left_gt.txt')
    gtLoaderR = GroundtruthLoader('cube+_right_gt.txt')

    gtsl = gtLoaderL.gt
    gtsr = gtLoaderR.gt
    gt_diff = np.abs(gtsl - gtsr)
    gt_diff_mean = gt_diff.mean(0)
    gt_diff_filter = np.array(list(
        map(lambda x: x[0] + 1,
            filter(lambda x: (x[1] < gt_diff_mean).all(),
                   enumerate(gt_diff)
                   )
            )
    ))
    return gt_diff_filter


def process_with_edges(img, gtLoader, folder_step):
    image = fu.load(img, folder_step, depth=14)
    height, width, _ = image.shape
    image = cv2.resize(image, (int(width / 5), int(height / 5)))
    image = iu.process_image(image, 14)
    image = iu.color_correct_single(image, c_ill=1, u_ill=gtLoader[img - 1])
    edges, closing = iu.find_edges(image, 100, 200)
    contours, mask, identation_index = iu.cv2_contours(closing, method=-1, upper=np.array([128, 255, 255]))

    if 5 * mask.size / 6 < np.count_nonzero(mask) or np.count_nonzero(mask) < mask.size / 6:
        return False, image, mask, None

    if identation_index < 12.5:
        print(identation_index)
        return False, image, mask, None

    ill1, ill2 = ru.random_colors()
    relighted = iu.color_correct(image, mask=mask, ill1=1 / ill1, ill2=1 / ill2,
                                 c_ill=1)
    colored_mask = np.array(
        [[ill1 * pixel + ill2 * (1 - pixel) for pixel in row] for row in mask])

    return True, image, colored_mask, relighted


def main_process(data):
    img, gtLoader = data
    folder_step = 200
    draw = False
    save = True
    succ, image, colored_mask, relighted = process_with_edges(img, gtLoader, folder_step)
    if succ:
        if draw:
            pu.visualize([image, relighted, colored_mask], title=img)
        if save:
            cv2.imwrite(f'./data/relighted/images/{img}-4.png', cv2.cvtColor(relighted, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f'./data/relighted/gt/{img}-4.png', cv2.cvtColor((colored_mask * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            print(f'Saved {img}')
    else:
        if draw:
            pu.visualize([image, colored_mask], title=img)

if __name__ == '__main__':
    single_ill = get_ill_diffs()
    gtLoader = GroundtruthLoader('cube+_gt.txt')
    single_ill_gt = list(map(lambda x: (x, gtLoader), single_ill))
    num_threads = 8
    if num_threads < 2:
        for data in single_ill_gt:
            main_process(data)
    else:
        with mp.Pool(num_threads) as p:
            p.map(main_process, single_ill_gt)
            print('done')