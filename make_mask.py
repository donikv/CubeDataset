import os

import cv2

import numpy as np
import multiprocessing as mp

import utlis.image_utils as iu
import utlis.groundtruth_utils as gu
import utlis.file_utils as fu



def create_corrected_image(img: np.ndarray, gt: np.ndarray, mask: np.ndarray):
    result = [(i,j) if (mask[i][j] == 0).all() else None for j in range(mask.shape[1]) for i in range(mask.shape[0])]
    result = list(filter(lambda x: x is not None, result))
    if len(result) == 0:
        return img
    ind = result[int(len(result)/2)]
    gt_sum = gt[ind[0], ind[1]]
    center = np.array(gt_sum) / 255
    corrected = iu.color_correct_single(img, u_ill=center, c_ill=1)
    return corrected

def process_and_save(name):
    make_mask = True
    path = './data'
    folder = 'relighted'
    print(name)
    image, gt = fu.load_png(name, path=path, directory=f'{folder}/images'), fu.load_png(name, path=path, directory=f'{folder}/gt')
    for i in range(1, 3):
        if make_mask:
            filename = f"{path}/{folder}/gt_mask/{i}-{name}"
            mask = gu.get_mask_from_gt(gt)
            if i == 1:
                mask = 1 - mask
            cv2.imwrite(filename, mask)
        corrected = create_corrected_image(image, gt, mask).astype(int)
        r,g,b = cv2.split(corrected)
        corrected = np.dstack((b,g,r))
        filename = f"{path}/{folder}/img_corrected_1/{i}-{name}"
        cv2.imwrite(filename, corrected)


if __name__ == '__main__':
    path = './data'
    folder = 'relighted'
    special_folder = ''
    image_names = os.listdir(f"{path}/{folder}/images")
    cor_image_names = os.listdir(f"{path}/{folder}/img_corrected_1")
    image_names = list(filter(lambda x: x not in cor_image_names, image_names))

    with mp.Pool(8) as p:
        p.map(process_and_save, image_names)
        print('done')
