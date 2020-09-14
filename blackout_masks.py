import cv2
import numpy as np
import os
import multiprocessing as mp
import utils.plotting_utils as pu
import utils.image_utils as iu


def blackout(data):
    img = data
    pos = np.loadtxt(img+'/cube.txt').astype(int)
    image = cv2.imread(img + '/img.png')
    cb_size = 300
    image[pos[1]-cb_size:pos[1]+cb_size, pos[0] - cb_size:pos[0] + cb_size] = np.zeros(3)
    image[pos[3] - cb_size:pos[3] + cb_size, pos[2] - cb_size:pos[2] + cb_size] = np.zeros(3)
    image1 = cv2.imread(img+'/img_corrected_1.png')
    image1[pos[1]-cb_size:pos[1]+cb_size, pos[0] - cb_size:pos[0] + cb_size] = np.zeros(3)
    image1[pos[3] - cb_size:pos[3] + cb_size, pos[2] - cb_size:pos[2] + cb_size] = np.zeros(3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    mask = cv2.imread(img+'/gt_mask.png')
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2HLS)
    gt = cv2.imread(img+'/gt.png')
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2HLS)
    mask[:, :, 1] = np.where(mask[:, :, 1] < 128, 0, 255)
    mask[:,:,1] = np.where(image[:,:,1] == 0, 128, mask[:,:,1])
    gt[:,:,1] = np.where(image[:,:,1] == 0, 0, gt[:,:,1])
    mask = cv2.cvtColor(mask, cv2.COLOR_HLS2BGR)
    gt = cv2.cvtColor(gt, cv2.COLOR_HLS2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_HLS2BGR)
    # pu.visualize([image, gt, mask])
    cv2.imwrite(img + '/gt_mask_round.png', mask)
    cv2.imwrite(img + '/gt.png', gt)
    cv2.imwrite(img + '/img.png', image)
    cv2.imwrite(img+'/img_corrected_1.png', image1)

    return '1'


if __name__ == '__main__':
    path = 'D:/fax/Cube2/list.txt'
    images = np.loadtxt(path, str)
    images = list(filter(lambda x: x.find('outdoor2') != -1, images))
    blackout(images[0])
    with mp.Pool(8) as pool:
        masks = pool.map(blackout, images)
        exit(0)
