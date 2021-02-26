import cv2
import numpy as np
import os
import multiprocessing as mp
import utils.plotting_utils as pu
import utils.image_utils as iu


def blackout(data):
    img, invert = data
    pos = np.loadtxt(img+'/cube.txt').astype(int)
    image = cv2.imread(img + '/img.png')
    image1 = cv2.imread(img+'/img_corrected_1.png')
    # image_corr = cv2.imread(img+'/img_corrected.png')
    gt = cv2.imread(img + '/gt.png')
    mask = cv2.imread(img+'/gt_mask.png')
    if image1.shape != image.shape:
        image = cv2.resize(image, (0, 0), fx=1/5, fy=1/5)
        gt = cv2.resize(gt, (0, 0), fx=1/5, fy=1/5)
    if image.shape != mask.shape:
        mask = cv2.resize(mask, tuple(reversed(image.shape[0:2])))

    cb_size = 200
    image[pos[1]-cb_size:pos[1]+cb_size, pos[0] - cb_size:pos[0] + cb_size] = np.zeros(3)
    image[pos[3] - cb_size:pos[3] + cb_size, pos[2] - cb_size:pos[2] + cb_size] = np.zeros(3)

    image1[pos[1]-cb_size:pos[1]+cb_size, pos[0] - cb_size:pos[0] + cb_size] = np.zeros(3)
    image1[pos[3] - cb_size:pos[3] + cb_size, pos[2] - cb_size:pos[2] + cb_size] = np.zeros(3)
    # image_corr[pos[1]-cb_size:pos[1]+cb_size, pos[0] - cb_size:pos[0] + cb_size] = np.zeros(3)
    # image_corr[pos[3] - cb_size:pos[3] + cb_size, pos[2] - cb_size:pos[2] + cb_size] = np.zeros(3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2HLS)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2HLS)
    if invert:
        mask = 255 - mask
        cv2.imwrite(img + '/gt_mask.png', mask)
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
    # cv2.imwrite(img+'/img_corrected.png', image_corr)

    return '1'


if __name__ == '__main__':
    bpath = '/media/donik/Disk/cube_bounding/'
    path = bpath + 'list.txt'
    images = np.loadtxt(path, str)
    images = list(map(lambda x: bpath + x[2:], images))
    images = [(x, False) for x in images]
    # images = list(filter(lambda x: x.find('outdoor_slo') != -1, images))
    blackout(images[0])
    # for image in images:
    #     blackout(image)
    with mp.Pool(15) as pool:
        masks = pool.map(blackout, images)
        exit(0)
