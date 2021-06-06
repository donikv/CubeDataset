import cv2
import numpy as np

def convert_to_new(i):
    gt = np.loadtxt(i + '/gt.txt')
    # np.savetxt(i + '/gt_old.txt', gt.reshape((1, -1)))

    img = cv2.imread(i + '/img.png', cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 2 ** 16
    I = np.linalg.norm(img, axis=-1, keepdims=True)

    gt_mask = cv2.imread(i + '/gt_mask.png', cv2.IMREAD_UNCHANGED)[...,0:3]
    # cv2.imwrite(i + f'/gt_mask_old.png', gt_mask)
    gt_mask = np.where(gt_mask < 128, 2, 1)  # Invert and mark sun as 2 and shadow as 1
    gt_mask = np.where(I < 1e-5, np.zeros_like(gt_mask), gt_mask)
    cv2.imwrite(i + f'/gt_mask.png', gt_mask)

    gt = np.flip(gt.reshape((-1, 3)), axis=0)
    np.savetxt(i + '/gt.txt', gt)

def conver_to_old(i):
    gt = np.loadtxt(i + '/gt.txt')
    np.savetxt(i + '/gt_new.txt', gt)

    img = cv2.imread(i + '/img.png', cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 2 ** 16
    I = np.linalg.norm(img, axis=-1, keepdims=True)

    gt_mask = cv2.imread(i + '/gt_mask.png', cv2.IMREAD_UNCHANGED)
    cv2.imwrite(i + f'/gt_mask_new.png', gt_mask)
    gm = np.where(gt_mask == 2, 0, 255)  # Invert and mark sun as 0 and shadow as 255
    gt_mask = np.where(gt_mask == 0, 128, gm)  # Mask clipped regions as 128

    gt = np.flip(gt, axis=0).reshape((1,-1))
    np.savetxt(i + '/gt.txt', gt)
    cv2.imwrite(i + f'/gt_mask.png', gt_mask)

if __name__ == '__main__':
    import time

    start = time.time_ns()
    path = '/Volumes/Jolteon/fax/raws6/organized'
    l = np.loadtxt(f'{path}/list.txt', dtype=str)
    imgs = [path + x[1:] for x in l]

    for i in imgs:
        convert_to_new(i)
        #conver_to_old(i)

