import cv2
import numpy as np

if __name__ == '__main__':
    import time

    start = time.time_ns()
    path = '/Volumes/Jolteon/fax/to_process/organized'
    l = np.loadtxt(f'{path}/list.txt', dtype=str)
    imgs = [path + x[1:] for x in l]

    for i in imgs:
        img = cv2.imread(i + '/img.png', cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 2 ** 16
        I = np.linalg.norm(img, axis=-1, keepdims=True)
        gt_mask = cv2.imread(i + '/gt_mask.png', cv2.IMREAD_UNCHANGED)
        gt_mask = np.where(gt_mask < 128, 2, 1)  # Invert and mark sun as 1 and shadow as 2
        gt_mask = np.where(I < 1e-7, np.zeros_like(gt_mask), gt_mask)
        gt = np.loadtxt(i + '/gt.txt')
        gt = np.concatenate([gt[3:], gt[:3]], axis=0)
        np.savetxt(i + '/gt_new.txt', gt.reshape((1,-1)))
        cv2.imwrite(i + f'/gt_mask_new.png', gt_mask)
