import cv2
import numpy as np

if __name__ == '__main__':
    import time

    start = time.time_ns()
    path = '/Volumes/Jolteon/fax/to_process/organized2'
    l = np.loadtxt(f'{path}/list.txt', dtype=str)
    imgs = [path + x[1:] for x in l]

    for i in imgs:
        gt = np.loadtxt(i + '/gt.txt')
        np.savetxt(i + '/gt_old.txt', gt.reshape((1,-1)))
        print(i)
        img = cv2.imread(i + '/img.png', cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 2 ** 16
        I = np.linalg.norm(img, axis=-1, keepdims=True)
        gt_mask = cv2.imread(i + '/gt_mask.png', cv2.IMREAD_UNCHANGED)
        cv2.imwrite(i + f'/gt_mask_old.png', gt_mask)
        # gt_mask = cv2.imread(i + '/gt_mask_new.png', cv2.IMREAD_UNCHANGED)
        gt_mask = np.where(gt_mask < 128, 2, 1)  # Invert and mark sun as 1 and shadow as 2
        gt_mask = np.where(I < 1e-7, np.zeros_like(gt_mask), gt_mask)
        # np.savetxt(i + '/gt_new.txt', gt)
        gt = np.flip(gt.reshape((-1,3)), axis=0)
        np.savetxt(i + '/gt.txt', gt)
        cv2.imwrite(i + f'/gt_mask.png', gt_mask)
