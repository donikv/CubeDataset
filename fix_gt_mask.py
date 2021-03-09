import numpy as np
import cv2

def fix(i):
    gt_mask = cv2.imread(i + '/gt_mask.png', cv2.IMREAD_UNCHANGED)[..., 0:3]
    cv2.imwrite(i + '/gt_mask.png', gt_mask)
    gts = np.loadtxt(i + "/gt.txt")
    gts = gts.reshape((2,-1))
    n = np.linalg.norm(gts, axis=-1, keepdims=True)
    gts = gts / (n + 1e-10)
    np.savetxt(i + "/gt.txt", gts, fmt='%.7f')
    gts = gts * 255
    gts = gts.astype(np.uint8).reshape((2,-1))

    gt = np.where(gt_mask == 2, gts[1], gts[0])
    cv2.imwrite(i + f'/gt.png', cv2.cvtColor(gt, cv2.COLOR_RGB2BGR))



if __name__ == '__main__':
    import time

    # start = time.time_ns()
    path = '/Volumes/Jolteon/fax/to_process/organized'
    l = np.loadtxt(f'{path}/list.txt', dtype=str)

    # path = '/media/donik/Disk/Cube2_new'
    # path = '/Volumes/Jolteon/fax/to_process/organized2'
    # l = np.loadtxt(f'{path}/list_outdoor.txt', dtype=str)
    # l = list(filter(lambda x: x.find('outdoor6') != -1, l))

    imgs = [path + x[1:] for x in l]

    for i in imgs:
        # convert_to_new(i)
        fix(i)
