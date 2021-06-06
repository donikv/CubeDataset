import numpy as np
import cv2
import os
import utils.plotting_utils as plt


def fix(i):
    img = cv2.imread(i + '/img.png', cv2.IMREAD_UNCHANGED)[..., 0:3]
    print(i, img.min())
    img1 = np.minimum(2**14, img + 2048)
    print(i, img.min())
    plt.visualize([img / 2**14, img1 / 2**14])
    # ol = img.max() - 2
    # img = np.where(img.min(axis=-1) >= ol, np.zeros_like(img), img)
    #
    # cv2.imwrite(i + f'/img.png', img)
    return img.min()

def fix_t(i):
    # img = cv2.imread(i + '/gt_mask.png', cv2.IMREAD_UNCHANGED)
    # cv2.imwrite(i + f'/gt_mask.png', img[..., 0:3])
    try:
        os.rename(i + 'face_endpoints.txt', i + '/face_endpoints.txt')
    except Exception:
        pass
    #
    # try:
    #     os.remove(i + '/gt_mask_old.png')
    # except Exception:
    #     pass
    #
    # try:
    #     os.remove(i + '/cube.txt')
    # except Exception:
    #     pass
    #
    # try:
    #     os.remove(i + '/gt.txt.old1')
    # except Exception:
    #     pass
    #
    # try:
    #     os.remove(i + '/gt_old.txt')
    # except Exception:
    #     pass
    # return img.min()


if __name__ == '__main__':
    import time

    # start = time.time_ns()
    # path = '/Volumes/Jolteon/fax/to_process/organized'
    # l = np.loadtxt(f'{path}/list.txt', dtype=str)

    path = '/media/donik/Disk/Cube2_new_'
    # path = '/Volumes/Jolteon/fax/to_process/organized2'
    # imgs = list(map(lambda x: f'/media/donik/Disk/Cube2_new_/outdoor/canon_550d/outdoor4/{x}/', range(1,194)))
    l = np.loadtxt(f'{path}/list_outdoor.txt', dtype=str)
    l = list(filter(lambda x: x.find('nikon') != -1 or (x.find('canon') != -1 and x.find('outdoor1') != -1), l))

    imgs = [path + x[1:] for x in l]

    # mini = 3000
    for i in imgs:
        fix_t(i)
        print(i)
        # mini = min(mini, fix(i))
        # if i.find('outdoor6') != -1:
        #     os.rename(i + '/cube.txt', i + '/face_ednpoints.txt')
    # print(mini)