import utlis.projector_utils as pu
import utlis.plotting_utils as plt
import utlis.file_utils as fu
import utlis.image_utils as iu
import utlis.capture_utils as cu

import os
import cv2
import numpy as np
import multiprocessing as mp
from skimage.filters import gaussian

def saveImage(image, dir, name):
    if not os.path.exists(dir):
        os.mkdir(dir)
    pth = os.path.join(dir, f'{name}.png')
    cv2.imwrite(pth, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def correct(data):
    dir_name = './projector_test/projector2/images'
    img, gts_left = data
    idx = int(img[:-4]) - 1
    image = fu.load_png(img, dir_name, '', mask_cube=False)
    gt = np.clip(gts_left[idx], 1, 255) / 255
    image = iu.color_correct_single(image, gt, c_ill=1/3)
    cv2.imwrite(f'./projector_test/projector2/img_corrected_1/{idx + 1}.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def crop(data):
    def rect_for_img(idx):
        rect = None
        if idx < 11:
            rect = (426, 204, 582, 397)
        elif 11 <= idx < 22:
            rect = (357, 117, 628, 395)
        elif 22 <= idx < 33:
            rect = (322, 1, 606, 396)
        elif 33 <= idx < 44:
            rect = (412, 179, 586, 397)
        return rect

    names = ['images', 'img_corrected_1']
    img, gts_left = data
    for name in names:
        dir_name = f'./projector_test/projector1/{name}'
        idx = int(img[:-4]) - 1
        image = fu.load_png(img, dir_name, '', mask_cube=False)
        rect = rect_for_img(idx)
        if rect is None:
            continue
        cropped = pu.crop(image, rect, False)
        cropped = cv2.resize(cropped, (1000, 1000))
        cv2.imwrite(f'./projector_test/projector1_cropped_resized/{name}/{idx + 1}.png', cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))


def find_gt():
    dir = 'D:\\fax\\Dataset\\ambient/pngs'
    dir_name = f'{dir}/both'
    dir_name_left = f'{dir}/left'
    dir_name_right = f'{dir}/right'
    images = os.listdir(dir_name)
    images = list(filter(lambda x: str(x).lower().endswith('.png'), images))

    gts_left, gts_right = [], []
    for idx, img in enumerate(images):
        image = fu.load_png(img, dir_name, '', mask_cube=False)
        image = iu.process_image(image, depth=16, blacklevel=2048)
        image_r = fu.load_png(img, dir_name_right, '', mask_cube=False)
        image_r = iu.process_image(image_r, depth=16, blacklevel=2048)
        image_l = fu.load_png(img, dir_name_left, '', mask_cube=False)
        image_l = iu.process_image(image_l, depth=16, blacklevel=2048)
        gt_left, gt_right = np.zeros(3), np.zeros(3)
        n = 20
        for i in range(n):
            for j in range(n):
                x1, y1 = 1130, 2540#3861, 1912#1200, 2450
                x2, y2 = 1440, 2540#1368, 2217 #1350, 2450
                # if idx < 39 or idx > 43:
                #     x1, y1 = 1152, 1812
                #     x2, y2 = 4278, 1647
                # else:
                #     x1, y1 = 1089, 362
                #     x2, y2 = 2494, 351
                gt_left = gt_left + np.clip(image_l[y1 + i, x1 + j], 1, 255)
                gt_right = gt_right + np.clip(image_r[y2 + i, x2 + j], 1, 255)
        gt_left, gt_right = gt_left / n / n, gt_right / n / n
        image = cv2.resize(image, (0, 0), fx = 1/5, fy = 1/5)
        gts_left.append(gt_left)
        gts_right.append(gt_right)
        saveImage(image, f'{dir}/both/images', idx+1)
        saveImage(image_l, f'{dir}/left/images', idx + 1)
        saveImage(image_r, f'{dir}/right/images', idx + 1)
    np.savetxt(f'{dir}/gt_left.txt', np.array(gts_left, dtype=np.uint8), fmt='%d')
    np.savetxt(f'{dir}/gt_right.txt', np.array(gts_right, dtype=np.uint8), fmt='%d')


def par_create(data):
    idx, img, gts_left, gts_right = data
    dir = 'D:\\fax\\Dataset\\ambient/pngs'
    dir_name = f'{dir}/both'
    dir_name_left = f'{dir}/left'
    dir_name_right = f'{dir}/right'

    image = fu.load_png(img, dir_name, '', mask_cube=False)
    image_left = fu.load_png(img, dir_name_left, '', mask_cube=False)
    image_right = fu.load_png(img, dir_name_right, '', mask_cube=False)
    image = cv2.resize(image, (0, 0), fx = 1/5, fy = 1/5)
    image_left = cv2.resize(image_left, (0, 0), fx=1 / 5, fy=1 / 5)
    image_right = cv2.resize(image_right, (0, 0), fx=1 / 5, fy=1 / 5)
    gt_left = np.clip(gts_left[idx], 1, 255) / 255
    gt_right = np.clip(gts_right[idx], 1, 255) / 255
    gt_mask, ir, il, r = pu.create_gt_mask(image, image_right, image_left, gt_right, gt_left)
    ggt_mask = gt_mask#pu.denoise_mask(gt_mask)
    saveImage(ggt_mask.astype(np.uint8), f'{dir}/../gt_mask/', idx + 1)
    # gt_mask, ir, il, r = pu.create_gt_mask(image, image_right, image_left, gt_left, gt_right)
    # cv2.imwrite(f'D:\\fax\\Dataset\\ambient/pngs/gt_mask/{idx + 1}lr.png', cv2.cvtColor(gt_mask, cv2.COLOR_RGB2BGR))


def create_gt_mask():
    dir = 'D:\\fax\\Dataset\\ambient/pngs'
    dir_name = f'{dir}/both'
    dir_name_left = f'{dir}/left'
    dir_name_right = f'{dir}/right'
    images = os.listdir(dir_name)
    images = list(filter(lambda x: str(x).lower().endswith('.png'), images))
    gts_left = np.loadtxt(f'{dir}/gt_left.txt')
    gts_right = np.loadtxt(f'{dir}/gt_right.txt')
    data = list(map(lambda x: (int(x[:-4])-1, x, gts_left, gts_right), images))

    num_proc = 1

    if num_proc > 1:
        with mp.Pool(num_proc) as p:
            p.map(par_create, data)
            print('done')
        return

    for d in data:
        par_create(d)


def debayer():
    dir_name = 'D:\\fax\\Dataset\\two_ill/'
    images = os.listdir(dir_name)
    images = list(filter(lambda x: str(x).lower().endswith('.cr2'), images))
    images = sorted(images, key=lambda x: int(x[4:-4]))
    folds = ['both', 'left', 'right', 'white1', 'white2', 'allwhite']

    for idx in range(0, len(images), 1):
        img = images[idx]
        name = f'{int(idx/6) + 1}.png'
        fold = folds[idx%6]
        imageRaw = fu.load_cr2(img, path=dir_name, directory='', mask_cube=False)

        rgb = cv2.cvtColor(imageRaw, cv2.COLOR_RGB2BGR) #  cv2.cvtColor(imageRaw, cv2.COLOR_BAYER_RG2BGR)
        if not os.path.exists(f'{dir_name}pngs/{fold}/'):
            os.mkdir(f'{dir_name}pngs/{fold}/')
        cv2.imwrite(f'{dir_name}pngs/{fold}/{name}', rgb)


def line(image, colors):
    alpha = np.random.randint(1, 89, 1) * np.pi / 180
    alpha = 0.75
    return pu.line_image(image, colors, alpha)

if __name__ == '__main__':
    # size = 21
    dir = 'projector_test/third/'
    images = os.listdir(dir)
    for image in images:
        window = cu.show_full_screen(image, dir)
        k = cv2.waitKey(3000)
        if k != -1:
            exit()
        cv2.destroyWindow(window)
    # images = pu.create_image(1080, 1920, 1, pu.triangle_image)
    # for i, image in enumerate(images):
    #     plt.visualize([image], out_file=f'projector_test/third/triangle-white2-{i}.png')
    # debayer()
    # find_gt()
    # create_gt_mask()
    # dir_name = './projector_test/projector2/images'
    # images = os.listdir(dir_name)
    # # images = list(filter(lambda x: str(x).lower().endswith('.jpg'), images))
    #
    # gts_left = np.loadtxt('./projector_test/projector2/gt_left.txt')
    # data = list(map(lambda x: (x, gts_left), images))
    #
    # with mp.Pool(16) as p:
    #     p.map(correct, data)
    #     print('done')