import utlis.projector_utils as pu
import utlis.plotting_utils as plt
import utlis.file_utils as fu
import utlis.image_utils as iu

import os
import cv2
import numpy as np
import multiprocessing as mp


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
    dir_name = './projector_test/projector2/'
    images = os.listdir(dir_name)
    images = list(filter(lambda x: str(x).lower().endswith('.jpg'), images))

    gts_left, gts_right = [], []
    for idx, img in enumerate(images):
        if idx % 3 != 0 or img >= 'NIK_6896.JPG':
            continue
        image = fu.load_png(img, dir_name, '', mask_cube=False)
        gt_left, gt_right = np.zeros(3), np.zeros(3)
        n = 20
        for i in range(n):
            for j in range(n):
                gt_left = gt_left + np.clip(image[1812 + i, 1152 + j], 1, 255)
                gt_right = gt_right + np.clip(image[1647 + i, 4278 + j], 1, 255)
        gt_left, gt_right = gt_left / n / n, gt_right / n / n
        image = cv2.resize(image, (0, 0), fx = 1/5, fy = 1/5)
        gts_left.append(gt_left)
        gts_right.append(gt_right)
        idx = int(idx / 3)
        cv2.imwrite(f'./projector_test/projector2/images/{idx+1}.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    np.savetxt('./projector_test/projector2/gt_left.txt', np.array(gts_left, dtype=np.uint8), fmt='%d')
    np.savetxt('./projector_test/projector2/gt_right.txt', np.array(gts_right, dtype=np.uint8), fmt='%d')


def create_gt_mask():
    dir_name = './projector_test/projector2/'
    images = os.listdir(dir_name)
    images = list(filter(lambda x: str(x).lower().endswith('.jpg'), images))
    gts_left = np.loadtxt('./projector_test/projector2/gt_left.txt')
    gts_right = np.loadtxt('./projector_test/projector2/gt_right.txt')

    for idx in range(0, len(images), 3):
        img = images[idx]
        if 'NIK_6769.JPG' > img or img >= 'NIK_6896.JPG':
            continue
        image = fu.load_png(img, dir_name, '', mask_cube=False)
        image_left = fu.load_png(images[idx+1], dir_name, '', mask_cube=False)
        image_right = fu.load_png(images[idx+2], dir_name, '', mask_cube=False)
        # gt_left, gt_right = np.zeros(3), np.zeros(3)
        # n = 20
        # for i in range(n):
        #     for j in range(n):
        #         gt_left = gt_left + np.clip(image[1613 + i, 1115 + j], 1, 255) / 255
        #         gt_right = gt_right + np.clip(image[1460 + i, 4267 + j], 1, 255) / 255
        # gt_left, gt_right = gt_left / n / n, gt_right / n / n
        image = cv2.resize(image, (0, 0), fx = 1/10, fy = 1/10)
        image_left = cv2.resize(image_left, (0, 0), fx=1 / 10, fy=1 / 10)
        image_right = cv2.resize(image_right, (0, 0), fx=1 / 10, fy=1 / 10)
        idx = int(idx / 3)
        gt_left = np.clip(gts_left[idx], 1, 255) / 255
        gt_right = np.clip(gts_right[idx], 1, 255) / 255
        gtimg = np.ones((50, 50, 3)) * gt_right
        gtimg1 = np.ones((50, 50, 3)) * gt_left
        gtimg = np.concatenate((gtimg, gtimg1), axis=1)
        gt_mask, ir, il, r = pu.create_gt_mask(image, image_right, image_left, gt_right, gt_left)
        plt.visualize([image, gt_mask, ir, il, image_right, image_left, r, gtimg], out_file=dir_name + f'gt_mask/{idx+1}rl.png')

        gt_mask, ir, il, r = pu.create_gt_mask(image, image_right, image_left, gt_left, gt_right)
        plt.visualize([image, gt_mask, ir, il, image_right, image_left, r, gtimg], out_file=dir_name + f'gt_mask/{idx+1}lr.png')
        # cv2.imwrite(f'./projector_test/projector2/gt_mask/{idx+1}.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def debayer():
    dir_name = './projector_test/projector2/'
    images = os.listdir(dir_name)
    images = list(filter(lambda x: str(x).lower().endswith('.tiff'), images))
    gts_left = np.loadtxt('./projector_test/projector2/gt_left.txt')
    gts_right = np.loadtxt('./projector_test/projector2/gt_right.txt')

    for idx in range(0, len(images), 1):
        img = images[idx]
        name = f'{idx/3}.png'
        if idx % 3 == 1:
            fold = 'left'
        elif idx % 3 == 2:
            fold = 'right'
        else:
            fold = 'both'
        name = name + '.png'
        imageRaw = cv2.imread(os.path.join(dir_name, img), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)

        rgb = cv2.cvtColor(imageRaw, cv2.COLOR_BAYER_BG2BGR)
        cv2.imwrite(f'{dir_name}pngs/{fold}/{name}', rgb)


def line(image, colors):
    alpha = np.random.randint(1, 89, 1) * np.pi / 180
    alpha = 0.5
    return pu.line_image(image, colors, alpha)

if __name__ == '__main__':
    # size = 21
    # images = pu.create_image(1080, 1920, 1, line)
    # for i, image in enumerate(images):
    #     plt.visualize([image], out_file=f'projector_test/second/line5-{i}.png')
    # find_gt()
    debayer()
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