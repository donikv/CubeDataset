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
    dir_name = './projector_test/projector2/pngs/both'
    dir_name_left = './projector_test/projector2/pngs/left'
    dir_name_right = './projector_test/projector2/pngs/right'
    images = os.listdir(dir_name)
    images = list(filter(lambda x: str(x).lower().endswith('.png'), images))

    gts_left, gts_right = [], []
    for idx, img in enumerate(images):
        if idx < 39 or idx > 43:
            continue
        if idx == 40:
            dir_name = dir_name_right
        image = fu.load_png(img, dir_name, '', mask_cube=False)
        image = iu.process_image(image, depth=14)
        image_r = fu.load_png(img, dir_name_right, '', mask_cube=False)
        image_r = iu.process_image(image_r, depth=14)
        image_l = fu.load_png(img, dir_name_left, '', mask_cube=False)
        image_l = iu.process_image(image_l, depth=14)
        gt_left, gt_right = np.zeros(3), np.zeros(3)
        n = 20
        for i in range(n):
            for j in range(n):
                if idx < 39 or idx > 43:
                    x1, y1 = 1152, 1812
                    x2, y2 = 4278, 1647
                else:
                    x1, y1 = 1089, 362
                    x2, y2 = 2494, 351
                gt_left = gt_left + np.clip(image[y1 + i, x1 + j], 1, 255)
                gt_right = gt_right + np.clip(image[y2 + i, x2 + j], 1, 255)
        gt_left, gt_right = gt_left / n / n, gt_right / n / n
        image = cv2.resize(image, (0, 0), fx = 1/5, fy = 1/5)
        gts_left.append(gt_left)
        gts_right.append(gt_right)
        cv2.imwrite(f'./projector_test/projector2/pngs/both/images/{idx+1}.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'./projector_test/projector2/pngs/left/images/{idx + 1}.png', cv2.cvtColor(image_l, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'./projector_test/projector2/pngs/right/images/{idx + 1}.png', cv2.cvtColor(image_r, cv2.COLOR_RGB2BGR))
    np.savetxt('./projector_test/projector2/gt_left1.txt', np.array(gts_left, dtype=np.uint8), fmt='%d')
    np.savetxt('./projector_test/projector2/gt_right1.txt', np.array(gts_right, dtype=np.uint8), fmt='%d')


def par_create(data):
    idx, img, gts_left, gts_right = data
    dir_name = './projector_test/projector2/pngs/both'
    dir_name_left = './projector_test/projector2/pngs/left'
    dir_name_right = './projector_test/projector2/pngs/right'
    if idx < 39 or idx > 43:
        return

    image = fu.load_png(img, dir_name, '', mask_cube=False)
    image_left = fu.load_png(img, dir_name_left, '', mask_cube=False)
    image_right = fu.load_png(img, dir_name_right, '', mask_cube=False)
    image = cv2.resize(image, (0, 0), fx = 1/10, fy = 1/10)
    image_left = cv2.resize(image_left, (0, 0), fx=1 / 10, fy=1 / 10)
    image_right = cv2.resize(image_right, (0, 0), fx=1 / 10, fy=1 / 10)
    gt_left = np.clip(gts_left[idx], 1, 255) / 255
    gt_right = np.clip(gts_right[idx], 1, 255) / 255
    gt_mask, ir, il, r = pu.create_gt_mask(image, image_right, image_left, gt_right, gt_left)
    cv2.imwrite(f'./projector_test/projector2/gt_mask/{idx + 1}rl.png', cv2.cvtColor(gt_mask, cv2.COLOR_RGB2BGR))
    gt_mask, ir, il, r = pu.create_gt_mask(image, image_right, image_left, gt_left, gt_right)
    cv2.imwrite(f'./projector_test/projector2/gt_mask/{idx + 1}lr.png', cv2.cvtColor(gt_mask, cv2.COLOR_RGB2BGR))


def create_gt_mask():

    dir_name = './projector_test/projector2/pngs/both'
    dir_name_left = './projector_test/projector2/pngs/left'
    dir_name_right = './projector_test/projector2/pngs/right'
    images = os.listdir(dir_name)
    images = list(filter(lambda x: str(x).lower().endswith('.png'), images))
    gts_left = np.loadtxt('./projector_test/projector2/gt_left.txt')
    gts_right = np.loadtxt('./projector_test/projector2/gt_right.txt')
    data = list(map(lambda x: (x[0], x[1], gts_left, gts_right), enumerate(images)))

    num_proc = 16

    if num_proc > 1:
        with mp.Pool(num_proc) as p:
            p.map(par_create, data)
            print('done')
        return

    for idx, img in enumerate(images):
        image = fu.load_png(img, dir_name, '', mask_cube=False)
        image_left = fu.load_png(img, dir_name_left, '', mask_cube=False)
        image_right = fu.load_png(img, dir_name_right, '', mask_cube=False)
        image = cv2.resize(image, (0, 0), fx = 1/10, fy = 1/10)
        image_left = cv2.resize(image_left, (0, 0), fx=1 / 10, fy=1 / 10)
        image_right = cv2.resize(image_right, (0, 0), fx=1 / 10, fy=1 / 10)
        gt_left = np.clip(gts_left[idx], 1, 255) / 255
        gt_right = np.clip(gts_right[idx], 1, 255) / 255
        gtimg = np.ones((50, 50, 3)) * gt_right
        gtimg1 = np.ones((50, 50, 3)) * gt_left
        gtimg = np.concatenate((gtimg, gtimg1), axis=1)
        gt_mask, ir, il, r = pu.create_gt_mask(image, image_right, image_left, gt_right, gt_left)
        plt.visualize([image, gt_mask, ir, il, image_right, image_left, r, gtimg])

        gt_mask, ir, il, r = pu.create_gt_mask(image, image_right, image_left, gt_left, gt_right)
        plt.visualize([image, gt_mask, ir, il, image_right, image_left, r, gtimg])
        cv2.imwrite(f'./projector_test/projector2/gt_mask/{idx+1}.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def debayer():
    dir_name = './projector_test/projector2/'
    images = os.listdir(dir_name)
    images = list(filter(lambda x: str(x).lower().endswith('.tiff'), images))
    gts_left = np.loadtxt('./projector_test/projector2/gt_left.txt')
    gts_right = np.loadtxt('./projector_test/projector2/gt_right.txt')

    for idx in range(0, len(images), 1):
        img = images[idx]
        name = f'{int(idx/3) + 1}.png'
        if idx % 3 == 1:
            fold = 'left'
        elif idx % 3 == 2:
            fold = 'right'
        else:
            fold = 'both'
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
    # debayer()
    # find_gt()
    create_gt_mask()
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