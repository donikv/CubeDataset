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
    image = iu.color_correct_single(image, gt, c_ill=1)
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
        gts_left.append(image[1613, 1131])
        gts_right.append(image[1460, 4267])
        image = cv2.resize(image, (0, 0), fx = 1/5, fy = 1/5)
        idx = int(idx / 3)
        cv2.imwrite(f'./projector_test/projector2/images/{idx+1}.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    np.savetxt('./projector_test/projector2/gt_left.txt', np.array(gts_left, dtype=np.uint8), fmt='%d')
    np.savetxt('./projector_test/projector2/gt_right.txt', np.array(gts_right, dtype=np.uint8), fmt='%d')


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
    dir_name = './projector_test/projector2/images'
    images = os.listdir(dir_name)
    # images = list(filter(lambda x: str(x).lower().endswith('.jpg'), images))

    gts_left = np.loadtxt('./projector_test/projector2/gt_left.txt')
    data = list(map(lambda x: (x, gts_left), images))

    with mp.Pool(16) as p:
        p.map(correct, data)
        print('done')