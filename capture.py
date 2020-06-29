import utils.projector_utils as pu
import utils.plotting_utils as plt
import utils.file_utils as fu
import utils.image_utils as iu
import utils.capture_utils as cu
import utils.relighting_utils as ru

import os
import cv2
import numpy as np
import multiprocessing as mp
import time
from skimage.filters import gaussian
from datetime import datetime


def saveImage(image, dir, name, convert_rgb_2_bgr=True):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=False)
    pth = os.path.join(dir, f'{name}.png')
    if convert_rgb_2_bgr:
        r, g, b = cv2.split(image)
        image = np.dstack((b, g, r))
    cv2.imwrite(pth, image)


def debayer_single(img, dir_name):
    if img.endswith('tiff'):
        image_tiff = cv2.imread(f'{dir_name}{img}', cv2.IMREAD_UNCHANGED)
        imageRaw = cv2.cvtColor(image_tiff, cv2.COLOR_BAYER_RG2BGR)
    else:
        imageRaw = fu.load_cr2(img, path=dir_name, directory='', mask_cube=False)
    rgb = cv2.cvtColor(imageRaw, cv2.COLOR_RGB2BGR)
    return rgb


def line(image, colors):
    alpha = np.random.randint(1, 89, 1) * np.pi / 180
    alpha = 0.5
    return pu.line_image(image, colors, alpha)


def batch_process_images(folder_name, use_tiff=True):
    dir2 = 'G:\\fax\\diplomski\\Datasets\\third\\captures/'
    dir3 = f'G:\\fax\\diplomski\\Datasets\\third\\{folder_name}{"_tiff"if use_tiff else ""}/'
    if not os.path.exists(dir3):
        os.mkdir(dir3)

    if folder_name.startswith('ambient'):
        folds = ['both', 'left', 'right', 'both', 'left', 'right']
    else:
        folds = ['both', 'right', 'black_left', 'left', 'black_right']
    images = os.listdir(dir2)
    images = list(filter(lambda x: (x.startswith('cup-ambient-line-') or x.startswith('spyder-ambient-line-')) and x.endswith('.tiff' if use_tiff else '.cr2'), images))
    for img_name in images:
        deb_img = debayer_single(img_name, dir2)
        image_idx_pos = -6 if use_tiff else -5
        saveImage(deb_img, dir3 + folds[int(img_name[image_idx_pos])], img_name[:image_idx_pos+1])


def display_random_color():
    colors = ru.random_colors(desaturate=False)
    for color in colors:
        image = np.ones((90, 160, 3)) * color / 3
        window = "window"
        cv2.namedWindow(window, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(window, image)
        k = cv2.waitKey()
        if k == 27:
            exit()
        cv2.destroyWindow(window)

if __name__ == '__main__':
    while True:
        display_random_color()
    use_tiff = True
    batch_process_images('ambient', use_tiff)
    exit()
    dir = 'projector_test/third/ambient5'
    date = datetime.date(datetime.now())
    generate_new = True
    if generate_new:
        for it in range(0, 3):
            images = pu.create_image(1080, 1920, False, line)
            for i, image in enumerate(images):
                name = f'ambient-line-white-{date}-{it}-{i}'
                saveImage(image, dir, name)
                # plt.visualize([image], out_file=f'projector_test/third/triangle-white2-{i}.png')
        for it in range(0, 3):
            images = pu.create_image(1080, 1920, False, pu.circle_image)
            for i, image in enumerate(images):
                name = f'ambient-circle-white-{date}-{it}-{i}'
                saveImage(image, dir, name)
                # plt.visualize([image], out_file=f'projector_test/third/triangle-white2-{i}.png')
        for it in range(0, 5):
            images = pu.create_image(1080, 1920, False, pu.blur_image)
            for i, image in enumerate(images):
                saveImage(image, dir, f'ambient-blur-{date}-{it}-{i}')
        # size = 21
    time.sleep(5)
    # dir = 'projector_test/third/ambient'
    dir2 = "/media/donik/Disk/captures/"
    dir3 = "/media/donik/Disk/ambient5/"

    if dir.find('ambient') > -1:
        folds = ['both', 'left', 'right', 'both', 'left', 'right']
    else:
        folds = ['both', 'right', 'black_left', 'left', 'black_right']
    images = os.listdir(dir)
    for image in images:
        window = cu.show_full_screen(image, dir)
        k = cv2.waitKey(1000)
        if k != -1:
            exit()
        img_name = "spydercomb-" + image[:-4]
        cu.capture_from_camera(f"{dir2}/{img_name}.cr2")
        cv2.destroyWindow(window)

        if use_tiff:
            os.system(f'dcraw -D -4 -T {dir2}{img_name}.cr2')
            deb_img = debayer_single(img_name + ".tiff", dir2)
            saveImage(deb_img, dir3[:-1] + '_tiff/' + folds[int(image[-5])], img_name)

        deb_img = debayer_single(img_name + ".cr2", dir2)
        saveImage(deb_img, dir3 + folds[int(image[-5])], img_name)