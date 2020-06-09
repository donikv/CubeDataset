import utils.projector_utils as pu
import utils.plotting_utils as plt
import utils.file_utils as fu
import utils.image_utils as iu
import utils.capture_utils as cu

import os
import cv2
import numpy as np
import multiprocessing as mp
import time
from skimage.filters import gaussian
from datetime import datetime

def saveImage(image, dir, name, convert_rgb_2_bgr=True):
    if not os.path.exists(dir):
        os.mkdir(dir)
    pth = os.path.join(dir, f'{name}.png')
    if convert_rgb_2_bgr:
        r, g, b = cv2.split(image)
        image = np.dstack((b, g, r))
    cv2.imwrite(pth, image)

def debayer_single(img, dir_name):
    if img.endswith('tiff'):
        image_tiff = cv2.imread(f'{dir_name}{image}', cv2.IMREAD_UNCHANGED)
        imageRaw = cv2.cvtColor(image_tiff, cv2.COLOR_BAYER_RG2BGR)
    else:
        imageRaw = fu.load_cr2(img, path=dir_name, directory='', mask_cube=False)
    rgb = cv2.cvtColor(imageRaw, cv2.COLOR_RGB2BGR)
    return rgb


def line(image, colors):
    alpha = np.random.randint(1, 89, 1) * np.pi / 180
    alpha = 0.5
    return pu.line_image(image, colors, alpha)


if __name__ == '__main__':
    use_tiff = True
    dir = 'projector_test/third/ambient3'
    date = datetime.date(datetime.now())
    for it in range(0, 10):
        images = pu.create_image(1080, 1920, False, line)
        for i, image in enumerate(images):
            saveImage(image, dir, f'ambient-line-white-{date}-{it}-{i}')
            # plt.visualize([image], out_file=f'projector_test/third/triangle-white2-{i}.png')
        # size = 21
    time.sleep(5)
    # dir = 'projector_test/third/ambient'
    dir2 = "/media/donik/Disk/captures/"
    dir3 = "/media/donik/Disk/ambient3/"

    folds = ['both', 'left', 'black_left', 'right', 'black_right']
    folds_ambient = ['both', 'ambient', 'direct', 'both', 'ambient', 'direct']
    images = os.listdir(dir)
    for image in images:
        window = cu.show_full_screen(image, dir)
        k = cv2.waitKey(1000)
        if k != -1:
            exit()
        img_name = "cupcoffee-" + image[:-4]
        cu.capture_from_camera(f"{dir2}/{img_name}.cr2")
        cv2.destroyWindow(window)

        if use_tiff:
            os.system(f'dcraw -D -4 -T {dir2}{img_name}.cr2')
            deb_img = debayer_single(img_name + ".tiff", dir2)
        else:
            deb_img = debayer_single(img_name + ".cr2", dir2)
        saveImage(deb_img, dir3 + folds_ambient[int(image[-5])], img_name)