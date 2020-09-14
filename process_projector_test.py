import cv2
import numpy as np
import os
import shutil

import utils.image_utils as iu
import utils.file_utils as fu
import utils.relighting_utils as ru
import utils.projector_utils as pu
import utils.plotting_utils as plt

def select_image(idx, lr, images):
    image_index = (idx-1)*3

    image = images[image_index]
    image1 = images[image_index+1]
    image2 = images[image_index+2]

    if lr:
        return (image, image1, image2)
    else:
        return (image, image2, image1)

if __name__ == '__main__':
    path = 'projector_test/projector2/'
    images = os.listdir('projector_test/projector2/')
    images = list(filter(lambda x: x.endswith('.NEF'), images))
    images = sorted(images, key=lambda x: int(x[4:-4]))

    lrs = open(path + 'gt_mask_lr.txt').readlines()
    lrs = list(map(lambda x: (int(x.split()[0]), x.split()[1] == 'lr'), lrs))
    for idx, lr in lrs:
        if idx * 3 > len(images):
            continue
        imgs = select_image(idx, lr, images)
        shutil.copyfile(path + imgs[0], path + 'both/'+str(idx) + '.NEF')
        shutil.copyfile(path + imgs[1], path + 'right/' + str(idx) + '.NEF')
        shutil.copyfile(path + imgs[2], path + 'left/' + str(idx) + '.NEF')