import cv2
import utils.file_utils as fu
import utils.image_utils as iu
import utils.plotting_utils as pu
import utils.relighting_utils as ru
import numpy as np
import process_outdoor_multi as pom
import os

folders = [
'/Volumes/Jolteon/fax/raws6/organized',
'/Volumes/Jolteon/fax/raws1/organized',
'/Volumes/Jolteon/fax/raws2/organized',
'/Volumes/Jolteon/fax/raws3/organized',
'/Volumes/Jolteon/fax/raws4/nikon/organized'
]
folders = [
'/Volumes/Jolteon/fax/to_process/organized',
]

path = '/Volumes/Jolteon/fax/raws6/organized'
# paths = ['/media/donik/Disk/Cube2_new_/outdoor/nikon_d7000/outdoor1/39/', '/media/donik/Disk/Cube2_new_/outdoor/nikon_d7000/outdoor4/20/']

for path in folders:
    l = np.loadtxt(f'{path}/list.txt', dtype=str)
    paths = [path + x[1:] for x in l]
    bsun = []
    bshadow = []

    for i, p in enumerate(paths):
        try:
            gts = np.loadtxt(p + '/gt.txt')
        except:
            continue

        image = fu.load_png("img.png", path=p, directory='', mask_cube=False)
        gtm = fu.load_png("gt_mask.png", path=p, directory='', mask_cube=False)

        imgb = np.linalg.norm(image, ord=2, axis=-1)

        sun = np.ma.masked_where(gtm[...,0] == 0, imgb)
        shadow = np.ma.masked_where(gtm[...,0] != 0, imgb)
        bsun.append(np.ma.mean(sun))
        bshadow.append(shadow.mean())

    msun = np.array(bsun)
    mshadow = np.array(bshadow)
    ratio = msun.mean() / mshadow.mean()
    print(ratio)
