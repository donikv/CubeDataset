from math import sqrt
import cv2
import numpy as np

import utils.file_utils as fu
import utils.plotting_utils as pu
import utils.image_utils as iu
import utils.groundtruth_utils as gu

# PLOT SINGLE ILLUMINANT CORRECTED
# path = 'C:\\Users\\donik\\Dropbox\\Donik\\fax\\10_semestar\\Diplomski\\MultiIlluminant-Utils\\data\\test\\whatsapp\\images'
# idx = 1071
# name = '4eec980e-72ca-4150-ab4b-dd48592c3bb9'
# image = fu.load_png(f'{name}.jpg', path, '', landscape=True)
# image = cv2.resize(image, (0, 0), fx=1 / 5, fy=1 / 5)
# # image = iu.process_image(image, 14)
# # gt = gu.GroundtruthLoader('cube+_gt.txt', path)
#
# image_cor = iu.color_correct_single(image, np.array([175/255, 78/255, 36/255]), c_ill=1/sqrt(3))
# image_cor2 = iu.color_correct_single(image, np.array([174/255, 186/255, 191/255]), c_ill=1/sqrt(3))
# image_cor3 = iu.color_correct_single(image, np.array([255/255, 244/255, 113/255]) / 1.7, c_ill=1/sqrt(3))
# pu.visualize([image, image_cor, image_cor2, image_cor3], titles=['a)', 'b)', 'c)', 'd)'], out_file='./images/corrected_single.png', in_line=True)

# PLOT COMMON MULTI ILLUMINANT
# path = 'D:\\fax\\diplomski\\Datasets\\Cube+'
# path2 = 'C:\\Users\\donik\\Downloads'
# name2 = 'WhatsApp Image 2020-05-08 at 09.47.51'
# name = 'WhatsApp Image 2020-05-05 at 13.03.59'
# image3 = image
# image = fu.load_png(f'{name2}.jpg', path2, '', landscape=False)
# image = cv2.resize(image, (int(1600/5), int(1200/5)))
# # image = iu.process_image(image, 14)
# # gt = gu.GroundtruthLoader('cube+_gt.txt', path)
# # image = iu.color_correct_single(image, gt[idx - 1], c_ill=1/sqrt(3))
# image2 = fu.load_png(f'{name}.jpeg', path2, '', landscape=False)
# image2 = cv2.resize(image2, (0, 0), fx=1/5, fy=1/5)
#
#
# pu.visualize([image, image2, image3], titles=['a)', 'b)', 'c)', ], out_file='./images/common_multi.png', in_line=True)

# PLOT EXAMPLES
# path = '../MultiIlluminant-Utils/data/dataset_relighted'
# names_mask = [
#     f'complex/gt_mask/141.png',
#     f'complex/gt_mask/133.png',
#     f'gt_mask/201-inv-gray.png',
#     f'gt_mask/100-tresh-1.png',
#     f'complex4/gt_mask/171-4.png',
#     f'complex5/gt_mask/2-25-5-edge.png',
#     f'complex6/gt_mask/32-6-tresh.png',
# ]
#
# names_relighted = [
#     f'complex/images/141.png',
#     f'complex/images/133.png',
#     f'images/201-inv-gray.png',
#     f'images/100-tresh-1.png',
#     f'complex4/images/171-4.png',
#     f'complex5/images/25-5-edge.png',
#     f'complex6/images/32-6-tresh.png',
# ]
#
#
# def get_image(x):
#     x = fu.load_png(x, path, directory='', mask_cube=False)
#     return x
#
#
# images_good = list(map(get_image, names_mask))
# images_bad = list(map(get_image, names_relighted))
# pu.visualize(images_good, titles=['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)'], out_file='./images/examples_masks.png', in_line=True)
# pu.visualize(images_bad, titles=['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)'], out_file='./images/examples_relighted.png', in_line=True)

#PLOT CORRECTIONS
path = '../MultiIlluminant-Utils/data/dataset_relighted/valid'
folders = ['img_corrected_1', 'gt_mask', 'pmasks2', 'pmasks']
# path2 = 'C:\\Users\\donik\\Downloads'
names = ['2-815-5.png', '2-810-5-edge.png', '2-930-5.png']


def get_image(x, folder):
    x = fu.load_png(x, path, directory=folder, mask_cube=False)
    return x

for name in names:
    images = []
    for folder in folders:
        img = get_image(name, folder)
        images.append(img)
    pu.visualize(images, titles=['a)', 'b)', 'c)', 'd)'], out_file=f'./images/corrected_models3{name}.png', in_line=True)
# pu.visualize(images_bad, titles=['a)', 'b)', 'c)', 'd)'], out_file='./images/examples_bad.png', in_line=True)
