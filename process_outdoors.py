import cv2
import numpy as np
import os
from skimage.filters import gaussian

import utils.image_utils as iu
import utils.file_utils as fu
import utils.relighting_utils as ru
import utils.projector_utils as pu
import utils.plotting_utils as plt


def load_and_get_gt(path, idx, tiff):
    name = str(idx+1)
    if not tiff:
        im = fu.load_cr2(name+'.NEF', path, directory='', mask_cube=False)
    else:
        im = load_tiff(name+'.tiff', path, directory='')

    im = iu.process_image(im, depth=14, scale=True, blacklevel=0)

    x1, y1, x2, y2 = np.loadtxt(path+'/pos.txt').astype(int)[idx]
    if tiff:
        x1 = x1 // 2
        x2 = x2 // 2
        y1 = y1 // 2
        y2 = y2 // 2
    gt1 = im[y1-3:y1+3, x1-3:x1+3].mean(axis=1).mean(axis=0)
    gt1 = np.clip(gt1, 0.001, 1)
    gt2 = im[y2-3:y2+3, x2-3:x2+3].mean(axis=1).mean(axis=0)
    gt2 = np.clip(gt2, 0.001, 1)

    return im, gt1, gt2, np.array([x1, y1, x2, y2])


def color_mask(path, idx, size=None, gts=None):
    name = str(idx+1) + 'm'
    mask = fu.load_png(name+'.png', path, '', mask_cube=False)
    if size != None:
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    if gts is None:
        idx = idx if idx < 7 else idx - 1
        gts = np.loadtxt(path + '/gt.txt')[idx].reshape(2,3)

    cm = (np.where(mask == 0, gts[1], gts[0]) * 255).astype(np.uint8)
    return cm

def load_tiff(img, path, directory):
    image_tiff = cv2.imread(f'{path}/{directory}/{img}', cv2.IMREAD_UNCHANGED)
    # imageRaw = cv2.cvtColor(image_tiff, cv2.COLOR_BAYER_RG2BGR)
    imageRaw = iu.debayer(image_tiff)
    return imageRaw


def correct_with_mask(path, idx):
    img = cv2.imread(path + '/images/'+str(idx+1) + '.png', cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 2**16
    mask = cv2.imread(path + '/gt/' + str(idx + 1) + '.png', cv2.IMREAD_UNCHANGED)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) / 2**8
    mask = gaussian(mask, 15)

    imgc = np.multiply(img, 1/(1.713*mask))
    return imgc, img, mask

if __name__ == '__main__':
    import time
    start = time.time_ns()
    path = '/Volumes/Jolteon/fax/raws5_night'
    idx = 3
    sizes = []
    os.makedirs(path + '/organized', exist_ok=True)

    for idx in range(0, 20):
        i_path = path + '/organized' + f'/{idx+1}'
        os.makedirs(i_path, exist_ok=True)
        try:
            im, gt1, gt2, pos = load_and_get_gt(path, idx, tiff=True)
        except:
            continue
        gt1 = gt1 / gt1.sum()
        gt2 = gt2 / gt2.sum()
        size = tuple(reversed(im.shape[0:2]))
        sizes.append(size)
        
        np.savetxt(i_path + '/gt.txt', np.concatenate([gt1, gt2], axis=-1).reshape((1, -1)))
        np.savetxt(i_path + '/cube.txt', pos.reshape((1, -1)), fmt="%d")

        gt = color_mask(path, idx, size=size, gts=[gt1, gt2])
        gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
        cv2.imwrite(i_path + f'/gt.png', gt)

        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        # gt = color_mask(path, idx, (gt1, gt2))
        # gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
        cv2.imwrite(i_path + f'/img.png', (im * 2**14).astype(np.uint16))
        # cv2.imwrite(path + f'/gt/{idx + 1}.png', gt)
        
        gt_mask = cv2.imread(path + f'/{idx+1}m.png', cv2.IMREAD_UNCHANGED)
        gt_mask = cv2.resize(gt_mask, size, interpolation=cv2.INTER_NEAREST)
        gt_mask = np.where(gt_mask < 128, 0, 255)#cv2.threshold(gt_mask, 128, 255, cv2.THRESH_BINARY)
        cv2.imwrite(i_path + f'/gt_mask.png', gt_mask)

    end = time.time_ns()
    print((end - start) // 10**6)

#         f = open(path+'/gt.txt', 'a+')
#         f.write(f'{gt1[0]} {gt1[1]} {gt1[2]} {gt2[0]} {gt2[1]} {gt2[2]}\n')
#         f.close()
#     gts = np.loadtxt(path + '/gt.txt').reshape(-1, 2, 3)
#     #
#     for idx in range(0, 42):
#         gt = color_mask(path, idx, size=sizes[idx], gts=None)
#         gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
#         cv2.imwrite(path + f'/gt/{idx + 1}.png', gt)
# 
#     # for idx in range(0, 34):
#         imgc, img, mask = correct_with_mask(path, idx)
#         imgc1 = iu.color_correct_single_f32(img, gts[idx][0], c_ill=1/3)
#         # plt.visualize([img, imgc, mask, imgc1])
#         # imgc = imgc.astype(np.float32)
#         # imgc = cv2.cvtColor(imgc, cv2.COLOR_RGB2BGR)
#         # cv2.imwrite(path + f'/corrected/{idx + 1}.png', imgc * 255)
# 
#         gt_mask = cv2.imread(path + f'/{idx+1}m.png', cv2.IMREAD_UNCHANGED)
#         gt_mask = cv2.resize(gt_mask, sizes[idx], interpolation=cv2.INTER_NEAREST)
#         gt_mask = np.where(gt_mask < 128, 0, 255)#cv2.threshold(gt_mask, 128, 255, cv2.THRESH_BINARY)
#         imgc1 = imgc1.astype(np.float32)
#         imgc1 = cv2.cvtColor(imgc1, cv2.COLOR_RGB2BGR)
#         cv2.imwrite(path + f'/img_corrected_1/{idx+1}.png', (imgc1*65535).astype(np.uint16))
#         cv2.imwrite(path + f'/gt_mask/{idx+1}.png', gt_mask)
