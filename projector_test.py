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

def saveImage(image, dir, name, convert_rgb_2_bgr=True, scale=True):
    if not os.path.exists(dir):
        os.mkdir(dir)
    pth = os.path.join(dir, f'{name}.png')
    if convert_rgb_2_bgr:
        r, g, b = cv2.split(image)
        image = np.dstack((b, g, r))
    if image.max() < 2:
        image = (image * 65535).astype(np.uint16)
    cv2.imwrite(pth, image)

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


def create_cube_txt(dir, images):
    pos = []
    for idx, img in enumerate(images):
        if dir.endswith('projector2'):
            x1, y1, x2, y2 = 1019, 1902, 4256, 1477
        elif dir.endswith('ambient5') or dir.endswith('ambient5_tiff'):
            x1,y1,x2,y2 = 669, 2442, 4885, 2450
        elif dir.endswith('ambient') or dir.endswith('ambient_tiff'):
            x1, y1, x2, y2 = 1263, 2004, 3318, 2112 #ambient
        elif dir.endswith('ambient3_tiff'):
            if idx >= 60:
                x1, y1 = 720, 1850
                x2, y2 = 4785, 1737
            elif idx >= 40:  # ambient3_tiff
                x1, y1 = 965, 1639
                x2, y2 = 4904, 1495
            else:
                x1, y1 = 960, 1661  # 1379, 2336 #3241, 1955#3861, 1912#1200, 2450
                x2, y2 = 4585, 1576  # 3803, 2481#3092, 1974#1368, 2217 #1350, 2450
        elif dir.endswith('ambient3'):
            if idx >= 40: #ambient3
                x1, y1 = 928, 1844
                x2, y2 = 4659, 1762
            elif idx >= 20:
                x1, y1 = 720, 1850
                x2, y2 = 4785, 1737
            else:
                x1, y1 = 770, 1672#1379, 2336 #3241, 1955#3861, 1912#1200, 2450
                x2, y2 = 4723, 1535#3803, 2481#3092, 1974#1368, 2217 #1350, 2450
        pos.append([x1, y1, x2, y2])
    np.savetxt(f'{dir}/cube.txt', pos, fmt='%d')

def find_gt(dir):
    dir_name = f'{dir}/both'
    dir_name_left = f'{dir}/left'
    dir_name_right = f'{dir}/right'
    images = os.listdir(dir_name)
    images = list(filter(lambda x: str(x).lower().endswith('.tiff'), images))
    # images_del = list(filter(lambda x: str(x).lower().startswith('cup-'), images))
    # images = list(filter(lambda x: not str(x).lower().startswith('cup-'), images))

    if not os.path.exists(f'{dir}/cube.txt'):
        create_cube_txt(dir, images)
    gray_pos = np.loadtxt(f'{dir}/cube.txt').astype(int)

    if dir.find('ambient') >= 0:
        idx_offsets = (2, 1)
    else:
        idx_offsets = (1, 3)

    gts_left, gts_right = [], []
    for idx, img in enumerate(images):
        img_idx = int(img[-6])
        img_name_base = img[:-6]
        image = fu.load_tiff(img, dir_name, '')
        image = iu.process_image(image, depth=14, blacklevel=0, scale=True)
        right = img_name_base + str(img_idx+idx_offsets[0]) + '.tiff' if dir.find('projector2') == -1 else img
        left = img_name_base + str(img_idx+idx_offsets[1]) + '.tiff' if dir.find('projector2') == -1 else img
        image_r = fu.load_tiff(right, dir_name_right, '')
        image_l = fu.load_tiff(left, dir_name_left, '')
        image_l = iu.process_image(image_l, depth=14, blacklevel=0, scale=True)
        image_r = iu.process_image(image_r, depth=14, blacklevel=0, scale=True)
        gt_left, gt_right = np.zeros(3), np.zeros(3)
        n = 20
        if dir.find('ambient') >= 0:
            x2, y2, x1, y1 = gray_pos[idx]
        else:
            x1, y1, x2, y2 = gray_pos[idx]
        for i in range(n):
            for j in range(n):
                gt_left = gt_left + np.clip(image_l[y1 + i, x1 + j], 0, 1)
                gt_right = gt_right + np.clip(image_r[y2 + i, x2 + j], 0, 1)
        gt_left, gt_right = gt_left / n / n, gt_right / n / n
        # image = cv2.resize(image, (0, 0), fx = 1/5, fy = 1/5)
        gts_left.append(gt_left)
        gts_right.append(gt_right)
        saveImage(image, f'{dir}/both/images', idx+1, True)
        saveImage(image_l, f'{dir}/left/images', idx+1, True)
        saveImage(image_r, f'{dir}/right/images', idx+1, True)
        print(idx+1)
    np.savetxt(f'{dir}/gt_left.txt', np.array(gts_left, dtype=np.float32))
    np.savetxt(f'{dir}/gt_right.txt', np.array(gts_right, dtype=np.float32))



def par_create(data):
    idx, img, gts_left, gts_right, dir, tresh = data
    # dir = 'G:\\fax\\diplomski\\Datasets\\third\\ambient2'
    dir_name = f'{dir}/both/images'
    dir_name_left = f'{dir}/left/images'
    dir_name_right = f'{dir}/right/images'

    image = fu.load_png(img, dir_name, '', mask_cube=False) / 255
    image_left = fu.load_png(img, dir_name_left, '', mask_cube=False) / 255
    image_right = fu.load_png(img, dir_name_right, '', mask_cube=False) / 255
    # image = cv2.resize(image, (0, 0), fx = 1/5, fy = 1/5)
    # image_left = cv2.resize(image_left, (0, 0), fx=1 / 5, fy=1 / 5)
    # image_right = cv2.resize(image_right, (0, 0), fx=1 / 5, fy=1 / 5)
    gt_left = np.clip(gts_left[idx], 0.01, 1)
    gt_left[0], gt_left[2] = gt_left[2], gt_left[0]
    gt_right = np.clip(gts_right[idx], 0.01, 1)
    gt_right[0], gt_right[2] = gt_right[2], gt_right[0]
    gt_mask, ir, il, r = pu.create_gt_mask(image, image_right, image_left, gt_right, gt_left, thresh=tresh)
    ggt_mask = gt_mask / 255#pu.denoise_mask(gt_mask)
    saveImage(ggt_mask, f'{dir}/gt_mask/', idx + 1, False)
    # gt_mask, ir, il, r = pu.create_gt_mask(image, image_right, image_left, gt_left, gt_right)
    # cv2.imwrite(f'D:\\fax\\Dataset\\ambient/pngs/gt_mask/{idx + 1}lr.png', cv2.cvtColor(gt_mask, cv2.COLOR_RGB2BGR))


def create_gt_mask(dir, thresh):
    dir_name = f'{dir}/both/images'
    images = os.listdir(dir_name)
    images = list(filter(lambda x: str(x).lower().endswith('.png'), images))
    # if dir.find('processed') > -1:
    #     gts_right = np.loadtxt(f'{dir}/gt_left.txt')
    #     gts_left = np.loadtxt(f'{dir}/gt_right.txt')
    # else:
    gts_left = np.loadtxt(f'{dir}/gt_left.txt')
    gts_right = np.loadtxt(f'{dir}/gt_right.txt')
    data = list(map(lambda x: (int(x[:-4])-1, x, gts_left, gts_right, dir, thresh), images))

    num_proc = 8

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


def combine_for_training(fax, tiff, append):
    path = 'G:\\fax\\diplomski\\Datasets\\third\\' if not fax else '/media/donik/Jolteon/fax/diplomski/Datasets/third/'
    dirs = ['ambient', 'ambient3', 'ambient4', 'processed', 'realworld', 'ambient6']
    # path = 'G:\\fax\\diplomski\\Datasets\\' if not fax else '/media/donik/Jolteon/fax/diplomski/Datasets/third/'
    # dirs = ['third\\realworld', 'realworld']
    if tiff:
        dirs = list(map(lambda x: x+'_tiff', dirs))
    images_path = '/relighted/images'
    gt_path = '/relighted/gt'
    img1_path = '/relighted/img_corrected_1'
    gt_mask_path = '/relighted/gt_mask'

    if tiff:
        dest = 'G:\\fax\\diplomski\\Datasets\\third\\combined_tiff_relighted\\' if not fax else '/media/donik/Disk/combined_tiff/'
    else:
        # dest = 'G:\\fax\\diplomski\\Datasets\\realworld_combined\\' if not fax else '/media/donik/Disk/combined/'
        dest = 'G:\\fax\\diplomski\\Datasets\\third\\combined\\' if not fax else '/media/donik/Disk/combined/'

    gts = []
    pos = []
    current_relighted_images = os.listdir(dest+'images/')
    current_relighted_idxs = list(map(lambda x: int(x[:-4]), current_relighted_images))
    name_idx = sorted(current_relighted_idxs)[-1] if len(current_relighted_idxs) > 0 else 0
    name_idx = name_idx + 1
    if append and os.path.exists(dest + 'images'):
        current_images = os.listdir(dest + 'images')
        current_idxs = list(map(lambda x: int(x[:-4]), current_images))
        name_idx = max(current_idxs) + 1
    for dir in dirs:
        print(dir)
        image_names = os.listdir(path+dir+images_path)
        gts_old = np.loadtxt(f'{path+dir}/gt.txt')
        cube = np.loadtxt(f'{path+dir}/cube.txt')
        for name in image_names:
            idx = int(name[:-4]) - 1

            gt = gts_old[idx]
            gts.append(gt)
            # cb_pos = cube[idx]
            # pos.append(cb_pos)

            # img = fu.load_png(name, path+dir, images_path[1:], mask_cube=False)
            # gt = fu.load_png(name, path+dir, gt_path[1:], mask_cube=False)
            # img1 = fu.load_png(name, path+dir, img1_path[1:], mask_cube=False)
            # gt_mask = fu.load_png(name, path+dir, gt_mask_path[1:], mask_cube=False)
            # saveImage(img, dest+'images', str(name_idx), True)
            # saveImage(gt, dest + 'gt', str(name_idx), True)
            # saveImage(img1, dest+'img_corrected_1', str(name_idx), True)
            # saveImage(gt_mask, dest + 'gt_mask', str(name_idx), True)

            os.rename(path+dir+images_path+"/"+name, dest+'images/'+str(name_idx) + '.png')
            os.rename(path + dir + gt_path + "/" + name, dest + 'gt/' + str(name_idx) + '.png')
            os.rename(path + dir + img1_path + "/" + name, dest + 'img_corrected_1/' + str(name_idx) + '.png')
            os.rename(path + dir + gt_mask_path + "/" + name, dest + 'gt_mask/' + str(name_idx) + '.png')

            name_idx += 1
    np.savetxt(f'{dest}/gts.txt', np.array(gts, dtype=np.float32))
    np.savetxt(f'{dest}/pos.txt', np.array(cube, dtype=int), fmt='%d')


if __name__ == '__main__':
    fax = os.name != 'nt'
    combine_for_training(fax, True, False)
    exit()
    # if fax:
    #     dir = "/media/donik/Jolteon/fax/diplomski/Datasets/third/processed_tiff"
    # else:
    #     dir = "projector_test/projector2"
    # find_gt(dir)
    # tresh = 1 if dir.endswith('tiff') else 0
    # create_gt_mask(dir, thresh=tresh)


