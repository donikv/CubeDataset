import utils.sorting_utils as su
import utils.file_utils as fu
import os
import numpy as np
import cv2
import utils.groundtruth_utils as gu

if __name__ == '__main__':
    path = '/home/donik/'
    dirs = ['relighted_tau']
    new_dirs=['relighted']
    root = '/home/donik/tau'

    # dirs = list(map(lambda x: x+'_tiff', dirs))
    images_path = '/images'
    gt_path = '/gt'
    img1_path = '/img_corrected_1'
    gt_mask_path = '/gt_mask'
    ill_path = '/ill'
    save_locations = []

    for di, dir in enumerate(dirs):
        print(dir)
        image_names = os.listdir(path+dir+images_path)
        i = 0
        for name in image_names:
            nm = name[:-4]

            cb_pos = np.array([0, 0, 0, 0], dtype=int)
            gt = np.loadtxt(path + dir + ill_path + '/' + nm + '.txt')
            x1, y1, x2, y2 = cb_pos

            img = fu.load_png(name, path+dir, images_path[1:], mask_cube=False)
            gt_img = fu.load_png(name, path+dir, gt_path[1:], mask_cube=False)
            img1 = fu.load_png(name, path+dir, img1_path[1:], mask_cube=False)
            # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            gt_mask = fu.load_png(name, path+dir, gt_mask_path[1:], mask_cube=False)

            images = {'img.png':img, 'gt.png':gt_img, 'img_corrected_1.png':img1, 'gt_mask.png':gt_mask}
            # images = {'img.png': img, 'gt.png': gt_mask}
            camera = 'CANON_5DSR' if nm.startswith('C') else ('NIKON_D810' if nm.startswith('N') else 'SONY_IMX135')
            save_location = su.save_for_dataset(images, gt, cb_pos, root, "images", camera, new_dirs[di], nm)
            save_locations.append(save_location)
            i += 1

    # f = open(os.path.join(root, 'list.txt'), 'a+')
    # for item in save_locations:
    #     f.write(item+'\n')
    #     f.flush()
    # f.close()