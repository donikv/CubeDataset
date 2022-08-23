import utils.sorting_utils as su
import utils.file_utils as fu
import os
import numpy as np
import cv2

if __name__ == '__main__':
    path = 'C:/Users/Donik/Desktop/to_process/'
    dirs = ['indoor']
    new_dirs=['realworld2']
    root = 'D:/fax/Cube2/'

    # dirs = list(map(lambda x: x+'_tiff', dirs))
    images_path = '/images'
    gt_path = '/gt'
    img1_path = '/img_corrected_1'
    gt_mask_path = '/gt_mask'
    save_locations = []

    for di, dir in enumerate(dirs):
        print(dir)
        image_names = os.listdir(path+dir+images_path)
        gts_old = np.loadtxt(f'{path+dir}/gt.txt')
        cube = np.loadtxt(f'{path+dir}/pos.txt').astype(int)
        i = 0
        for name in image_names:
            idx = int(name[:-4])

            gt = gts_old[idx-1]
            gtl, gtr = gt[:3], gt[3:]
            gtl = gtl / np.max(gtl)
            gtr = gtr / np.max(gtr)
            gt = np.concatenate([gtl, gtr], 0)
            cb_pos = cube[idx-1]

            x1, y1, x2, y2 = cb_pos

            img = fu.load_png(name, path+dir, images_path[1:], mask_cube=False)
            gt_img = fu.load_png(name, path+dir, gt_path[1:], mask_cube=False)
            img1 = fu.load_png(str(idx)+'.png', path+dir, img1_path[1:], mask_cube=False)
            # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            gt_mask = fu.load_png(str(idx)+'.png', path+dir, gt_mask_path[1:], mask_cube=False)

            images = {'img.png':img, 'gt.png':gt_img, 'img_corrected_1.png':img1, 'gt_mask.png':gt_mask}
            # images = {'img.png': img, 'gt.png': gt_mask}
            # camera = 'CANON_5DSR' if nm.startswith('C') else ('NIKON_D810' if nm.startswith('N') else 'SONY_IMX135')
            camera = su.NIKON
            save_location = su.save_for_dataset(images, gt, cb_pos, root, "indoor", camera, new_dirs[di], str(i+1))
            save_locations.append(save_location)
            i += 1

    f = open(os.path.join(root, 'list.txt'), 'a+')
    for item in save_locations:
        f.write(item+'\n')
        f.flush()
    f.close()