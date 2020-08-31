import utils.sorting_utils as su
import utils.file_utils as fu
import os
import numpy as np

if __name__ == '__main__':
    path = 'G:\\fax\\diplomski\\Datasets\\third\\'
    dirs = ['ambient', 'ambient3', 'ambient4', 'processed']
    new_dirs=['lab1', 'lab3','lab4','labp']
    root = 'D:/fax/Cube2'

    dirs = list(map(lambda x: x+'_tiff', dirs))
    images_path = '/both/images'
    gt_path = '/gt'
    img1_path = '/img_corrected_1'
    gt_mask_path = '/gt_mask'
    save_locations = []

    for di, dir in enumerate(dirs):
        print(dir)
        image_names = os.listdir(path+dir+images_path)
        gts_oldl = np.loadtxt(f'{path+dir}/gt_left.txt')
        gts_oldr = np.loadtxt(f'{path + dir}/gt_right.txt')
        cube = np.loadtxt(f'{path+dir}/cube.txt').astype(int)
        for i, name in enumerate(image_names):
            idx = int(name[:-4]) - 1

            gtl = gts_oldl[idx]
            gtl = gtl / np.max(gtl)
            gtr = gts_oldr[idx]
            gtr = gtr / np.max(gtr)
            gt = np.concatenate([gtl, gtr], 0)
            cb_pos = cube[idx]

            x1, y1, x2, y2 = cb_pos

            img = fu.load_png(name, path+dir, images_path[1:], mask_cube=False)
            gt_img = fu.load_png(name, path+dir, gt_path[1:], mask_cube=False)
            img1 = fu.load_png(name, path+dir, img1_path[1:], mask_cube=False)
            gt_mask = fu.load_png(name, path+dir, gt_mask_path[1:], mask_cube=False)

            images = {'img.png':img, 'gt.png':gt_img, 'img_corrected_1.png':img1, 'gt_mask.png':gt_mask}
            # images = {'img.png': img, 'gt.png': gt_mask}
            save_location = su.save_for_dataset(images, gt, cb_pos, root, "lab", su.CANON, new_dirs[di], str(i))
            save_locations.append(save_location)

    f = open(os.path.join(root, 'list.txt'), 'a+')
    for item in save_locations:
        f.write(item+'\n')
        f.flush()
    f.close()