import cv2
import utils.file_utils as fu
import utils.image_utils as iu
import utils.plotting_utils as pu
import utils.relighting_utils as ru
import numpy as np
import process_outdoor_multi as pom


def get_white_side_correnction(gray_verts, img):
    def mirror_triangle(xa, ya, xb, yb, xc, yc):
        b = -(xa - xc)
        a = (ya - yc)
        c = (xa*yc - ya*xc)

        temp = -2 * (a * xb + b * yb + c) / (a * a + b * b)
        x = temp * a + xb
        y = temp * b + yb
        return np.array([xa,ya, x,y, xc,yc])

    white_verts = mirror_triangle(*gray_verts)
    gt, mask = pom.get_gt_from_cube_triangle(white_verts, img * 2**14, img.shape[0:2], return_mask=True)
    return gt, mask

path = '/Volumes/Jolteon/fax/to_process/organized'
l = np.loadtxt(f'{path}/list.txt', dtype=str)
paths = [path + x[1:] for x in l]


for p in paths:
    gts = np.loadtxt(p + '/gts.txt')
    verts = np.loadtxt(p + '/cube.txt')
    image = fu.load_png("img.png", path=p, directory='', mask_cube=False)
    image = iu.process_image(image, 14, 0, scale=True)
    gt_mask = cv2.cvtColor(cv2.imread(p + '/gt.png', cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB)
    gt_mask = cv2.GaussianBlur(gt_mask, (51,51), 100)
    images = []
    print(p)
    for gt, vert in zip(gts, verts):
        im = iu.color_correct_single_f32(image.copy(), gt)
        corrected_white, mask = get_white_side_correnction(vert, im)
        im = np.ma.masked_where(mask, im).filled(0)
        print(corrected_white, ru.angular_distance(np.ones_like(corrected_white), corrected_white))
        images.append(im * 4)
    images.append(gt_mask)
    images.append(image * 4 / 3 / (gt_mask / gt_mask.sum(axis=-1, keepdims=True)))
    pu.visualize(images)
    print()
