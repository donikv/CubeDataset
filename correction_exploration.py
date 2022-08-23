import cv2
import utils.file_utils as fu
import utils.image_utils as iu
import utils.plotting_utils as pu
import utils.relighting_utils as ru
import numpy as np
import process_outdoor_multi as pom
import os


def get_white_side_correnction(gray_verts, img):
    def mirror_triangle(xa, ya, xb, yb, xc, yc):
        b = -(xa - xc)
        a = (ya - yc)
        c = (xa*yc - ya*xc)

        temp = -2 * (a * xb + b * yb + c) / (a * a + b * b)
        x = temp * a + xb
        y = temp * b + yb
        return np.array([xa,ya, x,y, xc,yc])

    # white_verts = mirror_triangle(*gray_verts)
    gt, mask = pom.get_gt_from_cube_triangle(gray_verts, img * 2**14, img.shape[0:2], return_mask=True)
    return gt, mask

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

path = '/Volumes/Jolteon/fax/raws6/organized'
l = np.loadtxt(f'{path}/list.txt', dtype=str)
paths = [path + x[1:] for x in l]
# paths = ['/media/donik/Disk/Cube2_new_/outdoor/nikon_d7000/outdoor1/39/', '/media/donik/Disk/Cube2_new_/outdoor/nikon_d7000/outdoor4/20/']


bsun = []
bshadow = []

for i, p in enumerate(paths):
    gts = np.loadtxt(p + '/gt.txt')
    verts = np.loadtxt(p + '/face_endpoints.txt')
    tris_corr = np.tile(np.array([-8, -10]), verts.shape[-1] // 2)
    verts = verts + tris_corr

    image = fu.load_png("img.png", path=p, directory='', mask_cube=False)
    gtm = fu.load_png("gt_mask.png", path=p, directory='', mask_cube=False)


    # gtm = (gtm - 1) * 255

    # gtmask = fu.load_png("gt_mask2.png", path=p, directory='', mask_cube=False)
    image = iu.process_image(image, 14, 0, scale=True)

    # for gt in gts:
    #     im = iu.color_correct_single_f32(image.copy(), gt)
    #     pu.visualize([adjust_gamma((image * 255).astype("uint8"), 2.2) , adjust_gamma((im * 255).astype("uint8"), 2.2), gtm], in_line=True)
    #     pu.visualize([np.broadcast_to(gt[np.newaxis, np.newaxis, :], (im.shape))])
    # pu.visualize([image])
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
    # images.append(gt_mask)
    # images.append(image * 4 / 3 / (gt_mask / gt_mask.sum(axis=-1, keepdims=True)))
    pu.visualize(images)
    print()
