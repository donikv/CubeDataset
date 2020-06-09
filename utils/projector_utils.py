import utils.relighting_utils as ru
import utils.image_utils as iu
import utils.plotting_utils as pu

import numpy as np
import cv2
from statistics import mode


def create_image(height, width, invert_colors, coloring_f):
    image = np.zeros((height, width, 3))
    colors = ru.random_colors(desaturate=False)
    imgs = []
    color_combs = [colors,
                   (colors[0], np.zeros(3)),
                   (np.zeros(3), colors[1]),
                   (colors[1], colors[0]),
                   (colors[1], np.zeros(3)),
                   (np.zeros(3), colors[0]),
                   # (np.ones(3), np.ones(3))
                   ]
    # reducer = 1
    # ambient = np.array([255, 215, 0]) / 255
    # color_combs = [(colors[0], ambient * reducer),
    #                (colors[0], np.zeros(3)),
    #                (np.zeros(3), ambient * reducer),
    #                (colors[1], ambient * reducer),
    #                (colors[1], np.zeros(3)),
    #                (np.zeros(3), ambient * reducer),
    #                ]
    images = [image.copy() for _ in color_combs]
    for image, colors in zip(images, color_combs):
        if invert_colors:
            colors = (colors[1], colors[0])
        image = coloring_f(image, colors)
        imgs.append(image)
    return imgs


def line_image(image, colors, alpha):
    image = iu.add_gradient(image, (0, 0), colors, alpha=alpha)
    return image.astype(np.uint8)


def circle_image(image, colors):
    height, width, _ = image.shape
    image[:,:] = (colors[0] * 255).astype(np.uint8)
    color = (colors[1] * 255).astype(np.uint8)
    image = cv2.circle(image, (int(width/3), int(height)), int(width / 4), (int(color[0]), int(color[1]), int(color[2])), thickness=-1)
    return image.astype(np.uint8)


def triangle_image(image, colors):
    height, width, _ = image.shape
    image[:, :] = (colors[0] * 255).astype(np.uint8)
    color = (colors[1] * 255).astype(np.uint8)
    p1 = [int(width / 3), int(height / 3)]
    p2 = [int(width / 2), int(height)]
    p3 = [int(width *2 / 3), int(height / 3)]
    pnts = np.array([[p1, p2, p3]])
    image = cv2.fillPoly(image, pnts, (int(color[0]), int(color[1]), int(color[2])))

    p1 = [int(width * 3 / 4), int(height / 3)]
    p2 = [int(width), int(height / 3)]
    p3 = [int(width), int(height)]
    p4 = [int(width * 3 / 4), int(height)]
    pnts = np.array([[p1, p2, p3, p4]])
    image = cv2.fillPoly(image, pnts, (int(color[0]), int(color[1]), int(color[2])))

    return image.astype(np.uint8)


def crop(image, rect, black_out=True):
    x, y, x2, y2 = rect
    width, height, _ = image.shape
    if black_out:
        image[:y, :] = np.zeros(3)
        image[y2:, :] = np.zeros(3)
        image[:, :x] = np.zeros(3)
        image[:, x2:] = np.zeros(3)
    else:
        image = image[y:y2, x:x2, :]
    return image

def create_gt_mask(image, image_right, image_left, gt_right, gt_left, allwhite=None):
    gt_left, gt_right = gt_left / gt_left.sum(), gt_right / gt_right.sum()
    # img_right_norm = iu.color_correct_single(image_right, gt_right, c_ill=1/np.sqrt(3))
    img_right_norm = (image_right * 3 / np.sqrt(3)).astype(np.uint8)
    img_right_norm = cv2.cvtColor(img_right_norm, cv2.COLOR_RGB2HLS) #dodano
    # img_right_norm[:, :, 1] = img_right_norm[:, :, 1] + 1
    thresh = 0#mode(image[:,:,1].reshape(-1, 1))
    img_right_norm[:, :, 1] = np.where(img_right_norm[:, :, 1] < thresh, 0, img_right_norm[:, :, 1])
    # img_left_norm = iu.color_correct_single(image_left, gt_left, c_ill=1/np.sqrt(3))
    img_left_norm = (image_left * 3 / np.sqrt(3)).astype(np.uint8)
    img_left_norm = cv2.cvtColor(img_left_norm, cv2.COLOR_RGB2HLS) #dodano
    # img_left_norm[:, :, 1] = img_left_norm[:, :, 1] + 1
    img_left_norm[:, :, 1] = np.where(img_left_norm[:, :, 1] < thresh, 0, img_left_norm[:, :, 1])
    r = img_right_norm / (img_left_norm + img_right_norm)
    r = r.clip(0, 1)
    # mn = r.mean(axis=2)
    # mn = np.where(mn >= 1/2, 1, 0)
    mn = r[:, :, 1]
    r[:, :, 0], r[:, :, 1], r[:, :, 2] = mn, mn, mn
    iab = np.nan_to_num(np.clip(r * gt_right + (1-r) * gt_left, 0, 1))
    return (iab * 255).astype(np.uint8), img_right_norm, img_left_norm, r

def denoise_mask(mask):
    mask_cls = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return mask_cls