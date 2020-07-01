import utils.relighting_utils as ru
import utils.image_utils as iu
import utils.plotting_utils as pu

import numpy as np
import cv2
from skimage.filters import gaussian


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
    color_combs = [
                   (np.array([0,1,0]), np.zeros(3)),
                   (np.zeros(3), np.array([0,1,0])),
                   (np.array([0,1,0]), np.array([0,1,0])),
                   # (np.ones(3), np.ones(3))
                   ]
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


def blur_image(image, colors):
    height, width, _ = image.shape
    color1 = (colors[0] * 255).astype(np.uint8)
    color2 = (colors[1] * 255).astype(np.uint8)
    r = image / 255
    image[:, :] = np.clip(r[:,:] * color1 + (1 - r[:, :]) * color2, 0, 255)

    return image.astype(np.uint8)


def blur_image2(image, colors):
    height, width, _ = image.shape
    image[:, :] = (colors[0] / 3 * 255).astype(np.uint8)
    color = (colors[1] / 3 * 255).astype(np.uint8)
    pnts = [
        [0, 0],
        [int(width / 2), 0],
        [int(width / 2), height - 1],
        [0, height - 1]
    ]
    image = cv2.fillPoly(image, np.array([pnts]), (int(color[0]), int(color[1]), int(color[2])))

    return gaussian(image, 130).astype(np.uint8)


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


def create_gt_mask(image, image_right, image_left, gt_right, gt_left, allwhite=None, thresh=0):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    gt_left, gt_right = gt_left / gt_left.sum(), gt_right / gt_right.sum()
    img_right_norm = (image_right * 3 / np.sqrt(3)).astype(np.uint8)
    img_right_norm = cv2.cvtColor(img_right_norm, cv2.COLOR_RGB2HLS) #dodano


    img_right_norm[:, :, 1] = np.where(img_right_norm[:, :, 1] < thresh, 0, img_right_norm[:, :, 1])
    img_left_norm = (image_left * 3 / np.sqrt(3)).astype(np.uint8)
    img_left_norm = cv2.cvtColor(img_left_norm, cv2.COLOR_RGB2HLS) #dodano
    img_left_norm[:, :, 1] = np.where(img_left_norm[:, :, 1] < thresh, 0, img_left_norm[:, :, 1])

    r = img_right_norm / (img_left_norm + img_right_norm)
    r = r.clip(0, 1)
    mn = r[:, :, 1]
    r[:, :, 0], r[:, :, 1], r[:, :, 2] = mn, mn, mn
    iab = np.nan_to_num(np.clip(r * gt_right + (1-r) * gt_left, 0, 1))
    iab = (iab * 255).astype(np.uint8)
    iab = cv2.cvtColor(iab, cv2.COLOR_RGB2HLS)
    iab[:, :, 1] = np.where(image[:, :, 1] != 0, iab[:,:,1], 0)
    iab = cv2.cvtColor(iab, cv2.COLOR_HLS2RGB)
    return iab, img_right_norm, img_left_norm, r


def denoise_mask(mask):
    mask_cls = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return mask_cls


def combine_two_images(image_right, image_left, thresh=0):
    img_right_norm = (image_right).astype(np.uint8)
    img_right_norm = cv2.cvtColor(img_right_norm, cv2.COLOR_RGB2HLS) #dodano
    img_right_norm[:, :, 1] = np.where(img_right_norm[:, :, 1] < thresh, 0, img_right_norm[:, :, 1])

    img_left_norm = (image_left).astype(np.uint8)
    img_left_norm = cv2.cvtColor(img_left_norm, cv2.COLOR_RGB2HLS) #dodano
    img_left_norm[:, :, 1] = np.where(img_left_norm[:, :, 1] < thresh, 0, img_left_norm[:, :, 1])

    r = img_right_norm / (img_left_norm + img_right_norm)
    r = r.clip(0, 1)
    mn = r[:, :, 1]
    r[:, :, 0], r[:, :, 1], r[:, :, 2] = mn, mn, mn
    iab = np.nan_to_num(np.clip(r * image_right + (1-r) * image_left, 0, 255))
    iab = iab.astype(np.uint8)
    return iab #, img_right_norm, img_left_norm, r