import utlis.relighting_utils as ru
import utlis.image_utils as iu
import utlis.plotting_utils as pu

import numpy as np
import cv2
from skimage.filters import gaussian


def create_image(height, width, gradient, coloring_f):
    image = np.zeros((height, width, 3))
    colors = ru.random_colors(desaturate=False)
    imgs = []
    color_combs = [colors, (colors[0], np.zeros(3)), (np.zeros(3), colors[1])]
    images = [image.copy(), image.copy(), image.copy()]
    for image, colors in zip(images, color_combs):
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