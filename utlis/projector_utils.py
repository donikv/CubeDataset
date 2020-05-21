import utlis.relighting_utils as ru
import utlis.image_utils as iu
import utlis.plotting_utils as pu

import numpy as np
import cv2
from skimage.filters import gaussian


def create_image(height, width, gradient, coloring_f):
    image = np.zeros((height, width, 3))
    colors = ru.random_colors()
    imgs = coloring_f(image, colors)
    return imgs


def line_image(image, colors):
    color_combs = [colors, (colors[0], np.zeros(3)), (np.zeros(3), colors[1])]
    images = [image.copy(), image.copy(), image.copy()]
    images_ret = []
    alpha = np.random.randint(1, 89, 1) * np.pi / 180
    alpha = alpha[0]
    for image, colors in zip(images, color_combs):
        image = iu.add_gradient(image, (0, 0), colors, alpha=alpha)
        images_ret.append(image.astype(np.uint8))
    return images_ret


def circle_image(image, colors):
    height, width, _ = image.shape
    color_combs = [colors, (colors[0], np.zeros(3)), (np.zeros(3), colors[1])]
    images = [image.copy(), image.copy(), image.copy()]
    images_ret = []
    for image, colors in zip(images, color_combs):
        image[:,:] = (colors[0] * 255).astype(np.uint8)
        color = (colors[1] * 255).astype(np.uint8)
        image = cv2.circle(image, (int(width/3), int(height)), int(width / 4), (int(color[0]), int(color[1]), int(color[2])), thickness=-1)
        images_ret.append(image.astype(np.uint8))
    return images_ret


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