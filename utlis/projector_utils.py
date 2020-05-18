import utlis.relighting_utils as ru
import utlis.image_utils as iu
import utlis.plotting_utils as pu

import numpy as np
import cv2
from skimage.filters import gaussian


def create_image(height, width, gradient, coloring_f):
    image = np.zeros((height, width, 3))
    colors = ru.random_colors()
    image = coloring_f(image, colors)
    return gaussian(image, gradient)


def line_image(image, colors):
    return iu.add_gradient(image, (0, 0), colors)


def circle_image(image, colors):
    height, width, _ = image.shape
    image[:,:] = (colors[0] * 255).astype(np.uint8)
    color = (colors[1] * 255).astype(np.uint8)
    cv2.circle(image, (int(width/4 * 3), int(height/2)), int(width / 10), (int(color[0]), int(color[1]), int(color[2])), thickness=-1)
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