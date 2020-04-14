import math

import matplotlib.pyplot as plt
import cv2
import numpy as np


def visualize(images, custom_transform=lambda x: x, title=None):
    """
    Plot image and masks.
    If two pairs of images and masks are passes, show both.
    """

    rows = math.ceil((len(images) / 2))
    cols = 2 if len(images) > 1 else 1
    f, ax = plt.subplots(rows, cols, figsize=(30, 30), squeeze=False)
    if title is not None:
        f.suptitle(str(title), fontsize=64)
    for idx, img in enumerate(images):
        ax[int(idx / 2)][idx % 2].imshow(custom_transform(img))
    plt.show()


def plot_counturs(img, init, snake):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img)
    ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])

    plt.show()


def draw_number(number):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (25, 25)
    fontScale = 1
    color = (255, 255, 255)
    thickness = 1

    # Using cv2.putText() method
    image = np.zeros((50, 100, 3), np.uint8)
    image = cv2.putText(image, str(number), org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    return image