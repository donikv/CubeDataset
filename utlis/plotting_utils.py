import math

import matplotlib.pyplot as plt

def visualize(images, custom_transform=lambda x: x):
    """
    Plot image and masks.
    If two pairs of images and masks are passes, show both.
    """

    rows = math.ceil((len(images) / 2))
    cols = 2
    f, ax = plt.subplots(rows, cols, figsize=(30, 30), squeeze=False)
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
