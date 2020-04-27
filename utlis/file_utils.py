import numpy as np
import cv2
import os
import rawpy


def load_png(name, path = './data/Cube+', directory='PNG_1_200', mask_cube=True):
    image = f"{path}/{directory}"
    image_path = os.path.join(image, name)

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    r, g, b = cv2.split(img)
    rgb = np.dstack((b, g, r))

    if mask_cube:
        for i in range(2000, rgb.shape[0]):
            for j in range(4000, rgb.shape[1]):
                rgb[i][j] = np.zeros(3)

    return rgb


def load_image(name, path = './data/Cube+', directory='CR2_1_100', mask_cube=True, depth=8):
    image = f"{path}/{directory}"
    image_path = os.path.join(image, name)

    raw = rawpy.imread(image_path)  # access to the RAW image
    # rgb = raw.postprocess()  # a numpy RGB array
    rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=depth)  # a numpy RGB array

    if mask_cube:
        for i in range(2000, rgb.shape[0]):
            for j in range(4000, rgb.shape[1]):
                rgb[i][j] = np.zeros(3)

    return rgb


def load(index, folder_step=100, mask_cube=False, depth=8):
    start = int((index - 1) / folder_step) * folder_step + 1
    end = min(int((index - 1) / folder_step) * folder_step + folder_step, 1707)
    print(start, end, index)
    if folder_step == 100:
        folder = f'CR2_{start}_{end}'
        rgb = load_image(f"{index}.CR2", directory=folder, mask_cube=mask_cube,
                         depth=depth)
        return rgb
    else:
        folder = f'PNG_{start}_{end}'
        rgb = load_png(f"{index}.PNG", directory=folder, mask_cube=mask_cube)
        return rgb