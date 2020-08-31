import numpy as np
import cv2
import os
import rawpy
import shutil


def load_cr2(name, path = 'D:\\fax\\Dataset\\ambient', directory='PNG_1_200', mask_cube=True):
    image = f"{path}/{directory}"
    image_path = os.path.join(image, name)

    img = rawpy.imread(image_path)
    rgbg = img.raw_image_visible
    rgb = cv2.cvtColor(rgbg, cv2.COLOR_BAYER_BG2RGB)
    # rgb = img.postprocess(gamma=(1,0), no_auto_bright=True, output_bps=16, use_camera_wb=False, use_auto_wb=False)

    return rgb

def load_png(name, path = './data/Cube+', directory='PNG_1_200', mask_cube=True, landscape=True):
    image = f"{path}/{directory}"
    image_path = os.path.join(image, name)

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    r, g, b = cv2.split(img)
    rgb = np.dstack((b, g, r))

    if mask_cube:
        for i in range(2000, rgb.shape[0]):
            for j in range(4000, rgb.shape[1]):
                rgb[i][j] = np.zeros(3)

    if rgb.shape[0] > rgb.shape[1] and landscape:
        rgb = rgb.transpose((1, 0, 2))
        rgb = cv2.flip(rgb, 0)

    return rgb


def load_tiff(img, path, directory, bayer=cv2.COLOR_BAYER_RG2BGR):
    image_tiff = cv2.imread(f'{path}/{directory}/{img}', cv2.IMREAD_UNCHANGED)
    imageRaw = cv2.cvtColor(image_tiff, bayer)
    return imageRaw


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


def create_valid_set(path='./data/relighted', ratio=0.9):
    folders = ['images', 'gt', 'img_corrected_1', 'gt_mask', 'ill']
    valid_path = f'{path}/valid'
    if not os.path.exists(valid_path):
        os.mkdir(valid_path)
    for folder in folders:
        image_names = os.listdir(f'{path}/{folder}/')
        last = int(len(image_names) * ratio)
        valid_images = image_names[last:len(image_names) - 1]
        valid_folder = os.path.join(valid_path, folder)
        if not os.path.exists(valid_folder):
            os.mkdir(valid_folder)
        elif len(os.listdir(valid_folder)) > 0:
            continue
        for img in valid_images:
            image = os.path.join(path, folder, img)
            valid_image = os.path.join(valid_folder, img)
            os.rename(image, valid_image)
