import os

import cv2
import rawpy
import numpy as np
from segmentizer import Segmentizer

from skimage.filters import gaussian


def load_image(name, path = './data/Cube+', directory='CR2_1_100', mask_cube=True):
    image = f"{path}/{directory}"
    image_path = os.path.join(image, name)

    raw = rawpy.imread(image_path)  # access to the RAW image
    rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16)  # a numpy RGB array

    if mask_cube:
        for i in range(2000, rgb.shape[0]):
            for j in range(4000, rgb.shape[1]):
                rgb[i][j] = np.zeros(3)

    return rgb


def process_image(img: np.ndarray):
    blacklevel = 2048
    saturationLevel = img.max() - 2
    img = img.astype(int)
    img = np.clip(img - blacklevel, a_min=0, a_max=np.infty).astype(int)
    m = np.where(img >= saturationLevel - blacklevel, 1, 0).sum(axis=2, keepdims=True)
    max_val = np.iinfo(np.int32).max
    m = np.where(m > 0, [0, 0, 0], [max_val, max_val, max_val])
    result = cv2.bitwise_and(img, m)

    return (result / saturationLevel * 255).astype(np.uint8)

def segment(img: np.ndarray):
    if len(img.shape) < 3:
        img = np.array([[[pixel, pixel, pixel] for pixel in row] for row in img])
    height, width, _ = img.shape
    segmentizer = Segmentizer(width, height)
    segmented_img = segmentizer.fit_and_predict(img)

    return segmented_img

def cv2_contours(image, lower: np.ndarray = np.array([0, 0, 0]), upper: np.ndarray = np.array([100, 255, 255])):
    image = image.astype(np.uint8)
    blank_mask = np.zeros(image.shape, dtype=np.uint8)
    original = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    # lower = np.array([18, 42, 69])
    # upper = np.array([179, 255, 255])
    mask = cv2.inRange(image, lower, upper)

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        cv2.drawContours(blank_mask, [c], -1, (255, 255, 255), -1)
        break

    result = cv2.bitwise_and(original, blank_mask)
    return result, gaussian(blank_mask, 11)


def color_correct(img, mask, ill1, ill2, c_ill=1 / 3.):
    def correct_pixel(p, ill1, ill2, mask):
        ill = ill1 * mask + ill2 * (1 - mask)
        return np.clip(np.multiply(p, ill), a_min=0, a_max=255)

    ill1, ill2 = ill1 / np.linalg.norm(ill1), ill2 / np.linalg.norm(ill2)

    return np.array([
        np.array([
            correct_pixel(img[i][j], c_ill / ill1, c_ill / ill2, mask[i][j]) for j in range(img.shape[1])
        ]) for i in range(img.shape[0])
    ], dtype=np.uint8)


def color_correct_single(img, u_ill, c_ill=1 / 3.):
    def correct_pixel(p, ill):
        return np.clip(np.multiply(p, ill), a_min=0, a_max=255)

    # u_ill = u_ill / np.linalg.norm(u_ill)
    return np.array([np.array([correct_pixel(p, c_ill / u_ill) for p in row]) for row in img], dtype=np.uint8)
