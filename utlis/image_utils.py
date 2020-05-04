import cv2
import numpy as np

from skimage.filters import gaussian

from utlis.plotting_utils import visualize

def process_image(img: np.ndarray, depth=14):
    blacklevel = 2048
    saturationLevel = img.max() - 2
    img = img.astype(int)
    img = np.clip(img - blacklevel, a_min=0, a_max=np.infty).astype(int)
    m = np.where(img >= saturationLevel - blacklevel, 1, 0).sum(axis=2, keepdims=True)
    max_val = np.iinfo(np.int32).max
    m = np.where(m > 0, [0, 0, 0], [max_val, max_val, max_val])
    result = cv2.bitwise_and(img, m)

    return (result / 2**depth * 255).astype(np.uint8)


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def find_edges(image, lower, upper):
    image = image.astype(np.uint8)
    blank_mask = np.zeros(image.shape, dtype=np.uint8)
    original = image.copy()
    mask = None
    im_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    avg = int(im_bw.mean())
    edges = cv2.Canny(im_bw, avg / 2, avg * 2)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return edges, closing


def fill_holes(img):
    im_th = img.copy()

    im_floodfill = im_th.copy()

    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    start = (0, 0)
    color = 255
    if img[start] != 0:
        im_floodfill = cv2.bitwise_not(im_floodfill)
        # return img
        cv2.floodFill(im_floodfill, mask, start, color)
        im_floodfill = cv2.bitwise_not(im_floodfill)
    else:
        cv2.floodFill(im_floodfill, mask, start, color)

    # # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    #
    # # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    return im_out


def cv2_contours(image, lower: np.ndarray = np.array([0, 0, 0]), upper: np.ndarray = np.array([100, 255, 255]), method=1, invert=False, use_conv=False):
    image = image.astype(np.uint8)
    blank_mask = np.zeros(image.shape, dtype=np.uint8)
    original = image.copy()
    mask = None
    if method == -1:
        mask = image
    elif method == 1:
        im_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        (thresh, mask) = cv2.threshold(im_bw, lower[0], upper[0], 0)
    else:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        avg = np.array([hsv[:, :, 0].mean(), 255, 255]).astype(int)
        mask = cv2.inRange(hsv, lower, avg)
    mask = cv2.blur(mask, (5, 5))  # blur the image
    if method == -1:
        erosion = mask
    else:
        kernel = np.ones((1, 2), np.uint8)
        erosion = cv2.erode(mask, kernel, iterations=10)
        eroison = 255 - erosion
        kernel = np.ones((2, 1), np.uint8)
        erosion = cv2.erode(eroison, kernel, iterations=5)
    if invert:
        erosion = 255 - erosion

    cnts = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    blank_mask1 = np.zeros(image.shape, dtype=np.uint8)
    indetention_index = cv2.contourArea(cnts[0]) / cv2.arcLength(cnts[0], closed=False)

    for c in cnts:
        c = cv2.approxPolyDP(c, 1, True)
        cv2.fillConvexPoly(blank_mask, c, (255, 255, 255))
        # creating convex hull object for each contour
        # cv2.drawContours(blank_mask, [c], -1, (255, 255, 255), thickness=-1)
        h = (cv2.convexHull(c, False))
        cv2.drawContours(blank_mask1, [h], -1, (255, 255, 255), -1)
        break
    kernel_open = np.ones((10,10),np.uint8)
    if use_conv:
        res_mask = cv2.morphologyEx(blank_mask1, cv2.MORPH_OPEN, kernel_open)
    else:
        res_mask = cv2.morphologyEx(blank_mask, cv2.MORPH_OPEN, kernel_open)
    kernel_close = np.ones((20, 20), np.uint8)
    res_mask = cv2.morphologyEx(res_mask, cv2.MORPH_CLOSE, kernel_close)
    # res_mask = fill_holes(res_mask)
    # visualize([image, blank_mask, blank_mask1])
    result = cv2.bitwise_and(original, res_mask)
    return result, gaussian(res_mask, 9), indetention_index


def color_correct(img, mask, ill1, ill2, c_ill=1 / 3.):
    def correct_pixel(p, ill1, ill2, mask):
        ill = ill1 * mask + ill2 * (1 - mask)
        ill = ill / np.linalg.norm(ill)
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
