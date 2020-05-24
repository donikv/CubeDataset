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
    # blank_mask = gradient_mask(image.shape)
    blank_mask = np.zeros(image.shape, dtype=np.uint8)

    original = image.copy()
    mask = None
    if method == -1:
        mask = image
    elif method == 1:
        im_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        avg = im_bw.mean()
        if not invert:
            avg = 255 - avg
        (thresh, mask) = cv2.threshold(im_bw, avg, 255, 0)
    else:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        # avg = np.array([hsv[:, :, 0].mean(), 255, 255]).astype(int)
        mask = cv2.inRange(hsv, lower, upper)
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

    start = None
    for c in cnts:
        # c = cv2.approxPolyDP(c, 1, True)
        # cv2.fillConvexPoly(blank_mask, c, (255, 255, 255))
        # creating convex hull object for each contour
        # cv2.drawContours(blank_mask, [c], -1, (255, 255, 255), thickness=-1)
        blank_mask = gradient_fill(c, blank_mask)
        # h = (cv2.convexHull(c, False))
        # cv2.drawContours(blank_mask1, [h], -1, (255, 255, 255), -1)
        start = c[0][0]
        break
    kernel_open = np.ones((2,2),np.uint8)
    if use_conv:
        res_mask = cv2.morphologyEx(blank_mask1, cv2.MORPH_OPEN, kernel_open)
    else:
        res_mask = cv2.morphologyEx(blank_mask, cv2.MORPH_OPEN, kernel_open)
    kernel_close = np.ones((5, 5), np.uint8)
    res_mask = cv2.morphologyEx(res_mask, cv2.MORPH_CLOSE, kernel_close)
    res_mask = add_gradient(res_mask, start)
    # res_mask = fill_holes(res_mask)
    # visualize([image, blank_mask, res_mask])
    result = cv2.bitwise_and(original, blank_mask)
    return result, gaussian(blank_mask, 9), indetention_index


def color_correct(img, mask, ill1, ill2, c_ill=1 / 3., is_relighted=False):
    def correct_pixel(p, ill1, ill2, mask, is_relighted):
        ill1 = ill1 / [ill1[0] + ill1[1] + ill1[2]]
        ill2 = ill2 / [ill2[0] + ill2[1] + ill2[2]]
        if is_relighted:
            ill = ill1 if mask[0] > 0.5 else [c_ill, c_ill, c_ill]
            return np.clip(np.multiply(p, ill), a_min=0, a_max=255)
        else:
            ill = ill1 * mask + ill2 * (1 - mask)
            return np.clip(np.multiply(p, ill), a_min=0, a_max=255)

    # ill1, ill2 = ill1 / np.linalg.norm(ill1), ill2 / np.linalg.norm(ill2)

    return np.array([
        np.array([
            correct_pixel(img[i][j], c_ill / ill1, c_ill / ill2, mask[i][j], is_relighted) for j in range(img.shape[1])
        ]) for i in range(img.shape[0])
    ], dtype=np.uint8)


def color_correct_single(img, u_ill, c_ill=1 / 3.):
    u_ill = u_ill / u_ill.sum()
    # return np.clip(img / (c_ill / u_ill), 0, 255).astype(np.uint8)

    def correct_pixel(p, ill):
        # ill = ill / [ill[0] + ill[1] + ill[2]]
        return np.clip(np.multiply(p, ill), a_min=0, a_max=255)

    # u_ill = u_ill / np.linalg.norm(u_ill)
    return np.array([np.array([correct_pixel(p, c_ill / u_ill) for p in row]) for row in img], dtype=np.uint8)


def mask_image(image, mask, value = 0):
    masked = np.where(mask > 0.5, image, value)
    return masked


def combine_images_with_mask(image1, image2, mask):
    combined = np.where(mask > 0.5, image1, image2)
    return combined


def gradient_mask(shape):
    blank_mask = np.ones(shape, dtype=np.uint8) * 255
    blank_mask = add_gradient(blank_mask, (0, 0))
    return blank_mask


def add_gradient(image, start, colors=None, alpha=None):
    blank_mask = np.zeros(image.shape, dtype=np.uint8)
    if alpha is None:
        alpha = np.random.randint(1, 89, 1) * np.pi / 180
        alpha = alpha[0]
    tga = np.tan(alpha)
    if colors is None:
        colors = np.linspace(0, 90, blank_mask.shape[0] + blank_mask.shape[1])
        colors = np.ceil(colors).astype(np.uint8)
    color_idx = np.round(np.linspace(0, len(colors)-1, blank_mask.shape[0] + blank_mask.shape[1])).astype(int)
    for row in range(blank_mask.shape[0]):
        col = int((row*tga))
        p0 = (row, 0)
        p1 = (0, col) if col <= blank_mask.shape[1] else (int((row - blank_mask.shape[1]/tga)), blank_mask.shape[1])
        if len(colors[0]) > 1:
            c = (colors[color_idx[row]] * 255).astype(int)
            cv2.line(blank_mask, (p0[1], p0[0]), (p1[1], p1[0]), (int(c[0]), int(c[1]), int(c[2])), thickness=2)
        else:
            c = int(colors[color_idx[row]])
            cv2.line(blank_mask, (p0[1], p0[0]), (p1[1], p1[0]), (c, c, c), thickness=2)
    for col in range(blank_mask.shape[1]):
        row = blank_mask.shape[0]
        col2 = int(col + row * tga)
        p0 = (row, col)
        a = blank_mask.shape[1] - 1 - col
        p1 = (0, col2) if col2 <= blank_mask.shape[1] else (int((blank_mask.shape[0] - a/tga)), blank_mask.shape[1])
        if len(colors[0]) > 1:
            c = (colors[color_idx[col + blank_mask.shape[0]]] * 255).astype(int)
            cv2.line(blank_mask, (p0[1], p0[0]), (p1[1], p1[0]), (int(c[0]), int(c[1]), int(c[2])), thickness=2)
        else:
            c = int(colors[color_idx[col + blank_mask.shape[0]]])
            cv2.line(blank_mask, (p0[1], p0[0]), (p1[1], p1[0]), (c, c, c), thickness=2)
    if start[1] < blank_mask.shape[0] / 2:
        blank_mask = cv2.flip(blank_mask, 1)
    if start[0] < blank_mask.shape[0] / 2:
        blank_mask = cv2.flip(blank_mask, 0)
    image1 = (image / 255 * blank_mask).astype(np.uint8)
    image2 = np.where(image != 0, image / 255 * (255-blank_mask), image + blank_mask).astype(np.uint8)
    return image2



def gradient_fill(contour, blank):
    p0 = contour[0]
    colors = np.ceil(np.linspace(255, 110, len(contour))).astype(np.uint8)
    for i in range(len(contour) - 2):
        p1, p2 = contour[i + 1], contour[i + 2]
        cnt = np.array([p0, p1, p2])
        c = int(colors[i])
        cv2.fillConvexPoly(blank, cnt, (c, c, c))
        # cv2.drawContours(blank, [cnt], -1, (c, c, c), thickness=-1)
    return blank
