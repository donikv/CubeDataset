import cv2
import numpy as np

from skimage.filters import gaussian

from utils.plotting_utils import visualize


def debayer(rggb):
    red = rggb[:-1:2, :-1:2]
    green = (rggb[:-1:2, 1::2] + rggb[1::2, 0:-1:2]) / 2.0
    blue = rggb[1::2, 1::2]
    img = np.zeros((len(red), len(red[0]), 3))
    img[:, :, 0] = red
    img[:, :, 1] = green
    img[:, :, 2] = blue

    return img


def process_image(img: np.ndarray, depth=14, blacklevel=2048, scale=False, blackout=False):
    saturationLevel = img.max() - 2
    img = img.astype(int)
    img = np.clip(img - blacklevel, a_min=0, a_max=np.infty).astype(int)
    if blackout:
        m = np.where(img >= saturationLevel - blacklevel, 1, 0).sum(axis=2, keepdims=True)
        max_val = np.iinfo(np.int32).max
        m = np.where(m > 0, [0, 0, 0], [max_val, max_val, max_val])
        result = cv2.bitwise_and(img, m)
    else:
        result = img
    if scale:
        return (result / 2 ** depth).astype(np.float32)
    else:
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
    """
    Finds edges in the given image with upper an lower threshold values.
    Edges are found using Canny edge detection algorithm.

    :param image: The input image
    :param lower: The lower value for the luminance threshold
    :param upper: The upper value for the luminance threshold
    :return: (edges, s_edges) Tuple of edges mask and smoothed mask using closing operator
    """
    image = image.astype(np.uint8)
    im_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    p = np.random.random(1)
    if p > 0.5:
        avg = int(im_bw.mean() - lower * np.random.normal(1, 0.2, 1))
    else:
        avg = int(im_bw.mean() - upper * np.random.normal(1, 0.2, 1))
    edges = cv2.Canny(im_bw, avg / 2, avg * 2)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return edges, closing


def cv2_contours(image,
                 lower: np.ndarray = np.array([10, 0, 0]),
                 upper: np.ndarray = np.array([10, 255, 255]),
                 method=1, invert=False, use_conv=False, use_grad=True):
    """
    Finds the contours in the given image using luminance threshold using cv2 library and returns the contour mask.
    The default behaviour fills the contours as (255,255,255) values in the returned mask.

    :param image: The input image
    :param lower: The lower threshold value
    :param upper: The upper threshold value
    :param method: Thresholding method used: 1 - Random average threshold, 2 - Static threshold using `lower` and `upper` values, -1 - Uses the input image as the input for the contour algorithm (Input image must be grayscale).
    :param invert: Boolean indicating whether to invert the mask.
    :param use_conv: Boolean indicating whether to fit a convex hull around contours.
    :param use_grad: Boolean indicating whether to use gradient fill for contours. (When gradient is used the contours are filled into a convex polygon).
    :return: (mask, noisy_mask, indetention_index) - Contour mask, contour mask with added noise, indentation index of the produced mask
    """
    image = image.astype(np.uint8)
    blank_mask = np.zeros(image.shape, dtype=np.uint8)
    original = image.copy()

    if method == -1:
        mask = image
    elif method == 1:
        im_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        avg = im_bw.mean()
        if not invert:
            avg = 255 - avg
        p = np.random.random(1)
        if p > 0.5:
            avg = int(avg - lower[0] * np.random.normal(1, 0.2, 1))
        else:
            avg = int(avg - upper[0] * np.random.normal(1, 0.2, 1))
        (thresh, mask) = cv2.threshold(im_bw, avg, 255, 0)
    else:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
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
    indetention_index = cv2.contourArea(cnts[0]) / cv2.arcLength(cnts[0], closed=False)

    for c in cnts:
        if use_conv:
            c = (cv2.convexHull(c, False))
        if use_grad:
            blank_mask = gradient_fill(c, blank_mask)
        else:
            cv2.drawContours(blank_mask, [c], -1, (255, 255, 255), thickness=-1)
        break

    result = cv2.bitwise_and(original, blank_mask)
    return result, gaussian(blank_mask, 9), indetention_index


def color_correct(img, mask, ill1, ill2, c_ill=1 / 3., is_relighted=False):
    """
    Corrects the image for the two illuminants and the mask of illuminant distribution using the Von Kries model.

    :param img: Input image for correction
    :param mask: Mask of the spatial distribution of the illuminants
    :param ill1: First unknown illuminant
    :param ill2: Second unknown illuminant
    :param c_ill: Canonical illuminant
    :param is_relighted: Indicates whether the correction is applied in order to correct the image or synthetically relight it.
    :return: Corrected image
    """
    if mask.max() > 1:
        mask = mask / 255.
    img1 = mask_image(img, mask)
    img2 = mask_image(img, 1-mask)

    if img.dtype == np.uint8 or img.dtype == np.int32:
        c1 = color_correct_single(img1, ill1, c_ill, is_relighted)
        c2 = color_correct_single(img2, ill2, c_ill, is_relighted)
    elif img.dtype == np.uint16:
        c1 = color_correct_single_16(img1, ill1, c_ill, is_relighted)
        c2 = color_correct_single_16(img2, ill2, c_ill, is_relighted)
    elif img.dtype == np.float32:
        c1 = color_correct_single(img1, ill1, c_ill, is_relighted)
        c2 = color_correct_single(img2, ill2, c_ill, is_relighted)
    else:
        raise AssertionError(f"Type {img.dtype} not supported for conversion.")

    img_c = combine_images_with_mask(c1, c2, mask)
    return img_c


def color_correct_with_colored_mask(img, mask, ill1, ill2, c_ill=1 / 3., is_relighted=False):
    """
    Corrects the image for the two illuminants and the mask of illuminant distribution using the Von Kries model.

    :param img: Input image for correction
    :param mask: Mask of the spatial distribution of the illuminants
    :param ill1: First unknown illuminant
    :param ill2: Second unknown illuminant
    :param c_ill: Canonical illuminant
    :param is_relighted: Indicates whether the correction is applied in order to correct the image or synthetically relight it.
    :return: Corrected image
    """
    if mask.max() > 1:
        mask = mask / 255.
    img1 = mask_image(img, mask)
    img2 = mask_image(img, 1-mask)

    if img.dtype == np.uint8 or img.dtype == np.int32:
        c1 = color_correct_single(img1, ill1, c_ill, is_relighted)
        c2 = color_correct_single(img2, ill2, c_ill, is_relighted)
    elif img.dtype == np.uint16:
        c1 = color_correct_single_16(img1, ill1, c_ill, is_relighted)
        c2 = color_correct_single_16(img2, ill2, c_ill, is_relighted)
    elif img.dtype == np.float32:
        c1 = color_correct_single(img1, ill1, c_ill, is_relighted)
        c2 = color_correct_single(img2, ill2, c_ill, is_relighted)
    else:
        raise AssertionError(f"Type {img.dtype} not supported for conversion.")

    img_c = combine_images_with_mask(c1, c2, mask)
    return img_c


def color_correct_single_16(img, u_ill, c_ill=1 / 3., relight=False):
    """
    Corrects the image using the Von Kries model globally for one illuminant. Works on 16-bit unsigned images.

    :param img: Input image to be corrected
    :param u_ill: Unknown illuminant
    :param c_ill: Canonical illuminant
    :param relight: Indicates whether the correction is applied in order to correct the image or synthetically relight it.
    :return: Corrected image
    """
    if not relight:
        u_ill = u_ill / u_ill.sum()

    img[:, :, 0] = img[:, :, 0] * c_ill / u_ill[0]
    img[:, :, 1] = img[:, :, 1] * c_ill / u_ill[1]
    img[:, :, 2] = img[:, :, 2] * c_ill / u_ill[2]
    return np.where(img.max(axis=2, keepdims=True) > 65535, np.zeros(img.shape) * 65535, img).astype(np.uint16)


def color_correct_single_f32(img, u_ill, c_ill=1 / 3., relight=False):
    """
    Corrects the image using the Von Kries model globally for one illuminant. Works on 32-bit float images.

    :param img: Input image to be corrected
    :param u_ill: Unknown illuminant
    :param c_ill: Canonical illuminant
    :param relight: Indicates whether the correction is applied in order to correct the image or synthetically relight it.
    :return: Corrected image
    """
    if not relight:
        u_ill = u_ill / u_ill.sum()

    img[:, :, 0] = img[:, :, 0] * c_ill / u_ill[0]
    img[:, :, 1] = img[:, :, 1] * c_ill / u_ill[1]
    img[:, :, 2] = img[:, :, 2] * c_ill / u_ill[2]
    return np.where(img.max(axis=2, keepdims=True) > 1, np.zeros(img.shape), img).astype(np.float32)



def color_correct_single(img, u_ill, c_ill=1 / 3., relight=False):
    """
    Corrects the image using the Von Kries model globally for one illuminant. Works on standard 8-bit depth images.

    :param img: Input image to be corrected
    :param u_ill: Unknown illuminant
    :param c_ill: Canonical illuminant
    :param relight: Indicates whether the correction is applied in order to correct the image or synthetically relight it.
    :return: Corrected image
    """
    if not relight:
        u_ill = u_ill / u_ill.sum()

    img[:, :, 0] = img[:, :, 0] * c_ill / u_ill[0]
    img[:, :, 1] = img[:, :, 1] * c_ill / u_ill[1]
    img[:, :, 2] = img[:, :, 2] * c_ill / u_ill[2]
    return np.where(img.max(axis=2, keepdims=True) >= 255, np.zeros(img.shape) * 255, img).astype(np.uint8)


def mask_image(image, mask, value = 0):
    """
    Masks the parts of the images where the corresponding mask < 0.5.

    :param image: Input image to be masked
    :param mask: Mask used for masking
    :param value: Value with which to fill the masked image
    :return: Masked image
    """
    mask = np.expand_dims(mask, -1) if len(mask.shape) < 3 else mask
    masked = np.where(mask > 0.5, image, np.ones_like(image) * value)
    return masked


def combine_images_with_mask(image1, image2, mask):
    """
    Combines the images using given mask.

    :param image1: First image
    :param image2: Second image
    :param mask: Mask used for combination. Combined image is filled with the first image where the value in the mask is > 0.5 and the second images is used for the rest of the combined image
    :return: Combined image
    """
    mask = np.expand_dims(mask, -1) if len(mask.shape) < 3 else mask
    combined = np.where(mask > 0.5, image1, image2)
    return combined


def gradient_mask(shape):
    """
    Creates a mask where the gradient is applied form the top left part of the image at an random angle.

    :param shape: Dimensions of the output mask. Only the first two values are used for height and width of the mask.
    :return: Grayscale gradient mask
    """
    blank_mask = np.ones(shape, dtype=np.uint8) * 255
    blank_mask = add_gradient(blank_mask, (0, 0))
    return blank_mask


def add_gradient(image, start, colors=None, alpha=None):
    """
    Adds a gradient to the image. The resulting image has 8-bit depth.

    :param image: Image on which to draw the gradient.
    :param start: Coordinates from which the gradient will start.
    :param colors: Colors of the gradient. If None, then the mask will be grayscale with values from 0-255.
    :param alpha: The angle of the gradient. If None, then an random angle (from 1 to 89 degrees) will be used.
    :return: Image with the applied gradient
    """
    blank_mask = np.zeros(image.shape, dtype=np.uint8)
    if alpha is None:
        alpha = np.random.randint(1, 89, 1) * np.pi / 180
        alpha = alpha[0]
    tga = np.tan(alpha)
    if colors is None:
        colors = np.linspace(0, 90, blank_mask.shape[0] + blank_mask.shape[1]) #90
        colors = np.ceil(colors).astype(np.uint8)
        colors = np.expand_dims(colors, -1)
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
    image2 = np.where(image != 0, image / 255 * (255-blank_mask), image + blank_mask).astype(np.uint8)
    return image2



def gradient_fill(contour, blank):
    """
    Fills the convex hull of the given contour with a gradient fill on the given image.

    :param contour: Points corresponding to the contour that needs to be filled.
    :param blank: Image on which the contour fill be drawn and filled.
    :return: Image with the contour draw onto it.
    """
    p0 = contour[0]
    colors = np.ceil(np.linspace(255, 110, len(contour))).astype(np.uint8)
    for i in range(len(contour) - 2):
        p1, p2 = contour[i + 1], contour[i + 2]
        cnt = np.array([p0, p1, p2])
        c = int(colors[i])
        cv2.fillConvexPoly(blank, cnt, (c, c, c))
    return blank


def blackout(gt, im):
    """
    Blacks out parts of the mask where the luminance value in the image is 0.
    :param gt: Mask which will be blacked out.
    :param im: Image whose luminance values will be used.
    :return: Blacked out mask
    """
    im = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)
    gt = cv2.cvtColor(gt, cv2.COLOR_RGB2HLS)
    gt[:,:,1] = np.where(im[:,:,1] == 0, 0, gt[:,:,1])
    gt = cv2.cvtColor(gt, cv2.COLOR_HLS2RGB)
    return gt
