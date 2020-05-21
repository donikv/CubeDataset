import cv2
import numpy as np
from colormath.color_objects import LuvColor, sRGBColor
from colormath.color_conversions import convert_color

def xyz_from_xy(x, y):
    """Return the vector (x, y, 1-x-y)."""
    return np.array((x, y, 1-x-y))



def chromacity_planckian(T):
    u = 0.860117757 + 1.54118254e-4*T + 1.28641212e-7*T*T
    u /= 1 + 8.42420235e-4*T + 7.08145163e-7*T*T

    v = 0.317398726 + 4.22806245e-5*T + 4.20481691e-8*T*T
    v /= 1 - 2.89741816e-5*T + 1.61456053e-7*T*T

    return u, v


def chrom_to_rgb(uv: tuple):
    # l = 128
    u, v = uv
    #
    # luv = LuvColor(l, u * 255, v * 255)
    # rgb = convert_color(luv, sRGBColor)
    # rgb = np.array([rgb.rgb_r, rgb.rgb_g, rgb.rgb_b])
    # rgb = rgb / np.linalg.norm(rgb)
    # return rgb
    R = u + 2*u*v
    R /= 1 + u + v

    G = v * (1 + R)
    G /= 1 + v

    B = np.random.uniform(0, 1, 1)[0]

    rgb = np.array([R, G, B])
    # rgb = rgb / np.linalg.norm(rgb)
    return rgb

def angular_distance(a, b):
    a = a / (a[0] + a[1] + a[2])
    b = b / (b[0] + b[1] + b[2])
    return np.arccos(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)) * 180 / np.pi

def random_colors(offset=True):
    lam = np.arange(380., 781., 5)
    illuminant_D65 = xyz_from_xy(0.3127, 0.3291)
    cs_hdtv = ColourSystem(red=xyz_from_xy(0.67, 0.33),
                           green=xyz_from_xy(0.21, 0.71),
                           blue=xyz_from_xy(0.15, 0.06),
                           white=illuminant_D65)
    attempts = 0
    while True:
        T = np.random.uniform(1000, 11000, 1)
        dT = np.random.uniform(2000, 5000, 1)
        T = (T[0], (T[0] + dT[0]) % 11000 + 1000)
        spec1, spec2 = planck(lam, T[0]), planck(lam, T[1])

        attempts += 1
        random_offset = np.array([0, 0, 0]) if not offset else np.random.normal([0, 0, 0], [0.1, 0.1, 0.1])
        c1, c2 = cs_hdtv.spec_to_xyz(spec1), cs_hdtv.spec_to_xyz(spec2)
        c1, c2 = c1 + random_offset, c2 + random_offset
        c1, c2 = cs_hdtv.xyz_to_rgb(c1), cs_hdtv.xyz_to_rgb(c2)
        c1, c2 = np.clip(c1, a_min=1/255, a_max=254/255), np.clip(c2, a_min=1/255, a_max=254/255)
        ang = angular_distance(c1, c2)
        if ang > 5 or attempts > 10:
            return c1, c2



# colour_system.py
from scipy.constants import h, c, k


class ColourSystem:
    """A class representing a colour system.

    A colour system defined by the CIE x, y and z=1-x-y coordinates of
    its three primary illuminants and its "white point".


    """

    # The CIE colour matching function for 380 - 780 nm in 5 nm intervals
    try:
        cmf = np.loadtxt('./resources/cie-cmf.txt', usecols=(1,2,3))
    except OSError:
        cmf = None

    def __init__(self, red, green, blue, white):
        """Initialise the ColourSystem object.

        Pass vectors (ie NumPy arrays of shape (3,)) for each of the
        red, green, blue  chromaticities and the white illuminant
        defining the colour system.

        """

        # Chromaticities
        self.red, self.green, self.blue = red, green, blue
        self.white = white
        # The chromaticity matrix (rgb -> xyz) and its inverse
        self.M = np.vstack((self.red, self.green, self.blue)).T
        self.MI = np.linalg.inv(self.M)
        # White scaling array
        self.wscale = self.MI.dot(self.white)
        # xyz -> rgb transformation matrix
        self.T = self.MI / self.wscale[:, np.newaxis]

    def xyz_to_rgb(self, xyz, out_fmt=None):
        """Transform from xyz to rgb representation of colour.

        The output rgb components are normalized on their maximum
        value. If xyz is out the rgb gamut, it is desaturated until it
        comes into gamut.

        By default, fractional rgb components are returned; if
        out_fmt='html', the HTML hex string '#rrggbb' is returned.

        """

        rgb = self.T.dot(xyz)
        if np.any(rgb < 0):
            # We're not in the RGB gamut: approximate by desaturating
            w = - np.min(rgb)
            rgb += w
        if not np.all(rgb==0):
            # Normalize the rgb vector
            rgb /= np.max(rgb)

        if out_fmt == 'html':
            return self.rgb_to_hex(rgb)
        return rgb

    def rgb_to_hex(self, rgb):
        """Convert from fractional rgb values to HTML-style hex string."""

        hex_rgb = (255 * rgb).astype(int)
        return '#{:02x}{:02x}{:02x}'.format(*hex_rgb)

    def spec_to_xyz(self, spec):
        """Convert a spectrum to an xyz point.

        The spectrum must be on the same grid of points as the colour-matching
        function, self.cmf: 380-780 nm in 5 nm steps.

        """

        XYZ = np.sum(spec[:, np.newaxis] * self.cmf, axis=0)
        den = np.sum(XYZ)
        if den == 0.:
            return XYZ
        return XYZ / den

    def spec_to_rgb(self, spec, out_fmt=None):
        """Convert a spectrum to an rgb value."""

        xyz = self.spec_to_xyz(spec)
        return self.xyz_to_rgb(xyz, out_fmt)


def planck(lam, T):
    """ Returns the spectral radiance of a black body at temperature T.

    Returns the spectral radiance, B(lam, T), in W.sr-1.m-2 of a black body
    at temperature T (in K) at a wavelength lam (in nm), using Planck's law.

    """

    lam_m = lam / 1.e9
    fac = h*c/lam_m/k/T
    B = 2*h*c**2/lam_m**5 / (np.exp(fac) - 1)
    return B


def preprocess_for_estimation(img, mask):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    img[:, :, 0] = img[:, :, 0] * mask[:, :, 0]
    img = cv2.cvtColor(img, cv2.COLOR_LUV2RGB)
    return img.reshape((-1, 3))


def gray_world_estimation(img, mask=np.ones((1, 1, 1)), mask_hsv=True):
    if mask is not None:
        X = preprocess_for_estimation(img, mask)
    else:
        X = img.reshape((-1, 3))
    if mask_hsv:
        X = np.ma.masked_equal(X, np.array([255, 255, 255]))
        hsv = cv2.cvtColor(np.array([X]), cv2.COLOR_RGB2HLS).squeeze()
        hsv = np.ma.masked_less(hsv, [0, 80, 0])
        hsv = hsv.filled(0)
        X = cv2.cvtColor(np.array([hsv], dtype=np.uint8), cv2.COLOR_HLS2RGB).squeeze()
    X = np.ma.masked_equal(X, np.array([0, 0, 0]))
    X = X.mean(axis=0)

    return np.array(X, dtype=np.uint8)


def gray_edge_estimation(img, mask=np.ones((1, 1, 1))):
    edg = cv2.Canny(img, 0, 128)
    edg = np.dstack((edg, edg, edg))
    img = np.where(edg != 0, img, np.zeros(3,  dtype=np.uint8))
    X = preprocess_for_estimation(img, mask)
    X = np.ma.masked_equal(X, np.array([0, 0, 0]))
    X = np.ma.masked_less(X, np.array([50, 50, 50]))
    # X = np.ma.masked_equal(X, np.array([255, 255, 255]))
    X = X.mean(axis=0)

    return np.array(X, dtype=np.uint8)


def white_patch_estimation(img, mask=np.ones((1, 1, 1))):
    X = preprocess_for_estimation(img, mask)
    X = np.ma.masked_equal(X, np.array([0, 0, 0]))
    # X = np.ma.masked_equal(X, np.array([255, 255, 255]))
    # mn = X.mean(axis=0)
    # hsv = cv2.cvtColor(np.array([X]), cv2.COLOR_RGB2HLS).squeeze()
    # hsv = np.ma.masked_greater(hsv, [255, 250, 255])
    # hsv = hsv.filled(0)
    # pos = hsv[:, 1].argmax(axis=0)
    # X = cv2.cvtColor(np.array([hsv]), cv2.COLOR_HLS2RGB).squeeze()
    # pos = hsv[:, 1].argmax(axis=0)
    #
    # X = X[pos]
    X = X.max(axis=0)

    return np.array([[X]], dtype=np.uint8).squeeze().squeeze()


def white_balance(img, rgb, mask=np.ones((1, 1, 1))):
    if len(rgb.shape) == 1:
        rgb = np.array([[rgb]])
    if rgb.max() <= 1:
        rgb = (rgb * 255).astype(np.uint8)
    if mask.max() > 1:
        mask = mask / 255
    result = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab = lab.squeeze().squeeze()
    result[:, :, 1] = result[:, :, 1] - ((lab[1] - 128) * (result[:, :, 0] / 255.0) * 1.1) * mask[:, :, 0]
    result[:, :, 2] = result[:, :, 2] - ((lab[2] - 128) * (result[:, :, 0] / 255.0) * 1.1) * mask[:, :, 0]
    # result = np.clip(result, 1, 254)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
    return result