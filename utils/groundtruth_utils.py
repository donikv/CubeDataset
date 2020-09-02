import os

import cv2
import numpy as np

class GroundtruthLoader:
    """
    Class used to load and iterate over the groundtruth values in the given text file.
    """

    def __init__(self, name, path = './data/Cube+'):
        self.name = name
        self.path = path

        self.gt = np.loadtxt(os.path.join(path, name))
        self.len = self.gt.shape[0]

    def __iter__(self):
        return GTIterator(self)

    def __getitem__(self, item):
        return self.gt[item]


class GTIterator:

    def __init__(self, gtLoader: GroundtruthLoader):
        self.len = gtLoader.len
        self.gt = gtLoader.gt
        self.idx = 0

    def __next__(self):
        if self.idx < self.len:
            self.idx += 1
            return self.gt[self.idx - 1]
        raise StopIteration

    def __len__(self):
        return self.len


def get_mask_from_gt(gt):
    if type(gt) is not np.ndarray:
        gt = gt.cpu()
    _, _, centers = cluster(gt, draw=False)
    mask = np.array([[(np.array([255, 255, 255]) if np.linalg.norm(pixel - centers[0]) > np.linalg.norm(
        pixel - centers[1]) else np.array([0, 0, 0])) for pixel in row] for row in gt])
    return mask


def prep_img_for_clustering(img: np.ndarray):
    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)
    mask = (Z > -1) & (Z < 256)
    mask = np.array(list(map(lambda x: x.all(), mask)))
    # print(mask)
    Z = Z[mask]
    return Z


def cluster(img: np.ndarray, draw=True):
    Z = prep_img_for_clustering(img)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    A = Z[label.ravel() == 0]
    B = Z[label.ravel() == 1]
    # C = Z[label.ravel() == 2]
    return ret, label, center
