import os

import numpy as np

class GroundtruthLoader:

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