import cv2
import multiprocessing as mp
import numpy as np

import utils.image_utils as iu
import utils.file_utils as fu


def load_and_process(relative_path, base='/media/donik/Disk/intel_tau', ext='.tiff'):
    print(relative_path[1:]+ext)
    is_sony = relative_path.find('Sony') > -1
    bayer = cv2.COLOR_BAYER_GR2BGR if is_sony else cv2.COLOR_BAYER_RG2BGR
    img = fu.load_tiff(relative_path[1:]+ext, base, '', bayer=bayer)
    depth = 10 if is_sony else 14
    img_p = iu.process_image(img, depth=depth, scale=True)
    return img_p

def save(img, relative_path, base='/media/donik/Disk/intel_tau', ext='.png'):
    cv2.imwrite(base+relative_path+ext, img)

def process(relative_path):
    img = load_and_process(relative_path)
    save(img, relative_path)

if __name__ == '__main__':

    names = np.loadtxt('/media/donik/Disk/intel_tau/paths.txt', dtype='str')

    num_threads = 1
    with mp.Pool(num_threads) as p:
        p.map(process, names)
        print('done')