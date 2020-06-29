import cv2
import numpy as np
import os
import multiprocessing as mp


def blackout(data):
    path, img = data
    image = cv2.imread(path+'/images/'+img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    mask = cv2.imread(path+'/gt/'+img)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2HLS)
    mask[:,:,1] = np.where(image[:,:,1] == 0, 0, mask[:,:,1])
    mask = cv2.cvtColor(mask, cv2.COLOR_HLS2BGR)
    cv2.imwrite(path+'/gtb/'+img, mask)

    return '1'


if __name__ == '__main__':
    path = '/home/donik/train_dataset/valid'
    images = os.listdir(path + '/images')
    images = list(map(lambda x: (path, x), images))

    with mp.Pool(8) as pool:
        masks = pool.map(blackout, images)
        exit(0)
