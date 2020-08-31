import numpy as np
import cv2
import os


NIKON = 'nikon_d7000'
CANON = 'canon_550d'

def __create_save_location__(save_location):
    if not os.path.exists(save_location):
        os.mkdir(save_location)

def save_for_dataset(images:dict, gt:np.ndarray, pos: np.ndarray, root_folder:str, scene_type:str, camera:str, shoot_name:str, image_name:str):
    save_location = root_folder
    __create_save_location__(save_location)
    save_location = os.path.join(save_location, scene_type)
    __create_save_location__(save_location)
    save_location = os.path.join(save_location, camera)
    __create_save_location__(save_location)
    save_location = os.path.join(save_location, shoot_name)
    __create_save_location__(save_location)
    save_location = os.path.join(save_location, image_name)
    __create_save_location__(save_location)

    for name, image in images.items():
        filename = os.path.join(save_location, name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(filename, image)
    gt_fn = os.path.join(save_location, 'gt.txt')
    np.savetxt(gt_fn, np.array([gt]))
    pos_fn = os.path.join(save_location, 'cube.txt')
    np.savetxt(pos_fn, np.array([pos]))

    return save_location