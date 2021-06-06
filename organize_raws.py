import numpy as np
import cv2
import os
import shutil
import utils.plotting_utils as plt
import utils.image_utils as iu

def move(path, name, dest, mapping):
    new_dir_name = str(mapping(int(name[:-4]))) + '/'
    if int(new_dir_name[:-1]) <= 0:
        return 'skipped'
    new_img_name = 'img.NEF'
    os.makedirs(dest + new_dir_name, exist_ok=True)
    shutil.copyfile(path + name, dest + new_dir_name + new_img_name)
    return new_dir_name + new_img_name

def load_tiff(img):
    image_tiff = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    # imageRaw = cv2.cvtColor(image_tiff, cv2.COLOR_BAYER_RG2BGR)
    imageRaw = iu.debayer(image_tiff)
    return imageRaw

def create_debayered(path, name):
    try:
        os.remove(path + name[:-4] + '.png')
    except:
        pass
    return name[:-4] + '.png'
    os.system(f"cd {path} && dcraw -D -T -4 -K ../../d_nikon.pgm {name}")
    # os.system(f"cd {path} && dcraw -D -T -4 {name}")
    db = load_tiff(path + name[:-4] + '.tiff').round().astype(np.uint16)
    os.remove(path + name[:-4] + '.tiff')
    cv2.imwrite(path + name[:-4] + '.png', cv2.cvtColor(db, cv2.COLOR_RGB2BGR))
    return name[:-4] + '.png'

if __name__ == '__main__':
    import time

    # start = time.time_ns()
    path = '/Volumes/Jolteon/fax/'
    save_path = '/Volumes/Jolteon/fax/outdoor/raws/'
    folders_source = ['raws1/', 'raws2/', 'raws_slo/nikon/', 'raws4/nikon/', 'raws3/', 'to_process/']
    folders_dest = ['outdoor1/', 'outdoor2/', 'outdoor3/', 'outdoor4/', 'outdoor5/', 'outdoor6/']
    name_mapping = [lambda x: x, lambda x: x, lambda x: x, lambda x: x-11, lambda x: x, lambda x: x]
    # l = np.loadtxt(f'{path}/list.txt', dtype=str)

    for fs, fd, mp in list(zip(folders_source, folders_dest, name_mapping)):
        files = os.listdir(path + fs)
        raws = list(filter(lambda x: x.endswith('.NEF') and not x.startswith('.'), files))
        os.makedirs(save_path + fd, exist_ok=True)
        for raw in raws:
            # nname = move(path + fs, raw, save_path + fd, mp)
            nname = create_debayered(save_path + fd + raw[:-4] + '/', 'img.NEF')
            print(path + fs + raw, save_path + fd + nname)