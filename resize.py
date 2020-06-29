import cv2
import os

path = '../Datasets/fer_projector/'
folder = 'combined'
# folder = 'combined/valid'
dest = '../Datasets/fer_projector/resized'
images = os.listdir(path+folder+'/images/')

for img in images:
    if int(img[:-4]) < 201:
        continue
    image = cv2.imread(path+folder+'/images/'+img)
    gt = cv2.imread(path + folder + '/gt/' + img)
    image = cv2.resize(image, (0,0), fx=1/5, fy=1/5)
    gt = cv2.resize(gt, (0, 0), fx=1 / 5, fy=1 / 5)
    img_name = dest+'/images/'+img
    cv2.imwrite(img_name, image)
    cv2.imwrite(dest + '/gt/' + img, gt)
    print(img)