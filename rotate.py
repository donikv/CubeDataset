import cv2
import utils.file_utils as fu
import utils.plotting_utils as pu

import os
path = 'projector_test/projector2/pngs/both/images/'
# path = 'G:\\fax\\diplomski\\Datasets\\third\\ambient6\\both\\images/'
names = os.listdir(path)
left = 'projector_test/projector2/pngs/left/'
right = 'projector_test/projector2/pngs/right/'
for name in names:
    image = cv2.imread(path+name, cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (0,0), fx=1/2, fy=1/2)
    cv2.imwrite(path+name, image)
    # if name.find('lr') > -1:
    #     os.rename(left+image_name2, right+image_name2+'a')
    #     os.rename(left + 'images/'+image_name, right + 'images/'+ image_name + 'a')
    #
    #     os.rename(right+image_name2, left+image_name2)
    #     os.rename(right + 'images/'+ image_name, left + 'images/'+image_name)
    #
    #     os.rename(right+image_name2+'a', right+image_name2)
    #     os.rename(right + 'images/'+ image_name + 'a', right + 'images/'+ image_name)

exit(0)

names = [17, 18, 3, 5, 7, 10, 21, 9, 2]
names = [2]
names = list(map(lambda x: str(x) + '.jpg', names))
for name in names:
    image = fu.load_png(name, './data', 'custom_mask_nocor')

    rotated = cv2.flip(image, 0)
    pu.visualize([image, rotated])
    cv2.imwrite(f'./data/custom_mask_nocor/{name}', rotated)
exit(0)
