import cv2
import utils.file_utils as fu
import utils.plotting_utils as pu

names = [17, 18, 3, 5, 7, 10, 21, 9, 2]
names = [2]
names = list(map(lambda x: str(x) + '.jpg', names))
for name in names:
    image = fu.load_png(name, './data', 'custom_mask_nocor')

    rotated = cv2.flip(image, 0)
    pu.visualize([image, rotated])
    cv2.imwrite(f'./data/custom_mask_nocor/{name}', rotated)
exit(0)
