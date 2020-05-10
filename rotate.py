import cv2
import utlis.file_utils as fu
import utlis.plotting_utils as pu

names = ['17.jpg', '18.jpg', '3.jpg', '5.jpg', '7.jpg', '10.jpg', '21.jpg']
for name in names:
    image = fu.load_png(name, './data', 'custom_mask_nocor')

    rotated = cv2.rotate(image, cv2.ROTATE_180)
    pu.visualize([image, rotated])
    cv2.imwrite(f'./data/custom_mask_nocor/{name}', rotated)
exit(0)
