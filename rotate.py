import cv2
import utlis.file_utils as fu
import utlis.plotting_utils as pu


image = fu.load_png('10.png', './data', 'custom_mask')

rotated = cv2.rotate(image, cv2.ROTATE_180)
pu.visualize([image, rotated])
cv2.imwrite('./data/custom_mask/10.png', rotated)
exit(0)
