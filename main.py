import rawpy
import cv2
import numpy as np

from utlis.image_utils import load_image, color_correct, cv2_contours, color_correct_single, process_image
from utlis.plotting_utils import visualize, plot_counturs
from utlis.groundtruth_utils import GroundtruthLoader

img = cv2.imread('./data/1.png', cv2.COLOR_BGR2RGB)
r, g, b = cv2.split(img)
img = np.dstack((b, g, r))

gtLoader = GroundtruthLoader('cube+_gt.txt')
gtLoaderL = GroundtruthLoader('cube+_left_gt.txt')
gtLoaderR = GroundtruthLoader('cube+_right_gt.txt')
gts = gtLoader.gt
gt = gtLoader[100]

gtsl = gtLoaderL.gt
gtsr = gtLoaderR.gt
gt_diff = np.abs(gtsl - gtsr)
gt_diff_mean = gt_diff.mean(0)
gt_diff_filter = np.array(list(
    map(lambda x: x[0] + 1,
        filter(lambda x: (x[1] < gt_diff_mean).all(),
               enumerate(gt_diff)
               )
        )
))
print(gt_diff_filter)
# for idx, gt in enumerate(gt_diff):
#     if (gt < gt_diff_mean).all():
#         print(idx)

folder_step = 100

for idx in gt_diff_filter:
    start = int((idx - 1) / folder_step) * folder_step + 1
    end = min(int((idx - 1) / folder_step) * folder_step + folder_step, 1707)
    print(start, end, idx)
    folder = f'CR2_{start}_{end}'
    rgb = load_image(f"{idx}.CR2", directory=folder, mask_cube=False)
    rgb_masked = load_image(f"{idx}.CR2", directory=folder, mask_cube=True)
    rgb = process_image(rgb)
    # rgb = img
    height, width, _ = rgb.shape
    rgb = cv2.resize(rgb, (int(width/10), int(height/10)))
    gt = gts[idx-1]
    corrected = color_correct_single(rgb, gt, 1)
    foreground, mask = cv2_contours(corrected)
    relighted = color_correct(corrected, mask=mask, ill1=np.array([0.9, 0.1, 0.1]), ill2=np.array([0.1, 0.9, 0.1]), c_ill=1/10)
    visualize([rgb, corrected, relighted, mask])
