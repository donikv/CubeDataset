import pickle

import rawpy
import cv2
import numpy as np

from utlis.image_utils import load_image, color_correct, cv2_contours, color_correct_single, process_image
from utlis.plotting_utils import visualize, plot_counturs
from utlis.groundtruth_utils import GroundtruthLoader

# img = cv2.imread('./data/1.png', cv2.COLOR_BGR2RGB)
# r, g, b = cv2.split(img)
# img = np.dstack((b, g, r))

gtLoader = GroundtruthLoader('cube+_gt.txt')
gts = gtLoader.gt


def get_ill_diffs():
    gtLoaderL = GroundtruthLoader('cube+_left_gt.txt')
    gtLoaderR = GroundtruthLoader('cube+_right_gt.txt')

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
    return gt_diff_filter


folder_step = 100


def load(index, folder_step = 100, mask_cube=False, depth=8):
    start = int((index - 1) / folder_step) * folder_step + 1
    end = min(int((index - 1) / folder_step) * folder_step + folder_step, 1707)
    print(start, end, index)
    folder = f'CR2_{start}_{end}'
    rgb = load_image(f"{index}.CR2", directory=folder, mask_cube=mask_cube, depth=depth)
    return rgb


def get_diff_in_ill(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    image = image[:, :, 0]
    diff = image.max() - image.min()
    return diff


def process_and_visualize(image, idx, title=None):
    height, width, _ = image.shape
    image = cv2.resize(image, (int(width / 10), int(height / 10)))
    original = image.copy()
    image = process_image(image)
    # rgb = img
    gt = gts[idx - 1]
    corrected = color_correct_single(image, gt, 1)
    foreground, mask = cv2_contours(corrected, upper=np.array([128, 255, 255]))
    ill1 = np.random.uniform(0, 1, 3)
    ill2 = np.random.uniform(0, 1, 3)
    relighted = color_correct(image, mask=mask, ill1=1 / ill1, ill2=1 / ill2,
                              c_ill=2)
    colored_mask = np.array(
        [[ill1 * pixel + ill2 * (1 - pixel) for pixel in row] for row in mask])
    visualize([image, corrected, relighted, colored_mask], title=title)
    return relighted, mask, ill1, ill2


# pickle.dump([], open('./data/misc/illumination_diffs.pickle', 'wb'))
# pickle.dump(np.zeros(1), open('./data/misc/illumination_diffs.pickle', 'wb'))
# diffs = np.array(list(
#     map(lambda x: (get_diff_in_ill(load(x)), x), range(1, len(gts) + 1))))
diffs = pickle.load(open('./data/misc/illumination_diffs.pickle', 'rb'))
imgs = np.array(list(
    map(lambda x: x[1], filter(lambda x: x[0] < 160, diffs))))

exclusions = np.loadtxt('./data/misc/Exclusions.txt').astype(int)
possible = np.loadtxt('./data/misc/Possible_exclusions.txt').astype(int)
exclusions = np.append(exclusions, possible)
imgs = np.array(list(filter(lambda x: x not in exclusions, imgs)))
np.random.shuffle(imgs)
for img in imgs:
    image = load(img, depth=16)
    relighted, mask, ill1, ill2 = process_and_visualize(image, img)


gt_diff_filter = get_ill_diffs()
for idx in gt_diff_filter:
    rgb = load(idx)
    rgb_masked = load(idx, mask_cube=True)
