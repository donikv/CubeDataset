import os
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import itertools
import matplotlib as mpl
# import visualizer
from time import time

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange', 'red', 'm'])

# def plot_results(X, Y_, means, covariances, index, title):
#     splot = plt.subplot(2, 1, 1 + index)
#     for i, (mean, covar, color) in enumerate(zip(
#             means, covariances, color_iter)):
#         v, w = linalg.eigh(covar)
#         v = 2. * np.sqrt(2.) * np.sqrt(v)
#         u = w[0] / linalg.norm(w[0])
#         # as the DP will not use every component it has access to
#         # unless it needs it, we shouldn't plot the redundant
#         # components.
#         if not np.any(Y_ == i):
#             continue
#         plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)
#
#         # Plot an ellipse to show the Gaussian component
#         angle = np.arctan(u[1] / u[0])
#         angle = 180. * angle / np.pi  # convert to degrees
#         ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
#         ell.set_clip_box(splot.bbox)
#         ell.set_alpha(0.5)
#         splot.add_artist(ell)
#
#     plt.xticks(())
#     plt.yticks(())
#     plt.title(title)



dataset_path = '/Volumes/Jolteon/fax/'
folders = [
    'raws1/organized/',
    'raws3/organized/',
    'raws4/nikon/organized/',
    'raws6/organized/'
]
list_name = 'list.txt'

paths = [list(map(lambda x: dataset_path + f + x[2:], np.loadtxt(dataset_path + f + list_name, dtype=str))) for f in folders]

paths = np.concatenate(paths)

# image_paths = Cube2Dataset.load_paths(dataset_path, list_name)
# image_paths1 = list(filter(lambda x: x.find('canon') != -1, image_paths))
# image_paths2 = list(filter(lambda x: x.find('nikon') != -1, image_paths))
# image_paths3 = list(filter(lambda x: x.find('pixel') != -1, image_paths))


# path = '/home/donik/Disk/Cuben_collage/canon_550d/collage/1/'
# visualizer.visualize([Cube2Dataset.get_image(path, 'img.png', 256, 512, scaling=4)])
# image_paths_save = np.concatenate([image_paths1, image_paths2])
# np.savetxt(dataset_path + 'list_planck.txt', image_paths_save, fmt='%s')

# ip = [#("canon", image_paths1),
#       ("nikon", image_paths2),
#       #("pixel", image_paths3),
#       ]

avg_lum = []
fig = plt.figure()
ax = fig.add_subplot(111)

rg, bg = [], []

def normalize_brignthess(img):
    img = img / np.mean(np.linalg.norm(img, axis=-1))
    return img

#
rg1, rg2 = [], []
bg1, bg2 = [], []
# for name, paths in ip:
#     rg1, rg2 = [], []
#     bg1, bg2 = [], []
for image_path in paths:
    try:
        gt = np.flip(np.loadtxt(image_path + "/gt.txt"), axis=0).reshape((-1,))
    except:
        continue
    gt = np.array(gt)
    gt = np.reshape(gt, (-1, 3))
    # gt = gt / gw
    gt = np.array([np.array([x[0] / x[1], x[2] / x[1]]) for x in gt])
    # if gt[0][0] > 0.57 and gt[0][1] < 0.7 :
    #     img = NewCube2Dataset.get_image(image_path, 'img.png', 256, 512, scaling=4)
    #     mask = Cube2Dataset.get_image(image_path, 'gt.png', 256, 512, scaling=1)
    #     visualizer.visualize([img, mask], title=image_path)
    #     print(image_path)
    rg1.append(gt[0][0])
    bg1.append(gt[0][1])
    rg2.append(gt[1][0])
    bg2.append(gt[1][1])

    rg.append(rg1[-1])
    rg.append(rg2[-1])
    bg.append(bg1[-1])
    bg.append(bg2[-1])

ax.scatter(rg1, bg1, label='first', s=1)
ax.scatter(rg2, bg2, label='second', s=1)
# # rg1 = np.array(rg1)
# # rg2 = np.array(rg2)
# # bg1 = np.array(bg1)
# # bg2 = np.array(bg2)
# # rgbg1 = np.dstack([rg1, bg1])[0]
# # rgbg2 = np.dstack([rg2, bg2])[0]
# #
# # for a in [0.1, 0.5, 0.7, 0.95, 1.2, 1.5, 2.0, 2.5, 3.5]:
# #     # ash = alphashape.alphashape(rgbg1, a)
# #     # pp = PolygonPatch(ash, alpha=0.2)
# #     # ax.add_patch(pp)
# #
# #     ash1 = alphashape.alphashape(rgbg2, a)
# #     pp1 = PolygonPatch(ash1, alpha=0.2)
# #     ax.add_patch(pp1)
plt.legend(loc='upper right')
plt.show()


#
# mean1 = np.mean(rgbg1, axis=0)
# mean2 = np.mean(rgbg2, axis=0)
#
# cov1 = np.cov(rgbg1, rowvar=False)
# cov2 = np.cov(rgbg2, rowvar=False)
#
# means = np.stack([mean1, mean2])
# covs = np.stack([cov1, cov2])
#
# X = np.concatenate([rgbg1, rgbg2], axis=0)
# Y = np.concatenate([np.zeros_like(rg1), np.ones_like(rg2)])
#
# generated1 = np.random.multivariate_normal(means[0], covs[0], 1000)
# generated2 = np.random.multivariate_normal(means[1], covs[1], 1000)
# # generated2 = np.array(list(map(lambda y: y + np.random.normal(0, 0.1), yplt)))
# # generated2 = np.random.multivariate_normal(gmm.means_[1], gmm.covariances_[1], 1000)
# ax.scatter(generated1[..., 0], generated1[..., 1], s=3)
# ax.scatter(generated2[..., 0], generated2[..., 1], s=3)
# plt.show()
#
# plot_results(X, Y, means, covs, 0, "Gaussians")
# plt.show()
#
# np.savetxt('data/outdoor_meas.txt', means)
# np.savetxt('data/outdoor_covs.txt', covs.reshape((-1, 2)))
import CubeDataset
import histogram
cube_gts = np.loadtxt('/media/donik/Disk/Cube+/cube+_right_gt.txt')
cube_gts = tf.constant(cube_gts, dtype=tf.float32)

gts_hist1 = histogram.flatten_image(cube_gts, depth = 256)

cube_gts = np.loadtxt('/media/donik/Disk/Cube+/cube+_left_gt.txt')
cube_gts = tf.constant(cube_gts, dtype=tf.float32)

gts_hist2 = histogram.flatten_image(cube_gts, depth = 256)
diff = gts_hist2 - gts_hist1
visualizer.visualize([tf.reshape(gts_hist1, (256, 256)), tf.reshape(gts_hist2, (256, 256)), tf.reshape(diff, (256, 256))])
np.savetxt("data/uv_hist_bias.txt", gts_hist1.numpy().reshape((-1,1)))

cube_paths = CubeDataset.load_image_names('paths.txt', '/media/donik/Disk/Cube+')
indices = np.array(list(map(lambda x: int(x[x.rfind('/') + 1:-4]) - 1, cube_paths)))
rg = []
bg = []
for i, cube_path in zip(indices, cube_paths):
    gt = cube_gts[i]
    img = CubeDataset.get_image(cube_path)
    # visualizer.visualize([img])
    gw = tf.reduce_max(img, axis=[0, 1]).numpy()
    gw = gw / gw[1]

    gt = gt / gw

    rg.append(gt[0] / gt[1])
    bg.append(gt[2] / gt[1])
#
# # ax = fig.add_subplot(312)
# plt.legend(loc='upper right')
# plt.show()
#

import TauDataset
image_paths = list(map(lambda x: '/media/donik/Disk/intel_tau' + x, np.loadtxt('/media/donik/Disk/intel_tau/paths_field.txt', dtype=str)))
image_paths1 = list(filter(lambda x: x.find('Nikon') != -1, image_paths))
image_paths2 = list(filter(lambda x: x.find('Canon') != -1, image_paths))
image_paths3 = list(filter(lambda x: x.find('Sony') != -1, image_paths))
ip = [("nikon", image_paths1),
      ("canon", image_paths2),
      ("sony", image_paths3),]

# ax = fig.add_subplot(313)
rgt = []
bgt = []
with tf.device('/device:CPU:0'):
    for name, image_paths in ip:
        for image_path in image_paths:
            gt = np.loadtxt(image_path + '.wp', delimiter=',')
            img = TauDataset.get_image(image_path)
            gw = tf.reduce_max(img, axis=[0, 1]).numpy()
            gw = gw / gw[1]
            gt = gt / gw

            rgt.append(gt[0] / gt[1])
            bgt.append(gt[2] / gt[1])
            # rg.append(rgt[-1])
            # bg.append(bgt[-1])

ax.scatter(rgt, bgt, label='tau', s=1)
ax.scatter(rg, bg, label='cube+', s=1)
#
plt.legend(loc='upper right')
plt.show()


# feat = np.dstack((rg, bg)).reshape((-1,2))
# np.savetxt('data/rgbg.txt', feat)
feat = np.loadtxt('data/rgbg.txt')[:len(image_paths)*2]
# for a in [0.1, 0.5, 0.7, 0.95, 1.2, 1.5, 2.0, 2.5, 3.5]:
ash = alphashape.alphashape(feat, 15)
pp = PolygonPatch(ash, alpha=0.2)

hull = ConvexHull(feat)
hull_points = feat[hull.vertices]
fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(feat[1:len(image_paths)*2:2, 0], feat[1:len(image_paths)*2:2, 1], s=3)
ax.scatter(feat[0:len(image_paths)*2:2, 0], feat[0:len(image_paths)*2:2, 1], s=3)
ax.scatter(feat[1:len(image_paths)*2:2, 0].mean(), feat[1:len(image_paths)*2:2, 1].mean(), s=10)
ax.add_patch(pp)
plt.show()
# scaler = StandardScaler()
# feat = scaler.fit_transform(feat)
gmm = BayesianGaussianMixture(n_components=2, max_iter=10000)
gmm.fit(feat)

# lr = make_pipeline(PolynomialFeatures(4), Ridge(0.3))
#
# x = np.linspace(0, 2, 100)
# xplot = x[:, np.newaxis]
# lr.fit(feat[..., :1], feat[..., 1:])
# yplt = lr.predict(xplot)

plot_results(feat, gmm.predict(feat), gmm.means_, gmm.covariances_, 0, "Bayesian Gaussian n=6")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(211)

ax.scatter(feat[..., 0], feat[..., 1], s=3)
generated1 = np.random.multivariate_normal(gmm.means_[0], gmm.covariances_[0], 1000)
# generated2 = np.array(list(map(lambda y: y + np.random.normal(0, 0.1), yplt)))
# generated2 = np.random.multivariate_normal(gmm.means_[1], gmm.covariances_[1], 1000)
ax.scatter(generated1[..., 0], generated1[..., 1], s=3)
# ax.plot(xplot, yplt)
# ax.scatter(xplot[..., 0], generated2[..., 0], s=3)



import coloring

# ills = list(map(lambda x: coloring.create_illuminant(x), range(1500, 10000, 50)))
ills = np.loadtxt('data/planck.txt')
rg_planck = list(map(lambda x: x[0]/x[1], ills))
bg_planck = list(map(lambda x: x[2]/x[1], ills))
ax.scatter(rg_planck, bg_planck, s=1)

plt.show()

from joblib import dump, load
# dump(lr, 'data/gt_approx_poly2')
