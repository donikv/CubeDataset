#%%
import numpy as np
import matplotlib.pyplot as plt
import utils.relighting_utils as ru
import itertools
from mpl_toolkits import mplot3d
import process_outdoor_multi
import cv2

#paths = np.loadtxt('/media/donik/Disk/Cube2/list_outdoor.txt', dtype=str)
#%%
# path = '/media/donik/Disk/Cube2_new'
path = '/Volumes/Jolteon/fax/to_process/organized'
l = np.loadtxt(f'{path}/list.txt', dtype=str)
paths = [path + x[1:] for x in l]

#%%
def explore_gts(paths):
    diff_combination = list(itertools.combinations(range(3),2))
    diffs = {}
    for p in paths:
        gts = np.loadtxt(p + '/gts.txt')
        print("-" * 10)
        print(p)
        for comb in diff_combination:
            diff = ru.angular_distance(gts[comb[0]], gts[comb[1]])
            print(f"{comb} -> {diff}")
            if p not in diffs:
                diffs[p] = [diff]
            else:
                diffs[p].append(diff)
    return diffs

def create_new_gt_from_diff(pdiffs):
    diff_combination = list(itertools.combinations(range(3), 2))
    for p, diffs in pdiffs.items():
        gts = np.loadtxt(p + '/gts.txt')
        i = np.argmin(diffs)
        comb = diff_combination[i]
        gt = (gts[comb[0]] + gts[comb[1]]) / 2
        new_gts = np.array([gt, gts[-1]])
        np.savetxt(p + '/gt.txt', new_gts)


def plot_diff_histogram(diffs):
    vals = []
    for k,v in diffs.items():
        vals.append(v)
    vals = np.array(vals).reshape((-1,))
    hist = np.histogram(vals, 50, (0, 25))
    plt.hist(vals, 50, (0, 25))
    plt.show()
    return hist

def get_angles_from_verts(verts):
    verts = np.reshape(verts, (-1,2))
    a = verts[1] - verts[0]
    b = verts[2] - verts[0]
    c = verts[2] - verts[1]

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    nc = np.linalg.norm(c)

    def cos_dist(a,b):
        return np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)

    ca = cos_dist(a,b)
    cb = cos_dist(b, -c)
    cc = cos_dist(-a, -c)

    # ca = np.arccos(ca)
    # cb = np.arccos(cb)
    return ca, cb, cc

def get_area_from_verts(verts):
    verts = np.reshape(verts, (-1,2))
    a = verts[1] - verts[0]
    b = verts[2] - verts[0]
    c = verts[2] - verts[1]

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    nc = np.linalg.norm(c)

    p = na + nb + nc
    a = np.sqrt(p/2 * (p/2 - na) * (p/2 - nc) * (p/2 - nb))

    return p, a

def get_lum_from_verts(verts, img):
    verts = np.reshape(verts, (-1,2))
    gt = process_outdoor_multi.get_gt_from_cube_triangle(verts, img, size=img.shape[0:2])
    lum = np.linalg.norm(gt)
    return lum

def get_center_from_verts(verts):
    verts = np.reshape(verts, (-1,2))
    center = np.mean(verts, axis=0)
    return center[0], center[1]


def get_diff_to_angles(diffs):
    diff_combination = list(itertools.combinations(range(3), 2))
    xyz = []
    for file, angles in diffs.items():
        coordiantes = np.loadtxt(file + '/cube.txt')
        for i, comb in enumerate(diff_combination):
            ca1, cb1, cc1 = get_angles_from_verts(coordiantes[comb[0]])
            ca2, cb2, cc2 = get_angles_from_verts(coordiantes[comb[1]])
            diff = angles[i]
            xyz.append([abs(ca1-ca2), abs(cb1-cb2), abs(cc1-cc2), diff])
    return xyz

def get_diff_to_centers(diffs):
    diff_combination = list(itertools.combinations(range(3), 2))
    xyz = []
    for file, angles in diffs.items():
        coordiantes = np.loadtxt(file + '/cube.txt')
        for i, comb in enumerate(diff_combination):
            ca1, cb1 = get_center_from_verts(coordiantes[comb[0]])
            ca2, cb2 = get_center_from_verts(coordiantes[comb[1]])
            diff = angles[i]
            xyz.append([abs(ca1-ca2), abs(cb1-cb2), diff])
    return xyz

def get_diff_to_area(diffs):
    diff_combination = list(itertools.combinations(range(3), 2))
    xyz = []
    for file, angles in diffs.items():
        coordiantes = np.loadtxt(file + '/cube.txt')
        for i, comb in enumerate(diff_combination):
            p1, a1 = get_area_from_verts(coordiantes[comb[0]])
            p2, a2 = get_area_from_verts(coordiantes[comb[1]])
            diff = angles[i]
            xyz.append([a1-a2, diff])
    return xyz

def get_diff_to_lum(diffs):
    diff_combination = list(itertools.combinations(range(3), 2))
    xyz = []
    for file, angles in diffs.items():
        coordiantes = np.loadtxt(file + '/cube.txt')
        img = cv2.imread(file + '/img.png', cv2.IMREAD_UNCHANGED)
        for i, comb in enumerate(diff_combination):
            l1 = get_lum_from_verts(coordiantes[comb[0]], img)
            l2 = get_lum_from_verts(coordiantes[comb[1]], img)
            diff = angles[i]
            xyz.append([l1-l2, diff])
    return xyz

def plot3d_differences_to_angles(xyz, adaptive=False):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    if adaptive:
        plt.xlim((xyz[:,0].min(),xyz[:, 0].max()))
        plt.ylim((xyz[:,1].min(),xyz[:, 1].max()))
    else:
        plt.xlim((0,1))
        plt.ylim((0,1))
    ax.set_zlim((0,15))
    ax.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=xyz[:, 2])
    plt.show()

def plot3d_histogram(xy, bns = 25, rangex=(0,1), rangey=(0,25)):
    plt.figure()
    ax = plt.axes(projection='3d')
    x, y = xy[:,0], xy[:,1]
    # x = x / x.max()

    hist, xedges, yedges = np.histogram2d(x, y, bins=bns, range=[rangex, rangey])

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + (rangex[1]) / bns, yedges[:-1] + (rangey[1]) / bns, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dy = ((rangey[1]) / bns / 2) * np.ones_like(zpos)
    dx = ((rangex[1]) / bns / 2) * np.ones_like(zpos)
    dz = hist.ravel()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

    plt.show()

def plot2d_scatter(xy, xlim, ylim):
    x = xy[:, 0]
    # x = x / x.()
    y = xy[:, 1]
    plt.figure()
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.scatter(x,y)
#%%
diffs = explore_gts(paths)
# create_new_gt_from_diff(diffs)
# hist = plot_diff_histogram(diffs)
diffs2 = {}
diffs3 = {}
diffs4 = {}
diffs5 = {}
for k,v in diffs.items():
    diffs2[k] = [np.min(v)]
    diffs3[k] = [np.max(v)]
    diffs4[k] = [np.mean(v)]
    diffs5[k] = [np.std(v)]
# plot_diff_histogram(diffs2)
# plot_diff_histogram(diffs4)
# plot_diff_histogram(diffs3)
# plot_diff_histogram(diffs5)

features = []

diffslowest = list(diffs4.items())
diffslowest = list(sorted(diffslowest, key = lambda x: x[1]))
for fn, diff in diffslowest:
    if diff[0] > 1:
        break
    print(fn)


#%%
#LUMINANCE
# xyl = get_diff_to_lum(diffs)
# np.save('data/xyl.npy', xyl)
xyl = np.load('data/xyl.npy')
plot2d_scatter(xyl, (-5000,7000), (0,15))
features.append(xyl[:,:-1])
xyl = np.abs(xyl)
print(np.corrcoef(xyl[:,0], xyl[:, 1]))

# plot3d_histogram(xyl, 25, (0,7000), rangey=(0,15))

#%%
#ANGLES
xyz = np.array(get_diff_to_angles(diffs))
features.append(xyz[:, :-1])
xyz[:, 0] = xyz[:, 0] / xyz[:, 0].max()
xyz[:, 1] = xyz[:, 1] / xyz[:, 1].max()
xyz = np.concatenate([xyz[:,0:2], xyz[:, -1:]], axis=-1)
plot3d_differences_to_angles(xyz)

#%%
#CENTERS
xyc = np.array(get_diff_to_centers(diffs))
features.append(xyc[:,:-1])
xyc[:, 0] = xyc[:, 0] / xyc[:, 0].max()
xyc[:, 1] = xyc[:, 1] / xyc[:, 1].max()
plot3d_differences_to_angles(xyc)

cy = np.ones_like(xyc[:, :2])
cy[:,0] = np.sqrt(xyc[:,0]**2 + xyc[:,1]**2)
cy[:,1] = xyc[:, 2]

print(np.corrcoef(cy[:,0], cy[:, 1]))
plot2d_scatter(cy, (0,1.41), (0, 20))
plot3d_histogram(cy, rangex=(0,1.5), rangey=(0,10))

#%%
#AREA
xy = np.array(get_diff_to_area(diffs))
features.append(xy)
xy = np.abs(xy)
print(np.corrcoef(xy[:,0], xy[:, 1]))
# xy[:,0] = xy[:, 0] + xy[:, 0].min()
plot2d_scatter(xy, (0,300), (0,10))
plot3d_histogram(xy, rangex=(0,300), rangey=(0,10))

features = np.concatenate(features, axis=-1)

#%%
#AREA LUMINANCE
# xyl[:,0] = xyl[:,0] / xyl[:, 0].max()
cy[:,0] = (cy[:,0] - cy[:, 0].min()) / (cy[:, 0].max() - cy[:, 0].min())
xy[:,0] = (xy[:,0] - xy[:, 0].min()) / (xy[:, 0].max() - xy[:, 0].min())
alz = np.concatenate([xy[:, 0:1], cy], axis=-1)
plot3d_differences_to_angles(alz, adaptive=True)

#%%
def plt_gts(paths):
    gts_nikon_shadow = []
    gts_nikon_sun = []
    gts_canon_shadow = []
    gts_canon_sun = []
    for p in paths:
        p = '/media/donik/Disk/Cube2/' + p[2:] + '/gt.txt'
        try:
            gts = np.loadtxt(p)
            gts = gts.reshape([2,3])
        except:
            continue
        gt_shadow = gts[0]
        gt_sun = gts[1]

        if p.find('canon_550d/outdoor2') != -1:
            gts_canon_shadow.append(gt_shadow)
            gts_canon_sun.append(gt_sun)
        else:
            gts_nikon_shadow.append(gt_shadow)
            gts_nikon_sun.append(gt_sun)

    def map_gts(l):
        def f(gt):
            rg = gt[0] / gt[1]
            bg = gt[2] / gt[1]
            return np.array([rg,bg])
        return np.array(list(map(f, l)))

    gts_canon_shadow = map_gts(gts_canon_shadow)
    gts_nikon_shadow = map_gts(gts_nikon_shadow)
    gts_canon_sun = map_gts(gts_canon_sun)
    gts_nikon_sun = map_gts(gts_nikon_sun)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(gts_canon_shadow[...,0], gts_canon_shadow[...,1], label='Canon shadow', s=1)
    ax.scatter(gts_canon_sun[...,0], gts_canon_sun[...,1], label='Canon sun', s=1)
    ax.scatter(gts_nikon_shadow[...,0], gts_nikon_shadow[...,1], label='Nikon shadow', s=1)
    ax.scatter(gts_nikon_sun[...,0], gts_nikon_sun[...,1], label='Nikon sun', s=1)

    plt.legend(loc='upper right')
    plt.show()