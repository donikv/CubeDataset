import numpy as np
import matplotlib.pyplot as plt
import utils.relighting_utils as ru
import itertools
from mpl_toolkits import mplot3d

#paths = np.loadtxt('/media/donik/Disk/Cube2/list_outdoor.txt', dtype=str)

path = '/Volumes/Jolteon/fax/to_process/organized'
l = np.loadtxt(f'{path}/list.txt', dtype=str)
paths = [path + x[1:] for x in l]

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

    ca = np.arccos(cos_dist(a,b))
    cb = np.arccos(cos_dist(-a, c))
    return ca, cb

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

def get_diff_to_angles(diffs):
    diff_combination = list(itertools.combinations(range(3), 2))
    xyz = []
    for file, angles in diffs.items():
        coordiantes = np.loadtxt(file + '/cube.txt')
        for i, comb in enumerate(diff_combination):
            ca1, cb1 = get_angles_from_verts(coordiantes[comb[0]])
            ca2, cb2 = get_angles_from_verts(coordiantes[comb[1]])
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
            xyz.append([abs(a1-a2), diff])
    return xyz

def plot3d_differences_to_angles(xyz):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.xlim((0,1))
    plt.ylim((0,1))
    ax.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=xyz[:, 2])
    plt.show()

def plot3d_histogram(xy, bns = 25, rangex=(0,1), rangey=(0,25)):
    plt.figure()
    ax = plt.axes(projection='3d')
    x, y = xy[:,0], xy[:,1]
    x = x / x.max()

    hist, xedges, yedges = np.histogram2d(x, y, bins=bns, range=[rangex, rangey])

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + rangex[1] / bns, yedges[:-1] + + rangey[1] / bns, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dy = (rangey[1] / bns / 2) * np.ones_like(zpos)
    dx = (rangex[1] / bns / 2) * np.ones_like(zpos)
    dz = hist.ravel()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

    plt.show()

def plot2d_scatter(xy, xlim, ylim):
    plt.figure()
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.scatter(xy[:, 0], xy[:, 1])

diffs = explore_gts(paths)
hist = plot_diff_histogram(diffs)

xyz = np.array(get_diff_to_angles(diffs))
plot3d_differences_to_angles(xyz)

xy = np.array(get_diff_to_area(diffs))
plot2d_scatter(xy, (0,300), (0,15))

plot3d_histogram(xy)


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