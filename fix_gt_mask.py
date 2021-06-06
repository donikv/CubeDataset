import numpy as np
import cv2
import os

def fix(i):
    gt_mask = cv2.imread(i + '/gt_mask.png', cv2.IMREAD_UNCHANGED)[..., 0:3]
    cv2.imwrite(i + '/gt_mask.png', gt_mask)
    gts = np.loadtxt(i + "/gt.txt")
    gts = gts.reshape((2,-1))
    n = np.linalg.norm(gts, axis=-1, keepdims=True)
    gts = gts / (n + 1e-10)
    np.savetxt(i + "/gt.txt", gts, fmt='%.7f')
    gts = gts * 255
    gts = gts.astype(np.uint8).reshape((2,-1))

    gt = np.where(gt_mask == 2, gts[1], gts[0])
    cv2.imwrite(i + f'/gt.png', cv2.cvtColor(gt, cv2.COLOR_RGB2BGR))

def calculate_bb(i):
    triangles = np.loadtxt(i + '/cube.txt')
    gt_mask = cv2.imread(i + '/gt_mask.png', cv2.IMREAD_UNCHANGED)[..., 0:3]
    cubes = []
    for verts in triangles:
        verts = verts.reshape((-1,2))
        center = verts.mean(axis=0)
        o = 0
        for j in range(len(verts)):
            o += np.linalg.norm(verts[j] - verts[(j+1)%len(verts)])
        x1 = np.maximum(center[0] - o, 0)
        y1 = np.maximum(center[1] - o/2, 0)
        x2 = np.minimum(center[0] + o, gt_mask.shape[1])
        y2 = np.minimum(center[1] + o, gt_mask.shape[0])
        cube = np.array([x1,y1,x2,y2])
        cubes.append(cube)
    np.savetxt(i + '/cubes.txt', np.array(cubes))

def calculate_bb_old(i):
    centers = np.loadtxt(i + '/cube.txt').reshape((-1,2))
    gt_mask = cv2.imread(i + '/gt_mask.png', cv2.IMREAD_UNCHANGED)[..., 0:3]
    cubes = []
    for center in centers:
        o = 100
        x1 = np.maximum(center[0] - o, 0)
        y1 = np.maximum(center[1] - o / 2, 0)
        x2 = np.minimum(center[0] + o, gt_mask.shape[1])
        y2 = np.minimum(center[1] + o, gt_mask.shape[0])
        cube = np.array([x1, y1, x2, y2])
        cubes.append(cube)
    np.savetxt(i + '/cubes.txt', np.array(cubes))

def organize_gts(i):
    gts = np.loadtxt(i + '/gts.txt')
    np.savetxt(i + '/shadow.txt', gts[:-1])
    np.savetxt(i + '/sun.txt', gts[-1:])

def organize_gts_old(i):
    gts = np.loadtxt(i + '/gt.txt')
    np.savetxt(i + '/shadow.txt', gts[:-1])
    np.savetxt(i + '/sun.txt', gts[-1:])

def create_gt(i):
    gts = np.loadtxt(i + '/gt.txt')
    gt_mask = cv2.imread(i + '/gt_mask.png', cv2.IMREAD_UNCHANGED)[..., 0:1]
    gts = gts * 255
    gts = gts.astype(np.uint8).reshape((2, -1))

    gt = np.where(gt_mask > 1, gts[1], gts[0])
    cv2.imwrite(i + f'/gt.png', cv2.cvtColor(gt, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    import time

    # start = time.time_ns()
    # path = '/Volumes/Jolteon/fax/to_process/organized'
    # l = np.loadtxt(f'{path}/list.txt', dtype=str)

    path = '/media/donik/Disk/Cube2_new_/'
    # path = '/media/donik/Disk/Cube2'
    # path = '/Volumes/Jolteon/fax/to_process/organized2'
    l = np.loadtxt(f'{path}/list_0_5_deg.txt', dtype=str)
    # l = list(filter(lambda x: (x.find('canon') != -1 and x.find('outdoor1') == -1), l))
    l = np.array(list(filter(lambda x: x.find('outdoor4') != -1, l)))

    imgs = [path + x for x in l]

    for i in imgs:
        # img = cv2.imread(i + '/img.png', cv2.IMREAD_UNCHANGED)[..., 0:3]
        # print(img.min())
        create_gt(i)
