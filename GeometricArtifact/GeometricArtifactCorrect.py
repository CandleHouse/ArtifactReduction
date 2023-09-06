import math
import os
import cv2
import numpy as np
import tifffile
from crip.io import *
from crip.preprocess import *
from crip.postprocess import *
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
from utils import *
from findmaxima2d import find_maxima, find_local_maxima


def helix_points(radius, phi, screwPitch, numPerLap, number, startAngle=0, show=False):
    """
    :param radius: steel ball distribution radius (mm)
    :param phi: angle in slice direction between steel balls (degree)
    :param screwPitch: steel ball screw pitch (mm)
    :param numPerLap: steel ball number per lap
    :param number: steel ball number
    :param startAngle: first bottom steel ball start angle (degree)
    :return: steel balls' 3-D coordinates
    """
    x_list = []
    y_list = []
    z_list = []
    for i in range(number):
        rad = np.deg2rad(i * phi + startAngle)
        x = radius * np.cos(rad)
        y = radius * np.sin(rad)
        z = i * screwPitch / numPerLap
        x_list.append(x), y_list.append(y), z_list.append(z)

    if show:
        ax = plt.subplot(projection='3d')
        ax.scatter(x_list, y_list, z_list)
        ax.plot(x_list, y_list, z_list)
        ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
        ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
        ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
        plt.show()

    return x_list, y_list, z_list


def findImgPoints(img, physicalSize, show=False):
    balls_list = []
    ball_list = []
    for slice in range(img.shape[0]):
        if np.sum(img[slice]) != 0:
            ball_list.append(slice)

        elif ball_list:
            balls_list.append(ball_list)
            ball_list = []

    points = []
    ntol = 5
    for ball_list in balls_list:
        z = math.ceil(np.median(ball_list))
        # use ImageJ implemented function to find steel balls
        local_max = find_local_maxima(img_data=img[z])
        y, x, regs = find_maxima(img[z], local_max, ntol)  # regs now useless
        points.append([z, y[0], x[0]])
    points = np.array(points)

    if len(balls_list) != 25:
        print('Detect failed!')
        exit(0)

    dx, dy, dz = physicalSize
    x_list, y_list, z_list = points[:, 2] * dx, points[:, 1] * dy, points[:, 0] * dz

    if show:
        ax = plt.subplot(projection='3d')
        ax.scatter(x_list, y_list, z_list)
        ax.plot(x_list, y_list, z_list)
        ax.set_zlabel('iz×dz', fontdict={'size': 15, 'color': 'red'})
        ax.set_ylabel('iy×dy', fontdict={'size': 15, 'color': 'red'})
        ax.set_xlabel('ix×dx', fontdict={'size': 15, 'color': 'red'})
        plt.show()

    return x_list, y_list, z_list


if __name__ == '__main__':
    ### 1. Recon
    phantom = pack_dcm(r'/20220914_螺旋钢珠模体')  # (420, 976, 976)
    I0 = pack_dcm(r'/20220914_桌架')  # (420, 976, 976)

    # I0 = averageProjections  # if detector has broken line or point, drop this function
    for v in range(phantom.shape[0]):
        phantom[v] = -np.log(np.divide(phantom[v].T + 1e-5, I0[v].T + 1e-5) + 1e-5)
    sgm = projectionsToSinograms(phantom)
    sgm.tofile('./sgm/sgm_phantom0914.raw')

    ### 2. Blueprint to Recon
    src_x, src_y, src_z = helix_points(radius=52.2, phi=30, screwPitch=60, numPerLap=12, number=25, startAngle=0, show=False)

    rec_img = np.fromfile(r'./rec/rec_phantom0914_raw.raw', dtype=np.float32).reshape(512, 512, 512)
    retval, rec_binary = cv2.threshold(rec_img, thresh=0.3, maxval=255, type=cv2.THRESH_BINARY)
    rec_binary = rec_binary.astype(np.uint8)
    dst_x, dst_y, dst_z = findImgPoints(rec_binary, physicalSize=(0.3, 0.3, 0.3), show=False)

    matrix = getPerspectiveTransformMatrix3D((src_x, src_y, src_z), (dst_x, dst_y, dst_z))

    X, Y, Z = [], [], []
    for i in range(len(src_x)):
        mx, my, mz, m = matrix @ np.array([src_x[i], src_y[i], src_z[i], 1]).T
        X.append(mx / m), Y.append(my / m), Z.append(mz / m)

    ax = plt.subplot(projection='3d')
    ax.scatter(X, Y, Z)
    ax.plot(X, Y, Z)
    ax.scatter(dst_x, dst_y, dst_z)
    ax.plot(dst_x, dst_y, dst_z)
    plt.legend(['Blueprint', '', 'Recon'])
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.show()

    ### 3. projection fit
    sgm_phantom = np.fromfile(r'sgm/sgm_phantom0914.raw', dtype=np.float32).reshape(976, 420, 976)
    proj_phantom = sinogramsToProjections(sgm_phantom)

    proj_list = []
    for i in range(proj_phantom.shape[0]):
        proj = proj_phantom[i]
        proj[proj < 0] = 0
        proj = (proj / 5 * 255).astype(np.uint8)
        proj = cv2.GaussianBlur(proj, ksize=None, sigmaX=1, sigmaY=1)
        proj_list.append(proj)

    tifffile.imwrite('./smUint8/img_gray0914.tif', np.array(proj_list, dtype=np.uint8))

    proj_phantom = tifffile.imread('./smUint8/img_gray0914.tif')
    ntol = 20
    ROI_edge = 50  # pixel

    for view in trange(proj_phantom.shape[0]):
        proj = proj_phantom[view]
        # use ImageJ implemented function to find steel balls
        local_max = find_local_maxima(img_data=proj)
        v, u, regs = find_maxima(proj, local_max, ntol)  # regs now useless

        delete_list = []
        for points in range(len(v)):
            if v[points] < ROI_edge or v[points] > 976-ROI_edge or u[points] < ROI_edge or u[points] > 976-ROI_edge:
                delete_list.append([points])

        v = np.delete(v, delete_list)
        u = np.delete(u, delete_list)

        if len(u) == 25:
            writePoints('phantom0914', u, v, view)
        else:
            writePoints('phantom0914', u, v, view)
            print(f'view{view} is less than 25 points')

    ### 4. generate pMatrix
    Views = 420
    pMatrix = np.zeros((Views, 3, 4), dtype=np.float32)
    i = 0

    X = np.array(X) - 256 * 0.3
    Y = 256 * 0.3 - np.array(Y)
    Z = np.array(Z) - 256 * 0.3

    for filePath in listDirectory('./points/phantom0914', style='fullpath'):
        U, V = np.array(readPoints(filePath))
        pMatrix[i] = getProjectionTransformMatrix3D((X, Y, Z), (U, V), detectorElementSize=0.3036)
        i += 1

    pMatrix_file = dict(Value=pMatrix.flatten())
    with open(fr'./params/pmatrix_file0914.jsonc', 'w') as f:
        f.write(json.dumps(pMatrix_file, cls=NumpyEncoder))

    pMatrix_gt = read_paramsFile('./params/pmatrix_file_gt.jsonc').reshape(420, 3, 4)

    u_pred_list, u_gt_list = [], []
    v_pred_list, v_gt_list = [], []

    for v in range(Views):
        mu, mv, m = pMatrix[v] @ np.array([X[0], Y[0], Z[0], 1])
        u_pred_list.append(mu / m), v_pred_list.append(mv / m)
        mu, mv, m = pMatrix_gt[v] @ np.array([X[0], Y[0], Z[0], 1])
        u_gt_list.append(mu / m), v_gt_list.append(mv / m)

    plt.plot(range(Views), u_pred_list, range(Views), u_gt_list)
    plt.legend(['u_pred', 'u_gt'])
    plt.xlabel('views'), plt.ylabel('u_pos')
    plt.show()
    plt.plot(range(Views), v_pred_list, range(Views), v_gt_list)
    plt.legend(['v_pred', 'v_gt'])
    plt.xlabel('views'), plt.ylabel('v_pos')
    plt.show()

    gen_pMatrix(projOffsetU_list=np.zeros(420), projOffsetV_list=np.zeros(420))
