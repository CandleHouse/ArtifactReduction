import json
from crip.io import *
from crip.preprocess import *
from crip.postprocess import *
from crip.shared import *
from tqdm import tqdm
import pydicom
import numpy as np


def pack_dcm(dcm_path) -> np.array:
    img_list = []
    for file in tqdm(listDirectory(dcm_path, style='fullpath')):
        img = pydicom.read_file(file, force=True)
        img.SamplesPerPixel = 1
        img.PhotometricInterpretation = 1
        img = img.pixel_array.astype(np.float32)
        img_list.append(img)

    return stackImages(img_list)


def getPerspectiveTransformMatrix3D(src, dst):
    X, Y, Z = src
    A, B, C = dst
    n = len(X)

    S = np.zeros((3 * n, 15))
    for i in range(n):
        S[3*i+0][0:4] = np.array([X[i], Y[i], Z[i], 1])
        S[3*i+1][4:8] = np.array([X[i], Y[i], Z[i], 1])
        S[3*i+2][8:12] = np.array([X[i], Y[i], Z[i], 1])

        S[3*i+0][12:15] = np.array([-A[i]*X[i], -A[i]*Y[i], -A[i]*Z[i]])
        S[3*i+1][12:15] = np.array([-B[i]*X[i], -B[i]*Y[i], -B[i]*Z[i]])
        S[3*i+2][12:15] = np.array([-C[i]*X[i], -C[i]*Y[i], -C[i]*Z[i]])

    Bias = np.zeros(3 * n)
    for i in range(n):
        Bias[3*i+0] = A[i]
        Bias[3*i+1] = B[i]
        Bias[3*i+2] = C[i]

    Unknown = np.linalg.inv(S.T @ S) @ S.T @ Bias
    matrix = np.append(Unknown, 1).reshape(4, 4)

    return matrix


def getProjectionTransformMatrix3D(src, dst, detectorElementSize):
    X, Y, Z = src
    U, V = dst
    S = np.zeros((len(X) * 2, 11))
    for i in range(len(X)):
        S[2 * i][0:4] = np.array([-X[i], -Y[i], -Z[i], -1])
        S[2 * i][8:11] = np.array([U[i] * X[i], U[i] * Y[i], U[i] * Z[i]])

        S[2 * i + 1][4:8] = np.array([-X[i], -Y[i], -Z[i], -1])
        S[2 * i + 1][8:11] = np.array([V[i] * X[i], V[i] * Y[i], V[i] * Z[i]])

    C = np.zeros(len(X) * 2)
    for i in range(len(X)):
        C[2 * i] = -U[i]
        C[2 * i + 1] = -V[i]

    Unknown = np.linalg.inv(S.T @ S) @ S.T @ C
    P = np.append(Unknown, 1).reshape(3, 4)
    A_inv = P[:, :3]
    A = np.linalg.inv(A_inv)

    e_u = A[:, 0]
    L_u_cal = np.linalg.norm(e_u)
    ratio = L_u_cal / detectorElementSize

    new_P = ratio * P

    # new_A_inv = new_P[:, :3]
    # new_A = np.linalg.inv(new_A_inv)
    #
    # new_e_u = new_A[:, 0]
    # new_e_v = new_A[:, 1]
    # x_do_minus_x_s = new_A[:, 2]
    #
    # x_s = new_A @ -new_P[:, 3]
    # x_do = x_do_minus_x_s + x_s
    #
    # print(np.linalg.norm(new_e_u))
    # print(np.linalg.norm(new_e_v))
    # print(np.linalg.norm(x_do_minus_x_s))

    return new_P


def readPoints(filePath):
    X, Y = [], []
    with open(filePath, 'r') as f:
        next(f)
        for line in f.readlines():
            _, x, y = line.strip('\n').split(',')
            X.append(int(x)), Y.append(int(y))

    return X, Y


def writePoints(name, u, v, view):
    with open(f'./points/{name}/view{str(view).zfill(3)}.csv', 'w') as f:
        f.write(' ,X,Y\n')

        for i in range(len(u)):
            f.write(str(i+1) + ',' + str(int(u[i])) + ',' + str(int(v[i])) + '\n')


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def gen_pMatrix(projOffsetU_list, projOffsetV_list):
    Views = 420
    TotalScanAngle = 189  # inverse direction, same as config file
    anglePerView = TotalScanAngle / Views
    R = 640  # SOD
    D = 1130  # SDD
    du = dv = 0.3036
    nu = 976
    nv = 976
    ImageRotation = 0  # rotate again, commonly don't change

    pMatrix = np.zeros((Views, 3, 4), dtype=np.float32)
    for i in range(Views):
        beta = np.radians(anglePerView * i + ImageRotation)
        e_u = np.array([-np.sin(beta),  np.cos(beta),   0]) * du
        e_v = np.array([0,              0,              1]) * dv
        x_do = np.array([np.cos(beta),  np.sin(beta),   0]) * (R - D)
        x_s = np.array([np.cos(beta),   np.sin(beta),   0]) * R

        A = np.array([e_u, e_v, x_do - x_s], dtype=np.float32).T
        A_inv = np.linalg.pinv(A)

        det_center_side_u = nu // 2 - projOffsetU_list[i] / du
        det_center_side_v = nv // 2 - projOffsetV_list[i] / dv
        # mangoct detector coordinate system offset from virtual detector
        offsetMatrix = np.array([[-det_center_side_u, 0, 0, det_center_side_u * R],
                                 [-det_center_side_v, 0, 0, det_center_side_v * R],
                                 [0, 0, 0, 0]], dtype=np.float32) / D

        rotateMatrix = np.array([[np.cos(beta), np.sin(beta), 0, 0],
                                 [-np.sin(beta), np.cos(beta), 0, 0],
                                 [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

        pMatrix[i] = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32) \
                     @ np.concatenate((A_inv, (-A_inv @ x_s.T).reshape(3, 1)), axis=1) \
                     + offsetMatrix @ rotateMatrix

    pMatrix_file = dict(Value=pMatrix.flatten())
    with open(fr'./params/pmatrix_file2_sm.jsonc', 'w') as f:
        f.write(json.dumps(pMatrix_file, cls=NumpyEncoder))


def read_paramsFile(path):
    with open(path, 'r') as f:
        file = json.loads(f.read())
    return np.array(file['Value'])
