import json5
import json
import numpy as np


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


def save_paramsFile(arr, path):
    file = dict(Value=arr)
    with open(path, 'w') as f:
        f.write(json.dumps(file, cls=NumpyEncoder))


def read_paramsFile(path):
    with open(path, 'r') as f:
        file = json.loads(f.read())
    return np.array(file['Value'])


def gen_pMatrix(projOffsetD_list, projOffsetS_list, projOffsetEU_list, projOffsetEV_list,
                postfix='', path='./params', Views=360, TotalScanAngle=360):
    Views = Views
    TotalScanAngle = TotalScanAngle  # inverse direction corresponding to the config file
    anglePerView = TotalScanAngle / Views
    R = 750  # SID
    D = 1060  # SDD
    du = 0.4
    dv = 0.4
    nu = 750
    sliceCount = nv = 750
    ImageRotation = 0  # rotate again, commonly don't change

    pMatrix = np.zeros((Views, 3, 4), dtype=np.float32)
    for i in range(Views):
        beta = np.radians(anglePerView * i + ImageRotation)
        e_u = np.array([-np.sin(beta),  np.cos(beta),   0]) * du + projOffsetEU_list[i]
        e_v = np.array([0,              0,              1]) * dv + projOffsetEV_list[i]
        x_d = np.array([np.cos(beta),   np.sin(beta),   0]) * (R - D)
        x_s = np.array([np.cos(beta),   np.sin(beta),   0]) * R + projOffsetS_list[i]

        det_center_side_u = (nu-1) / 2 - projOffsetD_list[i] / du
        det_center_side_v = (sliceCount-1) / 2  # frozen in fan beam
        x_do = x_d - det_center_side_u * e_u - det_center_side_v * e_v

        # projOffsetD_list random direction vector condition
        if isinstance(projOffsetD_list[i], (list, tuple, np.ndarray)):
            x_do = x_do - (projOffsetD_list[i] / du * e_u) + projOffsetD_list[i]

        A = np.array([e_u, e_v, x_do - x_s], dtype=np.float32).T
        A_inv = np.linalg.pinv(A)

        # mangoct detector coordinate system offset from virtual detector
        pMatrix[i] = np.concatenate((A_inv, (-A_inv @ x_s.T).reshape(3, 1)), axis=1) \
                     @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

    pMatrix_file = dict(Value=pMatrix.flatten())
    with open(fr'{path}/pmatrix_file{postfix}.jsonc', 'w') as f:
        f.write(json.dumps(pMatrix_file, cls=NumpyEncoder))
