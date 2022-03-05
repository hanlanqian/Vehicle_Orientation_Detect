import cv2
import numpy as np
import math


def Cvt2ParallelSpace(mat):
    # 将平面坐标系的点转到Y, X, -Y轴的平行坐标系(u, v)
    # u = -Y 等价与 u = -1
    # u = Y 等价与 u = 1
    # u = X 等价与 u = 0

    mat[mat > 0] = mat[mat > 0]
    print(mat)


def draw_box(frame, box):
    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [0, 0, 255], thickness=3)
    return frame


def getFocal(vp1, vp2, pp):
    """
    返回焦距
    :param vp1: 第一消失点坐标(vp1_x, vp1_y, 1)
    :param vp2: 第二消失点坐标(vp1_x, vp2_y, 1)
    :param pp: 相机光心坐标(p_x, p_y, 1)
    :return: focal_length (f_x, f_y)
    """
    return math.sqrt(- np.dot(vp1[0:2] - pp[0:2], vp2[0:2] - pp[0:2]))


def getViewpoint(p, vp1, vp2, pp):
    try:
        focal = getFocal(vp1, vp2, pp)
    except ValueError:
        return None
    vp1W = np.concatenate((vp1[0:2] - pp[0:2], [focal]))
    vp2W = np.concatenate((vp2[0:2] - pp[0:2], [focal]))
    if vp1[0] < vp2[0]:
        vp2W = -vp2W
    vp3W = np.cross(vp1W, vp2W)
    vp1W, vp2W, vp3W = tuple(map(lambda u: u / np.linalg.norm(u), [vp1W, vp2W, vp3W]))
    pW = np.concatenate((p[0:2] - pp[0:2], [focal]))
    pW = pW / np.linalg.norm(pW)
    viewPoint = -np.dot(np.array([vp1W, vp2W, vp3W]), pW)
    return viewPoint
