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


def cvt_diamond_space(img, tracks):
    """
    :param img: original image plane
    :param tracks:
    :return:lines:
    [a, b, c]
    """
    h, w = img.shape[:2]
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray_img)
    lines = []
    for track in tracks:
        track = np.array(track)
        k, b = np.polyfit(track[:, 0], track[:, 1], deg=1)
        lines.append([k, -1, b])
    # cv2.polylines(mask, [np.int32(tr) for tr in tracks], isClosed=False, thickness=3, color=255)
    return np.array(lines)


def origin_to_diamond(point):
    """
    default:
        d = 1
        D = 1
        w = 1
    :param point:
    :return:
    """
    x, y = point
    return [-1, -x, np.sign(x * y) * x + y + np.sign(y)]


def diamond_to_origin(point):
    x, y, w = point
    return [y, np.sign(x) * x + np.sign(y) * y - 1, x]


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

    # variables end with W represent their world coordinate
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


def get_third_VP(vp1, vp2, f, width, height):
    Ut = np.array([vp1[0], vp1[1], f])
    Vt = np.array([vp2[0], vp2[1], f])
    Pt = np.array([width / 2 - 1, height / 2 - 1, 0])
    W = np.cross((Ut - Pt), (Vt - Pt))
    return W


def computeCameraCalibration(_vp1, _vp2, _pp):
    """
    Compute camera calibration from two van points and principal point. Variables end with W represent their world coordinate
    :param _vp1 first vanishing point (vp1_x, vp1_y)
    :param _vp2 first vanishing point (vp2_x, vp2_y)
    :param _pp first vanishing point (pp_x, pp_y)
    """
    vp1 = np.concatenate((_vp1, [1]))
    vp2 = np.concatenate((_vp2, [1]))
    pp = np.concatenate((_pp, [1]))
    focal = getFocal(vp1, vp2, pp)
    vp1W = np.concatenate((_vp1, [focal]))
    vp2W = np.concatenate((_vp2, [focal]))
    ppW = np.concatenate((_pp, [0]))
    vp3W = np.cross(vp1W - ppW, vp2W - ppW)
    vp3 = np.concatenate((vp3W[0:2] / vp3W[2] * focal + ppW[0:2], [1]))
    vp3Direction = np.concatenate((vp3[0:2], [focal])) - ppW
    roadPlane = np.concatenate((vp3Direction / np.linalg.norm(vp3Direction), [10]))
    return vp1, vp2, vp3, pp, roadPlane, focal


def oPoint_to_dPoint(tracks, ):
    return
