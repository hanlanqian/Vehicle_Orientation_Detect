import cv2
import numpy as np
import math


def start_end_line(points):
    lines = []
    for p in points:
        x1, y1, x2, y2 = p
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        lines.append([a, b, c])
    if lines:
        lines = np.vstack(lines)
    return lines


def draw_points(frame, points, visualize=True, color=(0, 255, 255)):
    for p in points:
        x, y = p
        if x < 0 or y < 0:
            continue
        else:
            cv2.circle(frame, (int(x), int(y)), 2, color, 2)
    if visualize:
        cv2.imshow('line', frame)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyWindow('line')
    return frame


def draw_point_line(frame, points, visualFlag=False):
    display = frame.copy()
    for p in points:
        x1, y1, x2, y2 = p
        cv2.line(display, (x1, y1), (x2, y2), (255, 255, 0), 2)
    if visualFlag:
        cv2.imshow('line', display)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
    return display


def draw_lines(img, lines):
    height, width = 720, 1280
    for line in lines:
        a, b, c = line
        if b != 0:
            x = np.array([0, width])
            y = (-a * x - c) / b
        else:
            x = np.full(2, -c / a)
            y = np.array([0, height])
        cv2.line(img, (int(x[0]), int(y[0])), (int(x[1]), int(y[1])), (0, 255, 0), thickness=2)
    return img


def cvt_diamond_space(tracks):
    """
    :param tracks:
    :return:lines: [ax+by+c=0]
    [a, b, c]
    """
    lines = []
    for track in tracks:
        track = np.array(track)
        k, b = np.polyfit(track[:, 0], track[:, 1], deg=1)
        lines.append([k, -1, b])
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


def check_int_type(point):
    res = point.astype(np.int32)[:2]
    return res


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


def getViewpointFromCalibration(p, principal_point, focal, rotation):
    # rotation : WCS -> CCS   rotation.T: CCS -> WCS
    pW = np.concatenate((p[:2] - principal_point[:2], [focal]))
    pW /= np.linalg.norm(pW)
    viewPoint = rotation.T @ pW
    return viewPoint


def drawViewpoint(img, p, vp1, vp2, vp3, scale=30):

    direction1 = vp1 - p
    direction1 /= 1 / scale * np.linalg.norm(direction1)
    direction2 = vp2 - p
    direction2 /= 1 / scale * np.linalg.norm(direction2)
    direction3 = vp3 - p
    direction3 /= 1 / scale * np.linalg.norm(direction3)
    p = check_int_type(p)
    direction1 = check_int_type(direction1)
    direction2 = check_int_type(direction2)
    direction3 = check_int_type(direction3)
    cv2.line(img, p, (p + direction1).astype(np.int32), (0, 255, 0), 2)
    cv2.line(img, p, (p + direction2).astype(np.int32), (255, 0, 0), 2)
    cv2.line(img, p, (p + direction3).astype(np.int32), (0, 0, 255), 2)

    return img


def drawCalibration(img, vp1, vp2, vp3, slices=3, offsetPercent=0.2, scale=30):
    display = img.copy()
    height, width = img.shape[:2]
    heightOffset = height * offsetPercent
    widthOffset = width * offsetPercent
    height *= 1 - offsetPercent
    width *= 1 - offsetPercent
    for i in range(slices):
        for j in range(slices):
            p = np.array([int(i / slices * width + widthOffset), int(j / slices * height + heightOffset)])
            direction1 = vp1 - p
            direction1 /= 1 / scale * np.linalg.norm(direction1)
            direction2 = vp2 - p
            direction2 /= 1 / scale * np.linalg.norm(direction2)
            direction3 = vp3 - p
            direction3 /= 1 / scale * np.linalg.norm(direction3)
            cv2.line(display, p, (p + direction1).astype(np.int32), (0, 255, 0), 2)
            cv2.line(display, p, (p + direction2).astype(np.int32), (255, 0, 0), 2)
            cv2.line(display, p, (p + direction3).astype(np.int32), (0, 0, 255), 2)

    return display


def get_third_VP(vp1, vp2, f, width, height):
    Ut = np.array([vp1[0], vp1[1], f])
    Vt = np.array([vp2[0], vp2[1], f])
    Pt = np.array([width / 2 - 1, height / 2 - 1, 0])
    W = np.cross((Ut - Pt), (Vt - Pt))
    return W


def computeCameraCalibration(_vp1, _vp2, _pp, cameraH=10):
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
    vp1_c = vp1W - ppW
    vp2_c = vp2W - ppW
    vp3W = np.cross(vp2_c, vp1_c)
    vp3 = np.concatenate((vp3W[0:2] / vp3W[2] * focal + ppW[0:2], [1]))
    vp3Direction = np.concatenate((vp3[0:2], [focal])) - ppW
    roadPlane = np.concatenate((vp3Direction / np.linalg.norm(vp3Direction), [cameraH]))
    P = np.array([[focal, 0, _pp[0]],
                  [0, focal, _pp[1], ],
                  [0, 0, 1]])

    # World Coordinate to Camera Coordinate
    R = np.stack([vp1_c / np.linalg.norm(vp1_c), vp2_c / np.linalg.norm(vp2_c), vp3W / np.linalg.norm(vp3W)], axis=1)

    return vp1, vp2, vp3, pp, roadPlane, focal, P, R


def coordinate_transform(p, focal, roadPlane, delta=10):
    pW = np.concatenate((p, [focal, 0]))
    pW_t = np.concatenate((p, [focal])).transpose()
    P = - delta / np.dot(pW, roadPlane) * pW_t
    return P


def get_intersections(points1, points2):
    """
    return the intersection of two lines.
    :param points1: the start and end point of first line
    :param points2: the start and end point of second line
    :return: intersection
    """
    x1, y1, x2, y2 = points1
    x3, y3, x4, y4 = points2
    a1 = y2 - y1
    b1 = x1 - x2
    c1 = x2 * y1 - x1 * y2
    a2 = y4 - y3
    b2 = x3 - x4
    c2 = x4 * y3 - x3 * y4
    x = (b2 * c1 / b1 - c2) / (a2 - a1 * b2 / b1)
    y = - (a1 * x + c1) / b1
    return (x, y)
