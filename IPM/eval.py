import math

import cv2
import numpy as np
from utils import get_scaled_homography, convertToBirdView, \
    modified_matrices_calculate_range_output_without_translation, get_scaled_matrix
from camera_calibration.calibration_utils import computeCameraCalibration


# points1 = np.array([[814, 849], [1145, 834], [1059, 918], [708, 933]])
# earth_data = np.array([[1.27548415e+00, 6.65416173e+00, -2.30653026e-02],
#                        [1.97012298e-01, 4.77941972e+00, -3.09488910e-02],
#                        [1.07432598e-04, 2.67615931e-03, 1.00000000e+00]])

# modified
# array([[-2.20253514e-02, -9.68560923e-02,  4.62626941e-04],
#        [-1.52702030e-02, -2.05552117e-01,  4.50651419e-05],
#        [-1.79486038e-05, -1.87279867e-04, -5.91611670e-02]])


def scale_image(shape, dst_shape, scaleUp=False, force=False):
    """

    :param shape: (height, width)
    :param dst_shape: (height, width)
    :param scaleUp:
    :param force:
    :return:
    """
    x_scale = dst_shape[1] / shape[1]
    y_scale = dst_shape[0] / shape[0]
    if not force:
        scale = x_scale if x_scale < y_scale else y_scale
        if not scaleUp:
            scale = min(scale, 1)
        return scale
    else:
        return x_scale, y_scale


if __name__ == '__main__':
    path = r"E:\datasets\0330数据集\eval\o1.mp4"
    # path = r"E:\datasets\0330数据集\eval\o2.mp4"
    # path = r"E:\datasets\surveilliance\v6.avi"

    camera = cv2.VideoCapture(path)

    calibration_path = '../results/new/calibrations.npy'
    # calibration_path = '../results/avi6/calibrations.npy'
    calibration_path = '../results/video4_t70/calibrations.npy'
    with open(calibration_path, 'rb') as f:
        calibration = np.load(f, allow_pickle=True).tolist()

    vp1 = calibration.get('vp1')
    vp2 = calibration.get('vp2')
    vp3 = calibration.get('vp3')
    principal_point = calibration.get('principal_point')
    focal = calibration.get('focal')
    K = calibration.get('intrinsic')
    rotation = calibration.get('rotation')
    height, width = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))

    ############################
    strict = False
    # target_shape = (width, height)
    target_shape = (1600, 2000)
    # target_shape = (1600, 900)

    ## video1
    # overhead_hmatrix = np.array([[4.77825655e+00, 5.85721771e+00, -9.01410392e+03],
    #                              [7.15577215e-01, 1.63481871e+01, -1.44693566e+04],
    #                              [1.07432598e-04, 2.67615931e-03, 1.00000000e+00]])
    ## video2
    # overhead_hmatrix = np.array([[-4.39995719e+00, - 5.14711974e+00, 8.66393080e+03],
    #                              [-6.93979612e-01, - 1.68290056e+01, 1.37317746e+04],
    #                              [-4.22788632e-04, - 3.27005650e-03, 1.00000000e+00]])
    # overhead_hmatrix = np.array([[-9.37738247e+01, - 1.14358323e+02, 1.89173825e+05],
    #                              [-1.47522236e+01, - 3.90933926e+02, 3.19164358e+05],
    #                              [1.53271919e-03, - 4.99285260e-02, 1.00000000e+00]])
    # overhead_hmatrix = np.array([[-1.45211933e+00, - 1.96693620e+00, 2.94800089e+03],
    #                              [-1.10548246e-01, - 2.32151317e+00, 1.29728367e+03],
    #                              [-2.44915563e-04, - 3.60979304e-03, 1.00000000e+00]])
    ## video3
    # overhead_hmatrix = np.array([[-5.00748865e+00, - 1.47279078e+00, 5.18716912e+03],
    #                              [-6.41713842e-01, - 2.39038406e+01, 2.00874080e+04],
    #                              [1.59162709e-04, - 3.68197694e-03, 1.00000000e+00]])

    ## video4
    # overhead_hmatrix = np.array([[1.42394822e+01, 9.92448759e+00, - 8.67443366e+03],
    #                              [3.60195921e-14, 6.64509169e+01, - 5.30278317e+04],
    #                              [2.67934434e-17, 5.39374326e-03, 1.00000000e+00]])
    overhead_hmatrix = np.array([[2.36025363e+00, 8.31708423e-01, - 1.79635532e+03],
                                 [3.26078892e-01, 1.14880102e+01, - 9.25567409e+03],
                                 [-8.81537417e-05, 2.25715584e-03, 1.00000000e+00]])
    est_range_u, est_range_v = modified_matrices_calculate_range_output_without_translation(height, width,
                                                                                            overhead_hmatrix,
                                                                                            verbose=False, opt=True)
    moveup_camera = np.array([[1, 0, -est_range_u[0]], [0, 1, -est_range_v[0]], [0, 0, 1]])
    overhead_hmatrix = np.dot(moveup_camera, overhead_hmatrix)

    scaled_overhead_hmatrix1 = get_scaled_matrix(overhead_hmatrix, target_shape, est_range_u, est_range_v,
                                                 strict=strict)
    scaled_overhead_hmatrix2 = convertToBirdView(K, rotation, (width, height), target_shape, strict)

    my = scaled_overhead_hmatrix2 / scaled_overhead_hmatrix2[-1, -1]


    # unit = 100
    # vector_y = np.array([unit, 0, 0])
    # vector_x = np.array([0, unit, 0])
    # unit_vector = np.vstack([vector_x, vector_y]).T
    #
    #
    # transform = np.linalg.inv(scaled_overhead_hmatrix1) @ unit_vector
    # my_transform = np.linalg.inv(my) @ unit_vector
    # transform /= transform[-1, :]
    # my_transform /= my_transform[-1, :]
    # transform[0, :] -= principal_point[0]
    # transform[1, :] -= principal_point[1]
    # my_transform[0, :] -= principal_point[0]
    # my_transform[1, :] -= principal_point[1]
    # # cosDistance =

    def cosine_distance(a, b):
        if a.shape != b.shape:
            raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
        if a.ndim == 1:
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
        elif a.ndim == 2:
            a_norm = np.linalg.norm(a, axis=1, keepdims=True)
            b_norm = np.linalg.norm(b, axis=1, keepdims=True)
        else:
            raise RuntimeError("array dimensions {} not right".format(a.ndim))

        # 原来代码b_norm 少了 T
        similiarity = np.dot(a, b.T) / (a_norm * b_norm.T)  # 计算相似度 [-1,1]
        dist = 1 - np.abs(similiarity)  # 计算余弦距离：[0,1]
        return dist


    w_g, v_g = np.linalg.eig(scaled_overhead_hmatrix1)
    w, v = np.linalg.eig(my)
    print('特征值：')
    print(w, w_g)
    diff_w = np.abs(np.abs(w) - np.abs(w_g))
    diff_w /= np.sum(diff_w)
    error = 0
    for i in range(3):
        # print(diff_w[i])
        # print(cosine_distance(v[:, i], v_g[:, i]))
        error += diff_w[i] * cosine_distance(v[:, i], v_g[:, i])
    print(error*100)
# diff = (my - scaled_overhead_hmatrix1) / scaled_overhead_hmatrix1
# error = np.average(np.power(diff, 2))
# print(error)

# path = '../test/imgs/o4.jpg'
path = '../test/imgs/o1.jpg'
img = cv2.imread(path)
# img = cv2.resize(img, (-1, -1), fx=1.6, fy=1.6)
my_warp = cv2.warpPerspective(img, scaled_overhead_hmatrix2, dsize=target_shape)
warp = cv2.warpPerspective(img, scaled_overhead_hmatrix1, dsize=target_shape)

cv2.imwrite('my_transform.jpg', my_warp)
cv2.imwrite('transform.jpg', warp)
