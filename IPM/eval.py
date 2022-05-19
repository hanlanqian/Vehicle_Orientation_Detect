import math

import cv2
import numpy as np
from utils import get_scaled_homography, convertToBirdView, \
    modified_matrices_calculate_range_output_without_translation, get_scaled_matrix
from detection_model.calibration_utils import computeCameraCalibration

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

    ## video1
    # overhead_hmatrix = np.array([[4.77825655e+00, 5.85721771e+00, -9.01410392e+03],
    #                              [7.15577215e-01, 1.63481871e+01, -1.44693566e+04],
    #                              [1.07432598e-04, 2.67615931e-03, 1.00000000e+00]])
    ## video2
    # overhead_hmatrix = np.array([[-1.45211933e+00, - 1.96693620e+00, 2.94800089e+03],
    #                              [-1.10548246e-01, - 2.32151317e+00, 1.29728367e+03],
    #                              [-2.44915563e-04, - 3.60979304e-03, 1.00000000e+00]])
    ## video3
    # overhead_hmatrix = np.array([[-5.00748865e+00, - 1.47279078e+00, 5.18716912e+03],
    #                              [-6.41713842e-01, - 2.39038406e+01, 2.00874080e+04],
    #                              [1.59162709e-04, - 3.68197694e-03, 1.00000000e+00]])
    ## video4
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
    diff_w = np.abs(np.abs(w) - np.abs(w_g))
    diff_w /= np.sum(diff_w)
    error = 0
    for i in range(3):
        error += diff_w[i] * cosine_distance(v[:, i], v_g[:, i])
    print(f"the calibration error is {error * 100}")
# diff = (my - scaled_overhead_hmatrix1) / scaled_overhead_hmatrix1
# error = np.average(np.power(diff, 2))
# print(error)

# path = '../test/imgs/o4.jpg'
path = '../test/imgs/o1.jpg'
img = cv2.imread(path)
my_warp = cv2.warpPerspective(img, scaled_overhead_hmatrix2, dsize=target_shape)
warp = cv2.warpPerspective(img, scaled_overhead_hmatrix1, dsize=target_shape)

cv2.imwrite('my_transform.jpg', my_warp)
cv2.imwrite('transform.jpg', warp)
