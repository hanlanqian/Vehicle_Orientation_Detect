import math

import cv2
import numpy as np
from utils import get_scaled_homography, convertToBirdView, \
    modified_matrices_calculate_range_output_without_translation, get_scaled_matrix

if __name__ == '__main__':

    outputFlag = False

    path = '../frame.png'
    # path = './cars.jpg'
    calibration_path = '../camera_calibration/pics/calibrations.npy'
    with open(calibration_path, 'rb') as f:
        calibration = np.load(f, allow_pickle=True).tolist()

    vp1 = calibration.get('vp1')
    vp2 = calibration.get('vp2')
    vp3 = calibration.get('vp3')
    principal_point = calibration.get('principal_point')
    focal = calibration.get('focal')
    K = calibration.get('intrinsic')
    rotation = calibration.get('rotation')
    original = cv2.imread(path)
    height, width = original.shape[:2]

    ############################
    strict = False

    overhead_hmatrix = np.array([[4.77825655e+00, 5.85721771e+00, -9.01410392e+03],
                                 [7.15577215e-01, 1.63481871e+01, -1.44693566e+04],
                                 [1.07432598e-04, 2.67615931e-03, 1.00000000e+00]])

    # overhead_hmatrix = np.array([[4.61218604e+00, 5.96348187e+00, -8.00681186e+03],
    #                              [6.54095130e-01, 1.10774175e+01, -1.55501622e+03],
    #                              [2.36361997e-04, 3.12583569e-03, 1.00000000e+00]])
    est_range_u, est_range_v = modified_matrices_calculate_range_output_without_translation(height, width,
                                                                                            overhead_hmatrix,
                                                                                            verbose=False)
    moveup_camera = np.array([[1, 0, -est_range_u[0]], [0, 1, -est_range_v[0]], [0, 0, 1]])
    overhead_hmatrix = np.dot(moveup_camera, overhead_hmatrix)

    scaled_overhead_hmatrix1 = get_scaled_matrix(overhead_hmatrix, (1600, 900), est_range_u, est_range_v, strict=strict)
    # print(scaled_overhead_hmatrix)
    # print(width, height)
    scaled_overhead_hmatrix2 = convertToBirdView(K, rotation, (width, height), (1600, 900), strict=strict)
    # print(scaled_overhead_hmatrix)
    camera = cv2.VideoCapture(r"E:\datasets\0330数据集\o1.mp4")
    if outputFlag:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output = cv2.VideoWriter('./IPM.avi', fourcc, 25, (1600, 900))
    while True:
        _, frame = camera.read()
        warped1 = cv2.warpPerspective(frame, scaled_overhead_hmatrix1, dsize=(1600, 900))
        warped2 = cv2.warpPerspective(frame, scaled_overhead_hmatrix2, dsize=(1600, 900))
        if outputFlag:
            output.write(warped1)
        frame = cv2.resize(frame, (-1, -1), fx=0.5, fy=0.5)
        cv2.imshow('warp1', warped1)
        cv2.imshow('warp2', warped2)
        cv2.imshow('origin', frame)
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            camera.release()
            if outputFlag:
                output.release()
            break
