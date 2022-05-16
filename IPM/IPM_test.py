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
    outputFlag = False
    path = r"E:\datasets\0330数据集\o1.mp4"
    # path = r"E:\datasets\0330数据集\o2.mp4"
    # path = r"E:\datasets\surveilliance\v6.avi"

    camera = cv2.VideoCapture(path)

    calibration_path = '../results/new/calibrations.npy'
    # calibration_path = '../results/avi6/calibrations.npy'
    # calibration_path = '../results/video1/calibrations.npy'
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
    target_shape = (1600, 1700)
    # target_shape = (1600, 900)

    ## video1
    # overhead_hmatrix = np.array([[4.77825655e+00, 5.85721771e+00, -9.01410392e+03],
    #                              [7.15577215e-01, 1.63481871e+01, -1.44693566e+04],
    #                              [1.07432598e-04, 2.67615931e-03, 1.00000000e+00]])
    ## video2
    # overhead_hmatrix = np.array([[-4.39995719e+00, - 5.14711974e+00, 8.66393080e+03],
    #                              [-6.93979612e-01, - 1.68290056e+01, 1.37317746e+04],
    #                              [-4.22788632e-04, - 3.27005650e-03, 1.00000000e+00]])
    # overhead_hmatrix = np.array([[-6.13831606e+00, - 7.63141996e+00, 1.25941655e+04],
    #                              [-1.04311421e+00, - 2.53824458e+01, 2.09269573e+04],
    #                              [-2.75777412e-04, - 3.60123953e-03, 1.00000000e+00]])
    ## video3
    overhead_hmatrix = np.array([[-5.00748865e+00, - 1.47279078e+00, 5.18716912e+03],
                                 [-6.41713842e-01, - 2.39038406e+01, 2.00874080e+04],
                                 [1.59162709e-04, - 3.68197694e-03, 1.00000000e+00]])

    est_range_u, est_range_v = modified_matrices_calculate_range_output_without_translation(height, width,
                                                                                            overhead_hmatrix,
                                                                                            verbose=False, opt=True)
    moveup_camera = np.array([[1, 0, -est_range_u[0]], [0, 1, -est_range_v[0]], [0, 0, 1]])
    overhead_hmatrix = np.dot(moveup_camera, overhead_hmatrix)

    scaled_overhead_hmatrix1 = get_scaled_matrix(overhead_hmatrix, target_shape, est_range_u, est_range_v,
                                                 strict=strict)
    scaled_overhead_hmatrix2 = convertToBirdView(K, rotation, (width, height), target_shape, strict)

    my = scaled_overhead_hmatrix2 / scaled_overhead_hmatrix2[-1, -1]
    diff = (my - scaled_overhead_hmatrix1) / scaled_overhead_hmatrix1
    error = np.average(np.power(diff, 2))
    print(error)

    path = '../test/imgs/calibration.jpg'
    img = cv2.imread(path)
    # img = cv2.resize(img, (-1, -1), fx=1.6, fy=1.6)
    my_warp = cv2.warpPerspective(img, scaled_overhead_hmatrix2, dsize=target_shape)
    warp = cv2.warpPerspective(img, scaled_overhead_hmatrix1, dsize=target_shape)

    cv2.imwrite('my_transform.jpg', my_warp)
    cv2.imwrite('transform.jpg', warp)

    # if outputFlag:
    #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #     output = cv2.VideoWriter('./IPM.avi', fourcc, 25, target_shape)
    # while True:
    #     _, frame = camera.read()
    #     # scale = scale_image(frame.shape[:2], (900, 1600), scaleUp=True)
    #     # frame = cv2.resize(frame, (-1, -1), fx=scale, fy=scale)
    #     warped1 = cv2.warpPerspective(frame, scaled_overhead_hmatrix1, dsize=target_shape)
    #     # warped2 = cv2.warpPerspective(frame, scaled_overhead_hmatrix1, dsize=target_shape)
    #     if outputFlag:
    #         output.write(warped1)
    #     frame = cv2.resize(frame, (-1, -1), fx=0.5, fy=0.5)
    #     cv2.imshow('warp1', warped1)
    #     # cv2.imshow('warp2', warped2)
    #     cv2.imshow('origin', frame)
    #     if cv2.waitKey(1) == 27:
    #         cv2.destroyAllWindows()
    #         camera.release()
    #         if outputFlag:
    #             output.release()
    #         break
