import cv2
import numpy as np

if __name__ == '__main__':
    ## video2 calibration board coordinates
    # points = np.array([[1066, 772], [1260, 764], [1211, 818], [1004, 825]], dtype=np.float32)
    points = np.array([[1082, 780], [1228, 774], [1190, 811], [1036, 817]], dtype=np.float32)
    dst_points = np.array([[0, 0], [400, 0], [400, 400], [0, 400]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(points, dst_points)
    print(matrix)
