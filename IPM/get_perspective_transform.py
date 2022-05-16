import cv2
import numpy as np

if __name__ == '__main__':
    ## video2 calibration board coordinates
    # points = np.array([[1066, 772], [1260, 764], [1211, 818], [1004, 825]], dtype=np.float32)
    # points = np.array([[1082, 780], [1228, 774], [1190, 811], [1036, 817]], dtype=np.float32)
    # points = np.array([[1071, 776], [1230, 770], [1190, 810], [1021, 817]], dtype=np.float32)
    # points = np.array([[1361, 494], [1697, 478],  [1495, 1001], [658, 1013]], dtype=np.float32)

    ## video3
    # points = np.array([[795, 819], [944, 815], [944, 848], [785, 853]], dtype=np.float32)


    ## video4
    # points = np.array([[540, 804], [878, 789], [876, 867], [516, 880]], dtype=np.float32)
    # points = np.array([[53, 798], [202, 798], [184, 831], [30, 831]], dtype=np.float32)
    points = np.array([[482, 792], [940, 779], [943, 880], [445, 897]], dtype=np.float32)


    dst_points = np.array([[0, 0], [400, 0], [400, 400], [0, 400]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(points, dst_points)
    print(matrix)
