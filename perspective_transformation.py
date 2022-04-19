import cv2
import numpy as np

height = 2000
width = 1000
img = cv2.imread('./test.jpg')
# matrix1 = np.float32([[1351, 639], [1373, 638], [1572, 795], [1596, 794]])
matrix1 = np.float32([[207, 25], [538, 16], [644, 745], [1743, 548]])  # four points of image plane
matrix2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])  # the corresponding points in bird view plane
Transform_matrix = cv2.getPerspectiveTransform(matrix1, matrix2)  # get the perspective transform matrix
output = cv2.warpPerspective(img, Transform_matrix, img.shape[:-1])  # warp the image plan to bird view plane
cv2.imwrite('test_output.jpg', output)
# cap = cv2.VideoCapture(r"E:\Pycharm\Car_ReIdentification_application\1-2-981_Trim.mp4")
#
# cap.set(0, 3)
# flag, frame = cap.read()
# cv2.imwrite('IPM.jpg', frame)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
