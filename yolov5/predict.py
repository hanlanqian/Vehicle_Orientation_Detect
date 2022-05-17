from model import YoloTensorrt
# from model import YoloTensorrt
import cv2
from time import time


def test():
    img_path = '../frame.jpg'
    # img_path = '../test/imgs/calibration.jpg'
    path = r'E:\datasets\surveilliance\v4.mp4'
    engine_path = 'weights/yolov5lfp16.engine'
    # engine_path = './weights/yolov5m_new.engine'
    classes_path = './classes.json'
    yolo = YoloTensorrt(engine_path, classes_path)
    yolo.warm_up()
    yolo.reload_images(img_path)
    classes, boxes = yolo.infer(visualize=True)


if __name__ == '__main__':
    test()
