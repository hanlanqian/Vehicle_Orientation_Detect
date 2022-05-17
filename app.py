import argparse
import cv2

from detection_model.calibration_yolo_model import Calibration_Yolo
from utils import getROIMouseEvent


def run(source, classes, calibration, engine, threshold, savePath='./result', caliFlag=False, visualize=True,
        view=False,
        verbose=False, **kwargs):
    app = Calibration_Yolo(source, detect_interval=15)
    if kwargs.get('roi', None):
        roi = []
        _, frame = app.camera.read()
        frame = cv2.resize(frame, (-1, -1), fx=app.scale, fy=app.scale)
        cv2.imshow('getROI', frame)
        cv2.setMouseCallback('getROI', getROIMouseEvent, [frame, roi])
        if cv2.waitKey(0) == 27:
            app.setROI(roi)
            cv2.destroyWindow('getROI')

    if caliFlag:
        app.load_yolo_model(engine, classes, verbose=verbose)
        app.run(view_process=view, threshold=threshold)
        app.calibrate(visualize=visualize, save_path=savePath)
    else:
        app.load_calibration(calibration, (1600, 1600))
        app.load_keypoint_model()
        app.load_yolo_model(engine, classes, verbose=verbose)
        app.detect_orientation()


if __name__ == '__main__':
    ################# test data ####################

    json_path = "./yolov5/classes.json"
    engine_path = "./yolov5/weights/yolov5m16.engine"
    # video_path = "../../Car_ReIdentification_application/1-2-981_Trim.mp4"
    video_path = r"E:\datasets\0330数据集\eval\o4.mp4"
    # video_path = r"E:\datasets\0330数据集\calibrate\o4.mp4"
    # video_path = r"E:\datasets\0330数据集\o2_Trim.mp4"
    # video_path = r"E:\datasets\surveilliance\v6.avi"
    # calibration_path = './results/new/calibrations.npy'
    calibration_path = './results/video4/calibrations.npy'
    cali_Flag = False
    visualizeFlag = True
    selectROI = False

    save_path = './results/'

    ####################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', default=video_path, help='the media source path')
    parser.add_argument('--engine', '-e', default=engine_path, help='the tensorrt engine path')
    parser.add_argument('--classes', '-c', default=json_path, help='all clasess of object detector model')
    parser.add_argument('--roi', '-r', default=selectROI, help='whether select the ROI from source video.')
    parser.add_argument('--calibration', '-C', default=calibration_path, help='the calibration file path')
    parser.add_argument('--caliFlag', '-f', default=cali_Flag, help='calibration or inference')
    parser.add_argument('--threshold', '-t', default=0.7, help='the threshold of determining high quality edge')
    parser.add_argument('--visualize', '-V', default=visualizeFlag, help='visualize the calibration process')
    parser.add_argument('--savePath', '-S', default=save_path, help='visualize the calibration process')
    parser.add_argument('--view', default=False, help='visualize the detect edges process')
    parser.add_argument('--verbose', '-v', default=False, help='print the verbose information')

    args = parser.parse_args()
    print(vars(args))
    run(**vars(args))
