import argparse

from camera_calibration.calibration_yolo_model import Calibration_Yolo


def run(source, classes, calibration, engine, savePath='./result', caliFlag=False, visualize=True, verbose=False):
    app = Calibration_Yolo(source, detect_interval=10)

    if caliFlag:
        app.load_yolo_model(engine, classes, verbose=verbose)
        app.run()
        app.calibrate(visualize=visualize, save_path=savePath)
    else:
        app.load_calibration(calibration, (1600, 900))
        app.load_keypoint_model()
        app.load_yolo_model(engine, classes, verbose=verbose)
        app.detect_orientation()
    return


if __name__ == '__main__':
    ################# test data ####################

    json_path = "./yolov5/classes.json"
    engine_path = "./yolov5/weights/yolov5m_fp16.engine"
    video_path = r"E:\datasets\0330数据集\o1.mp4"
    # video_path = r"E:\datasets\surveilliance\v6.avi"
    calibration_path = './results/video1/calibrations.npy'
    # calibration_path = './camera_calibration/pics/calibrations.npy'
    cali_Flag = False
    visualizeFlag = True

    save_path = './results/'

    ####################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', default=video_path, help='the media source path')
    parser.add_argument('--classes', '-c', default=json_path, help='the ')
    parser.add_argument('--engine', '-e', default=engine_path, help='the tensorrt engine path')
    parser.add_argument('--calibration', '-C', default=calibration_path, help='the calibration file path')
    parser.add_argument('--caliFlag', '-f', default=cali_Flag, help='calibration or inference')
    parser.add_argument('--visualize', '-V', default=visualizeFlag, help='visualize the calibration process')
    parser.add_argument('--savePath', '-S', default=save_path, help='visualize the calibration process')
    parser.add_argument('--verbose', '-v', default=False, help='print the verbose information')

    args = parser.parse_args()
    print(vars(args))
    run(**vars(args))
