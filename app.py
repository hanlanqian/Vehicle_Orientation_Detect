from camera_calibration.calibration_yolo_model import Calibration_Yolo

if __name__ == '__main__':
    json_path = "./yolov5/classes.json"
    engine_path = "./yolov5/weights/yolov5m_fp16.engine"
    video_path = r"E:\datasets\0330数据集\o1.mp4"
    # video_path = r"E:\datasets\surveilliance\v6.avi"
    calibration = Calibration_Yolo(video_path, detect_interval=20)
    RunFlag = False
    visualizeFlag = True
    if RunFlag:
        calibration.load_yolo_model(engine_path, json_path)
        calibration.run()
        calibration.get_vp1(visualize=visualizeFlag)
        calibration.get_vp2(visualize=visualizeFlag)
        calibration.save_calibration(visualize=visualizeFlag)
    else:
        calibration.load_calibration('./camera_calibration/pics/calibrations.npy')
        calibration.load_keypoint_model()
        calibration.load_yolo_model(engine_path, json_path)
        calibration.detect_orientation()
