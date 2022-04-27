import sys

sys.path.append(r'E:\Pycharm\mine\Vehicle_Orientation_Detect\SSD')

from calibration_model import Calibration

if __name__ == '__main__':
    json_path = "../SSD/pascal_voc_classes.json"
    # video_path = '../1-2-981_Trim.mp4'
    # video_path = r"E:\datasets\surveilliance\v6.avi"
    video_path = r"E:\datasets\0330数据集\o1.mp4"
    # video_path = '../sherbrooked_video.avi'
    # video_path = '../rouen_video.avi'
    cameraTrack = Calibration(video_path)
    RunFlag = False
    visualizeFlag = True
    if RunFlag:
        cameraTrack.load_ssd_model(json_path)
        cameraTrack.run()
        # cameraTrack.draw_all_tracks()
        cameraTrack.get_vp1(visualize=visualizeFlag)
        # print(cameraTrack.vp_1)
        cameraTrack.get_vp2(visualize=visualizeFlag)
        # print(cameraTrack.vp_2)
        cameraTrack.save_calibration(visualize=visualizeFlag)
    else:
        cameraTrack.load_calibration('./avi6/calibrations.npy')
        cameraTrack.load_ssd_model(json_path)
        cameraTrack.load_keypoint_model()
        cameraTrack.detect_orientation(0.5)
