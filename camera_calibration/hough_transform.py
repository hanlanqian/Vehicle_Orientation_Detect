import sys

sys.path.append(r'E:\Pycharm\mine\Vehicle_Orientation_Detect\SSD')

from calibration_model import Calibration

if __name__ == '__main__':
    json_path = "../SSD/pascal_voc_classes.json"
    # video_path = '../1-2-981_Trim.mp4'
    video_path = r"E:\datasets\surveilliance\v5.avi"
    # roi = [(221, 9), (789, 1431), (2560, 1440), (2560, 0)]
    roi = [(278, 1), (90, 700), (88, 783), (477, 679)]
    # video_path = '../sherbrooke_video.avi'
    # video_path = '../rouen_video.avi'
    cameraTrack = Calibration(video_path, roi)
    RunFlag = False
    # RunFlag = True
    if RunFlag:
        cameraTrack.load_ssd_model(json_path)
        cameraTrack.run()
        # cameraTrack.draw_all_tracks()
        cameraTrack.get_vp1(visualize=True)
        # print(cameraTrack.vp_1)
        cameraTrack.get_vp2(visualize=True)
        # print(cameraTrack.vp_2)
        cameraTrack.save_calibration(visualize=True)
    else:
        cameraTrack.load_calibration('./pics/calibrations.npy')
        cameraTrack.load_ssd_model(json_path)
        cameraTrack.detect_orientation()
