import sys
import cv2
import os
import torch
import json
import pickle
from time import time
from PIL import Image
from diamondSpace import DiamondSpace
from vpd_utils import cvt_diamond_space
import matplotlib.pylab as plt

import numpy as np

sys.path.append(r'E:\Pycharm\mine\Vehicle_Orientation_Detect\SSD')

from SSD.inference import time_synchronized, create_model, get_data_transform

# SSD detect interval
DetectInterval = 20

# Draw Line
DrawInterval = 10
colors = np.random.randint(0, 255, (100, 3))
good_features_parameters = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7, useHarrisDetector=True,
                                k=0.04)
draw_circle_parameters = dict(radius=3, color=(0, 0, 255), thickness=3, )
optical_flow_parameters = dict(winSize=(21, 21), minEigThreshold=1e-4)


class Tracks(object):
    def __init__(self, video_src, detect_interval=20, draw_interval=10, VP_interval=30):
        self.tracks = []
        self.detectInterval = detect_interval
        self.draw_interval = draw_interval
        self.updateVP_interval = VP_interval
        self.camera = cv2.VideoCapture(video_src)
        self.frame_count = 0
        self.detectModel = load_model()
        self.init_frame = None
        self.current_frame = None
        self.previous_frame = None
        self.features = []
        self.diff_threshold = 1
        self.maxFeatureNum = 50
        self.frame_height = 0
        self.frame_width = 0
        self.feature_determine_threshold = 0.1
        self.tracks_filter_threshold = 0.005

        # vanish point
        self.vp_1 = []
        self.vp_2 = []
        self.vp_3 = []

        self.DiamondSpace = None

    def run(self):
        self.frame_count = 0
        while True:
            start = time()

            flag, frame = self.camera.read()
            if self.frame_count == 0:
                self.init_frame = frame
                self.frame_height, self.frame_width = frame.shape[:2]
                self.DiamondSpace = DiamondSpace(d=min(self.frame_height, self.frame_width) / 2, size=256)

            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if not flag:
                print('no frame grabbed!')
                return 0
            if len(self.features) > 0:
                pimg, cimg = self.previous_frame, self.current_frame
                last_feature = np.float32([f[-1] for f in self.features])
                feature_c, status_c, _ = cv2.calcOpticalFlowPyrLK(pimg, cimg, last_feature, None,
                                                                  **optical_flow_parameters)
                feature_recover, status_r, _ = cv2.calcOpticalFlowPyrLK(cimg, pimg, feature_c, None,
                                                                        **optical_flow_parameters)

                good_features_index = [1e-5 < diff < self.diff_threshold for diff in
                                       abs(feature_recover - last_feature).reshape(-1, 2).max(-1)]
                new_tracks = []
                for fp, fc, isGood in zip(self.features, feature_c, good_features_index):
                    if not isGood:
                        continue
                    fp.append(tuple(fc.ravel().tolist()))

                    if len(fp) > self.maxFeatureNum:
                        del fp[0]
                        if self.get_track_length(fp) > 30:
                            temp = fp.copy()
                            self.tracks.append(temp)
                        # x, y = fp[-1]
                        # print(f"the length is {self.get_track_length(fp)}")
                        # if 0 + self.frame_width * self.feature_determine_threshold < x < (
                        #         1 - self.feature_determine_threshold) * self.frame_width and (
                        #         0 + self.frame_height * self.feature_determine_threshold < y < (
                        #         1 - self.feature_determine_threshold) * self.frame_height):
                        #     del fp[0]
                        # else:
                        #     continue
                    new_tracks.append(fp)
                # 开头的tracks在KLT的作用下已经成为一点
                # start_index = int(0.4 * len(new_tracks))
                # self.tracks += new_tracks[start_index:]
                # self.tracks += new_tracks
                cv2.polylines(frame, [np.int32(tr) for tr in new_tracks], False, (0, 255, 0))

            if self.frame_count % self.detectInterval == 0:
                mask = np.zeros_like(self.current_frame)
                boxes, classes_str = self.detect_car(frame)
                for box, box_class in zip(boxes, classes_str):
                    if box_class in ['car', "bus"]:
                        mask[box[1]:box[3], box[0]:box[2]] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.features]:
                    cv2.circle(mask, (x, y), 5, 0, -1)

                p = cv2.goodFeaturesToTrack(self.current_frame, mask=mask, **good_features_parameters)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.features.append([(x, y)])

            # if (self.frame_count + 1) % self.updateVP_interval == 0:
            #     frame = self.get_vps(frame)
            self.frame_count += 1
            self.previous_frame = self.current_frame.copy()
            cv2.imshow('vehicle tracks', frame)
            print(f'fps:{1 / (time() - start)}')
            if cv2.waitKey(1) & 0xFF == 27:
                self.camera.release()
                cv2.destroyAllWindows()
                break

    def detect_car(self, frame, threshold=0.5):
        image_RBG = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_img = Image.fromarray(image_RBG)  # 网络输入为rgb顺序
        data_transform = get_data_transform()
        img, _ = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            time_start = time_synchronized()
            predictions = self.detectModel(img.to(device))[0]  # bboxes_out, labels_out, scores_out
            time_end = time_synchronized()
            print("inference+NMS time: {}".format(time_end - time_start))
            predict_boxes = predictions[0].to("cpu").numpy()
            predict_boxes[:, [0, 2]] = predict_boxes[:, [0, 2]] * original_img.size[0]
            predict_boxes[:, [1, 3]] = predict_boxes[:, [1, 3]] * original_img.size[1]
            predict_classes = predictions[1].to("cpu").numpy()
            predict_scores = predictions[2].to("cpu").numpy()
            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")
            filter_boxes = predict_boxes[predict_scores > threshold].astype(int)
            classes_str = [category_index[i] for i in predict_classes[predict_scores > threshold]]
        return filter_boxes, classes_str

    @staticmethod
    def get_track_length(track):
        x, y = track[0]
        length = 0
        for j in np.arange(1, len(track), 1):
            xn = track[j][0]
            yn = track[j][1]
            diff_x = (xn - x)
            diff_y = (yn - y)
            length += np.sqrt(diff_x * diff_x + diff_y * diff_y)
        return length / len(track)

    def draw_all_tracks(self, save_name='all_tracks.jpg', save_path='./', save_tracks_data=False):
        display = self.init_frame.copy()
        print(len(self.tracks))
        self.filter_tracks()
        print(len(self.tracks))
        cv2.polylines(display, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
        cv2.imwrite(os.path.join(save_path, save_name), display)
        if save_tracks_data:
            with open(os.path.join(save_path, 'tracks.data'), 'wb') as f:
                pickle.dump(self.tracks, f)

    def filter_tracks(self):
        new_tracks = []
        for track in self.tracks:
            track_length = np.linalg.norm(np.array(track[0]) - np.array(track[-1]))
            if track_length < self.tracks_filter_threshold * np.linalg.norm([self.frame_width, self.frame_height]):
                continue
            else:
                new_tracks.append(track)
        self.tracks = new_tracks

    def get_vps(self):
        lines = cvt_diamond_space(self.init_frame, self.tracks)
        self.DiamondSpace.insert(lines)
        vps, weight, vpd_s = self.DiamondSpace.find_peaks(t=0.9, )

        self.vp_1 = vpd_s[0]

        img = cv2.cvtColor(self.init_frame, cv2.COLOR_BGR2RGB)
        # for vp in vps:
        #     cv2.circle(frame, (int(vp[0]), int(vp[1])), radius=3, color=(255, 0, 0))
        # return frame
        print("numbers of vps", len(vps))
        size = self.DiamondSpace.size
        scale = self.DiamondSpace.scale
        _, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(self.DiamondSpace.attach_spaces(), cmap="Greys", extent=(
            (-size + 0.5) / scale, (size - 0.5) / scale, (size - 0.5) / scale,
            (-size + 0.5) / scale))
        ax[0].set(title="Accumulator", xticks=np.linspace(-size + 1, size - 1, 5) / scale,
                  yticks=np.linspace(-size + 1, size - 1, 5) / scale)
        ax[0].plot(vpd_s[:, 0] / scale, vpd_s[:, 1] / scale, "r+")
        ax[0].invert_yaxis()

        ax[1].imshow(img)
        ax[1].set(title="first vanishing point in image")
        ax[1].plot(vps[0, 0], vps[0, 1], 'r+')

        # plt.imshow(img)
        # plt.plot(vps[:, 0], vps[:, 1], 'r+')
        # # plt.show()
        plt.savefig('first_vp.jpg')


def load_model():
    """
    加载SSD目标检测模型
    :return:
        SSD model
    """
    # create model
    # 目标检测数 + 背景
    num_classes = 20 + 1
    model = create_model(num_classes=num_classes)

    # load train weights
    train_weights = "../SSD/save_weights/ssd300-14.pth"
    train_weights_dict = torch.load(train_weights, map_location=device)['model']

    model.load_state_dict(train_weights_dict)
    model.to(device)
    model.eval()
    with torch.no_grad():
        # initial model
        init_img = torch.zeros((1, 3, 300, 300), device=device)
        model(init_img)

    return model


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # read class_indict
    json_path = "../SSD/pascal_voc_classes.json"
    assert os.path.exists(json_path), "file '{}' dose not exist.".format(json_path)
    json_file = open(json_path, 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}
    video_path = '../1-2-981_Trim.mp4'
    # video_path = '../sherbrooke_video.avi'
    # video_path = '../rouen_video.avi'
    cameraTrack = Tracks(video_path)

    cameraTrack.run()
    # cameraTrack.filter_tracks()
    cameraTrack.draw_all_tracks(save_tracks_data=True)
    cameraTrack.get_vps()
