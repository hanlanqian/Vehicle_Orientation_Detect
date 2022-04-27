import math
import cv2
import os
import torch
import pickle
import numpy as np
import matplotlib.pylab as plt

from camera_calibration.calibration_utils import cvt_diamond_space, start_end_line, draw_point_line, \
    computeCameraCalibration
from utils import get_pair_keypoints, scale_image
from camera_calibration.edgelets import neighborhood, accumulate_orientation
from camera_calibration.diamondSpace import DiamondSpace
from IPM.utils import convertToBirdView
from yolov5.last_model import YoloTensorrt
from openpifpaf.predictor import Predictor
from time import time

good_features_parameters = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7, useHarrisDetector=True,
                                k=0.04)
optical_flow_parameters = dict(winSize=(21, 21), minEigThreshold=1e-4)


class Calibration_Yolo(object):
    def __init__(self, video_src, detect_interval=20, track_interval=10):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tracks = []  # vehicle tracks
        self.edgelets = []  # vehicle edges
        self.features = []  # Harris corners
        self.background = None
        self.detectInterval = detect_interval
        self.track_interval = track_interval
        self.camera = cv2.VideoCapture(video_src)

        self.frame_count = 0
        self.detectModel = None
        self.init_frame = None
        self.current_frame = None
        self.previous_frame = None
        self.diff_threshold = 1
        self.maxFeatureNum = 50

        self.feature_determine_threshold = 0.1
        self.tracks_filter_threshold = 0.005

        # vanish point
        self.vp_1 = []
        self.vp_2 = []
        self.vp_3 = []
        self.principal_point = None

        # calibration
        self.frame_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.scale = scale_image((self.frame_height, self.frame_width), (900, 1600))
        # DiamondSpace
        self.DiamondSpace = None

    def run(self):
        self.frame_count = 0
        while True:
            start = time()

            flag, frame = self.camera.read()

            if not flag:
                print('no frame grabbed!')
                break

            if self.scale != 1:
                frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)

            if self.frame_count == 0:
                self.init_frame = frame
                self.principal_point = np.array([self.frame_height * self.scale / 2, self.frame_width * self.scale / 2])
                self.DiamondSpace = DiamondSpace(
                    d=min(self.frame_height * self.scale, self.frame_width * self.scale) / 2, size=256)

            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if len(self.features) > 0:
                # KLT Tracker
                pimg, cimg = self.previous_frame, self.current_frame
                last_feature = np.float32([f[-1] for f in self.features])
                feature_c, status_c, _ = cv2.calcOpticalFlowPyrLK(pimg, cimg, last_feature, None,
                                                                  **optical_flow_parameters)
                feature_recover, status_r, _ = cv2.calcOpticalFlowPyrLK(cimg, pimg, feature_c, None,
                                                                        **optical_flow_parameters)

                good_features_index = [1e-5 < diff < self.diff_threshold for diff in
                                       abs(feature_recover - last_feature).reshape(-1, 2).max(-1)]
                # new_tracks = []
                for fp, fc, isGood in zip(self.features, feature_c, good_features_index):
                    if not isGood:
                        continue
                    fp.append(tuple(fc.ravel().tolist()))

                    if len(fp) > self.maxFeatureNum:
                        del fp[0]
                        if self.frame_count % self.track_interval == 0:
                            if self.get_track_length(fp) > 30:
                                temp = fp.copy()
                                self.tracks.append(temp)

            if self.frame_count % self.detectInterval == 0:
                mask = np.zeros_like(self.current_frame)
                boxes = self.detect_car(frame)
                if len(boxes) > 0:
                    for box in boxes:
                        mask[box[1]:box[3], box[0]:box[2]] = 255
                        points = self.lines_from_box(frame, box)
                        if points is not None:
                            self.edgelets.append(points)
                    for x, y in [np.int32(tr[-1]) for tr in self.features]:
                        cv2.circle(mask, (x, y), 5, 0, -1)

                # good tracker
                p = cv2.goodFeaturesToTrack(self.current_frame, mask=mask, **good_features_parameters)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.features.append([(x, y)])
                print(f'final_time:{(time() - start):.6f}')

            self.frame_count += 1
            self.previous_frame = self.current_frame.copy()
            cv2.imshow('vehicle tracks', frame)
            print(f'fps:{1 / (time() - start)}')
            if cv2.waitKey(1) == 27:
                self.yolo.release()
                self.camera.release()
                cv2.destroyAllWindows()
                break

    def detect_orientation(self):
        self.frame_count = 0
        while True:
            start = time()
            flag, frame = self.camera.read()
            if not flag:
                print('no frame grabbed!')
                break
            else:
                if self.scale != 1:
                    frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)

                boxes = self.detect_car(frame)
                print(f"detect: {time() - start}")
                for box in boxes:
                    flag, points = self.get_keypoints(frame[box[1]:box[3], box[0]:box[2]])
                    print(f"keypoint: {time() - start}")
                    if flag:
                        points[:, 0] += box[0]
                        points[:, 1] += box[1]
                        pointsW = np.concatenate([points[:, :2], np.ones((len(points), 1))], axis=1)
                        points_IPM = pointsW @ self.perspective.T
                        points_IPM = points_IPM / points_IPM[:, -1].reshape(-1, 1)
                        diff = points_IPM[0] - points_IPM[1]
                        orientation = math.degreges(np.arctan(diff[1] / diff[0]))
                        cv2.putText(frame, "angle:" + str(orientation), box[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 0), 2)

                # warp = cv2.warpPerspective(display, self.perspective, self.targets_shape)
                # x_scale, y_scale = scale_image(warp.shape, (900, 1600), force=True)
                # warp = cv2.resize(warp, (-1, -1), fx=x_scale, fy=y_scale)

                print(f"fps:{(1 / (time() - start)):.3f}")
                cv2.imshow('detect', frame)
                # cv2.imshow('warp', warp)

                if cv2.waitKey(1) & 0xFF == 27:
                    self.yolo.release()
                    self.camera.release()
                    cv2.destroyAllWindows()
                    break

    def load_yolo_model(self, engine_file, class_json):
        self.yolo = YoloTensorrt(engine_file, class_json)

    def detect_car(self, frame, threshold=0.5):
        self.yolo.reload_images(frame)
        cls, bs = self.yolo.infer(threshold=threshold)
        if cls:
            [classes], [boxes] = cls, bs
            # filter cars
            index = [True if c in ['car', 'bus', 'truck', ] else False for c in classes]
            cars_boxes = boxes[index]
            return cars_boxes
        else:
            return []

    def load_keypoint_model(self, ):
        self.perdictor = Predictor(checkpoint="shufflenetv2k16-apollo-24")

        # warmup
        img = np.ones((640, 640, 3), dtype=np.uint8)
        self.perdictor.numpy_image(img)

    def get_keypoints(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictions, gt_anns, image_meta = self.perdictor.numpy_image(img)
        if predictions:
            data = np.vstack(predictions[0].data)
            index = np.where(data[:, -1] > 0)[0]
            # pair_flag, pair_keypoints = get_pair_keypoints(index, ktype='vertical')
            pair_flag, pair_keypoints = get_pair_keypoints(index, ktype='horizontal')
            if pair_flag:
                return pair_flag, data[pair_keypoints[0]]
                # return pair_flag, data[index]
            else:
                return pair_flag, None
        else:
            return False, None

    @staticmethod
    def get_track_length(track):
        """
        获得track的平均长度，用于过滤已消失在监控中的特征点仍在跟踪的情况
        :param track: the track of the feature point
        :return: the average length of track
        """
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
        cv2.polylines(display, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
        cv2.imwrite(os.path.join(save_path, save_name), display)
        if save_tracks_data:
            with open(os.path.join(save_path, 'tracks.data'), 'wb') as f:
                pickle.dump(self.tracks, f)

    def lines_from_box(self, frame, box, threshold=0.25, drawFlag=False):
        vehicle_img = frame[box[1]:box[3], box[0]:box[2]]
        edges = cv2.Canny(vehicle_img, 200, 200, L2gradient=True)
        orientation, quality = neighborhood(edges)
        accumulation, t = accumulate_orientation(orientation, quality)
        res = cv2.addWeighted(accumulation, 0.8, edges, 0.2, 0)
        # _, res = cv2.threshold(res, np.percentile(res[res!=0], 100 * (1 - threshold)), 255, cv2.THRESH_BINARY)
        # thres, edges = cv2.threshold(quality, 0, 255, cv2.THRESH_OTSU)
        # lines = get_lines(edges, orientation, box)
        points = cv2.HoughLinesP(res, 1.0, np.pi / 180, 30, minLineLength=30, maxLineGap=20)
        if points is not None:
            points = points.reshape(-1, 4)
            points[:, [0, 2]] += box[0]
            points[:, [1, 3]] += box[1]
            if drawFlag:
                for p in points:
                    x1, y1, x2, y2 = p
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.imshow('frame', frame)
                cv2.waitKey(0)
                cv2.destroyWindow('frame')
                return frame
        return points

    def get_vp1(self, visualize=False):
        lines = cvt_diamond_space(self.tracks)
        self.DiamondSpace.insert(lines)
        vps, values, vpd_s = self.DiamondSpace.find_peaks(t=0.9, )
        # vps中权重最大的一个点取为第一消失点
        self.vp_1 = vps[0][:2]

        if visualize:
            img = cv2.cvtColor(self.init_frame, cv2.COLOR_BGR2RGB)
            print("numbers of vps", len(vps))
            size = self.DiamondSpace.size
            scale = self.DiamondSpace.scale

            # 第一消失点可视化
            _, ax = plt.subplots(1, 2, figsize=(20, 10))
            ax[0].imshow(self.DiamondSpace.attach_spaces(), cmap="Greys", extent=(
                (-size + 0.5) / scale, (size - 0.5) / scale, (size - 0.5) / scale,
                (-size + 0.5) / scale))
            ax[0].set(title="Accumulator", xticks=np.linspace(-size + 1, size - 1, 5) / scale,
                      yticks=np.linspace(-size + 1, size - 1, 5) / scale)
            ax[0].plot(vpd_s[0, 0] / scale, vpd_s[0, 1] / scale, "ro", markersize=11)
            # ax[0].plot(vpd_s[1:, 0] / scale, vpd_s[1:, 1] / scale, "go", markersize=11)
            ax[0].invert_yaxis()

            ax[1].imshow(img)
            ax[1].set(title="first vanishing point in image")
            ax[1].plot(vps[0, 0], vps[0, 1], 'ro', markersize=11)
            # ax[1].plot(vps[1:, 0], vps[1:, 1], 'go', markersize=11)

            plt.savefig('./camera_calibration/avi6/first_vp1.jpg')

    def get_vp2(self, visualize=False):
        points = np.vstack(self.edgelets)
        lines = start_end_line(points)
        index = self.DiamondSpace.filter_lines_from_peak(self.vp_1, lines, min(self.frame_width, self.frame_height) / 2)
        vps, values, vpd_s = self.DiamondSpace.find_peaks(t=0.9, )
        # vps中权重最大的一个点取为第二消失点
        self.vp_2 = vps[0][:2]
        if visualize:
            img = cv2.cvtColor(self.init_frame, cv2.COLOR_BGR2RGB)
            print("numbers of vps", len(vps))
            size = self.DiamondSpace.size
            scale = self.DiamondSpace.scale
            edgelets = draw_point_line(self.init_frame, points, visualFlag=False)
            edgelets_filter = draw_point_line(self.init_frame, points[index], visualFlag=False)
            cv2.imwrite('./camera_calibration/avi6/edgelets.jpg', edgelets)
            cv2.imwrite('./camera_calibration/avi6/edgelets_filter.jpg', edgelets_filter)
            # 第一消失点可视化
            _, ax = plt.subplots(1, 2, figsize=(20, 10))
            ax[0].imshow(self.DiamondSpace.attach_spaces(), cmap="Greys", extent=(
                (-size + 0.5) / scale, (size - 0.5) / scale, (size - 0.5) / scale,
                (-size + 0.5) / scale))
            ax[0].set(title="Accumulator", xticks=np.linspace(-size + 1, size - 1, 5) / scale,
                      yticks=np.linspace(-size + 1, size - 1, 5) / scale)
            ax[0].plot(vpd_s[0, 0] / scale, vpd_s[0, 1] / scale, "ro", markersize=11)
            # ax[0].plot(vpd_s[1:, 0] / scale, vpd_s[1:, 1] / scale, "go", markersize=11)
            ax[0].invert_yaxis()

            ax[1].imshow(img)
            ax[1].set(title="first vanishing point in image")

            ax[1].plot(vps[0, 0], vps[0, 1], 'ro', markersize=11)
            # ax[1].plot(vps[1:, 0], vps[1:, 1], 'go', markersize=11)
            # ax[1].plot()
            plt.savefig('./camera_calibration/avi6/first_vp2.jpg')

    def save_calibration(self, visualize=False):
        vp1, vp2, vp3, pp, roadPlane, focal, intrinsic_matrix, rotation_matrix = computeCameraCalibration(
            self.vp_1 / self.scale,
            self.vp_2 / self.scale,
            self.principal_point / self.scale)

        calibration = dict(vp1=vp1, vp2=vp2, vp3=vp3, principal_point=pp,
                           roadPlane=roadPlane, focal=focal, intrinsic=intrinsic_matrix, rotation=rotation_matrix)
        with open('./camera_calibration/avi6/calibrations.npy', 'wb') as f:
            np.save(f, calibration)

        if visualize:
            img = cv2.cvtColor(self.init_frame, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(20, 20))
            plt.imshow(img)
            plt.plot(vp1[0], vp1[1], 'ro', markersize=11)
            plt.plot(vp2[0], vp2[1], 'ro', markersize=11)
            plt.plot(vp3[0], vp3[1], 'ro', markersize=11)
            plt.savefig("./camera_calibration/avi6/all_vps.jpg")
        return calibration

    def load_calibration(self, path):
        try:
            with open(path, 'rb') as f:
                calibration = np.load(f, allow_pickle=True).tolist()
            print(calibration)
            self.vp_1 = calibration.get("vp1")
            self.vp_2 = calibration.get("vp2")
            self.vp_3 = calibration.get("vp3")
            self.principal_point = calibration.get('principal_point')
            self.focal = calibration.get('focal')
            self.roadPlane = calibration.get('roadPlane')
            self.intrinsic = calibration.get('intrinsic')
            self.rotation = calibration.get('rotation')
            self.perspective = convertToBirdView(self.intrinsic, self.rotation, (self.frame_width, self.frame_height),
                                                 target_shape=(self.frame_width, self.frame_height))
        except:
            print('failed to load calibration')
            return "Error"
