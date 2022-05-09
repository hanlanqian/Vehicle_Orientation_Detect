import math
import cv2
import os
import torch
import pickle
import numpy as np
import matplotlib.pylab as plt

from camera_calibration.calibration_utils import cvt_diamond_space, start_end_line, draw_point_line, \
    computeCameraCalibration, draw_points
from utils import get_pair_keypoints, scale_image
from camera_calibration.edgelets import neighborhood, accumulate_orientation
from camera_calibration.diamondSpace import DiamondSpace
from IPM.utils import convertToBirdView
from yolov5.model import YoloTensorrt
from openpifpaf.predictor import Predictor
from time import time

good_features_parameters = dict(maxCorners=50, qualityLevel=0.3, minDistance=7, blockSize=7, useHarrisDetector=True,
                                k=0.04)
optical_flow_parameters = dict(winSize=(21, 21), minEigThreshold=1e-4)


def CLAHE(img, ):
    b, g, r = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=0.2, tileGridSize=(8, 8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    new = cv2.merge([b, g, r])
    return new


class Calibration_Yolo(object):
    def __init__(self, video_src, input_shape=(900, 1600), roi=None, detect_interval=20, track_interval=10):

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

        # calibration
        self.frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.scale = scale_image((self.frame_height, self.frame_width), input_shape, scaleUp=True)
        # DiamondSpace
        self.DiamondSpace = None
        # vanish point
        self.vp_1 = []
        self.vp_2 = []
        self.vp_3 = []
        self.principal_point = None

        # preprocess
        # todo 需要增加roi区域输入
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 均衡化
        # self.roi = np.zeros()

    def run(self, threshold=0.5, view_process=False):
        self.frame_count = 0
        _, frame = self.camera.read()
        # if not self.roi:
        #
        if self.scale != 1:
            frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        self.init_frame = frame
        background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        background = cv2.medianBlur(background, 3)
        self.principal_point = np.array([self.frame_height * self.scale / 2, self.frame_width * self.scale / 2])
        self.DiamondSpace = DiamondSpace(
            d=min(self.frame_height * self.scale, self.frame_width * self.scale) / 2, size=256)
        while True:
            start = time()

            flag, frame = self.camera.read()

            if not flag:
                print('no frame grabbed!')
                break

            if self.scale != 1:
                frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)

            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_medianBlur = cv2.medianBlur(self.current_frame, 3)
            background = cv2.addWeighted(current_medianBlur, 0.05, background, 0.95, 0)

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
                        points = self.lines_from_box(current_medianBlur, background, box, threshold=threshold,
                                                     winSize=9,
                                                     drawFlag=view_process)
                        if points is not None:
                            self.edgelets.append(points)
                    for x, y in [np.int32(tr[-1]) for tr in self.features]:
                        cv2.circle(mask, (x, y), 10, 0, -1)

                # good tracker
                p = cv2.goodFeaturesToTrack(self.current_frame, mask=mask, **good_features_parameters)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.features.append([(x, y)])
                print(f'final_time:{(time() - start):.6f}')

            self.frame_count += 1
            self.previous_frame = self.current_frame.copy()
            cv2.imshow('frame', frame)
            print(f'fps:{1 / (time() - start)}')
            if cv2.waitKey(1) == 27:
                # cv2.imwrite('frame1.jpg', frame)
                self.camera.release()
                cv2.destroyAllWindows()
                break

    def detect_orientation(self):
        self.frame_count = 0
        orientations = []
        while True:
            flag, original_frame = self.camera.read()
            frame = original_frame.copy()
            if not flag:
                print('no frame grabbed!')
                break
            else:
                if self.scale != 1:
                    frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
                start = time()
                boxes = self.detect_car(frame)
                print(f"detect: {time() - start}")
                for box in boxes:
                    flag, points = self.get_keypoints(frame[box[1]:box[3], box[0]:box[2]])
                    print(f"keypoint: {time() - start}")
                    if flag:
                        points[:, 0] += box[0]
                        points[:, 1] += box[1]
                        frame = draw_points(frame, points[:, :2], visualize=False)

                        points /= self.scale
                        pointsW = np.concatenate([points[:, :2], np.ones((len(points), 1))], axis=1)
                        points_IPM = pointsW @ self.perspective.T
                        points_IPM = points_IPM / points_IPM[:, -1].reshape(-1, 1)
                        diff = points_IPM[0] - points_IPM[1]
                        orientation = math.degrees(np.arctan(diff[1] / diff[0]))
                        orientations.append(orientation)
                        cv2.putText(frame, "angle:" + str(np.around(orientation, 3)), box[:2], cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 255, 0), 2)
                        self.frame_count += 1

                # warp = cv2.warpPerspective(display, self.perspective, self.targets_shape)
                # x_scale, y_scale = scale_image(warp.shape, (900, 1600), force=True)
                # warp = cv2.resize(warp, (-1, -1), fx=x_scale, fy=y_scale)

                print(f"fps:{(1 / (time() - start)):.3f}")
                cv2.imshow('detect', frame)
                # cv2.imshow('warp', warp)

                if cv2.waitKey(1) & 0xFF == 27:
                    print(f"平均角度为{np.average(np.abs(orientations))}")
                    cv2.imwrite('frame1.jpg', original_frame)
                    self.camera.release()
                    cv2.destroyAllWindows()
                    break

    def load_yolo_model(self, engine_file, class_json, verbose=False):
        self.yolo = YoloTensorrt(engine_file, class_json, verbose=verbose)

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

    def get_keypoints(self, image, all=False):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictions, gt_anns, image_meta = self.perdictor.numpy_image(img)
        if predictions:
            data = np.vstack(predictions[0].data)
            index = np.where(data[:, -1] > 0)[0]
            # pair_flag, pair_keypoints = get_pair_keypoints(index, ktype='vertical')
            pair_flag, pair_keypoints = get_pair_keypoints(index, ktype='horizontal')
            if pair_flag:
                if all:
                    return pair_flag, data[index]
                else:
                    return pair_flag, data[pair_keypoints[0]]
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

    def draw_all_tracks(self, save_name='all_tracks.jpg', save_path='./'):
        display = self.init_frame.copy()
        cv2.polylines(display, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
        cv2.imwrite(os.path.join(save_path, save_name), display)

    def lines_from_box(self, current, background, box, winSize=9, threshold=0.5, drawFlag=False):

        vehicle_current = current[box[1]:box[3], box[0]:box[2]]

        vehicle_background = background[box[1]:box[3], box[0]:box[2]]
        # vehicle_current_preprocess = self.clahe.apply(vehicle_current)
        # vehicle_background_preprocess = self.clahe.apply(vehicle_background)
        origin_edges = cv2.Canny(vehicle_current, 255 / 2, 255)
        edges = cv2.Canny(vehicle_current, 255 / 2, 255) - cv2.Canny(vehicle_background, 255 / 2, 255)
        orientation, quality = neighborhood(edges, winSize=winSize)
        accumulation, t = accumulate_orientation(orientation, quality, threshold=threshold)
        res = cv2.addWeighted(accumulation, 0.9, edges, 0.1, 0)
        _, res = cv2.threshold(res, 0, 255, cv2.THRESH_OTSU)
        # _, res = cv2.threshold(res, np.percentile(res[res!=0], 100 * (1 - threshold)), 255, cv2.THRESH_BINARY)
        # thres, edges = cv2.threshold(quality, 0, 255, cv2.THRESH_OTSU)
        # lines = get_lines(edges, orientation, box)
        points = cv2.HoughLinesP(res, 1.0, np.pi / 180, 30, minLineLength=30, maxLineGap=20)
        if points is not None:
            points = points.reshape(-1, 4)
            points[:, [0, 2]] += box[0]
            points[:, [1, 3]] += box[1]
            if drawFlag:
                mask = np.zeros(current.shape)
                for p in points:
                    x1, y1, x2, y2 = p
                    cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
                cv2.imshow('frame', mask)
                cv2.imshow('car', vehicle_current)
                cv2.imshow('edges', edges)
                cv2.imshow('res', res)
                if cv2.waitKey(0) == 27:
                    cv2.imwrite('./origin_edge.jpg', origin_edges)
                    cv2.imwrite('./edges.jpg', edges)
                    cv2.imwrite('./res.jpg', res)
                cv2.destroyWindow('frame')
                cv2.destroyWindow('car')
                cv2.destroyWindow('edges')
                cv2.destroyWindow('res')
        return points

    def get_vp1(self, save_path, visualize=False):
        lines = cvt_diamond_space(self.tracks)
        self.DiamondSpace.insert(lines)
        vps, values, vpd_s = self.DiamondSpace.find_peaks(t=0.9, )
        if len(vps) <= 0:
            raise Exception("Fail to detect the first vanishing point.")
        # vps中权重最大的一个点取为第一消失点
        self.vp_1 = vps[0][:2]

        if visualize:
            self.draw_all_tracks(save_path=f"{save_path}/new/")
            img = cv2.cvtColor(self.init_frame, cv2.COLOR_BGR2RGB)
            print("numbers of vps", len(vps))
            size = self.DiamondSpace.size
            scale = self.DiamondSpace.scale

            # 第一消失点可视化
            _, ax = plt.subplots(1, 2, figsize=(20, 10))
            ax[0].imshow(self.DiamondSpace.attach_spaces(), cmap="Greys", extent=(
                (-size + 0.5) / scale, (size - 0.5) / scale, (size - 0.5) / scale,
                (-size + 0.5) / scale))
            ax[0].set(xticks=np.linspace(-size + 1, size - 1, 5) / scale,
                      yticks=np.linspace(-size + 1, size - 1, 5) / scale)
            ax[0].plot(vpd_s[0, 0] / scale, vpd_s[0, 1] / scale, "ro", markersize=11)
            # ax[0].plot(vpd_s[1:, 0] / scale, vpd_s[1:, 1] / scale, "go", markersize=11)
            ax[0].invert_yaxis()

            ax[1].imshow(img)
            ax[1].set(title="first vanishing point in image")
            ax[1].plot(vps[0, 0], vps[0, 1], 'ro', markersize=11)
            # ax[1].plot(vps[1:, 0], vps[1:, 1], 'go', markersize=11)

            ax[0].set_title(label="Accumulator", fontsize=30)
            ax[0].tick_params(axis='x', labelsize=20)
            ax[0].tick_params(axis='y', labelsize=20)
            ax[1].tick_params(axis='x', labelsize=20)
            ax[1].tick_params(axis='y', labelsize=20)
            ax[1].set_title("first vanishing point in image", fontsize=30)

            plt.savefig(f'{save_path}/new/first_vp1.jpg')

    def get_vp2(self, save_path, visualize=False):
        points = np.vstack(self.edgelets)
        lines = start_end_line(points)
        index = self.DiamondSpace.filter_lines_from_peak(self.vp_1, lines, min(self.frame_width * self.scale,
                                                                               self.frame_height * self.scale) / 2)
        vps, values, vpd_s = self.DiamondSpace.find_peaks(t=0.9, )
        # vps中权重最大的一个点取为第二消失点
        if len(vps) <= 0:
            raise Exception("Fail to detect the second vanishing point.")

        ## get the best
        # first_filter = vps[np.bitwise_and(values == values.max(), vps[:, -1] == 1)]
        # self.vp_2 = first_filter[np.linalg.norm(first_filter, axis=1).argmin()][:2]

        self.vp_2 = vps[0][:2]

        if visualize:
            img = cv2.cvtColor(self.init_frame, cv2.COLOR_BGR2RGB)
            print("numbers of vps", len(vps))
            size = self.DiamondSpace.size
            scale = self.DiamondSpace.scale
            edgelets = draw_point_line(self.init_frame, points, visualFlag=False)
            edgelets_filter = draw_point_line(self.init_frame, points[index], visualFlag=False)
            cv2.imwrite(f'{save_path}/new/edgelets.jpg', edgelets)
            cv2.imwrite(f'{save_path}/new/edgelets_filter.jpg', edgelets_filter)
            # 第二消失点可视化
            _, ax = plt.subplots(1, 2, figsize=(20, 10))
            ax[0].imshow(self.DiamondSpace.attach_spaces(), cmap="Greys", extent=(
                (-size + 0.5) / scale, (size - 0.5) / scale, (size - 0.5) / scale,
                (-size + 0.5) / scale))
            ax[0].set(xticks=np.linspace(-size + 1, size - 1, 5) / scale,
                      yticks=np.linspace(-size + 1, size - 1, 5) / scale)
            ax[0].plot(vpd_s[0, 0] / scale, vpd_s[0, 1] / scale, "ro", markersize=11)
            # ax[0].plot(vpd_s[1:, 0] / scale, vpd_s[1:, 1] / scale, "go", markersize=11)
            ax[0].invert_yaxis()

            ax[1].imshow(img)
            ax[1].set(xticks=np.linspace(-18e4, 0, 5))
            # ax[1].set(title=)

            ax[1].plot(vps[0, 0], vps[0, 1], 'ro', markersize=11)

            ax[0].set_title(label="Accumulator", fontsize=30)
            ax[0].tick_params(axis='x', labelsize=20)
            ax[0].tick_params(axis='y', labelsize=20)
            ax[1].tick_params(axis='x', labelsize=20)
            ax[1].tick_params(axis='y', labelsize=20)
            ax[1].set_title("second vanishing point in image", fontsize=30)
            plt.savefig(f'{save_path}/new/first_vp2.jpg')

    def save_calibration(self, save_path, visualize=False):
        vp1, vp2, vp3, pp, roadPlane, focal, intrinsic_matrix, rotation_matrix = computeCameraCalibration(
            self.vp_1 / self.scale,
            self.vp_2 / self.scale,
            self.principal_point / self.scale)

        calibration = dict(vp1=vp1, vp2=vp2, vp3=vp3, principal_point=pp,
                           roadPlane=roadPlane, focal=focal, intrinsic=intrinsic_matrix, rotation=rotation_matrix)
        with open(f'{save_path}/new/calibrations.npy', 'wb') as f:
            np.save(f, calibration)

        if visualize:
            img = cv2.cvtColor(self.init_frame, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(20, 20))
            plt.imshow(img)
            plt.plot(vp1[0], vp1[1], 'ro', markersize=11)
            plt.plot(vp2[0], vp2[1], 'ro', markersize=11)
            plt.plot(vp3[0], vp3[1], 'ro', markersize=11)
            plt.savefig(f'{save_path}/new/all_vps.jpg')
        return calibration

    def calibrate(self, save_path, visualize=False, ):
        if not os.path.exists(os.path.join(save_path, 'new')):
            os.makedirs(os.path.join(save_path, 'new'), )
        self.get_vp1(save_path, visualize)
        self.get_vp2(save_path, visualize)
        self.save_calibration(save_path, visualize)

    def load_calibration(self, path, dst_shape=(1600, 900), strict=False):
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
            target_shape = dst_shape if dst_shape[0] * dst_shape[1] > self.frame_width * self.frame_height else (
                self.frame_width, self.frame_height)
            self.perspective = convertToBirdView(self.intrinsic, self.rotation, (self.frame_width, self.frame_height),
                                                 target_shape=target_shape, strict=strict)
        except Exception as e:
            print('failed to load calibration')
            raise e
