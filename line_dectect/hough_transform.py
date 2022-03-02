import sys
import cv2
import os
import torch
import json
from time import time
from PIL import Image

import numpy as np

sys.path.append(r'E:\Pycharm\mine\Vehicle_Orientation_Detect\SSD')

from SSD.inference import time_synchronized, create_model, get_data_transform

# SSD detect interval
FrameInterval = 20

# Draw Line
DrawInterval = 10
colors = np.random.randint(0, 255, (100, 3))


class Vehicle:
    def __init__(self, frame, box, features=None, ):
        self.frame = frame
        self.box = box  # SSD detect box
        self.good_features_parameters = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.draw_circle_parameters = dict(radius=3, color=(0, 0, 255), thickness=3, )
        self.optical_flow_parameters = dict(winSize=(21, 21), minEigThreshold=1e-4)
        self.features = self.getFeatures()
        self.next_features = None
        self.isDelete = False

    def delete(self):
        # 对象软删除
        self.isDelete = True

    def getFeatures(self):
        # box = [xmin, ymin, xmax, ymax]
        gray_car = cv2.cvtColor(self.frame[self.box[1]:self.box[3], self.box[0]:self.box[2]], cv2.COLOR_BGR2GRAY)
        # return cv2.goodFeaturesToTrack(gray_car, **self.good_features_parameters).astype(int).reshape(-1, 2)
        points = cv2.goodFeaturesToTrack(gray_car, **self.good_features_parameters)
        global_points = (points + self.box[:2]).astype(np.float32)
        return global_points

    def draw_features(self, img):
        for point in self.features.astype(int).reshape(-1, 2):
            cv2.circle(img, tuple(point), **self.draw_circle_parameters)
        return img

    def KLT_tracker(self, previousImg, currentImg):
        previous_gray = cv2.cvtColor(previousImg, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(currentImg, cv2.COLOR_BGR2GRAY)
        self.next_features, status, err = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, self.features,
                                                                   self.next_features, **self.optical_flow_parameters)
        return status

    def draw_tracker(self, status, mask):
        if self.features is not None:
            good_new = self.next_features[status == 1]
            good_old = self.features[status == 1]
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), colors[i].tolist(), 2)
        self.features = self.next_features.copy()
        return mask

    def checkDisappear(self, new_box):
        for point in self.next_features:
            a, b = point.ravel()
            if not (new_box[1] < b < new_box[3] and new_box[0] < a < new_box[2]):
                return True

        return False


def Cvt2ParallelSpace(mat):
    # 将平面坐标系的点转到Y, X, -Y轴的平行坐标系(u, v)
    # u = -Y 等价与 u = -1
    # u = Y 等价与 u = 1
    # u = X 等价与 u = 0

    mat[mat > 0] = mat[mat > 0]
    print(mat)


def load_model():
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


def dectect_car(model, image_array, data_transform, threshold=0.5):
    start = time()
    original_img = Image.fromarray(image_array)

    img, _ = data_transform(original_img)
    print(time() - start)

    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    with torch.no_grad():
        time_start = time_synchronized()
        predictions = model(img.to(device))[0]  # bboxes_out, labels_out, scores_out
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


def process_video(video_path):
    # parameters setting
    camera = cv2.VideoCapture(video_path)
    _, frame = camera.read()
    count_frame = 1
    last_frame = frame

    # SSD 车辆检测模型加载
    data_transform = get_data_transform()
    model = load_model()
    cars = []

    # 画图掩膜
    mask = np.zeros_like(frame)

    boxes, classes = dectect_car(model, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), data_transform)
    for box in boxes:
        detected_car = Vehicle(frame, box)
        cars.append(detected_car)
    while True:
        start = time()
        flag, frame = camera.read()
        if count_frame & DrawInterval == 0:
            if cars:
                for car in cars:
                    if car.isDelete:
                        continue
                    status = car.KLT_tracker(last_frame, frame)
                    mask = car.draw_tracker(status, mask)
        frame = cv2.add(mask, frame)

        if count_frame % FrameInterval == 0:
            boxes, classes = dectect_car(model, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), data_transform)
            checkFlag = False
            for box in boxes:
                ## To-do
                # 如果cars中已有boxes包含的车辆，则不会创建新的车辆实例Vehicle
                for car in cars:
                    if car.isDelete:
                        continue
                    if car.checkDisappear(box):
                        checkFlag = True
                        car.delete()
                        break
                if checkFlag:
                    checkFlag = False
                    continue
                else:
                    detected_car = Vehicle(frame, box, )
                    frame = detected_car.draw_features(frame)
                    cars.append(detected_car)
        if flag:
            display = cv2.resize(frame, (1600, 900))
            cv2.imshow('video', display)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
        count_frame += 1
        print(f"fps: {1 / (time() - start)}")
        last_frame = frame.copy()

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # read class_indict
    json_path = "../SSD/pascal_voc_classes.json"
    assert os.path.exists(json_path), "file '{}' dose not exist.".format(json_path)
    json_file = open(json_path, 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}
    video_path = '../1-2-981_Trim.mp4'
    process_video(video_path)
# original_img = cv2.imread('../test.jpg')
# gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
# boxes, classes = inference.main('../test.jpg')
# for i, box in enumerate(boxes):
#     cv2.cornerHarris()
# detect = cv2.cvtColor(detect, cv2.COLOR_RGB2BGR)
# gray_medium_img = cv2.medianBlur(gray_img, 5)
# edges = cv2.Canny(gray_medium_img, 300, 300)
# # x = Cvt2ParallelSpace(edges)
# cv2.imshow('original', original_img)
# cv2.imshow('detect', detect)
# cv2.imshow('gray_medium_img', gray_medium_img)
# cv2.imshow('edges', edges)
# cv2.waitKey(-1)
# cv2.destroyAllWindows()
