import cv2
import sys
import json
from time import time
import numpy as np
from openpifpaf.predictor import Predictor
from PIL import Image

sys.path.append(r'E:\Pycharm\mine\Vehicle_Orientation_Detect\SSD')
from SSD.inference import load_model, get_data_transform, inference

class_json = "../SSD/pascal_voc_classes.json"
json_file = open(class_json, 'r')
class_dict = json.load(json_file)
category_index = {v: k for k, v in class_dict.items()}

device = 'cuda'
model = load_model(device=device)
transfrom = get_data_transform()

# path = '../cars.jpg'
video_path = r"E:\datasets\surveilliance\v5.avi"
predictor = Predictor(checkpoint="shufflenetv2k16-apollo-24")

cam = cv2.VideoCapture(video_path)
l = []
while True:
    _, frame = cam.read()

    display = frame.copy()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgs = np.expand_dims(img, axis=0)
    filter_boxes, classes_str = inference(model, transfrom, imgs, category_index, device=device)
    print(len(filter_boxes))
    count = 0
    correct = 0
    for box in filter_boxes:
        count += 1
        start = time()
        car = display[box[1]:box[3], box[0]:box[2]]
        rgb_car = cv2.cvtColor(car, cv2.COLOR_BGR2RGB)
        carimg = Image.fromarray(rgb_car)
        predictions, _, _ = predictor.pil_image(carimg)
        # print(f"prediction time is {time() - start}")
        if predictions:
            correct += 1
            data = np.vstack(predictions[0].data)
            index = np.where(data[:, -1] > 0)[0]
            keypoint = [key for i, key in enumerate(predictions[0].keypoints) if i in index]
            true_data = data[index]

            for point, key in zip(true_data, keypoint):
                x, y = point[:2]
                cv2.circle(display, (int(x) + box[0], int(y) + box[1]), 3, (0, 255, 255), 2)
                # cv2.putText(img, str(key), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                #             (255, 255, 0), 2)
        # print(f"time is {time()-start}")
        l.append(correct / count)

    cv2.imshow('test', display)
    if cv2.waitKey(1) & 0xff == 27:
        l = np.vstack(l)

        cam.release()
        cv2.destroyAllWindows()
        print(np.average(l))
        break
