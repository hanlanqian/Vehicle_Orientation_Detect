import time

import torch
import numpy as np
import transforms
from models import SSD300, Backbone
from PIL import Image


def load_model(device, num_classes=20 + 1, checkpoint="../SSD/save_weights/ssd300-14.pth"):
    """
    加载SSD目标检测模型
    :return:
        SSD model
    """
    # create model
    # 目标检测数 + 背景
    model = create_model(num_classes=num_classes)

    # load train weights
    train_weights_dict = torch.load(checkpoint, map_location=device)['model']

    model.load_state_dict(train_weights_dict)
    model.to(device)
    model.eval()
    with torch.no_grad():
        # initial model
        init_img = torch.zeros((1, 3, 300, 300), device=device)
        model(init_img)

    return model


def inference(model, tf, images_batches, category_index, threshold=0.5, device='cpu'):
    height, width = images_batches[0].shape[:2]
    imgs = []
    for img in images_batches:
        imgPIL = Image.fromarray(img)
        _img, _ = tf(imgPIL)
        imgs.append(_img)
    images = torch.stack(imgs, dim=0)

    with torch.no_grad():
        predictions = model(images.to(device))[0]  # bboxes_out, labels_out, scores_out
        predict_boxes = predictions[0].to("cpu").numpy()
        predict_boxes[:, [0, 2]] = predict_boxes[:, [0, 2]] * width
        predict_boxes[:, [1, 3]] = predict_boxes[:, [1, 3]] * height
        predict_classes = predictions[1].to("cpu").numpy()
        predict_scores = predictions[2].to("cpu").numpy()
        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")
        filter_boxes = predict_boxes[predict_scores > threshold].astype(int)
        classes_str = [category_index[i] for i in predict_classes[predict_scores > threshold]]
    return filter_boxes, classes_str


def create_model(num_classes):
    backbone = Backbone()
    model = SSD300(backbone=backbone, num_classes=num_classes)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def get_data_transform():
    return transforms.Compose([transforms.Resize(),
                               transforms.ToTensor(),
                               transforms.Normalization()])
