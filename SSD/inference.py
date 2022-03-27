import time

import torch

import transforms
from models import SSD300, Backbone


def load_model(device):
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
