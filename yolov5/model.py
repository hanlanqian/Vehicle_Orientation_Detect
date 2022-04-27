import json
import cv2
import os
import glob
import torch
# import pycuda.driver as cuda
# import pycuda.autoinit
import numpy as np
import tensorrt as trt

from pathlib import Path
from collections import namedtuple, OrderedDict
from time import time
from yolov5.yolo_utils import letterbox, scale_coords, non_max_suppression

classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush']


class ImageList:
    def __init__(self, image_path, img_size=640, fill=True, auto=False):
        if isinstance(image_path, np.ndarray):
            if len(image_path.shape) != 4:
                files = image_path[None]
            else:
                files = image_path
        else:
            p = str(Path(image_path).resolve())
            if os.path.isdir(p):
                files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
            elif os.path.isfile(p):
                files = [p]  # files

        self.img_size = img_size
        self.auto = auto
        self.fill = fill
        self.files = files
        self.nums = len(files)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nums:
            raise StopIteration
        elif isinstance(self.files, np.ndarray):
            img0 = self.files[self.count]
            self.count += 1
            img = letterbox(img0, self.img_size, auto=self.auto, scaleFill=self.fill)[0]
            img = img.transpose((2, 0, 1))[::-1]
            img = np.ascontiguousarray(img)

            return None, img, img0
        else:
            path = self.files[self.count]
            self.count += 1
            img0 = cv2.imread(path)
            assert img0 is not None, f'Image Not Found {path}'

            # Padded resize
            img = letterbox(img0, self.img_size, auto=self.auto, scaleFill=self.fill)[0]

            # Convert
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)

            return path, img, img0


class YoloTensorrt():
    def __init__(self, engine_file, classes_json, device='cuda:0', verbose=False):
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.WARNING) if verbose else trt.Logger(trt.Logger.INFO)
        with open(engine_file, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        bindings = OrderedDict()
        fp16 = False  # default updated below
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = tuple(model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
            bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            if model.binding_is_input(index) and dtype == np.float16:
                fp16 = True
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        context = model.create_execution_context()
        batch_size = bindings['images'].shape[0]
        self.__dict__.update(locals())

        if os.path.exists(classes_json):
            with open(classes_json, 'r') as f:
                self.classes = list(json.load(f).keys())
        else:
            print("Using default classes")
            self.classes = classes

    def reload_images(self, images):
        self.imageList = ImageList(images, auto=True)

    def warm_up(self, img_shape=(1, 3, 640, 640)):
        if self.device != 'cpu':
            im = torch.zeros(*img_shape, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            self.forward(im)

    def forward(self, im):
        assert im.shape == self.bindings['images'].shape, (im.shape, self.bindings['images'].shape)
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = self.bindings['output'].data
        return y

    def infer(self, threshold=0.5, visualize=False, conf_thres=0.25, iou_thres=0.45, agnostic_nms=False,
              max_det=1000, ):
        boxes = []
        classses = []
        for path, im, im0s in self.imageList:
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]
            start = time()
            pred = self.forward(im)
            print(f"inference time:{time()-start}")
            pred = non_max_suppression(pred, conf_thres, iou_thres, None, agnostic_nms, max_det=max_det)
            for i, det in enumerate(pred):
                im0 = im0s.copy()
                if len(det):
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    filter_index = det[:, -2] > threshold
                    index = det[:, -1][filter_index].int()
                    boxes.append(det[:, :4][filter_index].int().cpu().numpy())
                    classses.append([self.classes[i] for i in index])  # add to string

                    if visualize:
                        for *xyxy, conf, cls in reversed(det):
                            xyxy = [t.int().item() for t in xyxy]
                            if self.classes[(cls.int())] == 'car':
                                cv2.rectangle(im0, xyxy[:2], xyxy[-2:], (0, 255, 255), 2)
            if visualize:
                cv2.imwrite(f'test{self.imageList.count}.jpg', im0)
        return classses, boxes
