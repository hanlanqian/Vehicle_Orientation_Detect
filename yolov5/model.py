import json
import cv2
import os
import glob
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

from pathlib import Path
from time import time
from yolov5 import common
from yolov5.yolo_utils import letterbox, scale_coords, postprocess

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
            return path, img, img0


class YoloTensorrt:
    def __init__(self, engine_file, classes_json, max_batch_size=1, dtype=np.float32):
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(engine_file)
        self.context = self.engine.create_execution_context()
        self.max_batch_size = max_batch_size
        if os.path.exists(classes_json):
            with open(classes_json, 'r') as f:
                self.classes = list(json.load(f).keys())
        else:
            print("Using default classes")
            self.classes = classes

    def load_engine(self, engine_file, ):

        with open(engine_file, 'rb') as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        return engine

    def reload_images(self, images):
        self.imageList = ImageList(images, auto=True)

    def infer(self, threshold=0.5, visualize=False):
        boxes = []
        classses = []
        for path, im, im0s in self.imageList:
            im = im.astype(self.dtype)
            im /= 255
            if len(im.shape) == 3:
                im = im[None]
            pred = self.infer_sigel(im)
            for i, det in enumerate(pred):
                im0 = im0s.copy()
                if len(det):
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    filter_index = det[:, -2] > threshold
                    index = det[:, -1][filter_index].int()
                    boxes.append(det[:, :4][filter_index].int().numpy())
                    classses.append([self.classes[i] for i in index])  # add to string

                    if visualize:
                        for *xyxy, conf, cls in reversed(det):
                            xyxy = [t.int().item() for t in xyxy]
                            cv2.rectangle(im0, xyxy[:2], xyxy[-2:], (0, 255, 255), 2)
            if visualize:
                cv2.imwrite(f'test{self.imageList.count}.jpg', im0)

        return classses, boxes

    def infer_sigel(self, input_image, device='cpu'):
        start = time()
        inputs, outputs, bindings, stream = common.allocate_buffers(self.engine)
        # with self.engine.create_execution_context() as context:
        np.copyto(inputs[0].host, input_image.ravel())
        [output] = common.do_inference_v2(self.context, bindings=bindings, inputs=inputs, outputs=outputs,
                                          stream=stream)

        output = postprocess(output, device)

        return output

    def release(self):
        ## 程序结束后需要释放engine和runtime
        del self.engine
        del self.runtime
        del self.context
