import time

import ctypes
import cv2
import numpy as np
import torch
import torchvision
import logging
logger = logging.getLogger(__name__)

try:
    import onnxruntime as rt
except:
    logger.exception('onnxruntime not found')

try:
    import pycuda.driver as cuda
    import tensorrt as trt
except Exception as e:
    logger.exception(f'PyCuda drivers not found: {e}')


def preprocess_image(images, img_size=640):
    img = np.array([(letterbox(x, new_shape=img_size, auto=False)[0])/255 for x in images])
    img = img.transpose(0, 3, 1, 2).astype(np.float32)

    return np.ascontiguousarray(img)


def postprocess(orig_img_shape, processed_img_shape, preds):
    preds[:, :4] = scale_coords(
        processed_img_shape, preds[:, :4], orig_img_shape).round()
    return preds


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / \
            shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # (pixels) minimum and maximum box width and height
    min_wh, max_wh = 2, 4096
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero().t()
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[
                conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(
        x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / \
            2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def add_padding(left, top, width, height, image, padding=0.15):
    left = left - int(width * padding / 2)
    left = 0 if left < 0 else left

    top = top - int(height * padding)
    top = 0 if top < 0 else top

    width = width + (int(width * padding / 2) * 2)
    width = image.shape[1] if width > image.shape[1] else width

    height = height + (int(height * padding) * 2)
    height = image.shape[0] if height > image.shape[0] else height
    return left, top, width, height


def get_extracted_image(image, bounding_boxes, padding=False):
    extracted_images = []
    boxes = []

    for bb in bounding_boxes:
        left = int(bb[0])
        top = int(bb[1])
        width = int(bb[2] - left)
        height = int(bb[3] - top)
        if padding:
            left, top, width, height = add_padding(left, top, width, height, image)
        extracted_images.append(image[top: top + height, left: left + width])
        boxes.append([left, top, left + width, top + height])

    return extracted_images, boxes


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class YoloOnnx:
    def __init__(self, model_path, img_size=640, conf_thresh=0.4, iou_thresh=0.5, classes=None, padding=False):

        self.padding = padding
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.classes = classes

        self.sess = rt.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.sess.get_inputs()[0].name

    def detect(self, img, classes=None):
        since = time.time()
        
        if classes is None:
            classes = self.classes
        
        processed_img = preprocess_image([img], self.img_size)

        preds = self.sess.run(None, {self.input_name: processed_img})[0]
        preds = torch.from_numpy(np.array(preds))
        preds = non_max_suppression(preds, self.conf_thresh, self.iou_thresh, classes=classes)

        if preds[0] is not None:
            preds = postprocess(img.shape, (self.img_size, self.img_size), preds[0])
            preds = preds.numpy()
            boxes = preds[:, :4].astype(np.int32)
            confs = preds[:, 4]
            classes = preds[:, 5].astype(np.int32)

            # Extract predicted images
            extracted_images, out_boxes = get_extracted_image(img, boxes, self.padding)

            return extracted_images, boxes, confs, classes, time.time() - since

        return np.array([]), np.array([]), np.array([]), np.array([]), time.time() - since
    
    def batch_detect(self, images, classes=None, conf_thresh=0.4):
        since = time.time()

        if classes is None:
            classes = self.classes
            
        if conf_thresh is None:
            conf_thresh = self.conf_thresh

        processed_images = preprocess_image(images, self.img_size)
        preds = self.sess.run(None, {self.input_name: processed_images})[0]
        preds = torch.from_numpy(np.array(preds))
        preds = non_max_suppression(preds, conf_thresh, self.iou_thresh, classes=classes)

        boxes = [None] * len(images)
        confs = [None] * len(images)
        classes = [None] * len(images)
        extracted_images = [None] * len(images)
        for i, pred in enumerate(preds):
            if i < len(images):
                if pred is not None:
                    pred = postprocess(images[i].shape, (self.img_size, self.img_size), pred)
                    pred = pred.numpy()
                    boxes[i] = pred[:, :4].astype(np.int32)
                    confs[i] = pred[:, 4]
                    classes[i] = pred[:, 5].astype(np.int32)
                    
                    # Extract predicted images
                    extracted_image, out_boxes = get_extracted_image(images[i], boxes[i], self.padding)
                    extracted_images[i] = extracted_image

        return extracted_images, np.array(boxes), np.array(confs), np.array(classes), time.time() - since




class YoloTrt:
    def _load_engine(self, model_path):
        with open(model_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _create_context(self):
        return self.engine.create_execution_context()

    def __init__(self, name, model_path, img_size=640, conf_thresh=0.4, iou_thresh=0.5, classes=None, padding=False):
        PLUGIN_LIBRARY = f'./{name}plugin.so'
        ctypes.CDLL(PLUGIN_LIBRARY)

        self.padding = padding
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.classes = classes

        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine(model_path)
        self.context = self._create_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)

    def detect(self, img):
        since = time.time()
        processed_img = preprocess_image([img], self.img_size)

        self.inputs[0].host = processed_img
        trt_outputs = self.do_inference(
            context=self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream)[0]

        num_det = int(trt_outputs[0])
        preds = np.reshape(trt_outputs[1:], (-1, 6))[:num_det, :]
        preds = torch.Tensor(preds)
        if self.classes:
            preds = preds[(preds[:, 5:6] == torch.tensor(self.classes, device=preds.device)).any(1)]

        boxes = preds[:, :4]
        scores = preds[:, 4]
        classid = preds[:, 5]

        si = scores > self.conf_thresh
        boxes = boxes[si, :]
        scores = scores[si]
        classid = classid[si]

        boxes = xywh2xyxy(boxes)
        indices = torchvision.ops.nms(boxes, scores, iou_threshold=self.iou_thresh)
        boxes = boxes[indices, :]
        scores = scores[indices]
        classes = classid[indices]

        if boxes is not None:
            boxes = postprocess(img.shape, (self.img_size, self.img_size), boxes)
            extracted_images, out_boxes = get_extracted_image(img, boxes, self.padding)

            return extracted_images, boxes, scores, classes, time.time() - since

        return np.array([]), np.array([]), np.array([]), np.array([]), time.time() - since

    @staticmethod
    def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async(batch_size=batch_size,
                              bindings=bindings,
                              stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    @staticmethod
    def allocate_buffers(engine):
        """Allocates all host/device in/out buffers required for an engine."""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream
    

class YoloTrt_v2:
    def _load_engine(self, model_path):
        with open(model_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _create_context(self):
        return self.engine.create_execution_context()

    def __init__(self, model_path, img_size=640, conf_thresh=0.4, iou_thresh=0.5, classes=None, padding=False):
        
        self.padding = padding
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.classes = classes

        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine(model_path)
        self.context = self._create_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)

    def detect(self, img):
        since = time.time()
        processed_img = preprocess_image([img], self.img_size)

        self.inputs[0].host = processed_img
        preds = self.do_inference(
            context=self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream)[3]

        preds = torch.from_numpy(np.array(preds))
        preds = non_max_suppression(preds, self.conf_thresh, self.iou_thresh, classes=self.classes)

        if preds[0] is not None:
            preds = postprocess(img.shape, (self.img_size, self.img_size), preds[0])
            preds = preds.numpy()
            boxes = preds[:, :4].astype(np.int32)
            confs = preds[:, 4]
            classes = preds[:, 5].astype(np.int32)

            # Extract predicted images
            extracted_images, out_boxes = get_extracted_image(img, boxes, self.padding)

            return extracted_images, boxes, confs, classes, time.time() - since

        return np.array([]), np.array([]), np.array([]), np.array([]), time.time() - since
    
    def batch_detect(self, images):
        since = time.time()
        processed_images = preprocess_image(images, self.img_size)

        self.inputs[0].host = processed_images
        preds = self.do_inference(
            context=self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream)[3]

        preds = torch.from_numpy(np.array(preds))
        preds = non_max_suppression(preds, self.conf_thresh, self.iou_thresh, classes=self.classes)

        boxes = [None] * len(images)
        confs = [None] * len(images)
        classes = [None] * len(images)
        extracted_images = [None] * len(images)
        for i, pred in enumerate(preds):
            if i < len(images):
                if pred is not None:
                    pred = postprocess(images[i].shape, (self.img_size, self.img_size), pred)
                    pred = pred.numpy()
                    boxes[i] = pred[:, :4].astype(np.int32)
                    confs[i] = pred[:, 4]
                    classes[i] = pred[:, 5].astype(np.int32)
                    
                    # Extract predicted images
                    extracted_image, out_boxes = get_extracted_image(images[i], boxes[i], self.padding)
                    extracted_images[i] = extracted_image

        return extracted_images, boxes, confs, classes, time.time() - since

    @staticmethod
    def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async_v2(
                              bindings=bindings,
                              stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    @staticmethod
    def allocate_buffers(engine):
        """Allocates all host/device in/out buffers required for an engine."""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in engine:
            # size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            size = tuple(engine.get_binding_shape(binding)) * engine.max_batch_size
            print(engine.max_batch_size)
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream


if __name__ == "__main__":
    import pycuda.autoinit
    # det = YoloTrt_v2('/home/mialo/workspace/github-repos/yolov5/runs/train/vehicle_20211207_0/weights/best.engine',
    #                img_size=320, classes=[0, 1, 2, 3, 4])
    
    det = YoloOnnx('frozen_models/yolov5s.onnx',
                   img_size=320, classes=[0])

    import os
    import math
    from tqdm import tqdm
    # from ocr import Ocr as crnn

    # ocr = crnn('/home/mialo/workspace/work/gatebot/models/frozen_models/ocr_v1.onnx')

    # folder = '/home/heet/workspace/oterra_people_counter/data/collection/oterra'
    # file = r'bulk_entry_oterra_1.mp4'
    # batch_size = 16
    
    # Warmup
    # for _ in range(10):
    #     img = np.random.randn(320, 320, 3)
    #     det_images, det_boxes, det_confs, det_class, det_time = det.detect(img)
    
    # Non batch inference
    # since = time.time()
    # # for file in tqdm(os.listdir(folder)):
    # orig_img = cv2.imread(os.path.join(folder, file))
    # print(orig_img)
    # img = orig_img[:, :, ::-1]
    # det_images, det_boxes, det_confs, det_class, det_time = det.detect(img)
    # print('Detect', det_time, det_boxes)

    # for i, box in enumerate(det_boxes):
    #     x1, y1, x2, y2 = box

    #     # Draw  box
    #     c1 = (x1, y1)
    #     c2 = (x2, y2)
    #     cv2.rectangle(orig_img, c1, c2, (0, 255, 0), thickness=2)

    #     (text_width, text_height) = cv2.getTextSize(f'{det_class[i]}', cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2)[0]
    #     cv2.rectangle(orig_img, c1, (c1[0] + text_width + 2, c1[1] +text_height + 10), (0, 255, 0), cv2.FILLED)
    #     cv2.putText(orig_img, f'{det_class[i]}', (c1[0], c1[1] + text_height + 3),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # orig_img = cv2.resize(orig_img, (1280, 720))
    # cv2.imshow('result', orig_img)
    # cv2.waitKey(0)
    # print('Single batch inference time', time.time()-since)

    # Batched inference
    # since = time.time()
    # files = os.listdir(folder)
    # total_steps = math.ceil(len(files)/batch_size)
    # for i in tqdm(range(total_steps)):
    #     images = []
    #     orig_images = []
    #     for j in range(i*batch_size, i*batch_size+batch_size):
    #         if j < len(files):
    #             orig_img = cv2.imread(os.path.join(folder, files[j]))
    #             img = orig_img[:, :, ::-1]
    #             images.append(img)
    #             orig_images.append(orig_img)
            
    #     det_images, det_boxes, det_confs, det_class, det_time = det.batch_detect(images)
    #     for i in range(len(images)):
    #         img = orig_images[i]
    #         for j, box in enumerate(det_boxes[i]):
    #             x1, y1, x2, y2 = box

    #             # Draw  box
    #             c1 = (x1, y1)
    #             c2 = (x2, y2)
                
    #             cv2.rectangle(img, c1, c2, (0, 255, 0), thickness=2)

    #             (text_width, text_height) = cv2.getTextSize(f'{det_class[i][j]}', cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2)[0]
    #             cv2.rectangle(img, c1, (c1[0] + text_width + 2, c1[1] +text_height + 10), (0, 255, 0), cv2.FILLED)
    #             cv2.putText(img, f'{det_class[i][j]}', (c1[0], c1[1] + text_height + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    #         img = cv2.resize(img, (1280, 720))
    #         cv2.imshow('result', img)
    #         cv2.waitKey(0)
    # print('Batch inference time', time.time()-since)


    cap = cv2.VideoCapture(r"data/bulk_entry_oterra_1.mp4")
    if (cap.isOpened() == False):
        print("Error while opeining the video!")

    while (cap.isOpened):
        ret, frame = cap.read() 
        if ret == True:
            
            orig_img = frame
            img = frame[:, :, ::-1]
            det_images, det_boxes, det_confs, det_class, det_time = det.detect(img)
            print('Detect', det_time, det_boxes)

            for i, box in enumerate(det_boxes):
                x1, y1, x2, y2 = box

                # Draw  box
                c1 = (x1, y1)
                c2 = (x2, y2)
                cv2.rectangle(orig_img, c1, c2, (0, 255, 0), thickness=2)

                (text_width, text_height) = cv2.getTextSize(f'{det_class[i]}', cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2)[0]
                cv2.rectangle(orig_img, c1, (c1[0] + text_width + 2, c1[1] +text_height + 10), (0, 255, 0), cv2.FILLED)
                cv2.putText(orig_img, f'{det_class[i]}', (c1[0], c1[1] + text_height + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            orig_img = cv2.resize(orig_img, (1280, 720))
            cv2.imshow('result', orig_img)


            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release
    # the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()