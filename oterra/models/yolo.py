import time
import numpy as np
import cv2
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
    img = np.array([(latterbox(x, new_shape=img_size, auto=False)[0])/255 for x in images])
    img = img.transpose(0, 3, 1, 2).astype(np.float32)

    return np.ascontiguousarray(img)


def postprocess(orig_img_shape, processed_img_shape, preds):
    preds[:, :4] = scale_coords(
        processed_img_shape, preds[:, :4], orig_img_shape).round()
    return preds


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


def latterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    #Resize image to a 32-pixll-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2] # current shape [height, weight]
    
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
        
    # sacle ratio (new/ old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
        
    #compute padding
    ratio = r, r #width, height ratio
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1] # wh padding
    
    if auto: # minimum rectangle
        dw , dh = np.mod(dw, 64), np.mod(dh, 64) # wh padding
    elif scaleFill: #stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0] # width, height ratios
        
    dw /= 2 # divide padding into 2 sides
    dh /= 2
    
    if shape[::-1] != new_unpad: #resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value = color
    )

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


class ObjectDetector:
    def __init__(self, model_path, img_size = 640, conf_thresh=0.4, iou_thresh=0.5, classes=None, padding=None):
        if model_path.split(".")[-1] == "onnx":
            self.object_detector = YoloOnnx(model_path, img_size, conf_thresh, iou_thresh, classes, padding)
        else:
            logger.info("Wrong Model")
            self.object_detector = None
            
    def detect(self, img, classes=None):
        return self.object_detector.detect(img, classes=classes)
                
            
class YoloOnnx:
    def __init__(self, model_path, img_size = 640, conf_thresh=0.4, iou_thresh=0.5, classes=None, padding=None):
        print("Process Start ...!")
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
        preds = non_max_suppression(preds, self.conf_thresh, self.iou_thresh, classes)

        if preds[0] is not None:
            preds = postprocess(img.shape, (self.img_size, self.img_size), preds[0])
            preds = preds.numpy()
            boxes = preds[:, :4].astype(np.int32)
            confs = preds[:, 4]
            classes = preds[:, 5].astype(np.int32)
            
            extracted_images, out_boxes = get_extracted_image(img, boxes, self.padding)
            
            return extracted_images, boxes, confs, classes, time.time() - since

        return np.array([]), np.array([]), np.array([]), np.array([]), time.time() - since