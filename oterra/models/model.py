from models.yolo import ObjectDetector
import logging

logger = logging.getLogger(__name__)

class Model:
    def __init__(self, object_detector=None) -> None:
        self.load_models(object_detector)
        
    def load_models(self, object_detector):
        if object_detector is not None:
            self.object_detector = ObjectDetector(object_detector, img_size=640)
            
    def detect_person(self, images):
        
        person_images, person_boxes, person_confs, _, _ = self.object_detector.detect(images[0], classes=[0])
        
        return [person_images], [person_boxes], [person_confs]