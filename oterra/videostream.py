from threading import Thread
import time
import cv2

from utils.time import Time

import logging
logger = logging.getLogger(__name__)


class VideoStream:
    def __init__(self, src=0, is_file=False):
        self.src = src
        self.stream = self.connect_to_camera()

        self.grabbed, self.frame = self.stream.read()
        # while not self.grabbed:
        #     logger.warning(f"Camera status of {src} is False, reconnecting to camera")
        #     self.stream = self.connect_to_camera()
        #     self.grabbed, self.frame = self.stream.read()

        self.stopped = False
        self.record = False
        self.out = None
        self.name = None
        self.is_file = is_file
        
        if self.is_file:
            self.time = Time()

    def connect_to_camera(self):
        stream = cv2.VideoCapture(self.src)
        return stream

    def start(self):
        if not self.is_file:
            Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        camera_status = False
        while True:
            if self.stopped:
                return

            self.grabbed, self.frame = self.stream.read()

            if not self.grabbed and camera_status:
                camera_status = False

            while not self.grabbed:
                logger.warning(f"Camera status of {self.src} is False, reconnecting to camera")
                self.stream = self.connect_to_camera()
                if self.stream.isOpened():
                    self.grabbed, self.frame = self.stream.read()
                else:
                    self.grabbed =False
                    logger.warning(f"Camera stream of {self.src} is not opened")
                    time.sleep(1)

            if self.grabbed and not camera_status:
                camera_status = True

    def read(self):
        if self.is_file:
            self.grabbed, self.frame = self.stream.read()
            self.time.update(40)
            return self.grabbed, self.frame, self.time.get_time()
        else:
            return self.grabbed, self.frame, None

    def stop(self):
        self.stopped = True

