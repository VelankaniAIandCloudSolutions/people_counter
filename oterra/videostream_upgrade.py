from threading import Thread, Lock
import time
import cv2
from utils.time import Time
import logging
import datetime

logger = logging.getLogger(__name__)

class VideoStream:
    def __init__(self, src=0, is_file=False):
        self.src = src
        self.stream = None
        self.grabbed = False
        self.frame = None
        self.stopped = False
        self.record = False
        self.out = None
        self.name = None
        self.is_file = is_file
        self.lock = Lock()
        self.last_reconnect_attempt = datetime.datetime.now()
        self.reconnect_interval = 5  # seconds between reconnection attempts
        
        # Initial connection
        self.stream = self.connect_to_camera()
        if self.stream and self.stream.isOpened():
            self.grabbed, self.frame = self.stream.read()
        
        if self.is_file:
            self.time = Time()

    def connect_to_camera(self):
        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES):
            try:
                stream = cv2.VideoCapture(self.src)
                if stream.isOpened():
                    # Configure stream settings
                    stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    stream.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS to reduce load
                    test_grab = stream.grab()  # Test if we can actually grab frames
                    if test_grab:
                        logger.info(f"Successfully connected to camera {self.src}")
                        return stream
                    else:
                        stream.release()
                
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed for camera {self.src}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1)
        
        logger.error(f"Failed to connect to camera {self.src} after {MAX_RETRIES} attempts")
        return None

    def start(self):
        if not self.is_file:
            Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            try:
                # Check if we need to attempt reconnection
                current_time = datetime.datetime.now()
                if (self.stream is None or not self.grabbed) and \
                   (current_time - self.last_reconnect_attempt).total_seconds() >= self.reconnect_interval:
                    
                    logger.warning(f"Attempting to reconnect to camera {self.src}")
                    if self.stream:
                        self.stream.release()
                    self.stream = self.connect_to_camera()
                    self.last_reconnect_attempt = current_time
                    
                    if self.stream is None:
                        time.sleep(1)
                        continue

                # Try to read frame
                if self.stream and self.stream.isOpened():
                    ret = self.stream.grab()  # First grab frame
                    if ret:
                        with self.lock:
                            self.grabbed, self.frame = self.stream.retrieve()  # Then retrieve it
                            if not self.grabbed:
                                self.stream.release()
                                self.stream = None
                    else:
                        self.stream.release()
                        self.stream = None
                
                time.sleep(0.01)  # Small delay to prevent CPU overload

            except Exception as e:
                logger.error(f"Error in update loop for camera {self.src}: {e}")
                if self.stream:
                    self.stream.release()
                self.stream = None
                time.sleep(1)

    def read(self):
        if self.is_file:
            if self.stream is None:
                return False, None, None
            self.grabbed, self.frame = self.stream.read()
            self.time.update(40)
            return self.grabbed, self.frame, self.time.get_time()
        else:
            with self.lock:
                return self.grabbed, self.frame, None

    def stop(self):
        self.stopped = True
        if self.stream:
            self.stream.release()