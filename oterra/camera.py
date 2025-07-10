from centroid_tracker import CentroidTracker
from videostream import VideoStream
import cv2
import datetime
import logging
from typing import List
from zones.footfall_zone import FootfallZone

import os
import sys
logger = logging.getLogger(__name__)

class Camera():
    def __init__(self, db, vepl, id, src, is_file, detect_people, zones) -> None:
        self.db = db
        self.vepl = vepl
        
        self.ct = None
        self.current_tracker = None
        self.id = id
        self.src = src
        self.is_file = is_file
        self.cap = None
        self.detect_people = detect_people
                
        self.recorder = None
        
        self.last_offline_check_time = datetime.datetime.now()
        self.init_zones(zones)
        
    def init_zones(self, zones):        
        self.footfall_zones: List[FootfallZone] = []
        for zone, values in zones.items():
           if zone == "footfall_zones":
               for value in values:
                    fz = FootfallZone(self.db,
                                     self.vepl,
                                     value["zone_id"],
                                     value["name"],
                                     value["roi"],
                                     value["people_counter_line"],
                                     value["point_for_count"],
                                     value["detect_gender"]
                                     ) 
                    self.footfall_zones.append(fz)

        
    def add_tracker(self, max_disappered):
        ct = CentroidTracker(maxDisappeared = max_disappered, nextObjectId=1)
        self.ct = ct
        self.current_tracker = {}
        
        
    def connect_camera(self):
        cap = VideoStream(self.src, is_file=self.is_file)
        cap.start()
        
        self.cap = cap
        
    def add_recorder(self, frame_shape):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        recorder = cv2.VideoWriter(f'data/output_{self.id}.avi', fourcc, 25.0, frame_shape)
        self.recorder = recorder
        
    def get_frame(self):
        status, frame, time = self.cap.read()
        
        if not status and (datetime.datetime.now() - self.last_offline_check_time).total_seconds() > 10:
            logger.warning(f"camera {self.id} is offline")
            self.last_offline_check_time = datetime.datetime.now()
            logger.info(f"Attempting to reconnect camera {self.id}")
            self.connect_camera()

        self.time = time
        return status, frame