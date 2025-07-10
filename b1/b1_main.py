import time
from typing import Dict, List
from camera import Camera
import os
import json
from database.database_config import Database
import config
from models.model import Model
from person import Person

from utils import utilities
from utils import drawing_functions

import cv2

from zones.footfall_zone import FootfallZone


import logging
from logging.handlers import RotatingFileHandler

logging.basicConfig(
    handlers=[RotatingFileHandler('data/logs/app.log', maxBytes=25 * 1024 * 1024, backupCount=3)],
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'    
)

logger = logging.getLogger(__name__)


class Main:
    
    def __init__(self) -> None:
        self.db = Database()
        self.vepl_people_counter = Model(config.OBJECT_DETECTOR)
        self.cameras: List[Camera] = self.get_camera_details()        
        self.start_video_streams()
        
        
    def get_camera_details(self):
        cameras = []
        
        #get camera config files from camera_config folder
        config_folder = './camera_config/'
        for config_file in os.listdir(config_folder):
            
            # check file ends with json
            if config_file.split(".")[-1] == "json":
                file_path = os.path.join(config_folder, config_file)
                f = open(file_path)
                json_data = json.load(f)
                
                camera = Camera(self.db,
                                self.vepl_people_counter,
                                json_data['camera_id'],
                                json_data['src'],
                                json_data['is_file'],
                                json_data['detect_people'],
                                json_data['zones'])
                camera.add_tracker(config.MAX_DISAPPEARED)
                cameras.append(camera)
        
        return cameras


    def start_video_streams(self):
        for camera in self.cameras:
            camera.connect_camera()
            
            #record video
            if config.RECORD_VIDEO:
                camera.add_recorder((config.RECORD_WIDTH, config.RECORD_HEIGHT))
                
            logger.info(f"Connected to camera with camera id {camera.id}")

    
    def get_frames_from_videos(self):
        frames = {}
        
        for i, camera in enumerate(self.cameras):
            status, frame = camera.get_frame()
            
            if status:
                frames[i] = frame
        
        return frames
        
       
    def process_footfall_zones(self, footfall_zones: Dict[int, List[FootfallZone]], track_ids):
        for camera_index, zones in footfall_zones.items():
            camera: Camera = self.cameras[camera_index]
            
            for zone in zones:
                persons = [camera.current_tracker[track_id] for track_id in track_ids[camera_index] if track_id > 0]
                zone.update(camera, persons)
       
        
    def process_zones(self, frames, track_ids):
        footfall_zones = {}
        
        #get zone details from the cameras
        for key, frame in frames.items():
            if len(self.cameras[key].footfall_zones) > 0:
                footfall_zones[key] = self.cameras[key].footfall_zones
                
        self.process_footfall_zones(footfall_zones, track_ids)
        
        
    def process(self, frames):
        if len(frames) > 0:
            #detect and track people from all cameras            
            person_boxs, person_images, track_ids = self.detect_track_people_all(frames)
            self.process_zones(frames, track_ids)
    
    
    def sort_footfall_zone(self, camera:Camera, track_id):
        for fz in camera.footfall_zones:
            fz.count(track_id, camera)
    
    
    def sort(self, camera, track_id):
        self.sort_footfall_zone(camera, track_id)
   
    
    def deregister_objects(self, camera, deregistered_objects):
        for i in deregistered_objects:
            if i in camera.current_tracker:
                self.sort(camera, i)
                del camera.current_tracker[i]
                
                
    def track_person(self, camera, person_boxes, person_images):
        #assign track ids for each person from a single camera
        track_ids = []
        
        if person_boxes is None:
            person_boxes = []
            person_images = []
            
        #update tracker
        object_tracker, deregistered_objects = camera.ct.update(person_boxes)
        self.deregister_objects(camera, deregistered_objects)
        
        final_person_images = person_images
        final_person_boxes = person_boxes
        
        for j in range(len(final_person_boxes)):
            #Get track id
            track_id = utilities.get_track_id(object_tracker, final_person_boxes[j])
            
            track_ids.append(track_id)
            
            if track_id <= 0:
                continue
            
            if track_id not in camera.current_tracker:
                person = Person(track_id)
                camera.current_tracker[track_id] = person
                
            #add bounding box to tracker
            centroid = object_tracker[track_id]
            camera.current_tracker[track_id].boxes.append(final_person_boxes[j])
            camera.current_tracker[track_id].centroids.append(centroid)
            camera.current_tracker[track_id].latest_person_img = final_person_images[j]
            
        return final_person_boxes, final_person_images, track_ids
    
    
    def detect_track_people_all(self, frames):
        #take images from all cameras
        # detects persons and give a tracking id
        
        # get images from all cameras
        images = []
        camera_ids = []
        camera_indexes = []
        # logger.info(f"entered detect and track people")
        #get ferames only from cameras which needs to detect people
        for key, frame in frames.items():
            if self.cameras[key].detect_people:
                images.append(frame[:,:,::-1])
                camera_ids.append(self.cameras[key].id)
                camera_indexes.append(key)
                
        if len(images) > 0:
            #detect person
            person_images, person_boxs, _ = self.vepl_people_counter.detect_person(images)
            
        track_ids = {}
        final_person_boxes = {}
        final_person_images = {}
        
        for i in range(len(images)):
            # Track person from a single camera
            camera = self.cameras[camera_indexes[i]]
            person_box, person_image, person_track_ids = self.track_person(camera, person_boxs[i], person_images[i])
            
            track_ids[i] = person_track_ids
            final_person_boxes[i] = person_box
            final_person_images[i] = person_image   
    
        return final_person_boxes, final_person_images, track_ids
            
            
    def draw_zones(self, camera, frame):
        self.draw_footfall_zones(camera, frame)


    def draw_footfall_zones(self, camera, frame):
        for fz in camera.footfall_zones:
            fz.draw(frame)
      
          
    def draw_bounding_boxes(self, camera:Camera, frame):
        #draw time
        if camera.time is not None:
            time_text = f'{camera.time.replace(microsecond=0)}'
            drawing_functions.draw_multiline_text(frame, text=time_text, org=(0, 0))

        self.draw_zones(camera, frame)
        
        
    def update(self):
        
        while True:
            since = time.time()
            
            # Get frames from all the cameras
            frames = self.get_frames_from_videos()
            
            # Process the frames
            try:
                self.process(frames)
            except:
                logger.exception("Error in process")
                
            for key, frame in frames.items():
                if config.DRAW_BB:
                    self.draw_bounding_boxes(self.cameras[key], frame)
                    
                if config.SHOW_VIDEO:
                    frame = cv2.resize(frame, (config.RECORD_WIDTH, config.RECORD_HEIGHT))
                    cv2.imshow(f"{self.cameras[key].id}", frame)
                    
                if config.RECORD_VIDEO:
                    frame = cv2.resize(frame, (config.RECORD_WIDTH, config.RECORD_HEIGHT))
                    self.cameras[key].recorder.write(frame)
                
            if cv2.waitKey(1) & 0xff == ord("q"):
                cv2.destroyAllWindows()
                break
            
            print("update time:", time.time()-since)
            

if __name__ == "__main__":
    program = Main()
    
    program.update()