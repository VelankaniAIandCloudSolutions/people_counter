import cv2

from threading import Thread
from typing import List

from zones.zone import Zone
from person import Person

from utils import utilities
from utils import drawing_functions

import logging
logger = logging.getLogger(__name__)

class FootfallZone(Zone):
    def __init__(self, db, rb, id, name, roi, counter_line, point_for_count, detect_gender):
        super().__init__(db, rb, id, name, roi)
        
        self.counter_line = counter_line
        self.point_for_count = point_for_count      # center, bottom
        self.detect_gender = detect_gender
        
        self.tracker = {}
        self.persons: List[Person] = []
        
        # Debug values
        self.entry_count = 0
        self.exit_count = 0
        
    def add_tracker(self, track_id):
        self.tracker[track_id] = {
            "boxes": [],
            "centroids": []
        }
        
    def get_gender(self, person_image):
        gender = self.rb.classify_gender([person_image])[0]
        return gender
        
    def update(self, camera, persons: List[Person]):
        self.persons: List[Person] = []
        for person in persons:
            # Check if the box is inside the roi
            box = person.boxes[-1]
            
            x1, y1, x2, y2 = box
            
            if self.point_for_count == "bottom":
                point = ((x1+x2)/2, y2)
            else:
                point = ((x1+x2)/2, (y1+y2)/2)
            
            if utilities.is_point_in_roi(point, self.roi):
                self.update_footfall(person)
                self.persons.append(Person)
            else:
                if person.track_id in self.tracker:
                    self.count(person.track_id, camera)
            
    def update_footfall(self, person: Person):
        if person.track_id not in self.tracker:
            self.add_tracker(person.track_id)
            
        self.tracker[person.track_id]["boxes"].append(person.boxes[-1])
        self.tracker[person.track_id]["centroids"].append(person.centroids[-1])
        
        if self.detect_gender:
            gender = self.get_gender(person.latest_person_img)
            person.genders.append(gender)
                
    def count(self, track_id, camera):
        if track_id not in self.tracker:
            return
        
        tracker = self.tracker[track_id]
        direction = self.get_direction(tracker["centroids"], tracker["boxes"])
        
        flag = False
        if direction == 0:
            self.entry_count += 1
            flag = True
        elif direction == 1:
            self.exit_count += 1
            flag = True
            
        if flag and track_id > 0:
             
            db_thread = Thread(target=self.add_to_db, args=(camera, camera.current_tracker[track_id], direction, ))
            db_thread.start()
            
        del self.tracker[track_id]
    
    def get_direction(self, centroids, boxes):
        if self.point_for_count == "bottom":
            y = [b[-1] for b in boxes]
            start = y[0]
            end = y[-1]
        else:
            y = [c[1] for c in centroids]
            start = y[0]
            end = y[-1]
            
        # Check start and end is in which half of the screen
        # If start is in first half and and end is in second half it's entry and vise versa is exit
        # If both are in same half set direction as -1
        
        direction = -1
        
        # Start in first half
        if start < self.counter_line:
            # End in second half
            if end > self.counter_line:
                direction = 0           # Entry
                
        # Start in second half
        else:
            # End in first half
            if end < self.counter_line:
                direction = 1           # Exit
                
        return direction
                
    def add_to_db(self, camera, person, direction):
        self.db.add_footfall_log(camera.id, self.id, person.track_id, None, person.get_gender(), None, person.init_time, direction, None, None)
        logger.info(f"{camera.id} - Footfall: {self.id, person.track_id, direction}")
        
    def draw(self, frame):
        x1, y1, x2, y2 = self.roi
        
        y_line = self.counter_line
        cv2.line(frame, (0, y_line), (2000, y_line), (209, 27, 103), thickness=2)
        
        text = f'{self.name}\nEntry: {self.entry_count}\nExit: {self.exit_count}'
        drawing_functions.draw_rectangle_with_text(frame, 
                                                    (x1, y1), 
                                                    (x2, y2), 
                                                    (209, 27, 103),
                                                    text=text)
        
        for track_id, values in self.tracker.items():
            text = f'{track_id}'
            
            x1, y1, x2, y2 = values['boxes'][-1]
            drawing_functions.draw_rectangle_with_text(frame, 
                                                    (x1, y1), 
                                                    (x2, y2), 
                                                    (0, 200, 0),
                                                    text=text)
            
            cv2.circle(frame, values['centroids'][-1], 4, (0, 200, 0),thickness=-1, lineType=cv2.FILLED)
            