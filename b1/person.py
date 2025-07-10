import numpy as np
import datetime

from typing import Dict

class Person():
    def __init__(self, track_id):
        self.track_id = track_id
        self.init_time = datetime.datetime.now()
        self.boxes = []
        self.centroids = []
        self.genders = []
        self.person_types = []
        
        self.latest_person_img = None
        self.color = list(np.random.random(size=3) * 256)
        
        self.dwell_events: Dict[int, dict] = {}
        
    def get_gender(self):
        if len(self.genders) > 0:
            return int(max(set(self.genders), key=self.genders.count))
        
        # if 1 in self.genders:
        #     print("Fem")
        
        return None
    
    def get_person_type(self):
        if len(self.person_types) > 0:
            return max(set(self.person_types), key=self.person_types.count)
        return ""
    
    
    # ===================== DWELL FUNCTIONS =============================================
    
    def add_dwell_event(self, zone_id):
        self.dwell_events[zone_id] = {
            "start_times": [None],
            "end_times": [],
            "duration": None
        }
    
    def update_dwell_event(self, zone_id, in_roi, time):
        if in_roi:
            if zone_id not in self.dwell_events:
                self.add_dwell_event(zone_id)
                
            # If last start time is None update the start time with current time
            if self.dwell_events[zone_id]["start_times"][-1] is None:
                self.dwell_events[zone_id]["start_times"][-1] = time
                self.dwell_events[zone_id]["end_times"].append(time)
            # Else update end time with current time
            else:
                self.dwell_events[zone_id]["end_times"][-1] = time
        else:
            if zone_id in self.dwell_events and self.dwell_events[zone_id]["start_times"][-1] is not None:
                self.dwell_events[zone_id]["start_times"].append(None)
                
    def sort_dwell(self):
        for zone_id, values in self.dwell_events.items():
            self.postprocess_dwell_times(zone_id)
            self.dwell_events[zone_id]["duration"] = self.get_dwell_duration(zone_id)
                
    def get_dwell_duration(self, zone_id) -> datetime.timedelta:
        duration = datetime.timedelta(seconds=0)      
        if zone_id in self.dwell_events:
            start_times = self.dwell_events[zone_id]["start_times"]
            end_times = self.dwell_events[zone_id]["end_times"]

            for start_time, end_time in zip(start_times, end_times):
                _duration = end_time - start_time
                duration += _duration
            
        return duration
    
    def postprocess_dwell_times(self, zone_id):
        # If the end and start time is of less than 30s difference
        # remove the next start time and set the exit time and next exit time
        
        start_times = self.dwell_events[zone_id]["start_times"]
        end_times = self.dwell_events[zone_id]["end_times"]
        
        new_start_times = []
        new_end_times = []
        for i in range(len(end_times)):
            if len(new_start_times) == 0:
                new_start_times.append(start_times[i])
                end_time = end_times[i]
            
            if i+1 < len(start_times):
                next_start_time = start_times[i+1]
                
                if next_start_time is None:
                    new_end_times.append(end_time)
                else:
                    if (next_start_time - end_time).total_seconds() < 30:
                        end_time = end_times[i+1]
                    else:
                        new_end_times.append(end_time)
                        new_start_times.append(next_start_time)
                        end_time = end_times[i+1]
            else:
                new_end_times.append(end_time)
                
        self.dwell_events[zone_id]["start_times"] = new_start_times
        self.dwell_events[zone_id]["end_times"] = new_end_times
    
    # ===================================================================================
                
    def __eq__(self, other: 'Person'):
        if isinstance(other, Person):
            return self.track_id == other.track_id
        return False
            