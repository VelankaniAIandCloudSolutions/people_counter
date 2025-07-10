from sqlalchemy.orm import sessionmaker, scoped_session
from database.database_init import *
import datetime
import logging
import numpy as np
from sqlalchemy import func
import sqlalchemy


logger = logging.getLogger(__name__)


class Database:

    def __init__(self):
        session_factory = sessionmaker(bind=engine)
        self.Session = scoped_session(session_factory)
                              
    def add_footfall_log(self, camera_id, zone_id, track_id, age, gender, image_path, timestamp, direction, person_type, person_distance):
        session = self.Session()               
        try:                                       
            log = FootfallZoneLogs(zone_id = zone_id,
                       timestamp=timestamp,
                       image_path=image_path,
                       age=age,
                       gender=gender,
                       camera_id=camera_id,
                       track_id=track_id,
                       direction=direction,
                       person_type=person_type,
                       person_distance=person_distance)
            session.add(log)                                                  
            session.commit()                                                       
        except Exception as e:                                                     
            logger.exception('Failed to add footfall log')                        
            session.rollback()                                                     
        session.close()
        
    def in_out_people_total_count(self, date):
        try:
            start_date = datetime.datetime.combine(date, datetime.time(0, 0, 0))
            end_date = datetime.datetime.combine(date, datetime.time(23, 59, 59))
                    
            result_In_Total = self.Session.query(
                                        FootfallZoneLogs.direction, 
                                        func.count(FootfallZoneLogs.log_id)).filter(
                                                FootfallZoneLogs.timestamp >= start_date, 
                                                FootfallZoneLogs.timestamp <= end_date).group_by(
                                                    FootfallZoneLogs.direction).all()
            
            # Convert results to dictionary for easier lookup
            result_dict = dict(result_In_Total)
            
            # Ensure we have both directions (0 for in, 1 for out) with default value 0
            formatted_result = [
                (0, result_dict.get(0, 0)),  # Entry count (direction 0)
                (1, result_dict.get(1, 0))   # Exit count (direction 1)
            ]
            
            return formatted_result
            
        except Exception as e:
            logger.error(f"Error in in_out_people_total_count: {str(e)}")
            return [(0, 0), (1, 0)]  # Return default structure in case of error
    
    def five_day_data(self, date):
        data = {}
        count = 6

        for i in range(count):
            new_date = date - datetime.timedelta(i)
            data[str(new_date.day)+'/'+str(new_date.month)+'/'+str(new_date.year)] = [str(new_date.strftime("%a")), self.in_out_people_total_count(new_date)]
            count -= 1

        for key, value in data.items():
            entry = value[1][0][1]
            exit = value[1][1][1]
            total = entry + exit
            # print(key, value[0], entry, exit, total)


    def percentage_up_down(self, date):
        yesterday = date - datetime.timedelta(1)
        day = date.strftime("%d %B %Y")
        (entry, today_In), (exit, today_Out) = self.in_out_people_total_count(date)        
        (entry, yesterday_In), (exit, yesterday_Out) = self.in_out_people_total_count(yesterday)
        total_today = today_In + today_Out
        total_yesterday = yesterday_In + yesterday_Out
        people_count_compair_yesterday = total_today - total_yesterday

        try:
            percentage = (people_count_compair_yesterday/total_yesterday)*100
        except:
            percentage = people_count_compair_yesterday

        if percentage > 0:
            text_color = str("text-success")
            img = "https://cdn-icons-png.flaticon.com/512/552/552922.png"
            more_or_less = "more </b> than"

        elif percentage < 0:
            text_color = str("text-danger")
            img = "https://cdn-icons-png.flaticon.com/512/2267/2267918.png"
            more_or_less = "less </b> than"

        else:
            text_color = ""
            img = "https://cdn-icons-png.flaticon.com/512/1716/1716838.png"
            more_or_less = "equal </b> to"

        return total_today, percentage, day, text_color, img, more_or_less

    def hourly_data(self, date, search_direction=[0]):
        start_date = datetime.datetime.combine(date, datetime.time(0, 0, 0))
        end_date = datetime.datetime.combine(date, datetime.time(23, 59, 59))
        result = self.Session.query(
                                FootfallZoneLogs.timestamp, sqlalchemy.func.count(FootfallZoneLogs.log_id))\
                                .filter(FootfallZoneLogs.direction.in_(search_direction), 
                                        FootfallZoneLogs.timestamp >= start_date, 
                                        FootfallZoneLogs.timestamp <= end_date)\
                                .group_by(sqlalchemy.func.strftime("%Y-%m-%d %H", FootfallZoneLogs.timestamp)).all()

        hourly_data = {}

        # get data for where vehicle count like 10:5,12:5 but 11 is not there
        for row in result:
            hourly_data[row[0].hour] = row[1]
        
        # add hours where vehicle count is 0
        for i in set(range(24)) - set(hourly_data.keys()):
            hourly_data[i] = 0

        # sort time from 0 to 24
        data_24_hours = {i: hourly_data[i] for i in sorted(hourly_data.keys())}

        # get only count of hours of the day
        hourly_data = [(hourly_data[i]) for i in data_24_hours]

        m = max(hourly_data)

        peak_hour_index = hourly_data.index(m)
        peak_hour_time = f"{peak_hour_index - 1}:00 - {peak_hour_index}:00"

        if peak_hour_time == '-1:00 - 0:00':
            peak_hour_time = '---'

        return hourly_data, peak_hour_time, m
