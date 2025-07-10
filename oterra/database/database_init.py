from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, NVARCHAR, ForeignKey, Float, DateTime
from sqlalchemy import create_engine


Base = declarative_base()

class ZoneCategory(Base):
    __tablename__ = "zone_category"
    __table_args__ = {'extend_existing' : True}
    
    zone_category_id = Column(Integer, primary_key = True)
    zone_name = Column(NVARCHAR(50))
    

class Zone(Base):
    __tablename__ = "zone"
    __table_args__ = {'extend_existing': True}
    
    zone_id = Column(Integer, primary_key = True)
    zone_name = Column(NVARCHAR(50))
    
    zone_category_id = Column(Integer, ForeignKey('zone_category.zone_category_id'))
    
    roi = Column(NVARCHAR(50))
    camera_id = Column(Integer)
    
    
class DwelZoneLogs(Base):
    __tablename__ = 'dwell_zone_logs'
    __table_args__ = {'extend_existing': True}
    
    log_id = Column(Integer, primary_key=True)
    
    camera_id = Column(Integer)
    zone_id = Column(Integer)
    track_id = Column(Integer)
    
    start_time = Column(NVARCHAR(None))
    end_time = Column(NVARCHAR(None))
    duration = Column(Float)    #in seconds
    
    person_type = Column(NVARCHAR(30))
    
    
class DewllZoneUnmannedLogs(Base):
    __tablename__ = 'dwell_zone_unmanned_logs'
    __table_args__ = {'extend_existing': True}
    
    log_id = Column(Integer, primary_key=True)
    
    camera_id = Column(Integer)
    zone_id = Column(Integer)
    
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    
    person_type = Column(NVARCHAR(30))
    alert_type = Column(Integer) # 0-unmanned, 1 - customer unattended
    
    
class MannedZoneLogs(Base):
    __tablename__ = 'manned_zone_logs'
    __table_args__ = {'extend_existing': True}
    
    log_id = Column(Integer, primary_key = True)
    
    camera_id = Column(Integer)
    zone_id = Column(Integer)
    
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    
    
class FootfallZoneLogs(Base):
    __tablename__ = 'footfall_zone_logs'
    __table_args__ = {'extend_existing': True}
    log_id = Column(Integer, primary_key = True)
    
    camera_id = Column(Integer)
    zone_id = Column(Integer)
    track_id = Column(Integer)
    
    timestamp = Column(DateTime)
    image_path = Column(String)
    
    age = Column(Integer)
    gender = Column(Integer)                                # 0=Female, 1=Male
    direction = Column(Integer)                             # 0=Entry, 1=Exit, -1=Unknown
    
    person_type = Column(String)                            # employee or customer
    person_distance = Column(Float)
    

class PersonType(Base):
    __tablename__ = 'person_type'
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key = True)
    
    person_type_name = Column(String)
    embedding = Column(String)
    
    
engine = create_engine('sqlite:///data/database.db', echo=False)

Base.metadata.create_all(engine)