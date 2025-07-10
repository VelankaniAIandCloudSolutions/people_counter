import datetime

class Time:
    
    def __init__(self) -> None:
        self._time: datetime.datetime = datetime.datetime.now()
        
    def update(self, milliseconds):
        self._time = self._time + datetime.timedelta(milliseconds=milliseconds)
        
    def get_time(self) -> datetime.datetime:
        return self._time