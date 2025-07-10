from database.database_config import Database
from models.model import Model

class Zone():
    def __init__(self, db: Database, rb: Model, id, name, roi):
        self.db = db
        self.rb = rb
        self.id = id
        self.name = name
        self.roi = roi