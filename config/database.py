from pymongo import MongoClient
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from env.mongo_credintials import *

url = "mongodb+srv://" + user_name + ":" + password + "@traditionalmlcluster.eotubcy.mongodb.net/?retryWrites=true&w=majority"

client = MongoClient(url)

db = client.raw_db

raw_data_collection = db["raw_uploaded_collection"]
evalution_colections = db["evalution_colections"]
variable_imp_colections = db["variable_imp_colections"]
predicted_train_data_colections = db["predicted_train_collections"]
predicted_production_data_colections = db["predicted_production_collections"]