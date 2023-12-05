from pymongo import MongoClient
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://tanvircmed:#include<stdio.h>Haramzada1@traditionalmlcluster.eotubcy.mongodb.net/?retryWrites=true&w=majority"

client = MongoClient(uri)

db = client.raw_db

raw_data_collection = db["raw_uploaded_collection"]
evalution_colections = db["evalution_colections"]
predicted_train_data_colections = db["predicted_train_collections"]
predicted_production_data_colections = db["predicted_production_collections"]