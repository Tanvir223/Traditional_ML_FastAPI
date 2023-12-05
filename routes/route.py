from fastapi import APIRouter, FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from config.database import raw_data_collection
import pandas as pd
from fastapi.responses import JSONResponse
import uuid
from helper.mongodb_helper_functions import *
from ml_codes.reg_feature_engineering import *
import json
router = APIRouter()

@router.post("/upload_data")
async def upload_csv(file: UploadFile, background_tasks: BackgroundTasks):
    df = pd.read_csv(file.file)
    df['guid'] = str(uuid.uuid4())
    background_tasks.add_task(save_data, df, 'raw_uploaded_collection')
    # save_data(df, 'raw_uploaded_collection')
    return JSONResponse(content={"msg": f"{file.filename} is uploaded, which length is {df.shape} and the guid is {df['guid'][0]}"})


@router.get("/get_data/{guid}/{collection_name}")
async def get_data_api(guid : str, collection_name: str):
    try: 
        data_dict = get_data(collection_name, guid)
    except:
        raise HTTPException(status_code=404, detail="Data not Found")
    return JSONResponse(data_dict)

@router.delete("/delete_data/{guid}")
async def delete_data(guid:str):
    data = raw_data_collection.delete_many({'guid':guid})

    if data.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Data not found")
    
    return {"message" : f"{data.deleted_count} number of records deleted"}


@router.post("/feature_engineering")
async def feature_engineering(guid: str, request, backgroundtask: BackgroundTasks):
    
    data = raw_data_collection.find({"guid":guid})
    if data:
        df = pd.DataFrame(data)
        print(df.head())
        # reg_feature_engineering(df, selected_feature, target_column, ml_algos)
    return {"message" : f"feature Engineering process accepted, finda the validationa and variable importance \
            data using the guid {guid} at collection : evalution_colections"}
