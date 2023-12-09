from fastapi import APIRouter, FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from config.database import raw_data_collection
import pandas as pd
from fastapi.responses import JSONResponse
import uuid
from helper.mongodb_helper_functions import *
from ml_codes.reg_feature_engineering import *
from ml_codes.reg_finalization import *
from ml_codes.reg_prediction import *
import json
from typing import List
from base_models.base_model import *

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

@router.delete("/delete_data/{guid}/{collection_name}")
async def delete_data(guid : str, collection_name: str):
    data = raw_data_collection.delete_many({'guid':guid})

    if data.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Data not found")
    
    return {"message" : f"{data.deleted_count} number of records deleted"}


@router.post("/feature_engineering")
async def feature_engineering(feature_engineering_params: Feature_Engineering_Params, backgroundtask: BackgroundTasks):
    
    data = raw_data_collection.find({"guid":feature_engineering_params['guid']})
    if data:
        df = pd.DataFrame(data)
        print(df.head())
        backgroundtask.add_task(reg_feature_engineering, df, feature_engineering_params['selected_feature'], feature_engineering_params['target_column'], feature_engineering_params['ml_algos'])
    return {"message" : f"feature Engineering process accepted, finda the validationa and variable importance \
            data using the guid {feature_engineering_params['guid']} at collection : evalution_colections"}


@router.post("/finalize_model")
async def finalizing_model(finaize_params : Finalize_Params, backgroundtask: BackgroundTasks):
    
    data_dict = raw_data_collection.find({"guid":finaize_params['guid']})
    validation_result_dict = evalution_colections.find({"guid":finaize_params['guid']})
    if data_dict:
        df = pd.DataFrame(data_dict)
        df.drop(columns=['_id', 'guid'], inplace=True)
        validation_result_df = pd.DataFrame(validation_result_dict)
        validation_result_df.drop(columns=['_id', 'guid'], inplace=True)
        print(df.head())
        backgroundtask.add_task(finalize_model, df, finaize_params['ml_algos'], finaize_params['selected_feature'], finaize_params['target_column'], validation_result_df)
    return {"message" : f"Finalizing the model , use now you can predict with the predict API"}

@router.post("/prediction")
async def prediction(file: UploadFile, prediction_params : Prediction_Params, backgroundtask: BackgroundTasks):
    
    df_dict = pd.read_csv(file.file)
    
    if df:
        df = pd.DataFrame(df_dict)
        backgroundtask.add_task(predict, df, prediction_params['selected_finalized_feature'], prediction_params['trained_ml_algos'])
    return {"message" : f"prediction is runing"}
