from config.database import raw_data_collection, variable_imp_colections, predicted_train_data_colections,predicted_production_data_colections, evalution_colections
import uuid
import pandas as pd
import numpy as np

def save_data(data, collection_name):
    if collection_name=='raw_uploaded_collection':
        if 'guid' not in data.columns:
            data['guid'] = str(uuid.uuid4())
        data_dict = data.to_dict(orient='records')
        raw_data_collection.insert_many(data_dict)

    elif collection_name=='evalution_colections':
        if 'guid' not in data.columns:
            data['guid'] = str(uuid.uuid4())
        data_dict = data.to_dict(orient='records')
        evalution_colections.insert_many(data_dict)

    elif collection_name=='variable_imp_colections':
        if 'guid' not in data.columns:
            data['guid'] = str(uuid.uuid4())
        data_dict = data.to_dict(orient='records')
        variable_imp_colections.insert_many(data_dict)

    elif collection_name=='predicted_train_collections':
        if 'guid' not in data.columns:
            data['guid'] = str(uuid.uuid4())
        data_dict = data.to_dict(orient='records')
        predicted_train_data_colections.insert_many(data_dict)

    elif collection_name=='predicted_production_collections':
        if 'guid' not in data.columns:
            data['guid'] = str(uuid.uuid4())
        data_dict = data.to_dict(orient='records')
        predicted_production_data_colections.insert_many(data_dict)


def update_data(data, collection_name, guid):
    if collection_name=='raw_uploaded_collection':
        raw_data_collection.delete_many({'guid':guid})
        if 'guid' not in data.columns:
            data['guid'] = str(uuid.uuid4())
        data_dict = data.to_dict(orient='records')
        raw_data_collection.insert_many(data_dict)

    elif collection_name=='evalution_colections':
        evalution_colections.delete_many({'guid':guid})
        if 'guid' not in data.columns:
            data['guid'] = str(uuid.uuid4())
        data_dict = data.to_dict(orient='records')
        evalution_colections.insert_many(data_dict)

    elif collection_name=='variable_imp_colections':
        variable_imp_colections.delete_many({'guid':guid})
        if 'guid' not in data.columns:
            data['guid'] = str(uuid.uuid4())
        data_dict = data.to_dict(orient='records')
        variable_imp_colections.insert_many(data_dict)

    elif collection_name=='predicted_train_collections':
        predicted_train_data_colections.delete_many({'guid':guid})
        if 'guid' not in data.columns:
            data['guid'] = str(uuid.uuid4())
        data_dict = data.to_dict(orient='records')
        predicted_train_data_colections.insert_many(data_dict)

    elif collection_name=='predicted_production_collections':
        predicted_production_data_colections.delete_many({'guid':guid})
        if 'guid' not in data.columns:
            data['guid'] = str(uuid.uuid4())
        data_dict = data.to_dict(orient='records')
        predicted_production_data_colections.insert_many(data_dict)

def get_data(collection_name, guid):
    if collection_name=='raw_uploaded_collection':
        data = raw_data_collection.find({"guid":guid})
        if data:
            df = pd.DataFrame(data)
            df.drop(columns=['guid', '_id'], inplace=True)
            print(df.head(3))
            data_dict = df.to_dict(orient='records')
            return data_dict

    elif collection_name=='evalution_colections':
        data = evalution_colections.find({"guid":guid})
        if data:
            df = pd.DataFrame(data)
            df.drop(columns=['guid', '_id'], inplace=True)
            df.replace(np.nan, None)
            print(df.head(3))
            
            data_dict = df.to_dict(orient='records')
            print(df)
            return data_dict

    elif collection_name=='variable_imp_colections':
        data = variable_imp_colections.find({"guid":guid})
        if data:
            df = pd.DataFrame(data)
            df.drop(columns=['guid', '_id'], inplace=True)
            df.replace(np.nan, None)
            print(df.head(3))
            
            data_dict = df.to_dict(orient='records')
            print(df)
            return data_dict

    elif collection_name=='predicted_train_collections':
        data = predicted_train_data_colections.find({"guid":guid})
        if data:
            df = pd.DataFrame(data)
            df.drop(columns=['guid', '_id'], inplace=True)
            print(df.head(3))
            data_dict = df.to_dict(orient='records')
            return data_dict

    elif collection_name=='predicted_production_collections':
        data = predicted_production_data_colections.find({"guid":guid})
        if data:
            df = pd.DataFrame(data)
            df.drop(columns=['guid', '_id'], inplace=True)
            print(df.head(3))
            data_dict = df.to_dict(orient='records')
            return data_dict

