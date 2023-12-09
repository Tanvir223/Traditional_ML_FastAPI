from pydantic import BaseModel
from typing import List

class Feature_Engineering_Params(BaseModel):
    guid: str
    selected_feature: List[str]
    target_column: str
    ml_algos : List[str]


class Finalize_Params(BaseModel):
    guid: str 
    selected_feature: List[str] 
    target_column: str
    ml_algos : List[str]

class Prediction_Params(BaseModel):
    selected_finalized_feature: List[str]
    trained_ml_algos : List[str]