from dataclasses import dataclass
import typing
import sklearn
import enum

@dataclass
class ModelParameters:
    pass 

@dataclass
class SvmParameters(ModelParameters):
    C:float
    kernel:str
    gamma:str

@dataclass
class LrParameters(ModelParameters):
    C:float
    max_iter:int    

@dataclass
class RfParameters(ModelParameters):
    max_depth:int
    n_estimators:int      
    bootstrap:bool  



@dataclass
class ModelArtifacts:
    model:sklearn.base.BaseEstimator
    accuracy:float
    precision:float 
    recall:float
    class_names:typing.List[str]
    data:typing.Tuple



class ModelTypes(enum.Enum):
    LR = "Logistic Regression"
    RF =  "Random Forest"
    SVM ="Support Vector Machine"


class ModelMetrics(enum.Enum):
    CM  = "Confusion Matrix"
    ROC =  "ROC Curve"
    PRC = "Precision-Recall Curve"




# class ModelTypes:
#     LR = "Logistic Regression"
#     RF = "Random Forest"
#     SVM = "Support Vector Machine"
   
   
#     @classmethod 
#     def available_models(cls):
        
#         return [cls.LR, cls.RF, cls.SVM]