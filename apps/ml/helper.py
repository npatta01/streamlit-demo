from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
import typing
import pandas as pd

from . import models
# import models

def _split(df:pd.DataFrame, seed = 0 ):
    y = df.type 
    x = df.drop(columns = ['type'])
    x_train , x_test , y_train , y_test =  train_test_split(x,y, test_size = 0.3, random_state = seed )

    return  x_train , x_test , y_train , y_test

def train_model(df:pd.DataFrame, model_params:models.ModelParameters,

    model_type:models.ModelTypes,
    class_names:typing.List) ->models.ModelArtifacts :
    x_train , x_test , y_train , y_test = _split(df)

    if model_type == models.ModelTypes.LR:
        model_params = typing.cast(models.LrParameters,model_params)
        model = LogisticRegression(C=model_params.C,max_iter=model_params.max_iter)

    elif model_type == models.ModelTypes.RF:
        model_params = typing.cast(models.RfParameters,model_params)
        model = RandomForestClassifier(n_estimators=model_params.n_estimators,
         max_depth=model_params.max_depth, bootstrap=model_params.bootstrap)

    else:
        model_params = typing.cast(models.SvmParameters,model_params)
        model = SVC(C=model_params.C, kernel = model_params.kernel, gamma=model_params.gamma)


    model.fit(x_train, y_train)

    accuracy = model.score(x_test,y_test)

    y_pred = model.predict(x_test)

    precision = precision_score(y_test, y_pred, labels=class_names).round(2)
    recall = recall_score(y_test,y_pred, labels=class_names).round(2)


    model_artifacts = models.ModelArtifacts(
        model=model
        ,accuracy = accuracy
        , precision = precision
        , recall=recall
        , data = (x_train , x_test , y_train , y_test )
        , class_names=class_names
    )
    

    return model_artifacts