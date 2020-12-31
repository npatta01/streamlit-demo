"""Home page shown when the user enters the application"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
import os
import typing
import matplotlib.pyplot as plt


from . import models, helper
# import models 
# import helper

from logging import getLogger
logger = getLogger(__file__)

#plt.rcParams["figure.figsize"] = [16,9]



@st.cache(persist=True)
def load_data():
    path = os.path.abspath(os.path.join(os.path.dirname( __file__ ),  "mushroom.parquet"))

    data = pd.read_parquet(path)
    label = LabelEncoder()
    for col in data.columns:
        data[col] = label.fit_transform(data[col])

    return data


def plot_metrics(metrics_list:typing.List[models.ModelMetrics],model_artifact):

    x_train , x_test , y_train , y_test  = model_artifact.data
    model = model_artifact.model

    for metric_choice in metrics_list:
        with st.beta_expander(metric_choice.value, expanded=True):

            #st.subheader(metric_choice.value)

            fig, ax = plt.subplots(figsize=(5, 5))

            if metric_choice == models.ModelMetrics.CM:
                display = plot_confusion_matrix(model,x_test,y_test,display_labels = model_artifact.class_names,ax=ax)
                
            elif metric_choice == models.ModelMetrics.ROC:
                display=plot_roc_curve(model,x_test,y_test , ax=ax)
            else:
                display = plot_precision_recall_curve(model,x_test,y_test ,ax=ax)

            # fig = display.figure_
            # fig.set_figheight(5)
            # fig.set_figwidth(5)

            # fig.set_dpi(100)
            st.pyplot(fig)


def display():
    

    section_padding_left, section_main, section_padding_right = st.beta_columns((1, 2, 1))

    section_sidebar = st.sidebar




    


    df = load_data()
    class_names = ['edible','poisonous']
    model_options =  list (models.ModelTypes.__members__.values() )

    
    
    def display_metrics(section):
        with section:

            st.write("Accuracy ", model_artifact.accuracy.round(2))

            st.write("Precision" , model_artifact.precision.round(2))
            st.write("Recall: ",  model_artifact.recall.round(2))

            plot_metrics(selected_metrics, model_artifact)

    def display_parameters(section):
        if model_type == models.ModelTypes.SVM:
            C = st.number_input("C (Regularization parameter)", 0.01, 10.0,step = 0.01,key='C')
            kernel = st.radio("kernel", ("rbf", "linear"), key='kernel')
            gamma = st.radio("Gamma Kernel Coefficient", ("scale","auto"),key="gamma")


            model_params = models.SvmParameters(C=C,kernel=kernel,gamma=gamma)
            

        elif model_type == models.ModelTypes.LR:
            C = st.number_input("C (Regularization parameter)", 0.01, 10.0,step = 0.01,key='C_LR')
            max_iter = st.slider("Maximum number od iterations", 100, 500, key ="max_iter")


            model_params = models.LrParameters(C=C,max_iter=max_iter)


        else:
            
            n_estimators = st.number_input("The number os trees in the forest",100,5000,step =10,key="N_E")
            max_depth = st.number_input("The maximum number depth of the tree",1,20,step =10,key="max_depth")

            bootstrap = st.radio("Bootstrap samples when building trees", ("True","False"), key ="bootstrap")

            model_params = models.RfParameters(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)

        return model_params

    
    with section_main:
        st.title("Binary Classification Web App")
        st.markdown("Are your mushrooms edible or poisonous ? 🍄")
        st.text('Click classify on the left to see model results.')

        section_main_metrics = st.beta_container()

        section_other = st.beta_container()


    with section_sidebar:
        st.subheader("Choose Classifier")
        model_type = st.selectbox ("Classifier", model_options, format_func=lambda x:x.value)
        
        st.subheader("Model Hyperparameters")
        
        section_sidebar_model_parameters = st.beta_container()

        model_params = display_parameters(section_sidebar_model_parameters)
        
            
        metrics_options = list (models.ModelMetrics.__members__.values() )
        selected_metrics = st.multiselect("What metrics to plot?", metrics_options,default=metrics_options , format_func=lambda x:x.value)

        if st.button("Classify", key = "classify"):
            
            #st.subheader("Random Forest (RF) Results")

            model_artifact = helper.train_model(df=df, model_params=model_params,
                model_type=model_type, 
                class_names=class_names) 
            
            display_metrics(section_main_metrics)

            
    

    with section_other:
        
        with st.beta_expander("Raw Data", expanded=True):
            st.subheader("Mushroom Data Set Classification")
            st.write(df)
 
        st.markdown(
            """
            ## Acknowledgments

            This app was inspired by the Coursera course [Build a Data Science Web App with Streamlit and Python](https://www.coursera.org/learn/data-science-streamlit-python/home/welcome)

            
            
            """,
            unsafe_allow_html=True,
        )

#display()