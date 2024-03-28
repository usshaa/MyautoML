import streamlit as st
import pandas as pd
import os

from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from pycaret.regression import setup, compare_models, pull, save_model
# from pycaret.classification import setup,compare_models, pull, save_model

with st.sidebar:
    st.image('DS.png')
    st.title("Usha Auto ML")
    choice = st.radio('Navigation',['Upload','Profiling','ML','Download'])
    st.info("This App allow you to build automated ML model")

if os.path.exists("source.csv"):
    df = pd.read_csv("source.csv",index_col=None)

if choice == "Upload":
    st.title("Upload your data for modeling!")
    file = st.file_uploader("Upload Dataset",type=['csv','xlsx'])
    if file is not None:
        df = pd.read_csv(file,index_col=None)
        df.to_csv("source.csv",index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report = ProfileReport(df)
    st_profile_report(profile_report)
    
if choice == "ML":
    st.title("Machine Learning")
    target = st.selectbox("Select your column",df.columns)
    df = df.convert_dtypes(convert_floating=True)
    if st.button("Train model"):
        setup(df,target=target,verbose=False)
        setup_df = pull()
        st.info("This is ML Experiment Settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML Model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model,'best_model')

if choice == "Download":
    with open("best_model.pkl",'rb') as f:
        st.download_button("Download the Model",f,"trained_model.pkl")
        
