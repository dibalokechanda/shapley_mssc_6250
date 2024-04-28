import streamlit as st
import pandas as pd
import xgboost
from streamlit_shap import st_shap
import shap

st.markdown("# Explainability of Machine Learning Models with Co-operative Game Theory")
st.divider()

st.write("This app is part of the project completed for MSSC 6250 : Statistical Machine Learning Spring 2024.")
st.write("The contributor of this project:")

contributors = ['Brigida Zhunio Cardenas', 'David Aguilera', 'Dibaloke Chanda']
for contributor in contributors:
    st.markdown("- " + contributor)
    
st.markdown('## Tabular Data')
st.divider()

st.markdown("### Upload Dataset")

data_file=st.file_uploader('Upload the Data Matrix CSV', type=['.csv'], accept_multiple_files=False, label_visibility="visible")

if data_file is not None:

    df1=pd.read_csv(data_file)
    
    st.markdown("#### Data Matrix ")
    st.write(df1)
    st.write("Shape of Data Matrix:",df1.to_numpy().shape)
    label_file=st.file_uploader('Upload the Label CSV', type=['.csv'], accept_multiple_files=False, label_visibility="visible")
    if label_file is not None:
        df2=pd.read_csv(label_file)
           
        st.markdown("#### Labels")
        st.write(df2)
        
        st.write("Shape of Labels:",df2.to_numpy().shape)


st.markdown("### Specify ML Model")

option = st.selectbox(
   "Specify the Machine Learning Model",
   ("XGBoost", "Random Forest","Decision Tree")
)


st.write('You selected:', option)

if data_file is not None and label_file is not None:
    X = df1.to_numpy()
    y = df2.to_numpy().squeeze()

    if option=="Random Forest":
        pass
    elif option=="Decision Tree":
        pass
    elif option=="XGBoost":
        model = xgboost.XGBRegressor().fit(X, y)

        # compute SHAP values
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        st_shap(shap.plots.waterfall(shap_values[0]), height=300)
        st_shap(shap.plots.beeswarm(shap_values), height=300)
    


