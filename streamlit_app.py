from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Boston House Price Prediction', page_icon='🏠', layout='wide')
ROOT=Path(__file__).resolve().parent
MODEL_PATH=ROOT/'best_model.pkl'
FEATURES=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists(): raise FileNotFoundError(MODEL_PATH)
    return joblib.load(MODEL_PATH)
st.title('Boston House Price Prediction')
st.caption('Machine Learning Internship Project — GAIL (India) Limited')
try: model=load_model()
except Exception as exc:
    st.error('Model could not be loaded.'); st.code(str(exc)); st.stop()
with st.form('prediction_form'):
    vals=[]
    for feature, default in zip(FEATURES,[0.1,0.0,7.0,0,0.5,6.0,60.0,4.0,1.0,300.0,18.0,390.0,12.0]):
        if feature=='CHAS': value=st.selectbox(feature,[0,1])
        else: value=st.number_input(feature,value=float(default),step=0.1)
        vals.append(value)
    submitted=st.form_submit_button('Predict House Price',type='primary',use_container_width=True)
if submitted:
    X=pd.DataFrame([vals],columns=FEATURES)
    try:
        prediction=float(model.predict(X)[0])
        st.success('Estimated median home value: ${:,.2f}'.format(prediction*1000))
    except Exception as exc: st.error('Prediction failed: '+str(exc))
st.caption('Historical Boston Housing benchmark; educational use only.')
