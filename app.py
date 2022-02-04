

import streamlit as st
import pandas as pd 
import numpy as np 
import time
import pickle
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

pickle_in = open("Randomforest.pkl","rb")
dec_cl = pickle.load(pickle_in)


st.header("predict wine quality")

fixed_acidity  = st.number_input("fixed_acidity",0.00,10.00,step = 0.50)

volatile_acidity  = st.number_input("volatile_acidity",0.00,1.50,step = 0.1)

citric_acid  = st.number_input("citric_acid",0.00,0.50,step = 0.05)

residual_sugar = st.number_input("residual_sugar",0.00,4.00,step = 0.50)

chlorides = st.number_input("chlorides",0.00,0.20,step = 0.01)

free_sulfur_dioxide = st.number_input("free_sulfur_dioxide",0.00,20.00,step = 1.00)

total_sulfur_dioxid = st.number_input("total_sulfur_dioxid",0.00,70.00,step = 5.00)

density = st.number_input("density",0.00,1.00,step = 0.05)

pH = st.number_input("pH",0.00,4.00,step = 0.4)

sulphates = st.number_input("sulphates",0.00,0.1,step = 0.01)

alcohol = st.number_input("alcohol",0.00,15.00,step = 0.5)


val = np.array([fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxid,density,pH,sulphates,alcohol]).reshape(1,-1)
pred = dec_cl.predict(val)


if st.button("Predict"):
        
        progress = st.progress(0)    # this is for progress bar
        for i in range(100):
            time.sleep(0.001)
            progress.progress(i+1)
        
        
        st.success('The output is {}'.format(pred))
        
        if pred ==0:
            st.write('***Bad Quality***')
        else:
            st.write('***Good Quality***')

