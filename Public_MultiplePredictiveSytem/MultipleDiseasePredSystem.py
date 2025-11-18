# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 15:32:43 2025

@author: HP
"""

import pickle
import streamlit as st 
from streamlit_option_menu import option_menu
import numpy as np

#loading the model
diabetes_model = pickle.load(open('diabetic_model.sav','rb'))
heart_model = pickle.load(open('heart_disease_model.sav','rb'))
parkinons_model = pickle.load(open('parkinsons_model.sav','rb'))

#sidebar for navigate

with st.sidebar:
    
    selected = option_menu('Disease Predictive System',
                           
                           ['Diabetes Prediction','Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           
                           icons=['activity','heart','person'],
                           
                           default_index=0)
    

    
if(selected == 'Diabetes Prediction'):
    
    #title
    st.title("Diabetes prediction using ML")
    
    #getting input from user
    col1,col2,col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input("Number of Pregnancies")
        
    with col2:
        Glucose = st.text_input("Glucose level")
        
    with col3:
        BloodPressure = st.text_input("BP value")
        
    with col1:    
        SkinThickness = st.text_input("SkinThickness value")
    with col2:
        Insulin = st.text_input("Insulin level")
    with col3:
        BMI = st.text_input("BMI Value")
    with col1:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value")
    with col2:
        Age = st.text_input("Age of person")
    
    #code for prediction
    diab_diagnosis = ''
    
    if st.button("Diabetes test results"):
       diagnosis = diabetes_model([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,
                                        BMI,DiabetesPedigreeFunction,Age])
       if (diagnosis[0] == 0):
           diab_diagnosis =  "Person is Non-diabetic"    
       else:
           diab_diagnosis = "Person is diabetic"
       
    st.success(diab_diagnosis)
    
if(selected == 'Heart Disease Prediction'):
    
    #title
    st.title("Heart Disease Prediction using ML")
    
    #getting input from user
    col1,col2,col3 = st.columns(3)
    
    with col1:
        age = st.text_input("Age")
        
    with col2:
        sex = st.text_input("Sex")
        
    with col3:
        cp = st.text_input("Chest Pain")
    with col3:
        trestbps = st.text_input("Resting Blood Pressure")
    with col1:
        chol = st.text_input("Cholestrol")
    with col2:
        fbs = st.text_input("Fasting Blood Pressure")
    with col3:
        restecg = st.text_input("Resting ElectrocardioGraphicResult")
    with col1:
        thalach = st.text_input("Maximum Heart Rate")
    with col2:
        exang = st.text_input("Exercise indusced angina")
    with col3:
        oldpeak = st.text_input("old peak")
    with col1:
        slope = st.text_input("slope of peak exercise")
    with col2:
        ca = st.text_input("colored by fluropsy")
    with col3:
        thal = st.text_input("thal")
        
    #code for prediction
    heart_diagnosis = ''
        
    if st.button("Heart disease test results"):
          heart_pred = heart_model([age,sex,cp,trestbps,chol,
                                            fbs,thalach,exang,oldpeak,slope,ca,thal])
          if (heart_pred[0] == 0):
              heart_diagnosis =  "The Person does not have a Heart Disease"    
          else:
              heart_diagnosis = "Person have a Heart Disease"
           
    st.success(heart_diagnosis)
    
if(selected == 'Parkinsons Prediction'):

    #title
    st.title("Parkinsons Prediction using ML")
    
    #getting input from user
    col1,col2,col3,col4,col5 = st.columns(5)
    
    with col1:
        fo = st.text_input("MDVP:Fo(Hz)")
        
    with col2:
        fhi = st.text_input("MDVP:Fhi(Hz)")
        
    with col3:
        flo = st.text_input("MDVP:Flo(Hz)")
        
    with col4:
     jitter_percent = st.text_input("MDVP:Jitter(%)")
        
    with col5:
        jitter_Abs = st.text_input("MDVP:Jitter(Abs)")
        
    with col1:
        rap = st.text_input("MDVP:RAP")
    with col2:
        ppq = st.text_input("MDVP:PPQ")
    with col3:
        Jitter_DDP = st.text_input("Jitter:DDP")
    with col4:
        Shimmer = st.text_input("MDVP:Shimmer")
    with col5:
        Shimmer_dB = st.text_input("MDVP:Shimmer(dB)")
    with col1:
        Shimmer_APQ3 = st.text_input("Shimmer:APQ3")
    with col2:
        Shimmer_APQ5 = st.text_input("Shimmer:APQ5")
    with col3:
        APQ = st.text_input("MDVP:APQ")
    with col4:
        Shimmer_DDA = st.text_input("Shimmer:DDA")
    with col5:
        NHR = st.text_input("NHR")
    with col1:
        HNR = st.text_input("HNR")
    with col2:
        RPDE = st.text_input("RPDE")
    with col3:
        DFA = st.text_input("DFA")
    with col4:
        spread1 = st.text_input("spread1")
    with col5:
        spread2 = st.text_input("spread2")
    with col1:
        D2 = st.text_input("D2")
    with col2:
        PPE = st.text_input("PPE")
        
    #code for prediction
    parkinsons_diagnosis = ''


    if st.button("Parkinsons disease test results"):
             park_pred = parkinons_model([fo,fhi,flo,jitter_percent,jitter_Abs,
                                     rap,ppq,Jitter_DDP,Shimmer,Shimmer_dB,Shimmer_APQ3,Shimmer_APQ5,APQ,
                                     Shimmer_DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE])
             if (heart_pred[0] == 0):
                 parkinsons_diagnosis =  "The Person does not have a Parkinson Disease"    
             else:
                 parkinsons_diagnosis = "Person have a Parkinson Disease"
               
    st.success(parkinsons_diagnosis)
        
    
    
    
    
    
    