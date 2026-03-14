import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle

st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background-color: #1a3c6e,#2e6db4 !important;
    }
    
    /* Global style for all buttons */
    div.stButton > button {
        background-color: #FF00B7 !important;
        color: #FFFFFF !important;
        border-radius: 5px;
    }

    /* Styling inputs */
    .stTextInput input, .stNumberInput input, .stSelectbox div {
        background-color: #D0F2F6 !important;
        color: #0A0A0A !important;
    }
    </style>
    """, unsafe_allow_html=True)

theme = {
        "textColor": "#262730",
    "font": "sans serif"
    }

#Load the trained modl
model = joblib.load('pregnancy_outcome.pkl')

# Streamlit application
st.title("PREGNANCY PREDICTION APP")
st.divider()
#st.subheader("Input Features")
st.write("Please provide the following information to predict the pregnancy outcome.")
st.set_page_config(page_title="Pregnancy Outcome App", layout="centered")

# Input Features
Age = st.number_input("Age ", min_value=18, max_value=50, value=30)
BMI = st.number_input("(BMI)", min_value=10.0, max_value=50.0, value=25.0)
Number_of_Embryos_transferred = st.number_input("Number of embryo(s) transfered? ", min_value=1, max_value=5, value=2)
Type_Embryo = st.number_input("Type of embryo transferred" , min_value=0, max_value=1, value=1) #(1 for D5, 0 for D3)
Embryo_Quality = st.number_input("Embryo Quality", min_value=0, max_value=4, value=1) #(0 for Grade1, 1 for Grade2, 2 for Grade2.5, 3 for Grade3, 4 for Grade4)"
Sperm_Quality = st.number_input("Sperm Quality", min_value=0, max_value=4, value=1) #(A for 0, B for 1, C for 2, D for 3, E for 4)

# Predict button
if st.button("Predict Outcome"):
    # Prepare the input data for prediction
    input_data = np.array([[Age, BMI, Number_of_Embryos_transferred, Type_Embryo, Embryo_Quality, Sperm_Quality]])
    
    # Make prediction using the loaded model
    prediction = model.predict(input_data)
    
    # Display the result
    if prediction[0] == 1:
        st.success("The predicted outcome is: Yes (Pregnancy Successful)")
    else:
        st.error("The predicted outcome is: No (Pregnancy Unsuccessful)")   



#st.markdown("**Disclaimer**: This app provides pregnancy predictions for ART patients. Consult a gynecologist for clinical decisions.")
st.caption("👨‍⚕️ Developed by Olumodeji Ibukun • Powered by Streamlit & Machine Learning")

st.balloons()  # Celebration balloons