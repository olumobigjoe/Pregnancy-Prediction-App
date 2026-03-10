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
        background-color: #C7F2FF;
    }
    
    /* Global style for all buttons */
    div.stButton > button {
        background-color: #FF00B7 !important;
        color: #FFFFFF !important;
        border-radius: 5px;
    }

    /* Styling inputs */
    .stTextInput input, .stNumberInput input, .stSelectbox div {
        background-color: #F0F2F6 !important;
        color: #262730 !important;
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

# Predict button
if st.button("Predict Outcome"):
    # Prepare the input data for prediction
    input_data = np.array([[Age, BMI, Number_of_Embryos_transferred]])
    
    # Make prediction using the loaded model
    prediction = model.predict(input_data)
    
    # Display the result
    if prediction[0] == 1:
        st.success("The predicted outcome is: Yes (Pregnancy Successful)")
    else:
        st.error("The predicted outcome is: No (Pregnancy Unsuccessful)")   


# RESULT
# Add a selectbox to the sidebar:
#st.sidebar.success("**RECORD UPDATE**")


#st.sidebar.text_input("Patient's ID", key="name")
#st.session_state.name

#st.sidebar.text_input("Patient's Address/City", key="address")
#st.session_state.address

#st.sidebar.text_input("Email", key="email")
#st.session_state.email

#st.divider()



#st.sidebar.success("FEEDBACK")


#add_selectbox = st.sidebar.multiselect(
 #   "ART Experience/Review",
   # ('Very Satisfied', 'Satisfied', 'Not Satisfied', 'Neutral')
#)

st.balloons()  # Celebration balloons
st.balloons()  # Celebration balloons
