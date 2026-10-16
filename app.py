import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# App Title & Description
st.set_page_config(page_title="QSAR Predictor", page_icon="🧪", layout="centered")
st.title("🧪 QSAR Model Prediction App")
st.write("Welcome! This app uses a Machine Learning model to predict Molecular Activity based on QSAR analysis.")

# Sidebar info
st.sidebar.header("About")
st.sidebar.write("This app deploys the QSAR model from GitHub onto Streamlit.")

# Section 1: Load Model
@st.cache_resource
def load_qsar_model():
    # Agar aapki repo me trained model file (e.g., qsar_model.pkl) mojood hai
    if os.path.exists("qsar_model.pkl"):
        with open("qsar_model.pkl", "rb") as f:
            return pickle.load(f)
    return None

model = load_qsar_model()

# Section 2: User Input
st.subheader("📝 Input Molecular Features")
st.write("Enter the chemical/molecular features below to get a prediction:")

# Example features (Aap apne dataset ke mutabiq in columns ko change kar sakte hain)
feature_1 = st.number_input("Molecular Weight (MW)", min_value=0.0, max_value=1000.0, value=250.0, step=0.1)
feature_2 = st.number_input("LogP (Octanol-Water Partition Coefficient)", min_value=-10.0, max_value=10.0, value=2.5, step=0.1)
feature_3 = st.number_input("Number of Hydrogen Bond Donors", min_value=0, max_value=20, value=2)
feature_4 = st.number_input("Number of Hydrogen Bond Acceptors", min_value=0, max_value=20, value=5)

# Predict Button
if st.button("🔮 Predict Activity"):
    # Input data frame matching model columns
    input_data = pd.DataFrame([[feature_1, feature_2, feature_3, feature_4]], 
                               columns=['MW', 'LogP', 'H-Donors', 'H-Acceptors'])
    
    st.write("### Input Summary:")
    st.dataframe(input_data)
    
    if model is not None:
        try:
            prediction = model.predict(input_data)
            st.success(f"🎉 **Predicted Value / Class:** {prediction[0]}")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        # Mock prediction if no saved .pkl file exists yet
        st.warning("⚠️ No trained `qsar_model.pkl` found. Displaying a dummy prediction for demonstration:")
        dummy_prediction = np.random.uniform(0.1, 9.9)
        st.info(f"✨ **Simulated Prediction:** {dummy_prediction:.3f}")
