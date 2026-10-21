import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import xgboost as xgb
import os

# App Configuration
st.set_page_config(page_title="QSAR Prediction App", page_icon="🧪", layout="wide")
st.title("🧪 QSAR Model Prediction App (File Upload Mode)")
st.write("Welcome! This application allows you to upload a SMILES file to predict Molecular Activity using QSAR analysis.")

# 1. File Uploader Component
uploaded_file = st.file_uploader("Choose a SMILES (.smi) or CSV file containing molecules", type=["smi", "csv"])

if uploaded_file is not None:
    # Handle different file formats
    if uploaded_file.name.endswith(".smi"):
        df = pd.read_csv(uploaded_file, sep="\t", names=["SMILES", "Name"], header=None)
    else:
        df = pd.read_csv(uploaded_file)
        
    st.write("### 📋 Uploaded Data Preview:", df.head())
    
    model_path = "mera_model.pkl"
    
    # 2. Prediction Trigger Button
    if st.button("🚀 Predict Activity for All Molecules"):
        with st.spinner("Processing molecules and generating molecular fingerprints..."):
            
            fps = []
            valid_rows = []
            
            # Generate Morgan Fingerprints for each molecule
            for index, row in df.iterrows():
                # Dynamically find a column named 'smiles' or fallback to the first column
                smiles_col = [col for col in df.columns if 'smile' in col.lower()]
                smiles = row[smiles_col[0]] if smiles_col else row.iloc[0]
                
                mol = Chem.MolFromSmiles(str(smiles))
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    fps.append(list(fp))
                    valid_rows.append(row)
            
            if len(fps) == 0:
                st.error("No valid SMILES structures were found in the uploaded file! Please check your column names.")
            else:
                # Create a DataFrame for valid processed rows
                result_df = pd.DataFrame(valid_rows)
                
                # Check for trained model file availability
                if os.path.exists(model_path):
                    try:
                        model = xgb.XGBRegressor()
                        model.load_model(model_path)
                        preds = model.predict(fps)
                        result_df["Predicted_Activity"] = preds
                        st.success(f"Successfully generated predictions for {len(fps)} molecules!")
                    except Exception as e:
                        st.warning("Error loading the model. Displaying baseline baseline values.")
                        result_df["Predicted_Activity"] = [5.5] * len(fps)
                else:
                    # Fallback notice if the model binary is missing from the repo
                    st.info("💡 Note: 'mera_model.pkl' not found in the repository. Displaying default demo predictions.")
                    result_df["Predicted_Activity"] = [5.2] * len(fps)
                
                # Display results on screen
                st.write("### 📊 Prediction Results:", result_df)
                
                # 3. CSV Data Download Action
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv,
                    file_name="qsar_predictions_output.csv",
                    mime="text/csv"
                )
