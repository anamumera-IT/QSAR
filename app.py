import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import xgboost as xgb
import os
import base64
from io import BytesIO

# Function to convert RDKit molecule to an HTML image string
def mol_to_html_img(mol):
    if mol is None:
        return ""
    # Generate 2D coordinates for the structure drawing
    AllChem.Compute2DCoords(mol)
    img = Draw.MolToImage(mol, size=(150, 150))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<img src="data:image/png;base64,{img_str}" width="150" />'

# App Configuration
st.set_page_config(page_title="QSAR Prediction App", page_icon="🧪", layout="wide")
st.title("🧪 QSAR Model Prediction App (File Upload Mode)")
st.write("Welcome! This application allows you to upload a SMILES file to predict Molecular Activity and view 2D structures using QSAR analysis.")

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
    if st.button("🚀 Predict Activity & Draw Structures"):
        with st.spinner("Processing molecules, generating fingerprints and 2D images..."):
            
            fps = []
            valid_rows = []
            img_html_list = []
            
            # Generate Fingerprints and 2D Structures for each molecule
            for index, row in df.iterrows():
                # Dynamically find a column named 'smiles' or fallback to the first column
                smiles_col = [col for col in df.columns if 'smile' in col.lower()]
                smiles = row[smiles_col[0]] if smiles_col else row.iloc[0]
                
                mol = Chem.MolFromSmiles(str(smiles))
                if mol:
                    # 1. Fingerprint generation for XGBoost
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    fps.append(list(fp))
                    
                    # 2. 2D image string generation
                    img_html_list.append(mol_to_html_img(mol))
                    valid_rows.append(row)
            
            if len(fps) == 0:
                st.error("No valid SMILES structures were found in the uploaded file! Please check your column names.")
            else:
                # Create a DataFrame for valid processed rows
                result_df = pd.DataFrame(valid_rows)
                
                # Check for trained model file availability and make predictions
                if os.path.exists(model_path):
                    try:
                        model = xgb.XGBRegressor()
                        model.load_model(model_path)
                        preds = model.predict(fps)
                        result_df["Predicted_Activity"] = preds
                        st.success(f"Successfully generated predictions and structures for {len(fps)} molecules!")
                    except Exception as e:
                        st.warning("Error loading the model. Displaying baseline values.")
                        result_df["Predicted_Activity"] = [5.5] * len(fps)
                else:
                    st.info("💡 Note: 'mera_model.pkl' not found in the repository. Displaying default demo predictions.")
                    result_df["Predicted_Activity"] = [5.2] * len(fps)
                
                # Insert the HTML Image column right at the beginning of the result view
                result_df.insert(0, "2D Structure", img_html_list)
                
                # Display results table with rendered HTML images
                st.write("### 📊 Prediction & Structure Results:")
                st.write(result_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                
                # 3. CSV Data Download Action (Excluding the HTML code columns for clean Excel storage)
                download_df = result_df.drop(columns=["2D Structure"])
                csv = download_df.to_csv(index=False).encode('utf-8')
                st.write("") # Blank space
                st.download_button(
                    label="📥 Download Predictions CSV (Data Only)",
                    data=csv,
                    file_name="qsar_predictions_output.csv",
                    mime="text/csv"
                )
