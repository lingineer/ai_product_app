import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

from utils import preprocessing

st.title("Churn Prediction")
# # # Preparation
artifact_dir = Path("../artifacts")

def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

artifacts = load_pickle(artifact_dir / "artifacts.pkl")
clf = load_pickle(artifact_dir / "model.pkl")


# # # Input
# # Simple layout
# with st.container():
#     gender = st.radio("Select gender", options=['Male', 'Female'])
#     payment_method = st.radio("Select payment method", options=['Credit Card', 'Bank Withdrawal', 'Mailed Check'])
#     age = st.text_input("Select age", 20)
#     download = st.slider("Select average monthly download in GB", min_value=0, max_value=100, value=10)
#     charge = st.slider("Select monthly charge in USD", min_value=0, max_value=200, value=50)
#     start_pred_btn = st.button("Let's predict")

# # Complex layout
with st.container():
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            gender = st.radio("Select gender", options=['Male', 'Female'])
        with col2:
            payment_method = st.radio("Select payment method", options=['Credit Card', 'Bank Withdrawal', 'Mailed Check'])
    with st.container():
        col1, col2, col3 = st.columns([2,4,4]) # [2/10, 4/10, 4/10]
        with col1:
            age = st.text_input("Select age", 20)
        with col2:            
            download = st.slider("Select average monthly download in GB", min_value=0, max_value=100, value=10)
        with col3:
            charge = st.slider("Select monthly charge in USD", min_value=0, max_value=200, value=50)
        start_pred_btn = st.button("Let's predict")

# # # Display
st.title("Prediction")
with st.container():
    if start_pred_btn:
        data = {
            "gender": gender,
            "payment_method": payment_method,
            "age": int(age),
            "download": download,
            "charge": charge,
        }
        df = pd.DataFrame([data])
        X, _ = preprocessing(df, artifacts)
        pred = clf.predict(X)
        status = {
            0: "Stayed",
            1: "Churned"
        }
        st.write(status[pred[0]])
    else:
        st.write("Please select customer profile and press [Let's predict] button")
