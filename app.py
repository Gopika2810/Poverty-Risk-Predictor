import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# --- Load model, scaler, encoder
# -------------------------------
model = joblib.load("best_poverty_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# -------------------------------
# --- Deployment features
# -------------------------------
deploy_features = [
    ("Cooking Fuel (Clean Use)", "number"),
    ("Sanitation Access", "number"),
    ("Pucca Housing (3+ Rooms)", "number"),
    ("Kuccha Housing (1 Room)", "number"),
    ("Education Years (10+)", "slider"),
    ("No Literate Adult (25+)", "number"),
    ("Child Malnutrition (Underweight <5)", "slider"),
    ("Maternal Support (JSY Assistance)", "number"),
    ("SC/ST Household", "number"),
    ("Household Disability", "number"),
    ("Improved Drinking Water Access", "slider")
]

feature_ranges = {
    "Cooking Fuel (Clean Use)": (0, 500),
    "Sanitation Access": (0, 500),
    "Pucca Housing (3+ Rooms)": (0, 200),
    "Kuccha Housing (1 Room)": (0, 200),
    "Education Years (10+)": (0, 20),
    "No Literate Adult (25+)": (0, 100),
    "Child Malnutrition (Underweight <5)": (0, 100),
    "Maternal Support (JSY Assistance)": (0, 150),
    "SC/ST Household": (0, 300),
    "Household Disability": (0, 50),
    "Improved Drinking Water Access": (0, 400)
}

# -------------------------------
# --- Streamlit page config
# -------------------------------
st.set_page_config(page_title="Poverty Risk Predictor", layout="wide")
st.title("🌍 Poverty Risk Prediction System")
st.markdown("Provide household-level counts to estimate **poverty risk**.")

# -------------------------------
# --- Input fields
# -------------------------------
st.subheader("Input Socio-Economic Indicators")
user_data = {}

slider_features = [feat for feat, typ in deploy_features if typ == "slider"]
number_features = [feat for feat, typ in deploy_features if typ != "slider"]

# Slider group
st.markdown("**Slider-based Inputs**")
slider_cols = st.columns(len(slider_features))
for i, feat in enumerate(slider_features):
    min_val, max_val = feature_ranges[feat]
    default_val = (min_val + max_val) // 2
    step_val = max(1, (max_val - min_val) // 20)
    user_data[feat] = slider_cols[i % len(slider_features)].slider(feat, min_val, max_val, default_val, step=step_val)

# Number inputs in 3 columns
st.markdown("**Number Inputs**")
num_cols = st.columns(3)
for i, feat in enumerate(number_features):
    min_val, max_val = feature_ranges[feat]
    default_val = (min_val + max_val) // 2
    step_val = max(1, (max_val - min_val) // 20)
    user_data[feat] = num_cols[i % 3].number_input(feat, min_val, max_val, default_val, step=step_val)

# -------------------------------
# --- Prediction
# -------------------------------
if st.button("🔍 Predict Poverty Risk"):
    user_df = pd.DataFrame([user_data])
    try:
        input_for_model = scaler.transform(user_df)
    except:
        input_for_model = user_df

    pred = model.predict(input_for_model)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_for_model)[0]
    else:
        proba = np.zeros(len(le.classes_))

    predicted_class = le.inverse_transform([pred])[0]

    st.session_state["pred"] = predicted_class
    st.session_state["proba"] = proba

# -------------------------------
# --- Display results (table only)
# -------------------------------
if "pred" in st.session_state:
    st.subheader("Prediction Results")
    color_map = {"Low Poverty Risk": "green",
                 "Medium Poverty Risk": "orange",
                 "High Poverty Risk": "red"}
    st.markdown(f"### ✅ Predicted Poverty Class: "
                f"<span style='color:{color_map.get(st.session_state['pred'],'black')}'>{st.session_state['pred']}</span>",
                unsafe_allow_html=True)

    results = pd.DataFrame({
        "Poverty Class": le.classes_,
        "Probability (%)": [round(p * 100, 2) for p in st.session_state["proba"]]
    })

    st.table(results)
