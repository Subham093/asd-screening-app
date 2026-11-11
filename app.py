import json, joblib
import numpy as np, pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="ASD Screening Aid", page_icon="🧩")

@st.cache_resource
def load():
    model = joblib.load("xgb_model.joblib")
    meta = json.load(open("preprocessing.json"))
    return model, meta, shap.TreeExplainer(model)

model, meta, explainer = load()
FEATURES = meta["features"]

st.title("ASD Screening Aid")
st.warning("This is a screening aid built for a college project. It is NOT a diagnosis. "
           "Please consult a qualified clinician.")

st.subheader("Screening answers")
st.caption("Enter the scored answer for each of the 10 screening questions (A1-A10). "
           "1 = the answer indicates the autism trait, 0 = it does not.")
cols = st.columns(2)
answers = {}
for i in range(1, 11):
    with cols[(i - 1) % 2]:
        answers[f"A{i}_Score"] = st.radio(f"Question {i} (A{i})", [0, 1],
                                          horizontal=True, key=f"q{i}")

st.subheader("About the person")
age = st.number_input("Age (years)", min_value=2, max_value=90, value=10)
gender = st.selectbox("Gender", ["Male", "Female"])
jaundice = st.selectbox("Born with jaundice?", ["No", "Yes"])
family = st.selectbox("Family member with autism?", ["No", "Yes"])

group = "child" if age <= 11 else ("adolescent" if age <= 17 else "adult")
row = dict(answers)
row.update({
    "age": age,
    "gender": 1 if gender == "Male" else 0,
    "jundice": 1 if jaundice == "Yes" else 0,
    "austim": 1 if family == "Yes" else 0,
    "group_child": int(group == "child"),
    "group_adolescent": int(group == "adolescent"),
    "group_adult": int(group == "adult"),
})
X = pd.DataFrame([row])[FEATURES].astype(float)

if st.button("Check"):
    p = float(model.predict_proba(X)[0, 1])
    st.metric("Model probability of ASD traits", f"{p:.0%}")
    if p >= 0.5:
        st.error("The screening answers show a high chance of ASD traits. "
                 "A professional assessment is recommended.")
    else:
        st.success("The screening answers show a low chance of ASD traits.")

    st.subheader("Why did the model say this?")
    sv = explainer(X)
    shap.plots.waterfall(sv[0], show=False)
    st.pyplot(plt.gcf())
    plt.close("all")
    st.caption("Red bars push the result toward ASD, blue bars push it away. "
               "Values are in model log-odds.")
