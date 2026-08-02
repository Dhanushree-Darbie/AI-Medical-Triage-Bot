import streamlit as st
import pickle
import pandas as pd
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

image_path = "image.jpg"   # make sure image is in same folder

img_base64 = get_base64_image(image_path)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* Optional: add transparent overlay */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.85);
        z-index: -1;
    }}
    </style>
    """,
    unsafe_allow_html=True
)




st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="AI Medical Triage Bot",
    page_icon="🩺",
    layout="centered"
)

# Load files
model = pickle.load(open("model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))
symptom_list = pickle.load(open("symptom_list.pkl", "rb"))

# -------------------------------
# FUNCTIONS
# -------------------------------

def extract_symptoms(user_input):
    user_input = user_input.lower()
    detected = []

    for symptom in symptom_list:
        if "_Yes" in symptom:
            base = symptom.replace("_Yes", "").replace("_", " ")
            if base in user_input:
                detected.append(symptom)

    return detected


def create_input_vector(symptoms):
    input_data = [0] * len(symptom_list)

    for i, col in enumerate(symptom_list):
        if col in symptoms:
            input_data[i] = 1

    return pd.DataFrame([input_data], columns=symptom_list)


def predict_disease(user_input):
    symptoms = extract_symptoms(user_input)

    if len(symptoms) == 0:
        return None, [], []

    input_vector = create_input_vector(symptoms)
    probs = model.predict_proba(input_vector)[0]

    top3_idx = probs.argsort()[-3:][::-1]

    results = []
    for idx in top3_idx:
        disease = le.inverse_transform([idx])[0]
        prob = round(probs[idx] * 100, 2)
        results.append((disease, prob))

    return results, symptoms


def triage(symptoms):
    if "difficulty breathing_Yes" in symptoms:
        return "🔴 Emergency – Seek immediate medical attention"

    score = 0
    if "fever_Yes" in symptoms:
        score += 2
    if "cough_Yes" in symptoms:
        score += 1
    if "fatigue_Yes" in symptoms:
        score += 1

    if score >= 4:
        return "🔴 Emergency"
    elif score >= 2:
        return "🟡 Moderate – Consult a doctor"
    else:
        return "🟢 Mild – Home care"


# -------------------------------
# UI
# -------------------------------

st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>🩺 AI Medical Triage Assistant</h1>
    <p style='text-align: center; font-size:18px;'>
    Enter your symptoms and get instant AI-based health insights
    </p>
    """,
    unsafe_allow_html=True
)

st.warning("⚠ This is not a medical diagnosis. Always consult a doctor.")
user_input = st.text_input(
    "📝 Describe your symptoms:",
    placeholder="e.g. I have fever, cough and fatigue"
)
if st.button("🔍 Analyze Symptoms"):    
    output = predict_disease(user_input)

    results = output[0]
    symptoms = output[1]

    if len(symptoms) == 0:
        st.error("⚠ No symptoms detected. Please try again.")
    else:
        st.markdown("### 🧾 Detected Symptoms")
        st.success(", ".join(symptoms))

        st.markdown("### 🧠 Top Predictions")
        for i, (disease, prob) in enumerate(results, 1):
            st.info(f"{i}. {disease} — {prob}%")

        st.markdown("### 🚦 Triage Result")
        triage_result = triage(symptoms)

        if "🔴" in triage_result:
            st.error(triage_result)
        elif "🟡" in triage_result:
            st.warning(triage_result)
        else:
            st.success(triage_result)