import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import os
import requests
import time
from werkzeug.security import generate_password_hash, check_password_hash
import base64
from io import BytesIO
from PIL import Image

# --- 1. CONFIGURATION ---
APP_TITLE = "🫶🏼 CardioPredict AI — Heart Disease Risk Analyzer"
DB_FILE = "heart_app.db"
MODEL_FILE = "heart_disease_model.pkl"
SCALER_FILE = "heart_disease_scaler.pkl"
FEATURES_FILE = "model_features.pkl"
LOTTIE_URL = "https://lottie.host/17498c40-0235-4384-9543-f6614138a0f8/R08oO5iE5x.json"

# --- GEMINI API CONFIG ---
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def get_gemini_api_key():
    """Load API key from Streamlit secrets or environment variable."""
    try:
        key = st.secrets["gemini"]["api_key"]
    except Exception:
        key = os.environ.get("GEMINI_API_KEY")
    if not key:
        st.warning("⚠️ Gemini API key not found! Set it in `.streamlit/secrets.toml` or environment variable `GEMINI_API_KEY`.")
    return key

GEMINI_API_KEY = get_gemini_api_key()

# --- 2. DATABASE FUNCTIONS ---
def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            age INTEGER,
            sex TEXT,
            chestpaintype TEXT,
            restingbp INTEGER,
            cholesterol INTEGER,
            fastingbs INTEGER,
            restingecg TEXT,
            maxhr INTEGER,
            exerciseangina TEXT,
            oldpeak REAL,
            st_slope TEXT,
            probability REAL,
            prediction INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()

@st.cache_resource
def setup_db():
    init_db()
    return True

DB_READY = setup_db()

def add_user(username, password):
    conn = get_db_connection()
    try:
        password_hash = generate_password_hash(password, method='pbkdf2:sha256:200000')
        conn.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
        conn.commit()
        return True, "Registration successful!"
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        conn.close()

def verify_user(username, password):
    conn = get_db_connection()
    user = conn.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    if user and check_password_hash(user['password_hash'], password):
        return user['id'], user['username']
    return None, None

def add_prediction(user_id, input_data, probability, prediction):
    conn = get_db_connection()
    input_data['probability'] = round(probability * 100, 2)
    input_data['prediction'] = prediction
    conn.execute("""
        INSERT INTO predictions (user_id, age, sex, chestpaintype, restingbp, cholesterol, fastingbs,
        restingecg, maxhr, exerciseangina, oldpeak, st_slope, probability, prediction)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        input_data['age'], input_data['sex'], input_data['chestpaintype'], input_data['restingbp'],
        input_data['cholesterol'], input_data['fastingbs'], input_data['restingecg'],
        input_data['maxhr'], input_data['exerciseangina'], input_data['oldpeak'], input_data['st_slope'],
        input_data['probability'], input_data['prediction']
    ))
    conn.commit()
    conn.close()

def get_history(user_id):
    conn = get_db_connection()
    history = conn.execute("SELECT age, cholesterol, maxhr, probability, prediction, timestamp FROM predictions WHERE user_id = ? ORDER BY timestamp DESC", (user_id,)).fetchall()
    conn.close()
    return pd.DataFrame(history, columns=['Age', 'Cholesterol', 'Max HR', 'Probability (%)', 'Risk (0=Low, 1=High)', 'Date & Time'])

# --- 3. ML MODEL LOADING ---
@st.cache_resource
def load_ml_assets():
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        model_features = joblib.load(FEATURES_FILE)
        st.sidebar.success("✅ ML Assets Loaded")
        return model, scaler, model_features
    except FileNotFoundError:
        st.error("⚠️ ML model files missing! Please ensure `.pkl` files exist.")
        return None, None, None

MODEL, SCALER, MODEL_FEATURES = load_ml_assets()

# --- 4. PREPROCESSING ---
def preprocess_input(input_data):
    if MODEL is None or SCALER is None or MODEL_FEATURES is None:
        return None
    df_raw = pd.DataFrame([input_data])
    categorical_cols = ['sex', 'chestpaintype', 'restingecg', 'exerciseangina', 'st_slope']
    df_encoded = pd.get_dummies(df_raw, columns=categorical_cols, drop_first=False)
    final_input = pd.DataFrame(0, index=[0], columns=MODEL_FEATURES)
    for col in df_encoded.columns:
        if col in final_input.columns:
            final_input[col] = df_encoded[col].values[0]
    numeric_cols = ['age', 'restingbp', 'cholesterol', 'maxhr', 'oldpeak']
    final_input[numeric_cols] = SCALER.transform(final_input[numeric_cols])
    return final_input

# --- 5. GOOGLE GEMINI API (REST) ---
def call_gemini_api(prompt, image_data=None):
    """Call Gemini 2.0 Flash REST API directly."""
    if not GEMINI_API_KEY:
        return "🚫 Gemini API key missing."
    
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GEMINI_API_KEY
    }

    # If image data is provided, handle it (base64 encoded)
    if image_data:
        body = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_data}}
                ]
            }]
        }
    else:
        body = {
            "contents": [{"parts": [{"text": prompt}]}]
        }

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=body, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"⚠️ Gemini API Error: {e}"

# --- 6. AUTH SCREENS ---
def sidebar_login_status():
    st.sidebar.markdown(f"**Logged in as:** `{st.session_state.username}`")
    if st.sidebar.button("🔓 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.user_id = None
        st.rerun()

def app_auth():
    st.title("CardioPredict AI Authentication")
    if 'auth_state' not in st.session_state:
        st.session_state.auth_state = 'login'
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Log In 🔑", use_container_width=True):
            st.session_state.auth_state = 'login'
    with col2:
        if st.button("Register ✨", use_container_width=True):
            st.session_state.auth_state = 'register'
    st.markdown("---")
    if st.session_state.auth_state == 'login':
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In", type="primary")
            if submitted:
                user_id, uname = verify_user(username, password)
                if user_id:
                    st.session_state.logged_in = True
                    st.session_state.username = uname
                    st.session_state.user_id = user_id
                    st.success(f"Welcome back, {uname}!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
    else:
        with st.form("register_form"):
            new_username = st.text_input("Choose Username")
            new_password = st.text_input("Choose Password", type="password")
            submitted = st.form_submit_button("Create Account", type="primary")
            if submitted:
                if len(new_password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    success, msg = add_user(new_username, new_password)
                    st.success(msg) if success else st.error(msg)

# --- 7. MAIN APP ---
def main_app():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.sidebar.title("User Profile")
    sidebar_login_status()
    st.sidebar.markdown("---")
    tab_predict, tab_vision, tab_history, tab_about = st.tabs(["Predict Risk", "Upload ECG/Scan", "History", "About"])

    # --- Prediction Tab ---
    with tab_predict:
        st.header("🩺 Patient Data Input")
        if MODEL is None:
            st.error("ML model not loaded.")
            return
        with st.form("prediction_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                age = st.slider("Age", 20, 90, 50)
                sex = st.selectbox("Sex", ['M', 'F'])
                restingbp = st.number_input("Resting BP", 90, 200, 120)
                cholesterol = st.number_input("Cholesterol", 0, 600, 250)
                fastingbs = st.selectbox("FastingBS > 120 mg/dl", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
            with col2:
                chestpaintype = st.selectbox("Chest Pain Type", ['ATA', 'NAP', 'ASY', 'TA'])
                restingecg = st.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
                maxhr = st.number_input("Max HR", 70, 220, 150)
                exerciseangina = st.selectbox("Exercise Angina", ['Y', 'N'])
                oldpeak = st.number_input("Oldpeak", 0.0, 6.2, 1.0, step=0.1)
                st_slope = st.selectbox("ST Slope", ['Up', 'Flat', 'Downsloping'])
            submitted = st.form_submit_button("Calculate Risk", type="primary")
            if submitted:
                raw_input = {
                    'age': age, 'sex': sex, 'chestpaintype': chestpaintype, 'restingbp': restingbp,
                    'cholesterol': cholesterol, 'fastingbs': fastingbs, 'restingecg': restingecg,
                    'maxhr': maxhr, 'exerciseangina': exerciseangina, 'oldpeak': oldpeak, 'st_slope': st_slope
                }
                processed_input = preprocess_input(raw_input)
                if processed_input is not None:
                    pred_proba = MODEL.predict_proba(processed_input)[:, 1][0]
                    pred_class = int(MODEL.predict(processed_input)[0])
                    probability_percent = round(pred_proba * 100, 2)
                    risk_level = "High Risk" if pred_class == 1 else "Low Risk"
                    icon = "🔴" if pred_class == 1 else "🟢"
                    st.success(f"{icon} {risk_level} ({probability_percent}%)")
                    add_prediction(st.session_state.user_id, raw_input, pred_proba, pred_class)
                    advice_prompt = f"A patient has a {probability_percent}% risk of heart disease. Suggest 3 practical preventive lifestyle or medical measures."
                    ai_advice = call_gemini_api(advice_prompt)
                    st.subheader("💡 AI-Generated Preventive Advice")
                    st.markdown(ai_advice)

    # --- Vision Tab ---
    with tab_vision:
        st.header("👁️ Upload ECG/Scan")
        uploaded_file = st.file_uploader("Upload image", type=['png', 'jpg', 'jpeg'])
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            base64_image = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
            if st.button("Analyze Image", type="primary"):
                vision_prompt = "Analyze this heart scan or ECG image. Describe neutral, objective indicators of cardiac risk."
                ai_analysis = call_gemini_api(vision_prompt, image_data=base64_image)
                st.subheader("AI Image Analysis Result")
                st.markdown(ai_analysis)

    # --- History Tab ---
    with tab_history:
        st.header("📜 Prediction History")
        df_history = get_history(st.session_state.user_id)
        if not df_history.empty:
            st.dataframe(df_history, use_container_width=True)
        else:
            st.info("No history found.")

    # --- About Tab ---
    with tab_about:
        st.header("ℹ️ About CardioPredict AI")
        st.markdown("""
        **CardioPredict AI** estimates heart disease risk using a Logistic Regression model.  
        It integrates **Google Gemini 2.0 Flash API** for AI-generated health insights and ECG image interpretation.
        """)

# --- 8. APP FLOW ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.user_id = None

if st.session_state.logged_in:
    main_app()
else:
    app_auth()
