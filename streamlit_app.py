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
import json # Added for Lottie animation embedding

# --- 1. CONFIGURATION AND ASSET PATHS ---
APP_TITLE = "🫶🏼 CardioPredict AI — Heart Disease Risk Analyzer"
DB_FILE = "heart_app.db"
MODEL_FILE = "heart_disease_model.pkl"
SCALER_FILE = "heart_disease_scaler.pkl"
FEATURES_FILE = "model_features.pkl"
OPENROUTER_MODEL_TEXT = "google/gemini-2.5-flash"
OPENROUTER_MODEL_VISION = "google/gemini-2.5-flash"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Lottie URL for a heart animation (New Feature)
LOTTIE_URL = "https://lottie.host/17498c40-0235-4384-9543-f6614138a0f8/R08oO5iE5x.json"

# Environment Variable Check
OPENROUTER_API_KEY = ""

if OPENROUTER_API_KEY is None:
    st.warning("🚨 WARNING: OPENROUTER_API_KEY environment variable is not set. AI functionalities will be disabled.")

# --- 2. DATABASE FUNCTIONS (Unchanged) ---

def get_db_connection():
    """Initializes and returns a database connection."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Creates tables if they don't exist."""
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
    """Run database setup once on startup."""
    init_db()
    return True # Return anything to confirm setup is done

DB_READY = setup_db()

def add_user(username, password):
    """Adds a new user to the database."""
    conn = get_db_connection()
    try:
        password_hash = generate_password_hash(password)
        conn.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                     (username, password_hash))
        conn.commit()
        return True, "Registration successful!"
    except sqlite3.IntegrityError:
        return False, "Username already exists. Please choose a different one."
    finally:
        conn.close()

def verify_user(username, password):
    """Verifies user credentials."""
    conn = get_db_connection()
    user = conn.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    if user and check_password_hash(user['password_hash'], password):
        return user['id'], user['username']
    return None, None

def add_prediction(user_id, input_data, probability, prediction):
    """Saves a prediction result to the database."""
    conn = get_db_connection()
    input_data['probability'] = round(probability * 100, 2)
    input_data['prediction'] = prediction
    
    conn.execute(
        """
        INSERT INTO predictions (user_id, age, sex, chestpaintype, restingbp, cholesterol, fastingbs, restingecg, maxhr, exerciseangina, oldpeak, st_slope, probability, prediction)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            input_data['age'], input_data['sex'], input_data['chestpaintype'], input_data['restingbp'],
            input_data['cholesterol'], input_data['fastingbs'], input_data['restingecg'],
            input_data['maxhr'], input_data['exerciseangina'], input_data['oldpeak'], input_data['st_slope'],
            input_data['probability'], input_data['prediction']
        )
    )
    conn.commit()
    conn.close()

def get_history(user_id):
    """Retrieves all prediction history for a user."""
    conn = get_db_connection()
    history = conn.execute("SELECT age, cholesterol, maxhr, probability, prediction, timestamp FROM predictions WHERE user_id = ? ORDER BY timestamp DESC", (user_id,)).fetchall()
    conn.close()
    return pd.DataFrame(history, columns=['Age', 'Cholesterol', 'Max HR', 'Probability (%)', 'Risk (0=Low, 1=High)', 'Date & Time'])

# --- 3. MACHINE LEARNING MODEL LOADING (Unchanged) ---

@st.cache_resource
def load_ml_assets():
    """Loads and caches the model, scaler, and feature list."""
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        model_features = joblib.load(FEATURES_FILE)
        st.sidebar.success("ML Assets Loaded")
        return model, scaler, model_features
    except FileNotFoundError:
        st.error(f"ML assets not found. Ensure {MODEL_FILE}, {SCALER_FILE}, and {FEATURES_FILE} are in the directory. Did you run `train_model.py`?")
        return None, None, None

MODEL, SCALER, MODEL_FEATURES = load_ml_assets()

# --- 4. PREPROCESSING LOGIC (Unchanged) ---

def preprocess_input(input_data):
    """
    Preprocesses raw user input into a format suitable for the model.
    Matches the scaling and one-hot encoding logic from train_model.py.
    """
    if MODEL is None or SCALER is None or MODEL_FEATURES is None:
        return None

    # 1. Create a raw DataFrame
    # Note: input_data is a dict from the form, need to wrap it for DataFrame
    df_raw = pd.DataFrame([input_data])

    # 2. Convert categorical (string) columns into numeric dummies
    categorical_cols = ['sex', 'chestpaintype', 'restingecg', 'exerciseangina', 'st_slope']
    df_encoded = pd.get_dummies(df_raw, columns=categorical_cols, drop_first=False)

    # 3. Align columns with the features the model was trained on
    # Add missing OHE columns and fill with 0
    final_input = pd.DataFrame(0, index=[0], columns=MODEL_FEATURES)
    
    # Update with the values present in the encoded data
    for col in df_encoded.columns:
        if col in final_input.columns:
            final_input[col] = df_encoded[col].values[0]

    # 4. Scale numerical columns
    # These columns should be the original numerical columns that were scaled:
    numeric_cols = ['age', 'restingbp', 'cholesterol', 'maxhr', 'oldpeak']
    
    final_input[numeric_cols] = SCALER.transform(final_input[numeric_cols])

    return final_input

# --- 5. OPENROUTER API COMMUNICATION (Unchanged) ---

def call_openrouter_api(prompt, model, image_data=None):
    """
    Sends a request to the OpenRouter API for text or vision generation.
    Implements simple exponential backoff.
    """
    if not OPENROUTER_API_KEY:
        return "AI is disabled because the OPENROUTER_API_KEY environment variable is missing."

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    # Construct messages payload, handling vision if image_data is provided
    messages = [{"role": "user", "content": []}]

    if image_data:
        # Add image part for vision model
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
        })
    
    # Add text prompt part
    messages[0]["content"].append({"type": "text", "text": prompt})


    payload = {
        "model": model,
        "messages": messages
    }
    
    # Retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            if result.get('choices'):
                return result['choices'][0]['message']['content']
            else:
                return f"AI Service Error: Could not parse response. Details: {result}"

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                return f"AI Service Error after {max_retries} attempts: {e}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"

# --- 6. STREAMLIT UI COMPONENTS (Modified) ---

def sidebar_login_status():
    """Renders the sidebar based on login status."""
    st.sidebar.markdown(f"**Logged in as:** `{st.session_state.username}`")
    if st.sidebar.button("🔓 Logout", type="secondary", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.user_id = None
        st.rerun()

def app_auth():
    """Handles the login and registration pages."""
    # Ensure CSS is injected even on the auth screen
    inject_custom_css() 
    
    st.title("CardioPredict AI Authentication")

    if 'auth_state' not in st.session_state:
        st.session_state.auth_state = 'login'

    col1, col2 = st.columns(2)

    with col1:
        # Style buttons with CSS classes via key
        if st.button("Log In 🔑", key="btn_login", use_container_width=True):
            st.session_state.auth_state = 'login'
    with col2:
        if st.button("Register ✨", key="btn_register", use_container_width=True):
            st.session_state.auth_state = 'register'

    st.markdown("---")

    # Use st.container for a visual card effect on the form
    with st.container(border=True):
        if st.session_state.auth_state == 'login':
            st.subheader("Login")
            with st.form("login_form"):
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True) # type="primary" triggers gradient button CSS

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
                        st.error("Invalid username or password.")

        elif st.session_state.auth_state == 'register':
            st.subheader("Register")
            with st.form("register_form"):
                new_username = st.text_input("Choose Username", key="reg_username")
                new_password = st.text_input("Choose Password", type="password", key="reg_password")
                submitted = st.form_submit_button("Create Account", type="primary", use_container_width=True) # type="primary" triggers gradient button CSS

                if submitted:
                    if len(new_password) < 6:
                        st.error("Password must be at least 6 characters.")
                    else:
                        success, message = add_user(new_username, new_password)
                        if success:
                            st.success(f"{message} You can now log in.")
                            st.session_state.auth_state = 'login'
                        else:
                            st.error(message)

def render_lottie_animation(url: str, height: int = 300):
    """Embeds a Lottie animation using a script and st.html."""
    html_content = f"""
    <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
    <lottie-player
        src="{url}"
        background="transparent"
        speed="1"
        style="width: 100%; height: {height}px; margin: 0 auto;"
        loop
        autoplay
    ></lottie-player>
    """
    st.html(html_content)

def inject_custom_css():
    """Injects custom CSS for rounded cards, gradients, and dark mode variables."""
    # Initialize theme if not present
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'

    # Define color variables based on theme
    is_dark = st.session_state.theme == 'dark'
    
    # Soft Blue/White Palette for Medical Aesthetic
    COLOR_PRIMARY = "#2563eb"  # Bright Blue
    COLOR_BG_LIGHT = "#ffffff"
    COLOR_BG_DARK = "#1a1a2e" # Dark Blue/Purple for deep aesthetic
    COLOR_SECONDARY_BG_LIGHT = "#f0f8ff" # Lightest Blue
    COLOR_SECONDARY_BG_DARK = "#2c2c54"
    COLOR_TEXT_LIGHT = "#1f2937"
    COLOR_TEXT_DARK = "#f3f4f6"
    COLOR_CARD_LIGHT = "#ffffff"
    COLOR_CARD_DARK = "#353569"

    # Theme switching logic
    if is_dark:
        bg_color = COLOR_BG_DARK
        secondary_bg = COLOR_SECONDARY_BG_DARK
        text_color = COLOR_TEXT_DARK
        card_bg = COLOR_CARD_DARK
        # Streamlit's main body color needs to be overridden via ::before and ::after for full coverage
        st_bg_override = f"""
            .stApp {{ background-color: {bg_color}; transition: background-color 0.3s; }}
            .main > div {{ background-color: {bg_color}; }}
        """
    else:
        bg_color = COLOR_BG_LIGHT
        secondary_bg = COLOR_SECONDARY_BG_LIGHT
        text_color = COLOR_TEXT_LIGHT
        card_bg = COLOR_CARD_LIGHT
        st_bg_override = ""

    
    st.markdown(f"""
    <style>
    /* Global Variables for Dark/Light Mode */
    :root {{
        --primary-color: {COLOR_PRIMARY};
        --background-color: {bg_color};
        --secondary-background-color: {secondary_bg};
        --text-color: {text_color};
        --card-background: {card_bg};
        --risk-high: #ef4444; /* Red */
        --risk-low: #10b981;  /* Green */
    }}

    /* Global Streamlit Overrides */
    {st_bg_override}
    
    /* Custom Card Style for st.container */
    .stContainer {{
        background-color: var(--card-background) !important;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
        border-left: 5px solid var(--primary-color);
        transition: all 0.3s ease-in-out;
        margin-bottom: 20px;
    }}
    .stContainer:hover {{
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.15);
    }}

    /* Gradient Primary Button */
    /* Target all primary buttons */
    .stButton button, [data-testid="stForm"] .stButton > button, 
    .stDownloadButton button {{
        background: linear-gradient(90deg, #3b82f6, var(--primary-color));
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transition: all 0.2s ease;
    }}
    .stButton button:hover, [data-testid="stForm"] .stButton > button:hover, 
    .stDownloadButton button:hover {{
        background: linear-gradient(90deg, #1d4ed8, #1e40af);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
    }}

    /* Secondary Button (Logout/Toggle) */
    [data-testid="stSidebar"] .stButton > button {{
        background: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid var(--primary-color);
        transition: all 0.2s;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        background: var(--primary-color);
        color: white;
    }}

    /* Result Display Styling with Transition (Framer Motion-like effect) */
    .result-box {{
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        padding: 25px;
        border-radius: 10px;
        margin-top: 20px;
        border-left: 6px solid;
        animation: fadeIn 0.8s ease-out;
        transition: background-color 0.3s, color 0.3s;
    }}
    .risk-high {{ border-left-color: var(--risk-high); }}
    .risk-low {{ border-left-color: var(--risk-low); }}
    
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* Custom Input Styles */
    label {{ font-weight: 600 !important; color: var(--text-color) !important; }}
    .stSlider > div > div > div:nth-child(2) {{ background: var(--primary-color); }}
    .stTextInput > div > div > input, .stNumberInput > div > input, .stSelectbox > div > div {{
        border-radius: 8px;
        background-color: var(--card-background); /* Keep inputs visible in dark mode */
        color: var(--text-color);
    }}

    /* Ensure responsiveness on smaller devices for sidebar toggle visibility */
    @media (max-width: 768px) {{
        /* Custom styles for mobile UI adaptation */
        .stContainer {{ padding: 15px; }}
        .stButton button {{ font-size: 14px; padding: 8px 15px; }}
    }}

    </style>
    """, unsafe_allow_html=True)


def main_app():
    """Renders the main Streamlit application dashboard."""
    # Must be the first call
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    
    # Inject CSS for design and theme variables
    inject_custom_css() 

    st.title(APP_TITLE)

    # --- Sidebar ---
    st.sidebar.title("User Profile")
    sidebar_login_status()
    st.sidebar.markdown("---")
    
    # Dark/Light Mode Toggle (New Feature)
    st.sidebar.subheader("App Settings")
    
    current_theme = st.session_state.theme.capitalize()
    
    # Rerun the app when the button is clicked to apply new CSS variables
    if st.sidebar.button(f"🎨 Toggle Theme: {current_theme}", use_container_width=True):
        st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Tabs
    tab_predict, tab_vision, tab_history, tab_about = st.tabs(["Predict Risk", "Upload ECG/Scan", "History", "About"])

    # --- TAB 1: PREDICTION DASHBOARD ---
    with tab_predict:
        st.header("Patient Data Input 🩺")

        # Add Lottie animation for visual engagement
        render_lottie_animation(LOTTIE_URL, height=150)
        
        if MODEL is None:
            st.error("The ML model assets could not be loaded. Please check the console for errors.")
            return

        with st.form("prediction_form", clear_on_submit=True):
            col_a, col_b = st.columns(2)

            with col_a:
                # Use st.container(border=True) for a styled card effect
                with st.container(border=True): 
                    st.subheader("Demographics & Vitals")
                    age = st.slider("Age", 20, 90, 50)
                    sex = st.selectbox("Sex", ['M', 'F'])
                    restingbp = st.number_input("Resting Blood Pressure (RestingBP)", 90, 200, 120)
                    cholesterol = st.number_input("Cholesterol", 0, 600, 250)
                    fastingbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (FastingBS)", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

            with col_b:
                # Use st.container(border=True) for a styled card effect
                with st.container(border=True):
                    st.subheader("Cardiac Indicators")
                    chestpaintype = st.selectbox("Chest Pain Type (ChestPainType)", ['ATA', 'NAP', 'ASY', 'TA'])
                    restingecg = st.selectbox("Resting ECG (RestingECG)", ['Normal', 'ST', 'LVH'])
                    maxhr = st.number_input("Maximum Heart Rate Achieved (MaxHR)", 70, 220, 150)
                    exerciseangina = st.selectbox("Exercise Induced Angina (ExerciseAngina)", ['Y', 'N'])
                    oldpeak = st.number_input("Oldpeak (ST depression induced by exercise relative to rest)", 0.0, 6.2, 1.0, step=0.1)
                    st_slope = st.selectbox("ST Slope (ST_Slope)", ['Up', 'Flat', 'Downsloping'])

            # type="primary" triggers the gradient button CSS
            submitted = st.form_submit_button("Calculate Risk Score", type="primary", use_container_width=True)

            if submitted:
                # 1. Gather raw data
                raw_input = {
                    'age': age, 'sex': sex, 'chestpaintype': chestpaintype,
                    'restingbp': restingbp, 'cholesterol': cholesterol,
                    'fastingbs': fastingbs, 'restingecg': restingecg,
                    'maxhr': maxhr, 'exerciseangina': exerciseangina,
                    'oldpeak': oldpeak, 'st_slope': st_slope
                }

                # 2. Preprocess
                processed_input = preprocess_input(raw_input)

                if processed_input is not None:
                    # 3. Predict
                    with st.spinner('Predicting and generating AI advice...'):
                        try:
                            # Prediction: probability and class
                            pred_proba = MODEL.predict_proba(processed_input)[:, 1][0]
                            pred_class = int(MODEL.predict(processed_input)[0])
                            probability_percent = round(pred_proba * 100, 2)

                            # 4. Display Result (Using custom CSS for aesthetic and transition)
                            risk_level = "High Risk" if pred_class == 1 else "Low Risk"
                            css_class = "risk-high" if pred_class == 1 else "risk-low"
                            icon = "🔴" if pred_class == 1 else "🟢"

                            st.subheader("Prediction Result")
                            st.markdown(
                                f"""
                                <div class="result-box {css_class}">
                                    <h3>{icon} {risk_level}</h3>
                                    <p>Predicted Heart Disease Risk Probability: <b>{probability_percent}%</b></p>
                                    <small>Risk calculated based on ML model. <b>This is not medical advice.</b></small>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )

                            # 5. Save History
                            add_prediction(st.session_state.user_id, raw_input, pred_proba, pred_class)

                            # 6. AI-based Preventive Measures
                            advice_prompt = f"A patient has a {probability_percent}% risk of heart disease. Suggest 3 practical preventive lifestyle or medical measures. Format the response as a clear markdown list."
                            ai_advice = call_openrouter_api(advice_prompt, OPENROUTER_MODEL_TEXT)

                            st.subheader("AI-Generated Preventive Measures 💡")
                            st.markdown(ai_advice)

                        except Exception as e:
                            st.error(f"An error occurred during prediction or AI call: {e}")
                else:
                    st.error("Preprocessing failed. Check model asset loading.")

    # --- TAB 2: IMAGE-BASED RISK ANALYSIS (AI VISION) ---
    with tab_vision:
        st.header("Image-Based Cardiac Analysis 👁️")
        st.markdown("Upload an image (e.g., ECG, heart scan) for basic AI interpretation.")

        # Use st.container for a visual card effect
        with st.container(border=True):
            uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])

            if uploaded_file is not None:
                # Display image
                st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

                # Convert image to Base64 for API
                base64_image = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

                if st.button("Analyze Image with AI 🧠", type="primary", use_container_width=True):
                    with st.spinner('Sending image for AI analysis...'):
                        vision_prompt = "Analyze this uploaded image (likely an ECG, X-ray, or scan). Describe any visible indicators of cardiac risk (e.g., arrhythmias, enlargement, calcification). Keep the description medically neutral, objective, and concise. State clearly that this is not a diagnosis."
                        
                        ai_analysis = call_openrouter_api(
                            prompt=vision_prompt,
                            model=OPENROUTER_MODEL_VISION,
                            image_data=base64_image
                        )

                        st.subheader("AI Vision Analysis Result")
                        st.markdown(ai_analysis)

    # --- TAB 3: HISTORY ---
    with tab_history:
        st.header("Prediction History 📜")
        df_history = get_history(st.session_state.user_id)

        # Use st.container for a visual card effect
        with st.container(border=True):
            if not df_history.empty:
                df_history['Risk Classification'] = df_history['Risk (0=Low, 1=High)'].apply(lambda x: "🔴 High Risk" if x == 1 else "🟢 Low Risk")
                df_history.rename(columns={'Risk (0=Low, 1=High)': 'Risk Code'}, inplace=True)
                
                # Reorder for better display
                display_cols = ['Date & Time', 'Age', 'Cholesterol', 'Max HR', 'Probability (%)', 'Risk Classification', 'Risk Code']
                df_history = df_history[display_cols]

                st.dataframe(df_history, use_container_width=True, hide_index=True)
                
                # Simple summary stats
                high_risk_count = (df_history['Risk Code'] == 1).sum()
                total_count = len(df_history)
                
                st.markdown(f"---")
                st.info(f"Summary: You have **{high_risk_count}** out of **{total_count}** past predictions classified as High Risk (Code 1).")
            else:
                st.info("No prediction history found. Use the 'Predict Risk' tab to start.")

    # --- TAB 4: ABOUT ---
    with tab_about:
        st.header("About CardioPredict AI")
        st.markdown("""
        CardioPredict AI is an experimental tool designed to calculate the risk of heart disease based on common clinical and lifestyle factors.
        
        ### ⚙️ Technical Details
        * **ML Model:** Logistic Regression trained on the `heart.csv` dataset.
        * **Preprocessing:** Data is scaled using `StandardScaler` and categorical features are handled via One-Hot Encoding (`pd.get_dummies`).
        * **AI Advice/Vision:** Utilizes the **OpenRouter** API (specifically the `google/gemini-2.5-flash` model) for real-time preventive suggestions and image analysis.
        * **Data Storage:** User credentials and prediction history are stored locally in an SQLite database (`heart_app.db`).
        
        ---
        **Disclaimer:** This application is for educational and experimental purposes only. The results and AI advice provided are not a substitute for professional medical consultation, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider.
        """)


# --- 7. MAIN APP EXECUTION FLOW ---

# Initialize session state for authentication and theme
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.user_id = None
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

if st.session_state.logged_in:
    main_app()
else:
    app_auth()
