import streamlit as st
import pandas as pd
import pickle
import json
import os
import time
import numpy as np

st.set_page_config(page_title="Career Predictor Dashboard", layout="wide", page_icon="🎓")

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
        margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.3);
        transition: all 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.25);
    }
    .main-header {
        font-size: 3.2rem;
        font-weight: 800;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
        animation: fadeInDown 0.8s ease;
    }
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: scale(1.03);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .prediction-result {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin-top: 20px;
        animation: pulse 2s infinite;
        text-align: center;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(56, 239, 125, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(56, 239, 125, 0); }
        100% { box-shadow: 0 0 0 0 rgba(56, 239, 125, 0); }
    }
</style>
""", unsafe_allow_html=True)

def load_model():
    """✅ FIXED: Properly extract model from dictionary format"""
    try:
        with open('model.pkl', 'rb') as f:
            model_package = pickle.load(f)
        
        # ✅ Extract model from dictionary
        if isinstance(model_package, dict) and 'model' in model_package:
            model = model_package['model']
            encoders = model_package.get('encoders', None)
            print(f"🔍 Model loaded: {type(model).__name__}")
            print(f"📐 Features expected: {model.n_features_in_}")
        else:
            # Fallback for old format
            model = model_package
            encoders = None
            
        # Load encoders separately if needed
        if encoders is None and os.path.exists('encoders.pkl'):
            with open('encoders.pkl', 'rb') as f:
                encoders = pickle.load(f)
                
        return model, encoders
        
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Run `train_model.py` first.")
        return None, None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None, None

def load_users():
    if os.path.exists("users.json"):
        with open("users.json", "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f, indent=4)

def animate_prediction():
    progress_bar = st.progress(0)
    status_text = st.empty()
    steps = [
        (10, "🔍 Loading your profile..."),
        (30, "📊 Analyzing academic background..."),
        (50, "🧠 Processing skill patterns..."),
        (70, "🔮 Running prediction model..."),
        (90, "✨ Generating recommendations..."),
        (100, "✅ Prediction complete!")
    ]
    for progress, message in steps:
        progress_bar.progress(progress)
        status_text.text(message)
        time.sleep(0.3)
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()

def login_page():
    st.markdown("<h1 class='main-header'>🎓 Career Predictor</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("""
            <div style='text-align: center; margin-bottom: 30px;'>
                <span style='font-size: 4rem;'>🚀</span>
                <h3 style='color: #2c3e50; margin: 10px 0;'>Welcome Back!</h3>
                <p style='color: #7f8c8d;'>Sign in to predict your dream career</p>
            </div>
            """, unsafe_allow_html=True)
            with st.form("login_form"):
                username = st.text_input("👤 Username", placeholder="Enter your username")
                password = st.text_input("🔐 Password", type="password", placeholder="Enter your password")
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    submit = st.form_submit_button("Login")
                with col_btn2:
                    if st.form_submit_button("Demo Login 🎭"):
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = 'demo_user'
                        st.rerun()
                if submit:
                    users = load_users()
                    if username in users and users[username]['password'] == password:
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = username
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials. Try again!")
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("""
            <div style='text-align: center; margin-top: 25px;'>
                <span style='color: white;'>New here? </span>
            </div>
            """, unsafe_allow_html=True)
            if st.button("✨ Create Account"):
                st.session_state['page'] = 'register'
                st.rerun()

def register_page():
    st.markdown("<h1 class='main-header'>🌟 Join Us Today</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("""
            <div style='text-align: center; margin-bottom: 25px;'>
                <span style='font-size: 4rem;'>🎯</span>
                <h3 style='color: #2c3e50;'>Start Your Journey</h3>
                <p style='color: #7f8c8d;'>Create your account in seconds</p>
            </div>
            """, unsafe_allow_html=True)
            with st.form("reg_form"):
                new_user = st.text_input("👤 Choose Username", placeholder="Desired username")
                new_pass = st.text_input("🔐 Create Password", type="password", placeholder="Min 6 characters")
                if new_pass:
                    strength = min(len(new_pass) * 20, 100)
                    st.progress(strength)
                    if strength < 40:
                        st.caption("🔴 Weak password")
                    elif strength < 70:
                        st.caption("🟡 Moderate password")
                    else:
                        st.caption("🟢 Strong password")
                submit = st.form_submit_button("🚀 Register Now")
                if submit:
                    if not new_user or not new_pass:
                        st.error("⚠️ All fields are required!")
                    elif len(new_pass) < 6:
                        st.error("⚠️ Password must be at least 6 characters!")
                    else:
                        users = load_users()
                        if new_user in users:
                            st.error("⚠️ Username already taken!")
                        else:
                            users[new_user] = {
                                "password": new_pass,
                                "profile": {
                                    "name": "", 
                                    "degree": "", 
                                    "spec": "", 
                                    "cgpa": 0.0
                                    # ✅ Technologies removed from profile
                                }
                            }
                            save_users(users)
                            st.success("🎉 Account created! Redirecting to login...")
                            time.sleep(1.5)
                            st.session_state['page'] = 'login'
                            st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
            if st.button("🔙 Back to Login"):
                st.session_state['page'] = 'login'
                st.rerun()

def dashboard_page():
    username = st.session_state['username']
    users = load_users()
    profile = users[username]['profile']
    st.markdown(f"<h1 class='main-header'>👋 Hello, {username}!</h1>", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        width: 80px; height: 80px; border-radius: 50%; 
                        margin: 0 auto 15px; display: flex; align-items: center; justify-content: center;
                        font-size: 2rem; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
                🎓
            </div>
            <h4 style='color: #2c3e50; margin: 0;'>Career Predictor</h4>
            <p style='color: #7f8c8d; font-size: 0.9rem; margin: 5px 0;'>AI-Powered Guidance</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        choice = st.radio(
            "🧭 Navigate", 
            ["📋 View Profile", "🔮 Job Prediction", "📊 Insights", "⚙️ Settings", "🚪 Logout"], 
            index=0,
            label_visibility="collapsed"
        )
        st.markdown("---")
        st.info(f"🔐 **Session:** {username}")
        if profile.get('name'):
            st.success("✅ Profile Complete")
        else:
            st.warning("⚠️ Complete your profile")
    if "Profile" in choice:
        view_profile(username, profile)
    elif "Prediction" in choice:
        prediction_page()
    elif "Insights" in choice:
        insights_page()
    elif "Settings" in choice:
        settings_page(username)
    elif "Logout" in choice:
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.session_state['page'] = 'login'
        st.toast("👋 Logged out successfully!", icon="✅")
        time.sleep(1)
        st.rerun()

def view_profile(username, profile):
    st.markdown("<div class='card'><h2>📋 My Profile</h2></div>", unsafe_allow_html=True)
    col_avatar, col_info = st.columns([1, 3])
    with col_avatar:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    width: 120px; height: 120px; border-radius: 50%; 
                    display: flex; align-items: center; justify-content: center;
                    font-size: 3rem; color: white; margin: 0 auto;
                    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);'>
            👤
        </div>
        """, unsafe_allow_html=True)
    with col_info:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{profile.get('name', 'Not Set')}</div><div class='metric-label'>Full Name</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{profile.get('degree', 'Not Set')}</div><div class='metric-label'>Degree</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{profile.get('spec', 'Not Set')}</div><div class='metric-label'>Specialization</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{profile.get('cgpa', 'Not Set')}/10</div><div class='metric-label'>CGPA</div></div>", unsafe_allow_html=True)
    
    # ✅ Technologies section removed
    
    st.markdown("---")
    with st.expander("✏️ Edit Profile", expanded=False):
        with st.form("edit_profile"):
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("Full Name", value=profile.get('name', ''))
                degree = st.text_input("Degree", value=profile.get('degree', ''))
            with c2:
                spec = st.text_input("Specialization", value=profile.get('spec', ''))
                cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=profile.get('cgpa', 0.0), step=0.1)
            # ✅ Technologies input removed
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                submit = st.form_submit_button("💾 Save Changes")
            with col_btn2:
                reset = st.form_submit_button("🔄 Reset")
            if submit:
                if not name or not degree:
                    st.error("⚠️ Name and Degree are required!")
                elif cgpa < 0 or cgpa > 10:
                    st.error("⚠️ CGPA must be between 0 and 10!")
                else:
                    users = load_users()
                    users[username]['profile'] = {
                        "name": name, 
                        "degree": degree, 
                        "spec": spec, 
                        "cgpa": cgpa
                    }
                    save_users(users)
                    st.success("✅ Profile updated successfully!")
                    time.sleep(1)
                    st.rerun()
            if reset:
                st.info("🔄 Form reset. Changes not saved.")

def prediction_page():
    st.markdown("<div class='card'><h2>🔮 AI Job Predictor</h2></div>", unsafe_allow_html=True)
    
    # Load model and encoders
    model, encoders = load_model()
    
    # ✅ Verify model loaded correctly
    if model is None:
        st.warning("⚠️ Please train the model first by running `train_model.py`")
        return
    
    # ✅ Verify model has predict method
    if not hasattr(model, 'predict'):
        st.error("❌ Model object invalid! Retrain with updated train_model.py")
        return
    
    # Get unique values for dropdowns
    if os.path.exists('job_dataset.csv'):
        df = pd.read_csv('job_dataset.csv')
        degrees = sorted(df['Degree'].unique())
        specs = sorted(df['Specialization'].unique())
    else:
        degrees = ["B.Tech", "M.Tech", "BCA", "MCA"]
        specs = ["CS", "IT", "ECE", "EE"]
    
    col_input, col_chart = st.columns([2, 1])
    
    with col_input:
        st.markdown("<div class='card'><h3>📝 Enter Your Details</h3></div>", unsafe_allow_html=True)
        
        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                input_degree = st.selectbox("🎓 Select Degree", degrees)
                input_cgpa = st.slider("📊 CGPA", 0.0, 10.0, 7.5, 0.1)
            with c2:
                input_spec = st.selectbox("🔬 Select Specialization", specs)
                # ✅ Technologies field REMOVED
            
            col_btn1, col_btn2 = st.columns([2, 1])
            with col_btn1:
                submit_pred = st.form_submit_button("🚀 Predict My Career")
            with col_btn2:
                if st.form_submit_button("🎲 Try Random"):
                    import random
                    input_degree = random.choice(degrees)
                    input_spec = random.choice(specs)
                    input_cgpa = round(random.uniform(6.0, 9.5), 1)
                    st.rerun()
            
            if submit_pred:
                animate_prediction()
                try:
                    # ✅ Encode inputs
                    deg_enc = encoders['degree'].transform([input_degree])[0]
                    spec_enc = encoders['spec'].transform([input_spec])[0]
                    
                    # ✅ Create input array with EXACTLY 3 features in correct order
                    input_array = np.array([[deg_enc, spec_enc, float(input_cgpa)]])
                    
                    # ✅ PREDICT - model is now the actual classifier!
                    prediction = model.predict(input_array)
                    job_role = encoders['job'].inverse_transform(prediction)[0]
                    
                    # Calculate confidence
                    confidence = min(85 + (input_cgpa - 6) * 3, 98)
                    
                    # Display result
                    st.markdown(f"""
                    <div class='prediction-result'>
                        <h2 style='margin: 0; font-size: 2rem;'>🎯 {job_role}</h2>
                        <p style='margin: 10px 0; opacity: 0.95;'>Based on: {input_degree} in {input_spec} with CGPA {input_cgpa}</p>
                        <div style='background: rgba(255,255,255,0.2); padding: 10px; border-radius: 8px; display: inline-block;'>
                            <strong>Confidence:</strong> {confidence:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Career tips
                    st.markdown("### 💡 Career Tips")
                    tips = {
                        "Software Engineer": ["Master DSA", "Build projects", "Contribute to OSS"],
                        "Data Analyst": ["Learn SQL deeply", "Practice visualization", "Understand statistics"],
                        "Web Developer": ["Master frameworks", "Learn responsive design", "Build portfolio"],
                        "Data Scientist": ["Study ML algorithms", "Work on Kaggle", "Learn deployment"]
                    }
                    for tip in tips.get(job_role, ["Keep learning!", "Network actively", "Stay updated"]):
                        st.markdown(f"• {tip}")
                    
                    st.markdown("### 📈 Your Path Forward")
                    st.progress(int(confidence))
                    st.caption(f"{100 - confidence:.1f}% more preparation recommended")
                    
                except Exception as e:
                    st.error(f"⚠️ Prediction error: {str(e)}")
                    st.info("💡 Tip: Delete model.pkl and retrain with updated train_model.py")
                    import traceback
                    traceback.print_exc()
    
    with col_chart:
        st.markdown("<div class='card'><h3>📊 Market Insights</h3></div>", unsafe_allow_html=True)
        if os.path.exists('job_dataset.csv'):
            chart_type = st.radio("View:", ["📈 By Role", "🎓 By Degree"], horizontal=True)
            if chart_type == "📈 By Role":
                job_counts = df['JobRole'].value_counts().head(8)
                st.bar_chart(job_counts, use_container_width=True)
            else:
                degree_counts = df['Degree'].value_counts()
                st.bar_chart(degree_counts, use_container_width=True)
            st.markdown("### 🔥 In-Demand Skills")
            skills = ["Python", "SQL", "JavaScript", "Machine Learning", "AWS", "React", "Java", "Data Analysis"]
            for i, skill in enumerate(skills[:5]):
                st.markdown(f"""
                <div style='display: flex; align-items: center; margin: 8px 0;'>
                    <div style='width: 30px; height: 30px; background: linear-gradient(135deg, #667eea, #764ba2); 
                                border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                                color: white; font-weight: bold; margin-right: 10px;'>{i+1}</div>
                    <span style='font-weight: 500;'>{skill}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📁 Dataset not found for analytics")

def insights_page():
    st.markdown("<div class='card'><h2>📊 Career Insights</h2></div>", unsafe_allow_html=True)
    if os.path.exists('job_dataset.csv'):
        df = pd.read_csv('job_dataset.csv')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📚 Total Records", len(df))
        with col2:
            st.metric("🎯 Job Roles", df['JobRole'].nunique())
        with col3:
            st.metric("🎓 Degrees", df['Degree'].nunique())
        with col4:
            st.metric("📊 Avg CGPA", f"{df['CGPA'].mean():.2f}")
        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["🎓 Degree Distribution", "🎯 Job Roles", "📈 CGPA Analysis"])
        with tab1:
            degree_dist = df['Degree'].value_counts()
            st.bar_chart(degree_dist, use_container_width=True)
        with tab2:
            job_dist = df['JobRole'].value_counts()
            st.bar_chart(job_dist, use_container_width=True)
        with tab3:
            st.subheader("CGPA by Job Role")
            cgpa_by_job = df.groupby('JobRole')['CGPA'].mean().sort_values(ascending=False).head(10)
            st.bar_chart(cgpa_by_job, use_container_width=True)
        st.markdown("### 💰 Estimated Salary Ranges")
        salary_data = {
            "Software Engineer": "₹6-15 LPA",
            "Data Analyst": "₹5-12 LPA", 
            "Web Developer": "₹4-10 LPA",
            "Data Scientist": "₹8-20 LPA",
            "AI Researcher": "₹10-25 LPA"
        }
        for role, salary in salary_data.items():
            if role in df['JobRole'].values:
                st.markdown(f"• **{role}**: {salary}")
    else:
        st.warning("📁 Please ensure job_dataset.csv exists for insights")

def settings_page(username):
    st.markdown("<div class='card'><h2>⚙️ Account Settings</h2></div>", unsafe_allow_html=True)
    users = load_users()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔐 Security")
        if st.button("🔄 Change Password"):
            st.info("Password change feature coming soon!")
        if st.button("🗑️ Delete Account", type="primary"):
            if st.checkbox("⚠️ I understand this is permanent"):
                if username in users:
                    del users[username]
                    save_users(users)
                    st.success("✅ Account deleted")
                    st.session_state['logged_in'] = False
                    time.sleep(2)
                    st.rerun()
    with col2:
        st.markdown("### 📧 Notifications")
        st.toggle("📊 Prediction alerts", value=True)
        st.toggle("📰 Career tips", value=True)
        st.toggle("🎓 New features", value=False)
    st.markdown("---")
    st.markdown("### 📤 Export Data")
    if st.button("📥 Download Profile as JSON"):
        profile_data = users.get(username, {})
        st.json(profile_data)
        st.download_button(
            "⬇️ Download File",
            data=json.dumps(profile_data, indent=2),
            file_name=f"{username}_profile.json",
            mime="application/json"
        )

def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'page' not in st.session_state:
        st.session_state['page'] = 'login'
    if st.session_state['logged_in']:
        dashboard_page()
    else:
        if st.session_state['page'] == 'register':
            register_page()
        else:
            login_page()
    st.markdown("""
    <div style='text-align: center; padding: 20px; color: white; font-size: 0.9rem; opacity: 0.9; margin-top: 40px;'>
        🎓 Career Predictor Dashboard | Powered by AI | © 2024
        <br><small>Remember: Predictions are guidance, not guarantees. Your effort matters most! 💪</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
