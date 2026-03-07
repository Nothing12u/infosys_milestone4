import streamlit as st
import pandas as pd
import pickle
import json
import os

st.set_page_config(page_title="Career Predictor Dashboard", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-image: linear-gradient(to right, #f5f7fa, #c3cfe2);
        background-color: #f0f2f6;
    }
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 20px;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)


def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        return model, encoders
    except FileNotFoundError:
        st.error("Model not found! Please run train_model.py first.")
        return None, None

def load_users():
    if os.path.exists("users.json"):
        with open("users.json", "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f)


def login_page():
    st.markdown("<h1 class='main-header'>Student Login</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                users = load_users()
                if username in users and users[username]['password'] == password:
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.rerun()
                else:
                    st.error("Invalid Username or Password")
        
        st.write("Don't have an account?")
        if st.button("Go to Registration"):
            st.session_state['page'] = 'register'
            st.rerun()

def register_page():
    st.markdown("<h1 class='main-header'>Register New User</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("reg_form"):
            new_user = st.text_input("Choose Username")
            new_pass = st.text_input("Choose Password", type="password")
            submit = st.form_submit_button("Register")
            
            if submit:
                if not new_user or not new_pass:
                    st.error("Fields cannot be empty.")
                else:
                    users = load_users()
                    if new_user in users:
                        st.error("Username already exists.")
                    else:
                        users[new_user] = {
                            "password": new_pass,
                            "profile": {"name": "", "degree": "", "spec": "", "cgpa": 0.0}
                        }
                        save_users(users)
                        st.success("Registration Successful! Please Login.")
                        st.session_state['page'] = 'login'
                        st.rerun()

def dashboard_page():
    username = st.session_state['username']
    users = load_users()
    profile = users[username]['profile']
    
    st.markdown(f"<h1 class='main-header'>Welcome, {username}!</h1>", unsafe_allow_html=True)
    
    st.sidebar.title("Menu")
    choice = st.sidebar.radio("Navigate", ["View Profile", "Job Prediction", "Logout"])
    
    if choice == "View Profile":
        view_profile(username, profile)
    elif choice == "Job Prediction":
        prediction_page()
    elif choice == "Logout":
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.rerun()

def view_profile(username, profile):
    st.header("My Profile")
    
    st.info(f"**Name:** {profile.get('name', 'Not Set')}")
    st.info(f"**Degree:** {profile.get('degree', 'Not Set')}")
    st.info(f"**Specialization:** {profile.get('spec', 'Not Set')}")
    st.info(f"**CGPA:** {profile.get('cgpa', 'Not Set')}")
    
    st.subheader("Edit Profile")
    with st.form("edit_profile"):
        name = st.text_input("Full Name", value=profile.get('name', ''))
        degree = st.text_input("Degree (e.g., B.Tech)", value=profile.get('degree', ''))
        spec = st.text_input("Specialization", value=profile.get('spec', ''))
        cgpa = st.number_input("CGPA (0-10)", min_value=0.0, max_value=10.0, value=profile.get('cgpa', 0.0))
        
        submit = st.form_submit_button("Update Profile")
        
        if submit:
            if not name or not degree:
                st.error("Name and Degree are required.")
            elif cgpa < 0 or cgpa > 10:
                st.error("CGPA must be between 0 and 10.")
            else:
                users = load_users()
                users[username]['profile'] = {
                    "name": name, "degree": degree, "spec": spec, "cgpa": cgpa
                }
                save_users(users)
                st.success("Profile Updated!")
                st.rerun()

def prediction_page():
    st.header("Job Role Prediction")
    model, encoders = load_model()
    
    if model is None:
        return

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Enter Details")
        df = pd.read_csv('job_dataset.csv')
        degrees = sorted(df['Degree'].unique())
        specs = sorted(df['Specialization'].unique())
        
        input_degree = st.selectbox("Select Degree", degrees)
        input_spec = st.selectbox("Select Specialization", specs)
        input_cgpa = st.number_input("Enter CGPA", 0.0, 10.0, 7.5)
        
        if st.button("Predict Job"):
            try:
                deg_enc = encoders['degree'].transform([input_degree])[0]
                spec_enc = encoders['spec'].transform([input_spec])[0]
                
                prediction = model.predict([[deg_enc, spec_enc, input_cgpa]])
                job_role = encoders['job'].inverse_transform(prediction)[0]
                
                st.success(f"Predicted Job Role: **{job_role}**")
            except Exception as e:
                st.error(f"Error in prediction: {e}")

    with col2:
        st.subheader("Job Market Distribution")
        job_counts = df['JobRole'].value_counts()
        st.bar_chart(job_counts)


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

if __name__ == "__main__":
    main()