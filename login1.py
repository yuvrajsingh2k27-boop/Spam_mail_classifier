import streamlit as st
from signup import load_users
import hashlib
def login():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    st.subheader("🔑 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    entered_hash = hashlib.sha256(password.encode()).hexdigest()
    if st.button("Login"):
        users = load_users()
        if username in users and users[username] == entered_hash:
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("❌ Invalid username or password..")
