import streamlit as st
import json
import os
import hashlib

USER_DB = "users.json"

def load_users():
    if os.path.exists(USER_DB):
        with open(USER_DB, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_DB, "w") as f:
        json.dump(users, f)

def signup():
    st.subheader("🔐 Signup")
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")
    entered_hash = hashlib.sha256(password.encode()).hexdigest()
    entered_hash1 = hashlib.sha256(confirm.encode()).hexdigest()
    if st.button("Create Account"):
        if not username or not password:
            st.error("Username and password cannot be empty.")
        elif entered_hash != entered_hash1:
            st.error("Passwords do not match.")
        else:
            users = load_users()
            if username in users:
                st.error("Username already exists.")
            else:
                users[username] = entered_hash
                save_users(users)
                st.success("Account created successfully! You can now log in.")