import time as t
import streamlit as st
import base64
def show_welcome():
    if 'welcome_shown' not in st.session_state:
        st.session_state.welcome_shown = True

        st.markdown("""
            <style>
            .welcome-overlay {
                position: fixed;
                top: 0; left: 0;
                width: 100%;
                height: 100%;
                background-color: #A1AAB1;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                z-index: 9999;
                animation: fadeOut 3s forwards;
                font-family: 'Segoe UI', sans-serif;
            }
            .welcome-logo {
                width: 100px;
                height: 100px;
                margin-bottom: 20px;
                border-radius: 20px;
                object-fit: contain;
                # animation: fadeIn 1s ease-in-out;
            }
            .welcome-text {
                font-size: 2.5rem;
                color: white;
                text-align: center;
                animation: fadeIn 1.5s ease-in-out;
            }
            @keyframes fadeIn {
                from {opacity: 0;}
                to {opacity: 1;}
            }
            </style>
        """, unsafe_allow_html=True)

        placeholder = st.empty()
        placeholder.markdown("""<div class="welcome-overlay">
                <div class="welcome-text">👋 Welcome to <i>Spamsilly<i></div>
            </div>
        """, unsafe_allow_html=True)

        t.sleep(2)
        placeholder.empty()
