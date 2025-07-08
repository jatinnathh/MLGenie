import streamlit as st


def apply_common_settings():


    st.markdown("""
        <style>
                
        section[data-testid="stSidebar"] {
            background-color: #121212;
            padding: 2rem 1rem;
        }
        .nav-title {
            font-size: 22px;
            color: #00c7b7;
            font-weight: 700;
            padding-bottom: 20px;
        }
        .nav-link {
            background-color: transparent;
            color: #dddddd;
            padding: 0.75rem 1rem;
            margin-bottom: 0.5rem;
            border: none;
            border-radius: 8px;
            text-align: left;
            width: 100%;
            font-size: 16px;
            transition: all 0.3s ease;
            box-shadow: none;
        }
        .nav-link:hover {
            background-color: #2e2e2e;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            transform: translateX(4px);
            cursor: pointer;
        }
        .nav-link-active {
            background-color: #00c7b7 !important;
            color: #000000 !important;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)