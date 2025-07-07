import streamlit as st
import visualize
import ML_training
import DL_training

st.set_page_config(page_title="AutoML Platform", layout="wide")

# Set up session state
if "page" not in st.session_state:
    st.session_state.page = "Home"

# --- Custom Styling ---
st.markdown("""
    <style>
    /* Hide sidebar collapse icon */
    [data-testid="collapsedControl"] { display: none; }

    /* Sidebar */
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

# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="nav-title">AutoML Navigator</div>', unsafe_allow_html=True)

    def nav_button(label, target):
        if st.session_state.page == target:
            st.markdown(f'<button class="nav-link nav-link-active">{label}</button>', unsafe_allow_html=True)
        else:
            if st.button(label, key=label):
                st.session_state.page = target

    nav_button("Home", "Home")
    nav_button("Data Visualization", "Visualize")
    nav_button("ML Training", "ML")
    nav_button("DL Training", "DL")

# --- Main Content ---
if st.session_state.page == "Home":
    st.title("AutoML Dashboard")
    st.markdown("""
        Welcome to your AutoML dashboard. 
        Use the sidebar to access modules for data visualization, machine learning, and deep learning training.
    """)

elif st.session_state.page == "Visualize":
    visualize.app()

elif st.session_state.page == "ML":
    ML_training.app()

elif st.session_state.page == "DL":
    DL_training.app()