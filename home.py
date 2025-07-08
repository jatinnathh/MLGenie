import streamlit as st

st.set_page_config(
    page_title="MLGenie",
    layout="wide",
    initial_sidebar_state="expanded"
)
import modules.visualize as visualize
import modules.ML_training as ML_training
import modules.DL_training as DL_training
import modules.Feature_eng as Feature_eng
from utils.shared import apply_common_settings
from utils.sidebar import custom_sidebar

# Apply common settings once
apply_common_settings()

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# Enhanced sidebar with toggle capability
selected = custom_sidebar()

# Map sidebar selection to page
page_map = {
    "Dashboard": "Dashboard",
    "Visualize": "Visualize",
    "Feature Engineering": "Feature Engineering",
    "ML Training": "ML Training",
    "DL Training": "DL Training"
}

if selected in page_map:
    st.session_state.page = page_map[selected]

# Render the selected page
if st.session_state.page == "Dashboard":
    st.title("AutoML Platform Dashboard")
    st.write("Welcome to your automated machine learning platform")
elif st.session_state.page == "Visualize":
    visualize.app()
elif st.session_state.page == "Feature Engineering":
    Feature_eng.app()
elif st.session_state.page == "ML Training":
    ML_training.app()
elif st.session_state.page == "DL Training":
    DL_training.app()