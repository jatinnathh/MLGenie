import streamlit as st

st.set_page_config(
    page_title="MLGenie",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modules correctly
from modules import visualize
from modules import ML_training
import modules.DL_training as DL_training
from modules import Feature_eng
from utils.shared import apply_common_settings
from utils.sidebar import custom_sidebar

# Apply common settings once
apply_common_settings()
custom_sidebar()

# Route based on page
if st.session_state.page == "Dashboard":
    st.title("AutoML Platform Dashboard")
    st.write("Welcome to your automated machine learning platform.")
elif st.session_state.page == "Visualize":
    visualize.app()
elif st.session_state.page == "Feature Engineering":
    Feature_eng.app()
elif st.session_state.page == "ML Training":
    ML_training.app()
elif st.session_state.page == "DL Training":
    DL_training.app()
