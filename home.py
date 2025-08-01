import streamlit as st

st.set_page_config(
    page_title="MLGenie - AutoML Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

from modules import visualize, ML_training, DL_training, Feature_eng, dashboard
from utils.shared import apply_common_settings
from utils.sidebar import custom_sidebar

apply_common_settings()
custom_sidebar()

if st.session_state.page == "Dashboard":
    dashboard.app()
elif st.session_state.page == "Visualize":
    visualize.app()
elif st.session_state.page == "Feature Engineering":
    Feature_eng.app()
elif st.session_state.page == "ML Training":
    ML_training.app()
elif st.session_state.page == "DL Training":
    DL_training.app()
