import streamlit as st
from utils.shared import apply_common_settings
from utils.global_dashboard_tracker import track_model_training, log_activity

apply_common_settings()
def app():
    st.title("DL Training")
    
    # Placeholder for when DL training is implemented
    if st.button("Coming soon"):
        track_model_training('dl')
     
