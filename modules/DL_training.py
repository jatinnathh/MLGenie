import streamlit as st
from utils.shared import apply_common_settings

apply_common_settings()
def app():
    st.title("DL Training")
    st.write("Here you will train deep learning models.")
