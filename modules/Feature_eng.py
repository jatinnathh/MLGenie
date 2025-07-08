import streamlit as st
import pandas as pd 
import numpy as np 
from utils.shared import apply_common_settings

def app():
    st.write(" Feature Engineering ")
    uploaded_file = st.file_uploader("Upload a CSV/Excel", type=["csv", "xlsx", "xls"])
