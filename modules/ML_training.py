import streamlit as st
import pandas as pd

def app():
    st.title("ML Training Module")

    # Sample DataFrame
    df = pd.DataFrame({
        'Feature 1': [1, 2, 3],
        'Feature 2': ['A', 'B', 'C']
    })

    # Editable data table
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="ml_editor")

    st.write("Edited Data:")
    st.write(edited_df)
