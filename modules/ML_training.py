import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np


def local_css():
    st.markdown("""
        <style>
        .stButton > button {
            background-color: #ffffff;
            color: #2e2e2e;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #f8f9fa;
            border-color: #2e2e2e;
            transform: translateY(-1px);
        }
        .main-action-btn > button {
            background-color: #1f77b4;
            color: white;
            border: none;
        }
        .main-action-btn > button:hover {
            background-color: #145c8e;
        }
        .section-header {
            padding: 1rem 0;
            border-bottom: 1px solid #f0f0f0;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

def app():
    local_css()
    
    # Main Title
    st.markdown('<h1 class="main-title">ML Training</h1>', unsafe_allow_html=True)
    st.markdown("Train and evaluate machine learning models on your prepared dataset")
    
    # Check if data is available from Feature Engineering
    if 'feature_eng_data' in st.session_state and st.session_state.get('from_feature_eng', False):
        # Data forwarded from Feature Engineering
        df=st.session_state['feature_eng_data']
        st.markdown('<h2 class="section-header">Data from Feature Engineering</h2>', unsafe_allow_html=True)
        col1,col2,col3=st.columns(2)
        with col1:
            st.metric("Records",f"{df.shape[0]:,}")
        with col2:
            st.metric("Features", f"{df.shape[1]:,}")

        with col3:
            st.metric("Data Usage",f"{df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
        
        st.session_state['from_feature_eng'] = False
    else:
        # Direct access to ML Training - show recommendation
        st.markdown("""
        <div class="status-card warning">
            <h3>Feature Engineering Recommended</h3>
            <p>For optimal model performance, we recommend using the Feature Engineering module first to prepare your data.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Still allow manual data upload
        uploaded_file = st.file_uploader(
            "Upload Dataset", 
            type=["csv", "xlsx"],
            help="Upload a dataset or use Feature Engineering for better results"
        )
        
        if uploaded_file is None:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <div class="nav-card primary">
                    <h4>Feature Engineering</h4>
                    <p>Prepare and optimize your data</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Go to Feature Engineering", use_container_width=True, type="primary"):
                    st.session_state['current_page'] = 'feature_eng'
                    st.rerun()
            
            with col2:
                st.markdown("""
                <div class="nav-card secondary">
                    <h4>Data Visualization</h4>
                    <p>Explore and analyze your data</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Go to Visualize", use_container_width=True, type="secondary"):
                    st.session_state['current_page'] = 'visualize'
                    st.rerun()
            return
        
        # Load uploaded file
        try:
            if uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success("Dataset loaded successfully")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return
    

    with st.expander("Data Preview",expanded=False):
        col1,col2=st.columns(2)
        with col1:
            st.dataframe((df.head(10)), use_container_width=True)
        with col2:
            st.write("* Data information")
            info_df=pd.DataFrame({
                "columns": df.columns,
                "Type":df.dtypes.astype(str),
                'Non Null Count': df.notnull().sum(),
                'Missing Values': df.isnull().sum(),
            })
            st.dataframe(info_df, use_container_width=True)
        
if __name__ == "__main__":
    app()