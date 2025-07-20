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
        df = st.session_state['feature_eng_data']
        st.markdown("""
        <div class="status-card success">
            <h3>Feature Engineered Data Loaded</h3>
            <p>Using optimized dataset from Feature Engineering module</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show data info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Records", f"{df.shape[0]:,}")
        with col2:
            st.metric("Features", f"{df.shape[1]:,}")
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Reset the flag
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
    
    # Data preview
    with st.expander("Data Preview", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df.head(), use_container_width=True)
        with col2:
            st.write("*Dataset Information*")
            info_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null': df.count(),
                'Missing': df.isnull().sum()
            })
            st.dataframe(info_df, use_container_width=True)

    # Model Configuration
    st.markdown('<div class="section-header"><h3>Model Configuration</h3></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            target_column = st.selectbox(
                "Target Column", 
                df.columns.tolist(),
                help="Select the column you want to predict"
            )
        
        with col2:
            problem_type = st.radio(
                "Problem Type",
                ["Classification", "Regression"],
                help="Choose based on your target variable type"
            )
        
        # Feature selection
        available_features = [col for col in df.columns if col != target_column]
        selected_features = st.multiselect(
            "Select Features (leave empty to use all features)",
            available_features,
            default=available_features[:10] if len(available_features) > 10 else available_features,
            help="Choose which features to use for training"
        )
        
        if not selected_features:
            selected_features = available_features
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Model Selection
    st.markdown('<div class="section-header"><h3>Model Selection</h3></div>', unsafe_allow_html=True)
    
    if problem_type == "Classification":
        model_options = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(random_state=42),
        }
    else:
        model_options = {
            "Random Forest": RandomForestRegressor(random_state=42),
            "Linear Regression": LinearRegression(),
        }
    
    selected_models = []
    model_cols = st.columns(len(model_options))
    
    for i, (model_name, model) in enumerate(model_options.items()):
        with model_cols[i]:
            st.markdown(f"""
            <div class="model-card">
                <h4>{model_name}</h4>
                <p>{'Tree-based ensemble method' if 'Random Forest' in model_name else 'Linear statistical model'}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.checkbox(f"Use {model_name}", value=True, key=f"model_{i}"):
                selected_models.append((model_name, model))

    # Training Parameters
    st.markdown('<div class="section-header"><h3>Training Parameters</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Split Ratio", 0.1, 0.5, 0.2, help="Proportion of data to use for testing")
    with col2:
        random_state = st.number_input("Random Seed", value=42, help="Set for reproducible results")

    # Train Models
    if st.button("Train Models", type="primary", use_container_width=True):
        if target_column and selected_models:
            train_models(df, target_column, selected_features, selected_models, test_size, random_state, problem_type)
        else:
            st.error("Please select a target column and at least one model")

def train_models(df, target_column, selected_features, selected_models, test_size, random_state, problem_type):
    try:
        # Prepare data
        X = df[selected_features]
        y = df[target_column]
        
        # Handle missing values
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
        
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "Unknown")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Train models
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (model_name, model) in enumerate(selected_models):
            status_text.text(f"Training {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            if problem_type == "Classification":
                score = accuracy_score(y_test, y_pred)
                metric_name = "Accuracy"
            else:
                score = mean_squared_error(y_test, y_pred)
                metric_name = "MSE"
            
            results.append({
                "Model": model_name,
                metric_name: score
            })
            
            progress_bar.progress((i + 1) / len(selected_models))
        
        status_text.text("Training completed successfully")
        progress_bar.empty()
        
        # Display results
        st.markdown('<div class="section-header"><h3>Training Results</h3></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        
        results_df = pd.DataFrame(results)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(results_df, use_container_width=True)
        
        with col2:
            best_model = results_df.loc[results_df[metric_name].idxmax() if problem_type == "Classification" else results_df[metric_name].idxmin()]
            
            st.markdown(f"""
            <div class="best-model-card">
                <h4>Best Performing Model</h4>
                <p><strong>{best_model['Model']}</strong></p>
                <p>{metric_name}: {best_model[metric_name]:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Training failed: {str(e)}")

if __name__ == "__main__":
    app()