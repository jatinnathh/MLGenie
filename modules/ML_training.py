import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

def load_professional_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 0.5rem;
    }
    
    /* Section Headers */
    .section-header {
        background: #f7fafc;
        border-left: 4px solid #4a5568;
        padding: 1rem 1.5rem;
        margin: 2rem 0 1.5rem 0;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .section-header h3 {
        margin: 0;
        color: #2d3748;
        font-size: 1.25rem;
        font-weight: 600;
    }
    
    /* Status Cards */
    .status-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .status-card.success {
        border-left: 4px solid #38a169;
        background: #f0fff4;
    }
    
    .status-card.warning {
        border-left: 4px solid #d69e2e;
        background: #fffaf0;
    }
    
    .status-card h3 {
        margin: 0 0 0.5rem 0;
        color: #2d3748;
        font-size: 1.1rem;
    }
    
    .status-card p {
        margin: 0;
        color: #4a5568;
        line-height: 1.5;
    }
    
    /* Model Cards */
    .model-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        transition: all 0.3s ease;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .model-card:hover {
        border-color: #4a5568;
        box-shadow: 0 4px 12px rgba(74, 85, 104, 0.15);
        transform: translateY(-2px);
    }
    
    .model-card h4 {
        margin: 0 0 0.5rem 0;
        color: #2d3748;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    .model-card p {
        margin: 0;
        color: #718096;
        font-size: 0.9rem;
    }
    
    /* Configuration Section */
    .config-section {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Results Section */
    .results-section {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .best-model-card {
        background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    
    .best-model-card h4 {
        margin: 0 0 0.5rem 0;
        color: white;
    }
    
    /* Navigation Cards */
    .nav-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .nav-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .nav-card.primary {
        border-left: 4px solid #4a5568;
    }
    
    .nav-card.secondary {
        border-left: 4px solid #38a169;
    }
    
    /* Buttons */
    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        border: none;
        font-size: 0.95rem;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
        color: white;
        box-shadow: 0 2px 8px rgba(74, 85, 104, 0.25);
    }
    
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(74, 85, 104, 0.35);
    }
    
    .stButton > button[kind="secondary"] {
        background: white;
        color: #4a5568;
        border: 1px solid #e2e8f0;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: #f7fafc;
        border-color: #cbd5e0;
        transform: translateY(-1px);
    }
    
    /* Form Controls */
    .stSelectbox > div > div {
        border-radius: 8px;
        border-color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    .stMultiSelect > div > div {
        border-radius: 8px;
        border-color: #e2e8f0;
    }
    
    .stRadio > div {
        gap: 1rem;
    }
    
    .stCheckbox > label {
        font-family: 'Inter', sans-serif;
        color: #4a5568;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
        border-radius: 4px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f7fafc;
        border-radius: 8px;
        font-weight: 500;
        color: #4a5568;
    }
    
    /* Success/Error Messages */
    .stAlert {
        border-radius: 8px;
        border: none;
        padding: 1rem;
    }
    
    .stSuccess {
        background: #f0fff4;
        color: #2f855a;
        border-left: 4px solid #38a169;
    }
    
    .stError {
        background: #fed7d7;
        color: #c53030;
        border-left: 4px solid #e53e3e;
    }
    
    .stInfo {
        background: #ebf8ff;
        color: #2b6cb0;
        border-left: 4px solid #3182ce;
    }
    
    .stWarning {
        background: #fffaf0;
        color: #b7791f;
        border-left: 4px solid #d69e2e;
    }
    </style>
    """, unsafe_allow_html=True)

def app():
    load_professional_css()
    
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
                    st.session_state['page'] = 'Feature Engineering'
                    st.rerun()
            
            with col2:
                st.markdown("""
                <div class="nav-card secondary">
                    <h4>Data Visualization</h4>
                    <p>Explore and analyze your data</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Go to Visualize", use_container_width=True, type="secondary"):
                    st.session_state['page'] = 'Visualize'
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
            st.write("**Dataset Information**")
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
