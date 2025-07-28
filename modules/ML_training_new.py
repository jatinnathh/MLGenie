import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import time
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

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
    st.title("MLGenie ML Training")
    st.markdown('<div class="section-header"><h3>Data Input</h3></div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'model_results' not in st.session_state:
        st.session_state.model_results = []
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = []
    if 'training_complete' not in st.session_state:
        st.session_state.training_complete = False
    
    # Check for dataset
    df = None
    
    if 'feature_eng_data' in st.session_state and st.session_state.get('from_feature_eng', False):
        df = st.session_state['feature_eng_data']
        st.session_state['from_feature_eng'] = False
    elif 'df_feature_eng' in st.session_state:
        df = st.session_state['df_feature_eng']
    
    if df is not None:
        st.success("Dataset loaded successfully")
        col1, col2 = st.columns([3,1])
        with col2:
            if st.button("Load New Dataset", use_container_width=True):
                if 'df_feature_eng' in st.session_state:
                    del st.session_state.df_feature_eng
                if 'feature_eng_data' in st.session_state:
                    del st.session_state.feature_eng_data
                st.rerun()
    else:
        uploaded_file = st.file_uploader("Select your dataset (CSV/Excel)", type=["csv", "xlsx", "xls"], key="ml_training_uploader")
        if uploaded_file is None:
            st.warning("Please upload a CSV or Excel file.")
            return
        
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")
                return
                
            if df.empty:
                st.warning("The uploaded file is empty. Please upload a valid file.")
                return
                
            st.session_state.df_feature_eng = df.copy()
            st.success("Data uploaded and saved for ML Training!")
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return
        
    ############## DATA OVERVIEW ##################
    st.markdown('<div class="section-header"><h3>Dataset Overview</h3></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        missing_percent = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
        st.metric("Data Quality", f"{100-missing_percent:.1f}%")
    
    # Show basic data info
    if st.expander("View Dataset Sample"):
        st.dataframe(df.head())
    
    if st.expander("Dataset Info"):
        st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
        st.write("**Data Types:**")
        st.write(df.dtypes)
        
        if df.isnull().sum().sum() > 0:
            st.write("**Missing Values:**")
            missing_data = df.isnull().sum()
            st.write(missing_data[missing_data > 0])
    
    ############## MODEL TRAINING ##################
    st.markdown('<div class="section-header"><h3>Model Training</h3></div>', unsafe_allow_html=True)
    
    # Target selection
    target_column = st.selectbox("Select Target Column", df.columns.tolist())
    
    if target_column:
        # Determine problem type
        if df[target_column].dtype == 'object' or df[target_column].nunique() <= 20:
            problem_type = "Classification"
            st.info(f"**Problem Type:** {problem_type} (Target has {df[target_column].nunique()} unique values)")
        else:
            problem_type = "Regression"
            st.info(f"**Problem Type:** {problem_type} (Continuous target variable)")
        
        # Feature selection
        feature_columns = [col for col in df.columns if col != target_column]
        selected_features = st.multiselect(
            "Select Features for Training", 
            feature_columns, 
            default=feature_columns[:10] if len(feature_columns) > 10 else feature_columns
        )
        
        if selected_features:
            # Model selection
            if problem_type == "Classification":
                available_models = {
                    "Random Forest": RandomForestClassifier(random_state=42),
                    "Logistic Regression": LogisticRegression(random_state=42),
                    "Support Vector Machine": SVC(random_state=42),
                    "Decision Tree": DecisionTreeClassifier(random_state=42),
                    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
                }
            else:
                available_models = {
                    "Random Forest": RandomForestRegressor(random_state=42),
                    "Linear Regression": LinearRegression(),
                    "Support Vector Regression": SVR(),
                    "Decision Tree": DecisionTreeRegressor(random_state=42),
                    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
                }
            
            selected_model_names = st.multiselect(
                "Select Models to Train", 
                list(available_models.keys()),
                default=[list(available_models.keys())[0]]
            )
            
            # Training parameters
            test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
            
            if st.button("Train Models", type="primary", use_container_width=True):
                if selected_model_names:
                    # Prepare data
                    X = df[selected_features]
                    y = df[target_column]
                    
                    # Handle missing values
                    if X.isnull().sum().sum() > 0:
                        X = X.fillna(X.mean(numeric_only=True))
                        X = X.fillna(X.mode().iloc[0])
                    
                    # Encode categorical features
                    categorical_features = X.select_dtypes(include=['object']).columns
                    if len(categorical_features) > 0:
                        le = LabelEncoder()
                        for col in categorical_features:
                            X[col] = le.fit_transform(X[col].astype(str))
                    
                    # Encode target if classification
                    if problem_type == "Classification" and y.dtype == 'object':
                        le_target = LabelEncoder()
                        y = le_target.fit_transform(y)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    results = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, model_name in enumerate(selected_model_names):
                        status_text.text(f"Training {model_name}...")
                        
                        model = available_models[model_name]
                        
                        # Train model
                        if model_name in ["Support Vector Machine", "Support Vector Regression", "Logistic Regression"]:
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)
                        else:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        if problem_type == "Classification":
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, average='weighted')
                            recall = recall_score(y_test, y_pred, average='weighted')
                            f1 = f1_score(y_test, y_pred, average='weighted')
                            
                            results.append({
                                'Model': model_name,
                                'Accuracy': f"{accuracy:.4f}",
                                'Precision': f"{precision:.4f}",
                                'Recall': f"{recall:.4f}",
                                'F1-Score': f"{f1:.4f}"
                            })
                        else:
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            
                            results.append({
                                'Model': model_name,
                                'RMSE': f"{rmse:.4f}",
                                'MAE': f"{mae:.4f}",
                                'R²': f"{r2:.4f}",
                                'MSE': f"{mse:.4f}"
                            })
                        
                        progress_bar.progress((i + 1) / len(selected_model_names))
                    
                    status_text.text("Training completed!")
                    
                    # Display results
                    st.markdown('<div class="section-header"><h3>Training Results</h3></div>', unsafe_allow_html=True)
                    
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Store results in session state
                    st.session_state.model_results = results_df
                    st.session_state.training_complete = True
                    
                    # Best model recommendation
                    if problem_type == "Classification":
                        best_model = results_df.loc[results_df['Accuracy'].astype(float).idxmax(), 'Model']
                        best_score = results_df.loc[results_df['Accuracy'].astype(float).idxmax(), 'Accuracy']
                        st.success(f"🏆 **Best Model:** {best_model} (Accuracy: {best_score})")
                    else:
                        best_model = results_df.loc[results_df['R²'].astype(float).idxmax(), 'Model']
                        best_score = results_df.loc[results_df['R²'].astype(float).idxmax(), 'R²']
                        st.success(f"🏆 **Best Model:** {best_model} (R²: {best_score})")
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "Download Results",
                        csv,
                        file_name="model_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                else:
                    st.warning("Please select at least one model to train.")
            
            # Show previous results if available
            if st.session_state.training_complete and 'model_results' in st.session_state:
                st.markdown('<div class="section-header"><h3>Previous Results</h3></div>', unsafe_allow_html=True)
                st.dataframe(st.session_state.model_results, use_container_width=True)
