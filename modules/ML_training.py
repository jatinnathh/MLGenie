import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import pickle
import json
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def local_css():
    st.markdown("""
        <style>
        .stButton > button {
            background-color: #2d2d2d;
            color: #ffffff;
            border: 1px solid #404040;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #404040;
            border-color: #1f77b4;
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
            border-bottom: 1px solid #404040;
            margin-bottom: 1rem;
        }
        
        /* Model Selection Cards */
        .model-card {
            border: 2px solid #404040;
            border-radius: 12px;
            padding: 25px;
            margin: 15px 0;
            background: #1e1e1e;
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        .model-card:hover {
            border-color: #1f77b4;
            background: #2a2a2a;
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(31, 119, 180, 0.3);
        }
        .model-card.selected {
            border-color: #1f77b4;
            background: #2d3748;
            box-shadow: 0 0 0 3px rgba(31, 119, 180, 0.2);
        }
        .model-name {
            font-weight: bold;
            font-size: 1.3rem;
            color: #ffffff;
            margin-bottom: 12px;
        }
        .model-description {
            color: #b0b0b0;
            font-size: 1rem;
            line-height: 1.6;
            margin-bottom: 15px;
        }
        .model-complexity {
            display: inline-block;
            padding: 6px 16px;
            border-radius: 25px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 15px;
        }
        .complexity-low { background: #2d5a27; color: #a3d977; }
        .complexity-medium { background: #5a4a2d; color: #f7d794; }
        .complexity-high { background: #5a2d2d; color: #ff7675; }
        
        .model-details {
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #404040;
        }
        .detail-item {
            margin-bottom: 12px;
            font-size: 0.95rem;
            line-height: 1.5;
        }
        .detail-label {
            font-weight: 600;
            color: #e2e8f0;
            margin-bottom: 4px;
        }
        .detail-content {
            color: #a0a0a0;
        }
        .pros { color: #81c784; }
        .cons { color: #f48fb1; }
        
        /* Training Progress */
        .training-container {
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            border-radius: 16px;
            padding: 30px;
            margin: 20px 0;
            border: 1px solid #404040;
        }
        .progress-step {
            display: flex;
            align-items: center;
            margin: 15px 0;
            padding: 15px;
            background: #2a2a2a;
            border-radius: 8px;
            border-left: 4px solid #404040;
        }
        .progress-step.active {
            border-left-color: #1f77b4;
            background: #1e2a3a;
        }
        .progress-step.completed {
            border-left-color: #28a745;
            background: #1e2f1e;
        }
        .step-icon {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background: #404040;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 15px;
            font-weight: bold;
            color: #ffffff;
        }
        .step-icon.active { background: #1f77b4; color: white; }
        .step-icon.completed { background: #28a745; color: white; }
        
        /* Metrics Display */
        .metric-card {
            background: #1e1e1e;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #404040;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #1f77b4;
            margin: 10px 0;
        }
        .metric-label {
            color: #b0b0b0;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Animated Progress Bar */
        .custom-progress {
            width: 100%;
            height: 12px;
            background: #404040;
            border-radius: 6px;
            overflow: hidden;
            margin: 15px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #1f77b4, #4dabf7);
            border-radius: 6px;
            transition: width 0.5s ease;
            position: relative;
        }
        .progress-fill::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            bottom: 0;
            right: 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            animation: shimmer 2s infinite;
        }
        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        </style>
    """, unsafe_allow_html=True)

def create_model_card(name, description, complexity, use_case, pros, cons, is_selected=False):
    """Create an interactive model selection card"""
    selected_class = "selected" if is_selected else ""
    return f"""
    <div class="model-card {selected_class}">
        <div class="model-name">{name}</div>
        <div class="model-description">{description}</div>
        <div class="model-complexity complexity-{complexity.lower()}">{complexity} Complexity</div>
        <div class="model-details">
            <div class="detail-item">
                <div class="detail-label">Best Use Cases:</div>
                <div class="detail-content">{use_case}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label pros">Advantages:</div>
                <div class="detail-content">{pros}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label cons">Considerations:</div>
                <div class="detail-content">{cons}</div>
            </div>
        </div>
    </div>
    """

def create_training_step(step_num, title, description, status="pending"):
    """Create a training step indicator"""
    icon_content = "✓" if status == "completed" else "⏳" if status == "active" else str(step_num)
    return f"""
    <div class="progress-step {status}">
        <div class="step-icon {status}">{icon_content}</div>
        <div>
            <div style="font-weight: bold; margin-bottom: 5px;">{title}</div>
            <div style="color: #666; font-size: 0.9rem;">{description}</div>
        </div>
    </div>
    """

def create_animated_progress(progress, label):
    """Create an animated progress bar"""
    return f"""
    <div style="margin: 20px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="font-weight: 500;">{label}</span>
            <span style="color: #1f77b4; font-weight: bold;">{progress:.1f}%</span>
        </div>
        <div class="custom-progress">
            <div class="progress-fill" style="width: {progress}%;"></div>
        </div>
    </div>
    """

def get_model_info():
    """Get comprehensive model information for selection"""
    return {
        "classification": {
            "Random Forest": {
                "description": "Ensemble method using multiple decision trees for robust predictions",
                "complexity": "Medium",
                "use_case": "Structured data, feature importance analysis",
                "pros": "Handles overfitting well, feature importance",
                "cons": "Can be slow on large datasets",
                "estimator": RandomForestClassifier(random_state=42, n_estimators=100)
            },
            "Gradient Boosting": {
                "description": "Sequential ensemble that corrects previous model errors",
                "complexity": "High",
                "use_case": "Complex patterns, high accuracy requirements",
                "pros": "Excellent performance, handles missing values",
                "cons": "Prone to overfitting, requires tuning",
                "estimator": GradientBoostingClassifier(random_state=42, n_estimators=100)
            },
            "Logistic Regression": {
                "description": "Linear model with probabilistic outputs for classification",
                "complexity": "Low",
                "use_case": "Linear relationships, interpretability needed",
                "pros": "Fast, interpretable, probabilistic outputs",
                "cons": "Assumes linear relationships",
                "estimator": LogisticRegression(random_state=42, max_iter=1000)
            },
            "Support Vector Machine": {
                "description": "Finds optimal decision boundary using support vectors",
                "complexity": "High",
                "use_case": "High-dimensional data, non-linear patterns",
                "pros": "Effective in high dimensions, memory efficient",
                "cons": "Slow on large datasets, requires scaling",
                "estimator": SVC(random_state=42, probability=True)
            },
            "Neural Network": {
                "description": "Multi-layer perceptron for complex pattern recognition",
                "complexity": "High",
                "use_case": "Complex non-linear patterns, large datasets",
                "pros": "Learns complex patterns, flexible architecture",
                "cons": "Requires more data, harder to interpret",
                "estimator": MLPClassifier(random_state=42, max_iter=200, hidden_layer_sizes=(100, 50))
            },
            "Decision Tree": {
                "description": "Tree-based model with interpretable decision rules",
                "complexity": "Low",
                "use_case": "Rule-based decisions, interpretability",
                "pros": "Highly interpretable, handles mixed data types",
                "cons": "Prone to overfitting, unstable",
                "estimator": DecisionTreeClassifier(random_state=42)
            }
        },
        "regression": {
            "Random Forest": {
                "description": "Ensemble of decision trees for robust regression predictions",
                "complexity": "Medium",
                "use_case": "Non-linear relationships, feature importance",
                "pros": "Reduces overfitting, feature importance",
                "cons": "Less interpretable than linear models",
                "estimator": RandomForestRegressor(random_state=42, n_estimators=100)
            },
            "Gradient Boosting": {
                "description": "Sequential ensemble optimizing for prediction accuracy",
                "complexity": "High",
                "use_case": "Complex patterns, maximum accuracy",
                "pros": "High accuracy, handles interactions well",
                "cons": "Computationally expensive, requires tuning",
                "estimator": GradientBoostingRegressor(random_state=42, n_estimators=100)
            },
            "Linear Regression": {
                "description": "Simple linear relationship modeling with interpretability",
                "complexity": "Low",
                "use_case": "Linear relationships, baseline model",
                "pros": "Fast, interpretable, well understood",
                "cons": "Limited to linear relationships",
                "estimator": LinearRegression()
            },
            "Support Vector Regression": {
                "description": "SVR with kernel tricks for non-linear regression",
                "complexity": "High",
                "use_case": "Non-linear patterns, robust predictions",
                "pros": "Effective with high dimensions, robust",
                "cons": "Requires parameter tuning, scaling needed",
                "estimator": SVR()
            },
            "Neural Network": {
                "description": "Multi-layer network for complex regression tasks",
                "complexity": "High",
                "use_case": "Complex non-linear relationships",
                "pros": "Learns complex patterns, flexible",
                "cons": "Requires more data, black box",
                "estimator": MLPRegressor(random_state=42, max_iter=200, hidden_layer_sizes=(100, 50))
            },
            "Ridge Regression": {
                "description": "Linear regression with L2 regularization",
                "complexity": "Low",
                "use_case": "Linear relationships with regularization",
                "pros": "Prevents overfitting, handles multicollinearity",
                "cons": "Still linear, doesn't do feature selection",
                "estimator": Ridge(random_state=42)
            }
        }
    }

class DataPreprocessor:
    """Unified preprocessing pipeline for consistent feature handling"""
    
    def __init__(self):
        self.label_encoders = {}
        self.feature_scaler = None
        self.target_scaler = None
        self.target_transform_config = {'type': 'none'}
        self.feature_names = []
        self.categorical_features = []
        self.numerical_features = []
        
    def fit_transform_features(self, X, y=None, target_scaling='auto'):
        """Fit preprocessing pipeline and transform features"""
        self.feature_names = list(X.columns)
        
        # Identify categorical and numerical features
        self.categorical_features = list(X.select_dtypes(include=['object', 'category']).columns)
        self.numerical_features = list(X.select_dtypes(include=[np.number]).columns)
        
        X_processed = X.copy()
        
        # Handle missing values first
        for col in X_processed.columns:
            if X_processed[col].isnull().sum() > 0:
                if col in self.numerical_features:
                    X_processed[col] = X_processed[col].fillna(X_processed[col].median())
                else:
                    X_processed[col] = X_processed[col].fillna(X_processed[col].mode().iloc[0] if not X_processed[col].mode().empty else 'Unknown')
        
        # Encode categorical features
        for col in self.categorical_features:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            self.label_encoders[col] = le
        
        # Scale features
        self.feature_scaler = StandardScaler()
        X_scaled = self.feature_scaler.fit_transform(X_processed)
        X_scaled = pd.DataFrame(X_scaled, columns=X_processed.columns, index=X_processed.index)
        
        # Handle target scaling for regression
        y_scaled = y
        if y is not None and target_scaling != 'none':
            y_scaled = self._fit_transform_target(y, target_scaling)
        
        return X_scaled, y_scaled
    
    def transform_features(self, X):
        """Transform new data using fitted preprocessors"""
        X_processed = X.copy()
        
        # Handle missing values
        for col in X_processed.columns:
            if X_processed[col].isnull().sum() > 0:
                if col in self.numerical_features:
                    X_processed[col] = X_processed[col].fillna(0)  # Use 0 for new data
                else:
                    X_processed[col] = X_processed[col].fillna('Unknown')
        
        # Apply label encoding
        for col in self.categorical_features:
            if col in X_processed.columns and col in self.label_encoders:
                try:
                    X_processed[col] = self.label_encoders[col].transform(X_processed[col].astype(str))
                except ValueError:
                    # Handle unseen categories
                    X_processed[col] = 0
        
        # Ensure all columns are present and in correct order
        for col in self.feature_names:
            if col not in X_processed.columns:
                X_processed[col] = 0
        
        X_processed = X_processed[self.feature_names]
        
        # Convert to numeric
        for col in X_processed.columns:
            if X_processed[col].dtype == 'object':
                X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce').fillna(0)
        
        # Scale features
        if self.feature_scaler:
            X_scaled = self.feature_scaler.transform(X_processed)
            X_scaled = pd.DataFrame(X_scaled, columns=X_processed.columns, index=X_processed.index)
            return X_scaled
        
        return X_processed
    
    def _fit_transform_target(self, y, scaling_method):
        """Handle target variable scaling"""
        if scaling_method == 'auto':
            # Auto-detect need for scaling
            if y.std() > 1000 or abs(y.mean()) > 10000:
                scaling_method = 'log'
            else:
                scaling_method = 'none'
        
        if scaling_method == 'log':
            if (y <= 0).any():
                offset = abs(y.min()) + 1
                y_scaled = np.log(y + offset)
                self.target_transform_config = {'type': 'log', 'offset': offset}
            else:
                y_scaled = np.log(y)
                self.target_transform_config = {'type': 'log', 'offset': 0}
            return pd.Series(y_scaled, index=y.index)
            
        elif scaling_method == 'standard':
            self.target_scaler = StandardScaler()
            y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
            self.target_transform_config = {'type': 'standard'}
            return pd.Series(y_scaled, index=y.index)
            
        elif scaling_method == 'minmax':
            self.target_scaler = MinMaxScaler()
            y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
            self.target_transform_config = {'type': 'minmax'}
            return pd.Series(y_scaled, index=y.index)
        
        self.target_transform_config = {'type': 'none'}
        return y
    
    def inverse_transform_target(self, y_scaled):
        """Apply inverse transformation to target predictions"""
        if self.target_transform_config['type'] == 'log':
            offset = self.target_transform_config.get('offset', 0)
            return np.exp(y_scaled) - offset
        elif self.target_transform_config['type'] in ['standard', 'minmax']:
            if self.target_scaler:
                if hasattr(y_scaled, 'reshape'):
                    return self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
                else:
                    return self.target_scaler.inverse_transform([[y_scaled]])[0][0]
        return y_scaled

def train_model_real(model_name, X_train, y_train, X_test, y_test, problem_type, max_iterations=100):
    """FIXED: Real training with actual model fitting and progress tracking"""
    
    model_info = get_model_info()
    problem_key = problem_type.lower()
    
    if problem_key not in model_info or model_name not in model_info[problem_key]:
        st.error(f"Model {model_name} not found for {problem_type}")
        return None, None
    
    model = model_info[problem_key][model_name]["estimator"]
    
    # Create progress containers
    progress_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    start_time = time.time()
    
    try:
        # For neural networks, do incremental training
        if "Neural Network" in model_name:
            model.max_iter = 1
            model.warm_start = True
            best_score = -np.inf
            patience_counter = 0
            patience = 10
            
            for iteration in range(1, max_iterations + 1):
                # Incremental training
                if iteration == 1:
                    model.fit(X_train, y_train)
                else:
                    model.max_iter += 1
                    model.fit(X_train, y_train)
                
                # Calculate current performance
                y_pred = model.predict(X_test)
                
                if problem_type.lower() == "classification":
                    current_score = accuracy_score(y_test, y_pred)
                    score_name = "Accuracy"
                else:
                    current_score = r2_score(y_test, y_pred)
                    score_name = "R²"
                
                # Early stopping logic
                if current_score > best_score:
                    best_score = current_score
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Update progress every 10 iterations
                if iteration % 10 == 0 or iteration <= 10 or iteration >= max_iterations - 5:
                    progress = (iteration / max_iterations) * 100
                    elapsed_time = time.time() - start_time
                    eta = (elapsed_time / iteration) * (max_iterations - iteration)
                    
                    with progress_placeholder.container():
                        st.markdown(create_animated_progress(progress, f"Training {model_name}"), unsafe_allow_html=True)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Iteration", f"{iteration}/{max_iterations}")
                        with col2:
                            st.metric("Progress", f"{progress:.1f}%")
                        with col3:
                            st.metric(f"Best {score_name}", f"{best_score:.3f}")
                        with col4:
                            st.metric("ETA", f"{eta:.1f}s")
                
                # Early stopping
                if patience_counter >= patience and iteration > 20:
                    st.info(f"⏹️ Early stopping at iteration {iteration}")
                    break
                
                time.sleep(0.01)  # Small delay for visual effect
                
        else:
            # For other models, do progressive training
            if hasattr(model, 'n_estimators'):
                original_n_estimators = model.n_estimators
                
                for step in range(1, 11):  # 10 progressive steps
                    current_n_estimators = max(1, (original_n_estimators * step) // 10)
                    model.n_estimators = current_n_estimators
                    
                    # Real training
                    model.fit(X_train, y_train)
                    
                    # Calculate metrics
                    y_pred = model.predict(X_test)
                    
                    if problem_type.lower() == "classification":
                        current_score = accuracy_score(y_test, y_pred)
                        score_name = "Accuracy"
                    else:
                        current_score = r2_score(y_test, y_pred)
                        score_name = "R²"
                    
                    progress = (step / 10) * 100
                    elapsed_time = time.time() - start_time
                    eta = (elapsed_time / step) * (10 - step)
                    
                    with progress_placeholder.container():
                        st.markdown(create_animated_progress(progress, f"Training {model_name}"), unsafe_allow_html=True)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Estimators", f"{current_n_estimators}/{original_n_estimators}")
                        with col2:
                            st.metric("Progress", f"{progress:.1f}%")
                        with col3:
                            st.metric(f"Current {score_name}", f"{current_score:.3f}")
                        with col4:
                            st.metric("ETA", f"{eta:.1f}s")
                    
                    time.sleep(0.1)
                
                # Final training with full estimators
                model.n_estimators = original_n_estimators
                model.fit(X_train, y_train)
                
            else:
                # Direct training for simple models
                with progress_placeholder.container():
                    st.markdown(create_animated_progress(50, f"Training {model_name}"), unsafe_allow_html=True)
                    st.info(f"Training {model_name}...")
                
                model.fit(X_train, y_train)
                
                with progress_placeholder.container():
                    st.markdown(create_animated_progress(100, f"Training {model_name}"), unsafe_allow_html=True)
        
        # Final evaluation with proper target inverse transformation
        st.success(f"{model_name} training completed!")
        
        y_pred = model.predict(X_test)
        
        # Get preprocessor for inverse transforms
        preprocessor = st.session_state.get('preprocessor')
        
        if problem_type.lower() == "classification":
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Display final metrics
            with metrics_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Accuracy</div>
                        <div class="metric-value">{accuracy:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Precision</div>
                        <div class="metric-value">{precision:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Recall</div>
                        <div class="metric-value">{recall:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">F1-Score</div>
                        <div class="metric-value">{f1:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            results = {
                'Model': model_name,
                'Accuracy': f"{accuracy:.4f}",
                'Precision': f"{precision:.4f}",
                'Recall': f"{recall:.4f}",
                'F1-Score': f"{f1:.4f}",
                'Training_Time': f"{time.time() - start_time:.2f}s"
            }
            
        else:  # Regression
            # Apply inverse transformation for meaningful metrics
            if preprocessor and preprocessor.target_transform_config['type'] != 'none':
                y_test_orig = preprocessor.inverse_transform_target(y_test)
                y_pred_orig = preprocessor.inverse_transform_target(y_pred)
                
                mse = mean_squared_error(y_test_orig, y_pred_orig)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test_orig, y_pred_orig)
                r2 = r2_score(y_test_orig, y_pred_orig)
                
                st.info("Metrics calculated on original scale")
            else:
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
            
            # Display final metrics
            with metrics_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">R² Score</div>
                        <div class="metric-value">{r2:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">RMSE</div>
                        <div class="metric-value">{rmse:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">MAE</div>
                        <div class="metric-value">{mae:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">MSE</div>
                        <div class="metric-value">{mse:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            results = {
                'Model': model_name,
                'RMSE': f"{rmse:.2f}",
                'MAE': f"{mae:.2f}",
                'R²': f"{r2:.4f}",
                'MSE': f"{mse:.1f}",
                'Training_Time': f"{time.time() - start_time:.2f}s"
            }
        
        total_time = time.time() - start_time
        st.success(f"Training completed in {total_time:.2f} seconds")
        
        return results, model
        
    except Exception as e:
        st.error(f"Training failed for {model_name}: {str(e)}")
        return None, None

def make_prediction_consistent(model_name, input_data, training_config):
    """FIXED: Consistent prediction with unified preprocessing"""
    try:
        # Get trained model and preprocessor
        if model_name not in st.session_state.get('trained_models', {}):
            st.error(f"Model {model_name} not found")
            return None
        
        model = st.session_state.trained_models[model_name]
        preprocessor = st.session_state.get('preprocessor')
        
        if not preprocessor:
            st.error("Preprocessor not found")
            return None
        
        # Convert input to DataFrame
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            st.error("Input data must be a dictionary")
            return None
        
        st.info(f"Processing input with {len(input_df.columns)} features")
        
        # Apply consistent preprocessing
        X_processed = preprocessor.transform_features(input_df)
        
        st.success(f"Features processed: {X_processed.shape}")
        
        # Make prediction
        prediction = model.predict(X_processed.values)
        
        # Handle output based on problem type
        problem_type = training_config.get('problem_type', '')
        
        if problem_type == "Classification":
            # Handle target decoding if needed
            target_encoder = st.session_state.get('target_encoder')
            if target_encoder:
                try:
                    decoded_prediction = target_encoder.inverse_transform(prediction)
                    prediction = decoded_prediction
                except:
                    pass
            
            # Get confidence if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_processed.values)
                confidence = max(probabilities[0])
                return f"Prediction: {prediction[0]} (Confidence: {confidence:.3f})"
            else:
                return f"Prediction: {prediction[0]}"
                
        else:  # Regression
            # Apply inverse target transformation
            if preprocessor.target_transform_config['type'] != 'none':
                final_prediction = preprocessor.inverse_transform_target(prediction[0])
                st.info(f"Applied inverse {preprocessor.target_transform_config['type']} transform")
            else:
                final_prediction = prediction[0]
            
            return f"Prediction: {final_prediction:.4f}"
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

def app():
    local_css()
    st.title("AutoML Studio ")
    

    with st.sidebar:
        st.markdown("### Controls")
        if st.button("Reset All", help="Clear all data and start fresh"):
            for key in list(st.session_state.keys()):
                if key.startswith(('training_', 'model_', 'selected_', 'preprocessor')):
                    del st.session_state[key]
            st.success("Session reset!")
            st.rerun()
        
        st.markdown("---")
        if st.checkbox("Debug Mode"):
            st.write("**Session State:**")
            st.write(f"• Step: {st.session_state.get('training_step', 1)}")
            st.write(f"• Models: {len(st.session_state.get('trained_models', {}))}")
            if 'preprocessor' in st.session_state:
                prep = st.session_state.preprocessor
                st.write(f"• Features: {len(prep.feature_names)}")
                st.write(f"• Target transform: {prep.target_transform_config['type']}")
    
    # Initialize session state
    if 'training_step' not in st.session_state:
        st.session_state.training_step = 1
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'model_results' not in st.session_state:
        st.session_state.model_results = []
    if 'selected_models_for_training' not in st.session_state:
        st.session_state.selected_models_for_training = []
    if 'training_config' not in st.session_state:
        st.session_state.training_config = {}
    
    # Progress Steps Visualization
    st.markdown("### Training Pipeline Progress")
    
    steps_html = f"""
    <div style="margin: 20px 0;">
        {create_training_step(1, "Data Upload", "Load and validate your dataset", 
                            "completed" if st.session_state.training_step > 1 else "active")}
        {create_training_step(2, "Data Analysis", "Explore data characteristics and quality", 
                            "completed" if st.session_state.training_step > 2 else "active" if st.session_state.training_step == 2 else "pending")}
        {create_training_step(3, "Model Selection", "Choose algorithms for your problem", 
                            "completed" if st.session_state.training_step > 3 else "active" if st.session_state.training_step == 3 else "pending")}
        {create_training_step(4, "Training Configuration", "Set hyperparameters and training options", 
                            "completed" if st.session_state.training_step > 4 else "active" if st.session_state.training_step == 4 else "pending")}
        {create_training_step(5, "Model Training", "Train and evaluate selected models", 
                            "completed" if st.session_state.training_step > 5 else "active" if st.session_state.training_step == 5 else "pending")}
        {create_training_step(6, "Results & Deployment", "Review results and export models", 
                            "active" if st.session_state.training_step == 6 else "pending")}
    </div>
    """
    st.markdown(steps_html, unsafe_allow_html=True)
    
    # Step 1: Data Upload
    if st.session_state.training_step == 1:
        st.markdown('<div class="section-header"><h3>Step 1: Data Upload</h3></div>', unsafe_allow_html=True)
        
        # Check for existing dataset
        df = None
        
        if 'df_feature_eng' in st.session_state and st.session_state['df_feature_eng'] is not None:
            df = st.session_state['df_feature_eng']
            st.success("Dataset loaded from previous session!")
        elif 'feature_eng_data' in st.session_state and st.session_state['feature_eng_data'] is not None:
            df = st.session_state['feature_eng_data']
            st.success("Dataset loaded from Feature Engineering!")
        
        if df is not None:
            col1, col2 = st.columns([3,1])
            with col1:
                st.info(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
            with col2:
                if st.button("Load Different Dataset", use_container_width=True):
                    if 'df_feature_eng' in st.session_state:
                        del st.session_state.df_feature_eng
                    if 'feature_eng_data' in st.session_state:
                        del st.session_state.feature_eng_data
                    st.rerun()
            
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("Continue to Data Analysis →", type="primary", use_container_width=True):
                st.session_state.training_step = 2
                st.rerun()
        else:
            uploaded_file = st.file_uploader(
                "Upload your dataset (CSV/Excel)", 
                type=["csv", "xlsx", "xls"], 
                help="Upload a clean dataset ready for machine learning"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    if df.empty:
                        st.error("The uploaded file is empty. Please upload a valid file.")
                        return
                    
                    st.session_state.df_feature_eng = df.copy()
                    st.success("Data uploaded successfully!")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    if st.button("Continue to Data Analysis →", type="primary", use_container_width=True):
                        st.session_state.training_step = 2
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
            else:
                st.info("Please upload your dataset to begin training")
        return
    
    # Get dataset for subsequent steps
    df = st.session_state.get('df_feature_eng')
    if df is None:
        df = st.session_state.get('feature_eng_data')

    
    if df is None and st.session_state.training_step > 1:
        st.error("No dataset found. Please return to Step 1.")
        if st.button("← Back to Data Upload"):
            st.session_state.training_step = 1
            st.rerun()
        return
    
    # Step 2: Data Analysis
    if st.session_state.training_step == 2:
        st.markdown('<div class="section-header"><h3>Step 2: Data Analysis</h3></div>', unsafe_allow_html=True)
        
        # Dataset overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Rows</div>
                <div class="metric-value">{df.shape[0]:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Features</div>
                <div class="metric-value">{df.shape[1]}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            missing_percent = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Data Quality</div>
                <div class="metric-value">{100-missing_percent:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Numeric Features</div>
                <div class="metric-value">{numeric_cols}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Data tabs
        tab1, tab2, tab3 = st.tabs(["Data Sample", "Data Info", "Data Quality"])
        
        with tab1:
            st.dataframe(df.head(10), use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Data Types:**")
                dtype_df = pd.DataFrame({
                    'Column': df.dtypes.index,
                    'Type': df.dtypes.values,
                    'Non-Null Count': df.count().values,
                    'Null Count': df.isnull().sum().values
                })
                st.dataframe(dtype_df, use_container_width=True)
            
            with col2:
                st.write("**Statistical Summary:**")
                st.dataframe(df.describe(), use_container_width=True)
        
        with tab3:
            if df.isnull().sum().sum() > 0:
                st.warning("Missing values detected")
                missing_data = df.isnull().sum()
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Missing %': (missing_data.values / len(df) * 100).round(2)
                })
                missing_df = missing_df[missing_df['Missing Count'] > 0]
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("No missing values found!")
        
        # Navigation
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Data Upload"):
                st.session_state.training_step = 1
                st.rerun()
        with col2:
            if st.button("Continue to Model Selection →", type="primary", use_container_width=True):
                st.session_state.training_step = 3
                st.rerun()
    
    # Step 3: Model Selection
    elif st.session_state.training_step == 3:
        st.markdown('<div class="section-header"><h3>Step 3: Model Selection</h3></div>', unsafe_allow_html=True)
        
        # Target selection
        target_column = st.selectbox(
            "Select Target Column (What do you want to predict?)", 
            df.columns.tolist(),
            help="Choose the column you want your model to predict"
        )
        
        if target_column:
            # Enhanced problem type detection
            unique_values = df[target_column].nunique()
            target_dtype = df[target_column].dtype
            
            if target_dtype == 'object':
                problem_type = "Classification"
                st.info(f"**Problem Type:** {problem_type} (Categorical target with {unique_values} classes)")
            elif unique_values <= 10 and target_dtype in ['int64', 'int32']:
                user_choice = st.radio(
                    f"Target has {unique_values} unique values. Choose problem type:",
                    ["Classification", "Regression"],
                    help="Classification: Categories/Classes. Regression: Continuous numbers."
                )
                problem_type = user_choice
            else:
                problem_type = "Regression"
                st.info(f"**Problem Type:** {problem_type} (Continuous target with {unique_values} unique values)")
            
            # Store configuration
            st.session_state.training_config.update({
                'target_column': target_column,
                'problem_type': problem_type
            })
            
            # Feature selection
            feature_columns = [col for col in df.columns if col != target_column]
            st.markdown("### Feature Selection")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_features = st.multiselect(
                    "Select features for training", 
                    feature_columns, 
                    default=feature_columns,
                    help="Choose input features for the model"
                )
            with col2:
                if st.button("Select All", use_container_width=True):
                    st.session_state.training_config['selected_features'] = feature_columns
                    st.rerun()
            
            st.session_state.training_config['selected_features'] = selected_features
            
            if selected_features:
                # Model selection
                st.markdown("### Choose Models to Train")
                
                model_info = get_model_info()
                available_models = model_info[problem_type.lower()]
                
                selected_models = []
                cols = st.columns(2)
                
                for i, (model_name, info) in enumerate(available_models.items()):
                    with cols[i % 2]:
                        is_selected = st.checkbox(
                            f"{model_name}", 
                            key=f"model_{model_name}",
                            help=f"{info['description']} - {info['complexity']} complexity"
                        )
                        
                        if is_selected:
                            selected_models.append(model_name)
                        
                        # Show model card
                        st.markdown(
                            create_model_card(
                                model_name, 
                                info["description"],
                                info["complexity"],
                                info["use_case"],
                                info["pros"],
                                info["cons"],
                                is_selected
                            ), 
                            unsafe_allow_html=True
                        )
                
                st.session_state.selected_models_for_training = selected_models
                
                # Navigation
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("← Back to Data Analysis"):
                        st.session_state.training_step = 2
                        st.rerun()
                with col2:
                    if selected_models:
                        if st.button("Configure Training →", type="primary", use_container_width=True):
                            st.session_state.training_step = 4
                            st.rerun()
                    else:
                        st.warning("Please select at least one model to continue")
    
    # Step 4: Training Configuration
    elif st.session_state.training_step == 4:
        st.markdown('<div class="section-header"><h3>Step 4: Training Configuration</h3></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Data Splitting")
            test_size = st.slider("Test Size (%)", 10, 40, 20, step=5) / 100
            
            st.markdown("### Training Parameters")
            max_iterations = st.number_input("Max Iterations", min_value=10, max_value=500, value=100, step=10,
                                           help="Maximum training iterations")
            
        with col2:
            st.markdown("### Target Scaling (Regression)")
            target_scaling = st.selectbox(
                "Target Scaling Method",
                ["auto", "log", "standard", "minmax", "none"],
                help="Auto will choose the best method based on data"
            )
            
            st.markdown("### Validation")
            enable_cv = st.checkbox("Cross Validation", value=True,
                                  help="Use cross-validation for robust evaluation")
            
        # Save configuration
        st.session_state.training_config.update({
            'test_size': test_size,
            'max_iterations': max_iterations,
            'target_scaling': target_scaling,
            'enable_cv': enable_cv
        })
        
        # Configuration summary
        with st.expander("Configuration Summary", expanded=True):
            config = st.session_state.training_config
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Data Configuration:**")
                st.write(f"• Target: `{config.get('target_column', 'Not set')}`")
                st.write(f"• Problem: `{config.get('problem_type', 'Not set')}`")
                st.write(f"• Features: `{len(config.get('selected_features', []))}`")
                st.write(f"• Test Size: `{test_size*100:.0f}%`")
                
            with col2:
                st.write("**Selected Models:**")
                for model in st.session_state.selected_models_for_training:
                    st.write(f"• {model}")
            
            with col3:
                st.write("**Training Settings:**")
                st.write(f"• Max Iterations: `{max_iterations}`")
                st.write(f"• Target Scaling: `{target_scaling}`")
                st.write(f"• Cross Validation: `{'Yes' if enable_cv else 'No'}`")
        
        # Navigation
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Model Selection"):
                st.session_state.training_step = 3
                st.rerun()
        with col2:
            if st.button("Start Training", type="primary", use_container_width=True):
                st.session_state.training_step = 5
                st.rerun()
    
# Step 5: COMPLETELY FIXED Model Training
    elif st.session_state.training_step == 5:
        st.markdown('<div class="section-header"><h3>Step 5: Real Model Training</h3></div>', unsafe_allow_html=True)
        
        config = st.session_state.training_config
        
        # Check if training is already completed
        if 'training_completed' not in st.session_state:
            st.session_state.training_completed = False
        
        if not st.session_state.training_completed:
            # Prepare data
            target_column = config['target_column']
            selected_features = config['selected_features']
            
            X = df[selected_features].copy()
            y = df[target_column].copy()
            
            # Data validation
            st.info("Validating data...")
            
            if config['problem_type'] == "Classification":
                unique_classes = y.nunique()
                if unique_classes < 2:
                    st.error("Classification needs at least 2 classes")
                    return
                st.success(f"Classification target: {unique_classes} classes")
            else:
                if y.dtype == 'object':
                    try:
                        y = pd.to_numeric(y, errors='coerce')
                        if y.isnull().sum() > 0:
                            st.error("Cannot convert target to numeric")
                            return
                    except:
                        st.error("Regression target must be numeric")
                        return
                st.success(f"Regression target: {y.nunique()} unique values")
            
            # Initialize unified preprocessor
            st.info("Setting up unified preprocessing pipeline...")
            preprocessor = DataPreprocessor()
            
            # Fit and transform data
            target_scaling = config.get('target_scaling', 'auto')
            X_processed, y_processed = preprocessor.fit_transform_features(
                X, y if config['problem_type'] == "Regression" else None, target_scaling
            )
            
            # Store preprocessor for consistent predictions
            st.session_state.preprocessor = preprocessor
            
            st.success(f"Preprocessing complete: {len(preprocessor.categorical_features)} categorical, {len(preprocessor.numerical_features)} numerical features")
            
            if config['problem_type'] == "Regression" and preprocessor.target_transform_config['type'] != 'none':
                st.info(f"Target scaling applied: {preprocessor.target_transform_config['type']}")
            
            # Handle target encoding for classification
            if config['problem_type'] == "Classification" and y.dtype == 'object':
                st.info("Encoding target...")
                le_target = LabelEncoder()
                y_processed = pd.Series(le_target.fit_transform(y.astype(str)), index=y.index)
                st.session_state.target_encoder = le_target
                st.success(f"Target encoded: {len(le_target.classes_)} classes")
            else:
                st.session_state.target_encoder = None
            
            # Data splitting
            st.info("Splitting data...")
            try:
                stratify = y_processed if config['problem_type'] == "Classification" and len(np.unique(y_processed)) > 1 else None
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_processed, 
                    test_size=config['test_size'], 
                    random_state=42, 
                    stratify=stratify
                )
                st.success(f"Data split: {X_train.shape[0]} train, {X_test.shape[0]} test")
            except Exception as e:
                st.error(f"Error splitting data: {str(e)}")
                return
            
            # REAL TRAINING PHASE - ONLY RUNS ONCE
            st.markdown('<div class="training-container">', unsafe_allow_html=True)
            st.markdown("### Real Model Training in Progress")
            
            all_results = []
            trained_models = {}
            
            # Start training button
            if st.button("START TRAINING", type="primary", use_container_width=True):
                for i, model_name in enumerate(st.session_state.selected_models_for_training):
                    st.markdown(f"#### Training {model_name} ({i+1}/{len(st.session_state.selected_models_for_training)})")
                    
                    # REAL training with actual progress tracking
                    results, trained_model = train_model_real(
                        model_name, 
                        X_train.values, y_train.values if hasattr(y_train, 'values') else y_train,
                        X_test.values, y_test.values if hasattr(y_test, 'values') else y_test,
                        config['problem_type'],
                        max_iterations=config['max_iterations']
                    )
                    
                    if results and trained_model:
                        all_results.append(results)
                        trained_models[model_name] = trained_model
                        st.success(f"{model_name} training completed!")
                    else:
                        st.error(f"{model_name} training failed")
                    
                    st.markdown("---")
                
                # Store results and mark training as completed
                st.session_state.model_results = all_results
                st.session_state.trained_models = trained_models
                st.session_state.training_completed = True
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                if all_results:
                    st.success(f"Training completed! {len(all_results)} models trained successfully.")

                else:
                    st.error("No models were trained successfully. Please try again.")
                    st.session_state.training_completed = False
            
            else:
                st.markdown('</div>', unsafe_allow_html=True)
                st.info("Click 'START TRAINING' to begin the training process")
        
        else:
            # Training already completed - show summary
            st.success("Training already completed!")
            
            if st.session_state.model_results:
                results_df = pd.DataFrame(st.session_state.model_results)
                st.markdown("### Training Summary")
                st.dataframe(results_df, use_container_width=True)
                
                st.info(f"{len(st.session_state.trained_models)} models trained successfully")
            
            # Option to retrain
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Retrain Models", use_container_width=True):
                    # Reset training state
                    st.session_state.training_completed = False
                    st.session_state.model_results = []
                    st.session_state.trained_models = {}
                    st.rerun()
        
        # Navigation - only show if training is completed
        if st.session_state.training_completed and st.session_state.model_results:
            if st.button("View Results & Deploy →", type="primary", use_container_width=True):
                st.session_state.training_step = 6
                st.rerun()
        elif not st.session_state.training_completed:
            if st.button("← Back to Configuration"):
                st.session_state.training_step = 4
                st.rerun()

    
    # Step 6: Results & Deployment
    elif st.session_state.training_step == 6:
        st.markdown('<div class="section-header"><h3>Step 6: Results & Model Deployment</h3></div>', unsafe_allow_html=True)
        
        if not st.session_state.model_results or not st.session_state.trained_models:
            st.error("No trained models found")
            if st.button("← Back to Training"):
                st.session_state.training_step = 5
                st.rerun()
            return
        
        results_df = pd.DataFrame(st.session_state.model_results)
        
        # Best model identification
        if st.session_state.training_config['problem_type'] == "Classification":
            best_model_idx = results_df['Accuracy'].astype(float).idxmax()
            best_metric = 'Accuracy'
        else:
            best_model_idx = results_df['R²'].astype(float).idxmax()
            best_metric = 'R²'
        
        best_model_name = results_df.loc[best_model_idx, 'Model']
        best_score = results_df.loc[best_model_idx, best_metric]
        
        # Champion model display
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #28a745, #20c997);
                   color: white; padding: 25px; border-radius: 15px; 
                   text-align: center; margin: 20px 0;">
            <h2 style="margin: 0;">Champion Model</h2>
            <h3 style="margin: 10px 0;">{best_model_name}</h3>
            <p style="margin: 0; font-size: 1.2rem;">{best_metric}: {best_score}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Results table
        st.markdown("### Complete Results")
        st.dataframe(results_df, use_container_width=True)
        
        # Model actions
        st.markdown("### Model Management")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Export Results")
            csv = results_df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                file_name=f"training_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### Download Model")
            model_to_download = st.selectbox("Select model:", list(st.session_state.trained_models.keys()))
            if st.button("Download Model Package", use_container_width=True):
                model_package = {
                    'model': st.session_state.trained_models[model_to_download],
                    'preprocessor': st.session_state.get('preprocessor'),
                    'config': st.session_state.training_config,
                    'target_encoder': st.session_state.get('target_encoder')
                }
                model_bytes = pickle.dumps(model_package)
                st.download_button(
                    "Download Package",
                    model_bytes,
                    file_name=f"{model_to_download.replace(' ', '_')}.pkl",
                    mime="application/octet-stream",
                    use_container_width=True
                )
        
        with col3:
            st.markdown("#### Actions")
            if st.button("Train More Models", use_container_width=True):
                st.session_state.training_step = 3
                st.rerun()
            if st.button("New Project", use_container_width=True):
                # Clear session
                for key in list(st.session_state.keys()):
                    if key.startswith(('training_', 'model_', 'selected_', 'preprocessor')):
                        del st.session_state[key]
                st.session_state.training_step = 1
                st.rerun()
        
        # FIXED Prediction Section
        st.markdown("### Model Prediction (COMPLETELY FIXED)")
        st.markdown("*Test your models with consistent, reliable predictions*")
        
        # Model selection
        prediction_model = st.selectbox(
            "Select model for prediction:",
            list(st.session_state.trained_models.keys())
        )
        
        # Input method
        input_method = st.radio(
            "Input method:",
            ["Interactive", "JSON"],
            horizontal=True
        )
        
        # Show required features
        with st.expander("Required Features", expanded=False):
            preprocessor = st.session_state.get('preprocessor')
            if preprocessor:
                feature_names = preprocessor.feature_names
                target_column = st.session_state.training_config.get('target_column')
                
                st.write(f"**Required Features ({len(feature_names)}):**")
                for i, feature in enumerate(feature_names, 1):
                    if feature in df.columns:
                        if df[feature].dtype == 'object':
                            unique_vals = df[feature].unique()[:3]
                            st.write(f"{i:2d}. `{feature}` (categories: {list(unique_vals)}{'...' if len(df[feature].unique()) > 3 else ''})")
                        else:
                            st.write(f"{i:2d}. `{feature}` (numeric: {df[feature].min():.2f} to {df[feature].max():.2f})")
                
                # Template
                template = {}
                for feature in feature_names:
                    if feature in df.columns:
                        sample_val = df[feature].iloc[0]
                        if pd.isna(sample_val):
                            template[feature] = "value_here"
                        elif df[feature].dtype in ['int64', 'float64']:
                            template[feature] = float(sample_val) if df[feature].dtype == 'float64' else int(sample_val)
                        else:
                            template[feature] = str(sample_val)
                
                st.code(json.dumps(template, indent=2), language="json")
        
        # Interactive input
        if input_method == "Interactive":
            st.markdown("#### Enter Feature Values")
            
            preprocessor = st.session_state.get('preprocessor')
            if preprocessor:
                feature_names = preprocessor.feature_names
                input_values = {}
                cols = st.columns(min(3, len(feature_names)))
                
                for i, feature in enumerate(feature_names):
                    with cols[i % len(cols)]:
                        if feature in df.columns:
                            if df[feature].dtype in ['int64', 'float64']:
                                sample_val = float(df[feature].iloc[0]) if not pd.isna(df[feature].iloc[0]) else 0.0
                                input_values[feature] = st.number_input(
                                    f"{feature}",
                                    value=sample_val,
                                    key=f"input_{feature}",
                                    help=f"Range: {df[feature].min():.2f} to {df[feature].max():.2f}"
                                )
                            else:
                                unique_vals = sorted(df[feature].dropna().unique())
                                if len(unique_vals) <= 20:
                                    input_values[feature] = st.selectbox(
                                        f"{feature}",
                                        unique_vals,
                                        key=f"input_{feature}"
                                    )
                                else:
                                    input_values[feature] = st.text_input(
                                        f"{feature}",
                                        value=str(df[feature].iloc[0]),
                                        key=f"input_{feature}"
                                    )
                
                if st.button("Make Prediction", type="primary"):
                    with st.spinner("Generating prediction..."):
                        prediction_result = make_prediction_consistent(
                            prediction_model, 
                            input_values, 
                            st.session_state.training_config
                        )
                        if prediction_result:
                            st.success(f"**{prediction_result}**")
                            st.info("Prediction generated using unified preprocessing pipeline")
        
        # JSON input
        else:
            st.markdown("#### JSON Input")
            
            json_input = st.text_area(
                "Enter JSON data:",
                height=200,
                placeholder='{"feature1": value1, "feature2": value2, ...}'
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Validate JSON"):
                    try:
                        json.loads(json_input.strip())
                        st.success("Valid JSON!")
                    except json.JSONDecodeError as e:
                        st.error(f"Invalid JSON: {str(e)}")
            
            with col2:
                if st.button("Predict", type="primary"):
                    try:
                        json_data = json.loads(json_input.strip())
                        with st.spinner("Processing..."):
                            prediction_result = make_prediction_consistent(
                                prediction_model, 
                                json_data, 
                                st.session_state.training_config
                            )
                            if prediction_result:
                                st.success(f"**{prediction_result}**")
                    except json.JSONDecodeError:
                        st.error("Fix JSON format first")
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")

if __name__ == "__main__":
    app()
