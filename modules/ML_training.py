import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pickle
import json
import re
import warnings
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
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
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
                "estimator": RandomForestClassifier(random_state=42)
            },
            "Gradient Boosting": {
                "description": "Sequential ensemble that corrects previous model errors",
                "complexity": "High",
                "use_case": "Complex patterns, high accuracy requirements",
                "pros": "Excellent performance, handles missing values",
                "cons": "Prone to overfitting, requires tuning",
                "estimator": GradientBoostingClassifier(random_state=42)
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
                "estimator": MLPClassifier(random_state=42, max_iter=500)
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
                "estimator": RandomForestRegressor(random_state=42)
            },
            "Gradient Boosting": {
                "description": "Sequential ensemble optimizing for prediction accuracy",
                "complexity": "High",
                "use_case": "Complex patterns, maximum accuracy",
                "pros": "High accuracy, handles interactions well",
                "cons": "Computationally expensive, requires tuning",
                "estimator": GradientBoostingRegressor(random_state=42)
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
                "estimator": MLPRegressor(random_state=42, max_iter=500)
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

def simulate_advanced_training(model_name, X_train, y_train, X_test, y_test, epochs=100, problem_type="classification"):
    """Simulate advanced training with epochs and progress tracking"""
    results = []
    
    # Create single containers for progress updates
    progress_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    # Initialize model
    model_info = get_model_info()
    
    # Ensure we're using the correct problem type key
    problem_key = problem_type.lower()
    if problem_key not in model_info:
        st.error(f"Problem type '{problem_type}' not supported. Available types: {list(model_info.keys())}")
        return None, None
    
    model_dict = model_info[problem_key]
    
    if model_name not in model_dict:
        st.error(f"Model '{model_name}' not found for {problem_type}. Available models: {list(model_dict.keys())}")
        return None, None
    
    model = model_dict[model_name]["estimator"]
    
    # For neural networks, we can actually use epochs
    if "Neural Network" in model_name:
        model.max_iter = epochs
        
    # Simulate training epochs with single progress bar
    for epoch in range(1, epochs + 1):
        # Update progress only every 5 epochs or on final epoch to reduce UI spam
        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            progress = (epoch / epochs) * 100
            eta = ((epochs - epoch) * 0.1)  # Simulate time estimation
            
            with progress_placeholder.container():
                st.markdown(create_animated_progress(progress, f"Training {model_name}"), unsafe_allow_html=True)
                
                # Show current epoch metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Epoch", f"{epoch}/{epochs}")
                with col2:
                    st.metric("Progress", f"{progress:.1f}%")
                with col3:
                    st.metric("ETA", f"{eta:.1f}s")
        
        # Simulate intermediate results every 10 epochs
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            # Partial training for demonstration
            if hasattr(model, 'partial_fit') and epoch < epochs:
                try:
                    if problem_type.lower() == "classification":
                        model.partial_fit(X_train, y_train, classes=np.unique(y_train))
                    else:
                        model.partial_fit(X_train, y_train)
                except Exception as e:
                    # If partial_fit fails, just continue to final training
                    pass
            elif epoch == epochs:
                # Final training - this is the most important part
                try:
                    st.info(f"Final training for {model_name}...")
                    model.fit(X_train, y_train)
                    st.success(f"✅ Model {model_name} fitted successfully")
                except Exception as e:
                    st.error(f"❌ Training failed for {model_name}: {str(e)}")
                    return None, None
                
            # Get predictions for current state
            try:
                y_pred = model.predict(X_test)
                
                # Validate predictions
                if len(y_pred) != len(y_test):
                    st.error(f"❌ Prediction length mismatch: {len(y_pred)} vs {len(y_test)}")
                    continue
                
                # Check for invalid predictions
                if np.isnan(y_pred).any() or np.isinf(y_pred).any():
                    st.error(f"❌ Invalid predictions (NaN/Inf) from {model_name}")
                    continue
                
                # Calculate metrics
                if problem_type.lower() == "classification":
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
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
                        'F1-Score': f"{f1:.4f}"
                    }
                    
                else:  # Regression
                    try:
                        # Handle target inverse transformation for meaningful metrics
                        y_test_original = y_test.copy()
                        y_pred_original = y_pred.copy()
                        
                        # If target was scaled, transform back for interpretable metrics
                        target_transform = st.session_state.get('target_transform', {'type': 'none'})
                        
                        if target_transform['type'] == 'log':
                            offset = target_transform.get('offset', 0)
                            y_test_original = np.exp(y_test) - offset
                            y_pred_original = np.exp(y_pred) - offset
                            st.info("📊 Metrics calculated on original scale (inverse log transform)")
                            
                        elif target_transform['type'] == 'standard':
                            target_scaler = st.session_state.get('target_scaler')
                            if target_scaler:
                                y_test_original = target_scaler.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
                                y_pred_original = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                                st.info("📊 Metrics calculated on original scale (inverse standard scaling)")
                                
                        elif target_transform['type'] == 'minmax':
                            target_scaler = st.session_state.get('target_scaler')
                            if target_scaler:
                                y_test_original = target_scaler.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
                                y_pred_original = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                                st.info("📊 Metrics calculated on original scale (inverse min-max scaling)")
                        
                        # Calculate metrics on both scales
                        # Scaled metrics (for model comparison)
                        mse_scaled = mean_squared_error(y_test, y_pred)
                        rmse_scaled = np.sqrt(mse_scaled)
                        mae_scaled = mean_absolute_error(y_test, y_pred)
                        r2_scaled = r2_score(y_test, y_pred)
                        
                        # Original scale metrics (for interpretation)
                        mse_original = mean_squared_error(y_test_original, y_pred_original)
                        rmse_original = np.sqrt(mse_original)
                        mae_original = mean_absolute_error(y_test_original, y_pred_original)
                        r2_original = r2_score(y_test_original, y_pred_original)
                        
                        # Validate metrics
                        if np.isnan([mse_scaled, rmse_scaled, mae_scaled, r2_scaled]).any() or np.isinf([mse_scaled, rmse_scaled, mae_scaled, r2_scaled]).any():
                            st.error(f"❌ Invalid regression metrics for {model_name}")
                            continue
                        
                        # Use original scale metrics for display (more interpretable)
                        display_rmse = rmse_original if target_transform['type'] != 'none' else rmse_scaled
                        display_mae = mae_original if target_transform['type'] != 'none' else mae_scaled
                        display_r2 = r2_original if target_transform['type'] != 'none' else r2_scaled
                        display_mse = mse_original if target_transform['type'] != 'none' else mse_scaled
                        
                        # Show warning for very high RMSE only on scaled values to avoid false positives
                        if rmse_scaled > np.std(y_test) * 5:
                            st.warning(f"⚠️ High RMSE on scaled data ({rmse_scaled:.2f}) for {model_name}.")
                        
                        # Display metrics
                        with metrics_placeholder.container():
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-label">R² Score</div>
                                    <div class="metric-value">{display_r2:.3f}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with col2:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-label">RMSE</div>
                                    <div class="metric-value">{display_rmse:.1f}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with col3:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-label">MAE</div>
                                    <div class="metric-value">{display_mae:.1f}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with col4:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-label">MSE</div>
                                    <div class="metric-value">{display_mse:.1f}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                        # Store results using original scale for interpretability
                        results = {
                            'Model': model_name,
                            'RMSE': f"{display_rmse:.2f}",
                            'MAE': f"{display_mae:.2f}",
                            'R²': f"{display_r2:.4f}",
                            'MSE': f"{display_mse:.1f}"
                        }
                        
                        # Show both scales if target was transformed
                        if target_transform['type'] != 'none':
                            with st.expander(f"📊 Detailed Metrics for {model_name}", expanded=False):
                                st.write("**Original Scale (Interpretable):**")
                                st.write(f"  • RMSE: {rmse_original:.2f}")
                                st.write(f"  • MAE: {mae_original:.2f}")
                                st.write(f"  • R²: {r2_original:.4f}")
                                st.write("**Scaled Values (Model Training):**")
                                st.write(f"  • RMSE: {rmse_scaled:.4f}")
                                st.write(f"  • MAE: {mae_scaled:.4f}")
                                st.write(f"  • R²: {r2_scaled:.4f}")
                        
                    except Exception as e:
                        st.error(f"❌ Error calculating regression metrics: {str(e)}")
                        results = {
                            'Model': model_name,
                            'RMSE': "Error",
                            'MAE': "Error", 
                            'R²': "Error",
                            'MSE': "Error"
                        }
            except Exception as e:
                if epoch % 20 == 0:  # Show progress message less frequently
                    st.warning(f"Metrics calculation in progress... (Epoch {epoch})")
        
        # Small delay to show progress (reduced frequency)
        if epoch % 10 == 0:  # Update every 10 epochs instead of every 5
            time.sleep(0.02)  # Very small delay
    
    return results, model

def make_prediction(model_name, input_data, training_config):
    """Make prediction using trained model with intelligent feature mapping"""
    try:
        # Get the trained model
        if model_name not in st.session_state.trained_models:
            st.error(f"Model {model_name} not found")
            return None
        
        model = st.session_state.trained_models[model_name]
        
        # Get the encoded features that the model expects
        encoded_features = st.session_state.get('encoded_features', training_config.get('selected_features', []))
        target_column = training_config.get('target_column', None)
        
        # Get original dataset to understand feature relationships
        original_df = None
        if 'df_feature_eng' in st.session_state and st.session_state['df_feature_eng'] is not None:
            original_df = st.session_state['df_feature_eng']
        
        # Convert input to DataFrame
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            st.error("Input data must be a dictionary")
            return None
        
        # Create a DataFrame that matches the training features exactly
        model_input_df = pd.DataFrame()
        
        # Identify which original features can be one-hot encoded
        one_hot_patterns = {
            'Sex': ['Sex_female', 'Sex_male'],
            'Embarked': ['Embarked_C', 'Embarked_Q', 'Embarked_S'],
            'Pclass': ['Pclass_1', 'Pclass_2', 'Pclass_3']
        }
        
        # Process each required encoded feature
        for encoded_feature in encoded_features:
            if encoded_feature == target_column:
                continue  # Skip target column
                
            # Check if this is a one-hot encoded feature
            is_one_hot = False
            original_feature = None
            
            # Check common one-hot encoding patterns
            for orig_col, encoded_cols in one_hot_patterns.items():
                if encoded_feature in encoded_cols:
                    is_one_hot = True
                    original_feature = orig_col
                    break
            
            # Also check for general one-hot patterns (feature_value)
            if not is_one_hot:
                for col in input_df.columns:
                    if encoded_feature.startswith(f"{col}_"):
                        is_one_hot = True
                        original_feature = col
                        break
            
            if is_one_hot and original_feature and original_feature in input_df.columns:
                # Handle one-hot encoding
                original_value = str(input_df[original_feature].iloc[0]).lower()
                
                # Extract the expected value from the encoded feature name
                # For Sex_male, we want to check if original value is "male"
                expected_value = encoded_feature.split('_')[-1].lower()
                
                # Set 1 if values match, 0 otherwise
                if original_value == expected_value:
                    model_input_df[encoded_feature] = 1
                else:
                    model_input_df[encoded_feature] = 0
                    
            elif encoded_feature in input_df.columns:
                # Direct feature mapping (no encoding needed)
                model_input_df[encoded_feature] = input_df[encoded_feature]
                
            elif encoded_feature.endswith('_encoded'):
                # Handle label encoded features
                base_feature = encoded_feature.replace('_encoded', '')
                if base_feature in input_df.columns:
                    # Apply label encoding if available
                    if 'label_encoders' in st.session_state and base_feature in st.session_state.label_encoders:
                        try:
                            encoder = st.session_state.label_encoders[base_feature]
                            encoded_value = encoder.transform([str(input_df[base_feature].iloc[0])])[0]
                            model_input_df[encoded_feature] = encoded_value
                        except ValueError:
                            # Handle unseen categories
                            st.warning(f"⚠️ Unknown category '{input_df[base_feature].iloc[0]}' for feature '{base_feature}'. Using default encoding.")
                            model_input_df[encoded_feature] = 0
                    else:
                        # No encoder available, try direct conversion
                        try:
                            model_input_df[encoded_feature] = pd.to_numeric(input_df[base_feature])
                        except:
                            model_input_df[encoded_feature] = 0
                else:
                    st.error(f"❌ Cannot find original feature '{base_feature}' for encoded feature '{encoded_feature}'")
                    return None
            else:
                # Try to find if this is a direct numeric feature
                if encoded_feature in input_df.columns:
                    model_input_df[encoded_feature] = input_df[encoded_feature]
                else:
                    # Feature not found - set to default value instead of failing
                    st.warning(f"⚠️ Feature '{encoded_feature}' not found in input, using default value 0")
                    model_input_df[encoded_feature] = 0
        
        # Ensure all required features are present with default values if missing
        missing_encoded_features = [f for f in encoded_features if f not in model_input_df.columns and f != target_column]
        if missing_encoded_features:
            st.warning(f"⚠️ Adding missing features with default values: {missing_encoded_features}")
            for missing_feat in missing_encoded_features:
                model_input_df[missing_feat] = 0
        
        # Reorder columns to match training order
        final_features = [f for f in encoded_features if f != target_column]
        model_input_df = model_input_df[final_features]
        
        # Debug information for user
        st.success(f"✅ Successfully mapped {len(input_df.columns)} input features to {len(model_input_df.columns)} model features")
        
        # Show the actual feature mapping for debugging
        with st.expander("🔍 Feature Mapping Debug Info", expanded=False):
            st.write("**Input Features:**", list(input_df.columns))
            st.write("**Model Features:**", list(model_input_df.columns))
            st.write("**Feature Values:**")
            for col in model_input_df.columns:
                st.write(f"  • {col}: {model_input_df[col].iloc[0]}")
            st.write("**Data Types:**")
            st.write(model_input_df.dtypes.to_dict())
        
        # Convert to numeric and handle missing values
        for col in model_input_df.columns:
            if model_input_df[col].dtype == 'object':
                try:
                    model_input_df[col] = pd.to_numeric(model_input_df[col])
                except:
                    model_input_df[col] = 0
            
            # Fill any remaining NaN values
            if model_input_df[col].isnull().any():
                # Use 0 for numeric columns, or the most common value if available from training
                if pd.api.types.is_numeric_dtype(model_input_df[col]):
                    model_input_df[col] = model_input_df[col].fillna(0)
                else:
                    model_input_df[col] = model_input_df[col].fillna(0)
        
        # Final safety check - ensure no NaN values remain
        if model_input_df.isnull().any().any():
            st.warning("⚠️ Filling remaining missing values with defaults...")
            model_input_df = model_input_df.fillna(0)
        
        # Handle scaling if needed
        if model_name in ["Support Vector Machine", "Support Vector Regression", "Logistic Regression", "Neural Network"]:
            if 'scaler' in st.session_state and st.session_state.scaler is not None:
                input_scaled = st.session_state.scaler.transform(model_input_df)
            else:
                input_scaled = model_input_df.values
        else:
            input_scaled = model_input_df.values
        
        # Make prediction
        prediction = model.predict(input_scaled)
        
        # Decode prediction if classification with target encoder
        if training_config['problem_type'] == "Classification" and 'target_encoder' in st.session_state and st.session_state.target_encoder is not None:
            try:
                decoded_prediction = st.session_state.target_encoder.inverse_transform(prediction)
                prediction = decoded_prediction
            except:
                pass  # Use numeric prediction if decoding fails
        
        # Format output based on problem type
        if training_config['problem_type'] == "Classification":
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_scaled)
                return f"Class: {prediction[0]}, Confidence: {max(probabilities[0]):.3f}"
            else:
                return f"Class: {prediction[0]}"
        else:  # Regression
            # Handle target inverse transformation for meaningful predictions
            final_prediction = prediction[0]
            target_transform = st.session_state.get('target_transform', {'type': 'none'})
            
            if target_transform['type'] == 'log':
                offset = target_transform.get('offset', 0)
                final_prediction = np.exp(final_prediction) - offset
                return f"{final_prediction:.2f}"
                
            elif target_transform['type'] == 'standard':
                target_scaler = st.session_state.get('target_scaler')
                if target_scaler:
                    final_prediction = target_scaler.inverse_transform([[final_prediction]])[0][0]
                return f"{final_prediction:.2f}"
                
            elif target_transform['type'] == 'minmax':
                target_scaler = st.session_state.get('target_scaler')
                if target_scaler:
                    final_prediction = target_scaler.inverse_transform([[final_prediction]])[0][0]
                return f"{final_prediction:.2f}"
            else:
                return f"{final_prediction:.4f}"
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def app():
    local_css()
    st.title("MLGenie AutoML Training Studio")
    st.markdown("*Train world-class machine learning models with intelligent automation*")
    
    # Add reset option in sidebar for debugging
    with st.sidebar:
        st.markdown("---")
        if st.button("🔄 Reset Training Session", help="Clear all training data and start fresh"):
            # Reset all training-related session state
            keys_to_reset = ['training_step', 'trained_models', 'model_results', 
                           'selected_models_for_training', 'training_config', 
                           'label_encoders', 'target_encoder', 'scaler']
            for key in keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("✅ Training session reset!")
            st.rerun()
    
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
    st.markdown("### Training Pipeline")
    
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
        
        # Check for dataset
        df = None
        
        if 'feature_eng_data' in st.session_state and st.session_state.get('from_feature_eng', False):
            df = st.session_state['feature_eng_data']
            st.session_state['from_feature_eng'] = False
        elif 'df_feature_eng' in st.session_state:
            df = st.session_state['df_feature_eng']
        
        if df is not None:
            st.success("Dataset loaded successfully from Feature Engineering!")
            col1, col2 = st.columns([3,1])
            with col2:
                if st.button("Load Different Dataset", use_container_width=True):
                    if 'df_feature_eng' in st.session_state:
                        del st.session_state.df_feature_eng
                    if 'feature_eng_data' in st.session_state:
                        del st.session_state.feature_eng_data
                    st.rerun()
            
            if st.button("Continue to Data Analysis →", type="primary", use_container_width=True):
                st.session_state.training_step = 2
                st.rerun()
                
        else:
            uploaded_file = st.file_uploader(
                "Upload your dataset (CSV/Excel)", 
                type=["csv", "xlsx", "xls"], 
                key="ml_training_uploader",
                help="Upload a clean dataset ready for machine learning"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith((".xlsx", ".xls")):
                        df = pd.read_excel(uploaded_file)
                    
                    if df.empty:
                        st.error("The uploaded file is empty. Please upload a valid file.")
                        return
                        
                    st.session_state.df_feature_eng = df.copy()
                    st.success("Data uploaded successfully!")
                    
                    if st.button("Continue to Data Analysis →", type="primary", use_container_width=True):
                        st.session_state.training_step = 2
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
                    return
            else:
                st.info("Please upload your dataset to begin the AutoML journey")
                return
    
    # Get the dataset for subsequent steps
    df = None
    if 'df_feature_eng' in st.session_state and st.session_state['df_feature_eng'] is not None:
        df = st.session_state['df_feature_eng']
    elif 'feature_eng_data' in st.session_state and st.session_state['feature_eng_data'] is not None:
        df = st.session_state['feature_eng_data']
    
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
        
        # Data sample and info
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
                st.warning("Missing values detected in your dataset")
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
        
        # Navigation buttons
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
        st.markdown('<div class="section-header"><h3>Step 3: Intelligent Model Selection</h3></div>', unsafe_allow_html=True)
        
        # Target selection
        # Check if target was protected in Feature Engineering
        protected_target = st.session_state.get('protected_target_column', None)
        if protected_target and protected_target in df.columns:
            default_target_index = df.columns.tolist().index(protected_target)
            st.info(f"💡 **Suggested Target:** '{protected_target}' (protected in Feature Engineering)")
        else:
            default_target_index = 0
        
        target_column = st.selectbox(
            "Select Target Column (What do you want to predict?)", 
            df.columns.tolist(),
            index=default_target_index,
            help="Choose the column you want your model to predict"
        )
        
        if target_column:
            # Determine problem type with improved logic
            unique_values = df[target_column].nunique()
            target_dtype = df[target_column].dtype
            
            # More sophisticated problem type detection
            if target_dtype == 'object':
                problem_type = "Classification"
                st.info(f"**Problem Type:** {problem_type} (Categorical target with {unique_values} unique classes)")
            elif target_dtype in ['bool', 'category']:
                problem_type = "Classification"
                st.info(f"**Problem Type:** {problem_type} (Boolean/Categorical target)")
            elif unique_values <= 10 and all(df[target_column].dropna() == df[target_column].dropna().astype(int)):
                # Integer values with few unique values - likely classification
                problem_type = "Classification"
                st.info(f"**Problem Type:** {problem_type} (Discrete target with {unique_values} unique values)")
            elif unique_values <= 20 and target_dtype in ['int64', 'int32']:
                # Ask user to confirm for ambiguous cases
                user_choice = st.radio(
                    f"Target has {unique_values} unique integer values. Please specify the problem type:",
                    ["Classification", "Regression"],
                    help="Classification: Predicting categories/classes. Regression: Predicting continuous numbers."
                )
                problem_type = user_choice
                st.info(f"**Problem Type:** {problem_type} (User specified)")
            else:
                problem_type = "Regression"
                st.info(f"**Problem Type:** {problem_type} (Continuous target variable with {unique_values} unique values)")
            
            st.session_state.training_config['target_column'] = target_column
            st.session_state.training_config['problem_type'] = problem_type
            
            # Feature selection
            feature_columns = [col for col in df.columns if col != target_column]
            st.markdown("### Feature Selection")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_features = st.multiselect(
                    "Select features for training", 
                    feature_columns, 
                    default=feature_columns[:15] if len(feature_columns) > 15 else feature_columns,
                    help="Choose which columns to use as input features"
                )
            with col2:
                if st.button("Select All Features"):
                    selected_features = feature_columns
                    st.rerun()
            
            st.session_state.training_config['selected_features'] = selected_features
            
            if selected_features:
                # Model selection with detailed cards
                st.markdown("### Choose Your Models")
                st.markdown("*Select multiple models to compare performance*")
                
                model_info = get_model_info()
                available_models = model_info[problem_type.lower()]
                
                # Create model selection grid
                model_cols = st.columns(2)
                selected_models = []
                
                for i, (model_name, info) in enumerate(available_models.items()):
                    col_idx = i % 2
                    with model_cols[col_idx]:
                        is_selected = st.checkbox(
                            f"Select {model_name}", 
                            key=f"model_select_{model_name}",
                            value=model_name in st.session_state.selected_models_for_training
                        )
                        
                        if is_selected and model_name not in selected_models:
                            selected_models.append(model_name)
                        
                        # Display model card
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
                        if st.button("Continue to Training Configuration →", type="primary", use_container_width=True):
                            st.session_state.training_step = 4
                            st.rerun()
                    else:
                        st.warning("Please select at least one model to continue")
    
    # Step 4: Training Configuration  
    elif st.session_state.training_step == 4:
        st.markdown('<div class="section-header"><h3>Step 4: Training Configuration</h3></div>', unsafe_allow_html=True)
        
        # Training parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Data Splitting")
            test_size = st.slider("Test Size (%)", 10, 40, 20, step=5) / 100
            validation_size = st.slider("Validation Size (%)", 0, 20, 10, step=5) / 100
            
            st.markdown("### Training Parameters")
            epochs = st.number_input("Training Epochs", min_value=10, max_value=1000, value=100, step=10,
                                   help="Number of training iterations (higher = more training time)")
            
        with col2:
            st.markdown("### Advanced Options")
            cross_validation = st.checkbox("Enable Cross Validation", value=True,
                                         help="Use k-fold cross validation for robust evaluation")
            
            if cross_validation:
                cv_folds = st.slider("CV Folds", 3, 10, 5)
            
            early_stopping = st.checkbox("Early Stopping", value=True,
                                       help="Stop training if no improvement detected")
            
            auto_feature_scaling = st.checkbox("Automatic Feature Scaling", value=True,
                                             help="Automatically scale features for better performance")
        
        # Save configuration
        st.session_state.training_config.update({
            'test_size': test_size,
            'validation_size': validation_size,
            'epochs': epochs,
            'cross_validation': cross_validation,
            'cv_folds': cv_folds if cross_validation else 5,
            'early_stopping': early_stopping,
            'auto_feature_scaling': auto_feature_scaling
        })
        
        # Configuration Summary
        with st.expander("Configuration Summary", expanded=True):
            config = st.session_state.training_config
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Data Configuration:**")
                st.write(f"• Target: {config.get('target_column', 'Not set')}")
                st.write(f"• Problem: {config.get('problem_type', 'Not set')}")
                st.write(f"• Features: {len(config.get('selected_features', []))}")
                
            with col2:
                st.write("**Selected Models:**")
                for model in st.session_state.selected_models_for_training:
                    st.write(f"• {model}")
            
            with col3:
                st.write("**Training Settings:**")
                st.write(f"• Test Size: {test_size*100:.0f}%")
                st.write(f"• Epochs: {epochs}")
                st.write(f"• Cross Validation: {'Yes' if cross_validation else 'No'}")
        
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
    
    # Step 5: Model Training
    elif st.session_state.training_step == 5:
        st.markdown('<div class="section-header"><h3>Step 5: Model Training in Progress</h3></div>', unsafe_allow_html=True)
        
        config = st.session_state.training_config
        
        # Prepare data
        target_column = config['target_column']
        selected_features = config['selected_features']
        
        X = df[selected_features].copy()
        y = df[target_column].copy()
        
        # Validate target variable for problem type
        st.info("Validating target variable...")
        unique_target_values = y.nunique()
        
        if config['problem_type'] == "Classification":
            if unique_target_values < 2:
                st.error("❌ Classification requires at least 2 classes in target variable")
                return
            elif unique_target_values > 100:
                st.warning(f"⚠️ Target has {unique_target_values} classes - this might be better suited for regression")
            else:
                st.success(f"✅ Classification target validated: {unique_target_values} classes")
        else:  # Regression
            if y.dtype == 'object':
                st.error("❌ Regression target cannot be text/categorical. Please select a numeric column or use Classification.")
                return
            elif unique_target_values < 10:
                st.warning(f"⚠️ Target has only {unique_target_values} unique values - consider Classification instead")
            else:
                st.success(f"✅ Regression target validated: {unique_target_values} unique values")
        
        # Handle missing values
        if X.isnull().sum().sum() > 0:
            st.info("Handling missing values...")
            X = X.fillna(X.mean(numeric_only=True))
            X = X.fillna(X.mode().iloc[0])
        
        # Store original feature names before any encoding - intelligently detect base features
        original_feature_names = set()
        
        # Get the original dataset columns to help identify base features
        original_df_columns = set()
        if 'df_feature_eng' in st.session_state and st.session_state['df_feature_eng'] is not None:
            original_df_columns = set(st.session_state['df_feature_eng'].columns)
        
        # Known one-hot encoding patterns for common features
        known_one_hot_patterns = {
            'Sex': ['Sex_female', 'Sex_male'],
            'Embarked': ['Embarked_C', 'Embarked_Q', 'Embarked_S'],
            'Pclass': ['Pclass_1', 'Pclass_2', 'Pclass_3']
        }
        
        # Detect original features from selected features
        for feature in selected_features:
            # Skip target column
            if feature == target_column:
                continue
                
            # Check if this is directly an original feature
            if feature in df.columns and not any(pattern in feature for pattern in ['_encoded', '_female', '_male', '_C', '_Q', '_S']):
                original_feature_names.add(feature)
                continue
            
            # Check known one-hot patterns
            found_original = False
            for orig_col, encoded_cols in known_one_hot_patterns.items():
                if feature in encoded_cols and orig_col in df.columns:
                    original_feature_names.add(orig_col)
                    found_original = True
                    break
            
            if found_original:
                continue
            
            # Check for label encoded features (ending with _encoded)
            if feature.endswith('_encoded'):
                base_feature = feature.replace('_encoded', '')
                if base_feature in df.columns:
                    original_feature_names.add(base_feature)
                    continue
            
            # Check for general one-hot encoded features (feature_value pattern)
            if '_' in feature:
                potential_base = feature.split('_')[0]
                if potential_base in df.columns:
                    original_feature_names.add(potential_base)
                    continue
            
            # If no pattern matches, keep the feature as is (might be already original)
            original_feature_names.add(feature)
        
        # Convert to list and filter valid features
        valid_original_features = [f for f in original_feature_names if f in df.columns and f != target_column]
        
        st.session_state.original_features = valid_original_features
        st.success(f"✅ Identified {len(valid_original_features)} original features for prediction input")
        
        # Show the mapping for user clarity
        if len(selected_features) > len(valid_original_features):
            st.info(f"📊 **Feature Mapping:** {len(selected_features)} selected features → {len(valid_original_features)} original input features")
            with st.expander("View Feature Mapping Details", expanded=False):
                st.write("**Selected Features (for training):**", selected_features)
                st.write("**Original Features (for prediction input):**", valid_original_features)
        
        # Encode categorical features
        categorical_features = X.select_dtypes(include=['object']).columns
        if len(categorical_features) > 0:
            st.info("Encoding categorical features...")
            le_dict = {}
            for col in categorical_features:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                le_dict[col] = le
            # Store encoders for predictions
            st.session_state.label_encoders = le_dict
            st.success(f"✅ Encoded {len(categorical_features)} categorical features")
        else:
            st.session_state.label_encoders = {}
            
        # Store the final feature names after encoding (for model training)
        st.session_state.encoded_features = list(X.columns)
        
        # Encode target if classification
        if config['problem_type'] == "Classification":
            if y.dtype == 'object' or not all(isinstance(x, (int, float, np.integer, np.floating)) for x in y.dropna()):
                st.info("Encoding target variable...")
                le_target = LabelEncoder()
                y_encoded = le_target.fit_transform(y.astype(str))
                y = pd.Series(y_encoded, index=y.index)
                st.session_state.target_encoder = le_target
                st.success(f"✅ Target encoded: {len(le_target.classes_)} classes")
            else:
                st.session_state.target_encoder = None
                st.info("Target is already numeric for classification")
        else:  # Regression
            st.session_state.target_encoder = None
            # Ensure target is numeric for regression
            if y.dtype == 'object':
                try:
                    y = pd.to_numeric(y, errors='coerce')
                    if y.isnull().sum() > 0:
                        st.error("❌ Cannot convert target to numeric values for regression")
                        return
                    st.success("✅ Target converted to numeric for regression")
                except:
                    st.error("❌ Target must be numeric for regression problems")
                    return
            else:
                st.info("✅ Target is already numeric for regression")
        
        # Final check: ensure no target column leaked into features
        if target_column in X.columns:
            st.error(f"❌ Target column '{target_column}' found in features! Removing it.")
            X = X.drop(columns=[target_column])
            st.success("✅ Target column removed from features")
        
        # Check for any remaining non-numeric columns after encoding
        remaining_object_cols = X.select_dtypes(include=['object']).columns
        if len(remaining_object_cols) > 0:
            st.warning(f"⚠️ Converting remaining object columns to numeric: {list(remaining_object_cols)}")
            for col in remaining_object_cols:
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    X[col] = X[col].fillna(0)
                except:
                    # If conversion fails, use label encoding
                    le_temp = LabelEncoder()
                    X[col] = le_temp.fit_transform(X[col].astype(str))
        
        # Ensure all data is numeric
        X = X.select_dtypes(include=[np.number])
        
        # Final data validation
        st.info("🔍 Final data validation...")
        st.write(f"**Features shape:** {X.shape}")
        st.write(f"**Target shape:** {y.shape}")
        st.write(f"**Features data types:** {X.dtypes.value_counts().to_dict()}")
        st.write(f"**Target data type:** {y.dtype}")
        st.write(f"**Missing values in features:** {X.isnull().sum().sum()}")
        st.write(f"**Missing values in target:** {y.isnull().sum()}")
        
        if X.shape[0] != y.shape[0]:
            st.error("❌ Mismatch between features and target sizes!")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config['test_size'], random_state=42, stratify=y if config['problem_type'] == "Classification" and len(np.unique(y)) > 1 else None
        )
        
        st.success(f"✅ Data split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        
        # Scale features if enabled - CRITICAL for good performance
        if config['auto_feature_scaling']:
            st.info("🔧 Scaling features for optimal performance...")
            scaler = StandardScaler()
            
            # Fit scaler only on training data to prevent data leakage
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Convert back to DataFrame to maintain feature names
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            
            # Store scaler for later predictions
            st.session_state.scaler = scaler
            st.success("✅ Features scaled using StandardScaler")
            
            # Show scaling stats
            st.write(f"**Feature means after scaling:** {X_train_scaled.mean().abs().max():.4f} (should be ~0)")
            st.write(f"**Feature stds after scaling:** {X_train_scaled.std().abs().max():.4f} (should be ~1)")
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
            st.session_state.scaler = None
            st.info("⚠️ Feature scaling disabled - this may affect model performance")
        
        # For regression, also show target statistics and handle scaling
        if config['problem_type'] == "Regression":
            st.write(f"**Target statistics:**")
            st.write(f"  • Mean: {y_train.mean():.2f}")
            st.write(f"  • Std: {y_train.std():.2f}")
            st.write(f"  • Range: {y_train.min():.2f} to {y_train.max():.2f}")
            
            # Critical fix: Handle target scaling for large values
            target_std = y_train.std()
            target_mean = abs(y_train.mean())
            
            # Apply target scaling if values are very large (std > 1000 or mean > 10000)
            if target_std > 1000 or target_mean > 10000:
                st.warning(f"⚠️ Target has large values (std: {target_std:.0f}, mean: {target_mean:.0f})")
                
                # Offer target scaling options
                target_scaling_option = st.radio(
                    "Choose target scaling approach:",
                    ["Log Transform (recommended for large values)", "Standard Scaling", "Min-Max Scaling", "No Scaling"],
                    help="Target scaling can significantly improve model performance for large target values"
                )
                
                if target_scaling_option == "Log Transform (recommended for large values)":
                    if (y_train <= 0).any():
                        st.error("❌ Log transform requires all positive values. Adding constant offset...")
                        offset = abs(y_train.min()) + 1
                        y_train_scaled = np.log(y_train + offset)
                        y_test_scaled = np.log(y_test + offset)
                        st.session_state.target_transform = {'type': 'log', 'offset': offset}
                    else:
                        y_train_scaled = np.log(y_train)
                        y_test_scaled = np.log(y_test)
                        st.session_state.target_transform = {'type': 'log', 'offset': 0}
                    st.success(f"✅ Log transform applied. New std: {y_train_scaled.std():.2f}")
                    
                elif target_scaling_option == "Standard Scaling":
                    target_scaler = StandardScaler()
                    y_train_scaled = pd.Series(target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten(), index=y_train.index)
                    y_test_scaled = pd.Series(target_scaler.transform(y_test.values.reshape(-1, 1)).flatten(), index=y_test.index)
                    st.session_state.target_scaler = target_scaler
                    st.session_state.target_transform = {'type': 'standard'}
                    st.success(f"✅ Standard scaling applied. New std: {y_train_scaled.std():.2f}")
                    
                elif target_scaling_option == "Min-Max Scaling":
                    from sklearn.preprocessing import MinMaxScaler
                    target_scaler = MinMaxScaler()
                    y_train_scaled = pd.Series(target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten(), index=y_train.index)
                    y_test_scaled = pd.Series(target_scaler.transform(y_test.values.reshape(-1, 1)).flatten(), index=y_test.index)
                    st.session_state.target_scaler = target_scaler
                    st.session_state.target_transform = {'type': 'minmax'}
                    st.success(f"✅ Min-Max scaling applied. Range: [{y_train_scaled.min():.2f}, {y_train_scaled.max():.2f}]")
                    
                else:  # No Scaling
                    y_train_scaled = y_train
                    y_test_scaled = y_test
                    st.session_state.target_transform = {'type': 'none'}
                    st.info("⚠️ No target scaling applied - expect high RMSE values")
                
                # Update the target variables
                y_train = y_train_scaled
                y_test = y_test_scaled
                
            else:
                # Target values are reasonable, no scaling needed
                st.session_state.target_transform = {'type': 'none'}
                st.success("✅ Target values are in reasonable range - no scaling needed")
        
        # Store processed data for model training
        st.session_state.processed_data = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': list(X.columns)
        }
        
        # Training container
        st.markdown('<div class="training-container">', unsafe_allow_html=True)
        st.markdown("### Training Multiple Models")
        
        all_results = []
        trained_models = {}
        
        # Get processed data
        processed_data = st.session_state.get('processed_data', {})
        X_train_scaled = processed_data.get('X_train', X_train_scaled)
        X_test_scaled = processed_data.get('X_test', X_test_scaled)
        y_train = processed_data.get('y_train', y_train)
        y_test = processed_data.get('y_test', y_test)
        
        for i, model_name in enumerate(st.session_state.selected_models_for_training):
            st.markdown(f"#### Training {model_name} ({i+1}/{len(st.session_state.selected_models_for_training)})")
            
            # Use appropriate data based on model requirements
            if model_name in ["Support Vector Machine", "Support Vector Regression", "Logistic Regression", "Neural Network"]:
                # These models require scaled data
                X_train_use = X_train_scaled.values if hasattr(X_train_scaled, 'values') else X_train_scaled
                X_test_use = X_test_scaled.values if hasattr(X_test_scaled, 'values') else X_test_scaled
                st.info(f"Using scaled features for {model_name}")
            else:
                # Tree-based models can work with or without scaling
                X_train_use = X_train_scaled.values if hasattr(X_train_scaled, 'values') else X_train_scaled
                X_test_use = X_test_scaled.values if hasattr(X_test_scaled, 'values') else X_test_scaled
                st.info(f"Using processed features for {model_name}")
            
            # Ensure data is numeric arrays
            if hasattr(X_train_use, 'values'):
                X_train_use = X_train_use.values
            if hasattr(X_test_use, 'values'):
                X_test_use = X_test_use.values
            if hasattr(y_train, 'values'):
                y_train_use = y_train.values
            else:
                y_train_use = y_train
            if hasattr(y_test, 'values'):
                y_test_use = y_test.values
            else:
                y_test_use = y_test
            
            # Validate data shapes and types
            st.write(f"**Training data shape:** {X_train_use.shape}")
            st.write(f"**Test data shape:** {X_test_use.shape}")
            st.write(f"**Target train shape:** {y_train_use.shape}")
            st.write(f"**Target test shape:** {y_test_use.shape}")
            
            # Check for any remaining issues
            if np.isnan(X_train_use).any() or np.isnan(X_test_use).any():
                st.error("❌ NaN values found in features after preprocessing!")
                continue
            
            if np.isnan(y_train_use).any() or np.isnan(y_test_use).any():
                st.error("❌ NaN values found in target after preprocessing!")
                continue
            
            # Train with progress simulation
            results, trained_model = simulate_advanced_training(
                model_name, X_train_use, y_train_use, X_test_use, y_test_use, 
                epochs=config['epochs'], problem_type=config['problem_type']
            )
            
            if results:
                all_results.append(results)
                trained_models[model_name] = trained_model
                st.success(f"✅ {model_name} training completed successfully!")
            else:
                st.error(f"❌ {model_name} training failed!")
            
            st.markdown("---")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Store results
        st.session_state.model_results = all_results
        st.session_state.trained_models = trained_models
        
        # Navigation
        if st.button("View Results & Deploy Models →", type="primary", use_container_width=True):
            st.session_state.training_step = 6
            st.rerun()
    
    # Step 6: Results & Deployment
    elif st.session_state.training_step == 6:
        st.markdown('<div class="section-header"><h3>Step 6: Results & Model Deployment</h3></div>', unsafe_allow_html=True)
        
        # Validate training configuration
        if 'training_config' not in st.session_state or not st.session_state.training_config:
            st.error("❌ No training configuration found. Please complete the training process first.")
            if st.button("← Start Training Process"):
                st.session_state.training_step = 1
                st.rerun()
            return
        
        # Validate that we have trained models
        if not st.session_state.trained_models:
            st.warning("⚠️ No trained models found. Please complete the training process.")
            if st.button("← Back to Training"):
                st.session_state.training_step = 5
                st.rerun()
            return
        
        # Validate training configuration has required fields
        required_config_fields = ['target_column', 'problem_type', 'selected_features']
        missing_config = [field for field in required_config_fields if field not in st.session_state.training_config]
        
        if missing_config:
            st.error(f"❌ Training configuration incomplete. Missing: {missing_config}")
            if st.button("← Restart Training Process"):
                st.session_state.training_step = 1
                st.rerun()
            return
        
        if st.session_state.model_results:
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
            
            # Champion model announcement
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
            st.markdown("### Complete Results Comparison")
            st.dataframe(results_df, use_container_width=True)
            
            # Model actions and prediction
            st.markdown("### Model Actions")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Export Results")
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "Download Results CSV",
                    csv,
                    file_name=f"ml_training_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                st.markdown("#### Download Model")
                model_to_download = st.selectbox(
                    "Select model to download",
                    list(st.session_state.trained_models.keys())
                )
                if st.button("Download Selected Model", use_container_width=True):
                    model_data = {
                        'model': st.session_state.trained_models[model_to_download],
                        'feature_names': st.session_state.training_config['selected_features'],
                        'target_column': st.session_state.training_config['target_column'],
                        'problem_type': st.session_state.training_config['problem_type'],
                        'scaler': st.session_state.get('scaler', None),
                        'label_encoders': st.session_state.get('label_encoders', {}),
                        'target_encoder': st.session_state.get('target_encoder', None)
                    }
                    model_bytes = pickle.dumps(model_data)
                    st.download_button(
                        "Click to Download",
                        model_bytes,
                        file_name=f"{model_to_download.replace(' ', '_').lower()}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
            
            with col3:
                st.markdown("#### Other Actions")
                if st.button("Train New Models", use_container_width=True):
                    st.session_state.training_step = 3
                    st.rerun()
                if st.button("New Project", use_container_width=True):
                    # Reset all session state
                    for key in list(st.session_state.keys()):
                        if key.startswith(('training_', 'model_', 'selected_')):
                            del st.session_state[key]
                    st.session_state.training_step = 1
                    st.rerun()
            
            # Prediction Section
            st.markdown("### Model Prediction")
            st.markdown("*Test your trained models with new data*")
            
            # Model selection for prediction
            prediction_model = st.selectbox(
                "Select model for prediction",
                list(st.session_state.trained_models.keys()),
                key="prediction_model_select"
            )
            
            # Prediction input method
            input_method = st.radio(
                "Choose input method:",
                ["Manual Input", "JSON Dictionary"],
                horizontal=True
            )
            
            # Show what features are expected
            with st.expander("📋 View Required Features", expanded=False):
                st.write("**Expected Features for Prediction:**")
                
                # Use original features (before encoding) for input
                original_features = st.session_state.get('original_features', st.session_state.training_config['selected_features'])
                target_column = st.session_state.training_config.get('target_column', None)
                
                # Filter out target column from features (safety check)
                feature_names = [f for f in original_features if f != target_column] if target_column else original_features
                
                if target_column and target_column in original_features:
                    st.info(f"ℹ️ Target column '{target_column}' excluded from input features (this is what we're predicting)")
                
                # Show the difference between original and encoded features
                encoded_features = st.session_state.get('encoded_features', [])
                if len(encoded_features) != len(feature_names):
                    st.info("ℹ️ **Note:** During training, some features were automatically encoded (e.g., categorical → numeric). You only need to provide the original feature values.")
                
                # Show features in a more readable format
                st.write("**Required Original Features (provide these values):**")
                for i, feature in enumerate(feature_names, 1):
                    # Show sample values if available
                    sample_info = ""
                    if feature in df.columns:
                        if df[feature].dtype == 'object':
                            unique_vals = df[feature].unique()[:3]  # Show first 3 unique values
                            sample_info = f" (e.g., {', '.join([str(v) for v in unique_vals])}{'...' if len(df[feature].unique()) > 3 else ''})"
                        else:
                            min_val, max_val = df[feature].min(), df[feature].max()
                            sample_info = f" (range: {min_val:.2f} to {max_val:.2f})"
                    
                    st.write(f"{i:2d}. `{feature}`{sample_info}")
                
                # Generate a complete JSON template with all required original features
                st.write("**Complete JSON Template (copy and modify values):**")
                template_data = {}
                for feature in feature_names:
                    if feature in df.columns:
                        sample_val = df[feature].iloc[0]
                        if pd.isna(sample_val):
                            template_data[feature] = "your_value_here"
                        elif df[feature].dtype in ['int64', 'float64']:
                            template_data[feature] = float(sample_val) if df[feature].dtype == 'float64' else int(sample_val)
                        else:
                            template_data[feature] = str(sample_val)
                    else:
                        template_data[feature] = "your_value_here"
                
                st.code(json.dumps(template_data, indent=2), language="json")
                
                # Show sample prediction formats
                st.write("**Alternative: DataFrame Records Format:**")
                sample_df_format = {
                    "dataframe_records": [template_data]
                }
                st.code(json.dumps(sample_df_format, indent=2), language="json")
                
                # Show tips for features with spaces
                features_with_spaces = [f for f in feature_names if ' ' in f]
                if features_with_spaces:
                    st.info(f"💡 **Note:** Features with spaces in names: {', '.join([f'`{f}`' for f in features_with_spaces])}")
                    st.write("Make sure to include the exact feature names with spaces in your JSON.")
            
            if input_method == "Manual Input":
                st.markdown("#### Enter values for each feature:")
                input_values = {}
                
                # Use original features (before encoding) for input
                original_features = st.session_state.get('original_features', st.session_state.training_config['selected_features'])
                target_column = st.session_state.training_config.get('target_column', None)
                
                # Filter out target column from features (safety check)
                feature_names = [f for f in original_features if f != target_column] if target_column else original_features
                
                if target_column and target_column in original_features:
                    st.info(f"ℹ️ Note: Target column '{target_column}' is excluded from input (this is what we're predicting)")
                
                # Create input fields for each original feature
                cols = st.columns(3)
                for i, feature in enumerate(feature_names):
                    with cols[i % 3]:
                        # Determine input type based on original feature data
                        if feature in df.columns:
                            sample_val = df[feature].iloc[0] if not pd.isna(df[feature].iloc[0]) else 0
                            if df[feature].dtype in ['int64', 'float64']:
                                input_values[feature] = st.number_input(
                                    f"{feature}",
                                    value=float(sample_val),
                                    key=f"input_{feature}"
                                )
                            else:
                                unique_vals = df[feature].unique()
                                if len(unique_vals) <= 20:  # Use selectbox for reasonable number of options
                                    input_values[feature] = st.selectbox(
                                        f"{feature}",
                                        unique_vals,
                                        key=f"input_{feature}"
                                    )
                                else:
                                    input_values[feature] = st.text_input(
                                        f"{feature}",
                                        value=str(sample_val),
                                        key=f"input_{feature}"
                                    )
                        else:
                            # Feature not in original dataset (shouldn't happen with original features)
                            input_values[feature] = st.text_input(
                                f"{feature}",
                                placeholder="Enter value",
                                key=f"input_{feature}"
                            )
                
                if st.button("Predict with Manual Input", type="primary"):
                    prediction_result = make_prediction(
                        prediction_model, 
                        input_values, 
                        st.session_state.training_config
                    )
                    if prediction_result is not None:
                        st.success(f"Prediction: {prediction_result}")
            
            else:  # JSON Dictionary
                st.markdown("#### Enter data as JSON:")
                st.markdown("**Example format (using original features):**")
                
                # Use original features for example
                original_features = st.session_state.get('original_features', st.session_state.training_config['selected_features'])
                target_column = st.session_state.training_config.get('target_column', None)
                feature_names = [f for f in original_features if f != target_column] if target_column else original_features
                
                example_json = {
                    "dataframe_records": [
                        {feature: f"value_for_{feature}" for feature in feature_names[:5]}
                    ]
                }
                st.code(json.dumps(example_json, indent=2), language="json")
                
                st.markdown("**Alternative format (single record):**")
                single_example = {feature: f"value_for_{feature}" for feature in feature_names[:5]}
                st.code(json.dumps(single_example, indent=2), language="json")
                
                json_input = st.text_area(
                    "Paste your JSON data here:",
                    height=300,
                    placeholder='{"dataframe_records": [{"feature1": value1, "feature2": value2, ...}]} \n\nOR\n\n{"feature1": value1, "feature2": value2, ...}'
                )
                
                # Add helpful buttons
                col_json1, col_json2 = st.columns(2)
                with col_json1:
                    if st.button("Validate JSON Format"):
                        try:
                            cleaned_input = json_input.strip()
                            import re
                            cleaned_input = re.sub(r',(\s*[}\]])', r'\1', cleaned_input)
                            json.loads(cleaned_input)
                            st.success("✅ Valid JSON format!")
                        except json.JSONDecodeError as e:
                            st.error(f"❌ Invalid JSON: {str(e)}")
                
                with col_json2:
                    if st.button("Show Sample Data"):
                        # Use original features (before encoding) for sample
                        original_features = st.session_state.get('original_features', st.session_state.training_config['selected_features'])
                        target_column = st.session_state.training_config.get('target_column', None)
                        
                        # Filter out target column from features (safety check)
                        feature_names = [f for f in original_features if f != target_column] if target_column else original_features
                        
                        sample_data = {}
                        for feature in feature_names:
                            if feature in df.columns:
                                sample_val = df[feature].iloc[0]
                                # Convert to serializable type
                                if pd.isna(sample_val):
                                    sample_data[feature] = None
                                elif isinstance(sample_val, (np.int64, np.int32)):
                                    sample_data[feature] = int(sample_val)
                                elif isinstance(sample_val, (np.float64, np.float32)):
                                    sample_data[feature] = float(sample_val)
                                else:
                                    sample_data[feature] = str(sample_val)
                            else:
                                # This shouldn't happen with original features
                                sample_data[feature] = "value_here"
                        
                        if target_column and target_column in original_features:
                            st.info(f"ℹ️ Target column '{target_column}' excluded from sample")
                        
                        st.write("**Copy this sample and modify the values:**")
                        st.code(json.dumps(sample_data, indent=2), language="json")
                        
                        # Also show dataframe_records format
                        st.write("**Or use this format for multiple predictions:**")
                        df_records_sample = {"dataframe_records": [sample_data]}
                        st.code(json.dumps(df_records_sample, indent=2), language="json")
                
                if st.button("Predict with JSON Input", type="primary"):
                    try:
                        # Clean the JSON input (remove trailing commas and fix common issues)
                        cleaned_input = json_input.strip()
                        # Remove trailing commas before closing braces/brackets
                        import re
                        cleaned_input = re.sub(r',(\s*[}\]])', r'\1', cleaned_input)
                        
                        json_data = json.loads(cleaned_input)
                        
                        # Handle both formats
                        records_to_predict = []
                        if "dataframe_records" in json_data:
                            records_to_predict = json_data["dataframe_records"]
                        elif isinstance(json_data, dict):
                            # Single record format
                            records_to_predict = [json_data]
                        else:
                            st.error("Invalid JSON structure. Use either single record or dataframe_records format.")
                            return
                        
                        # Make predictions for each record
                        for i, record in enumerate(records_to_predict):
                            st.markdown(f"**Prediction {i+1}:**")
                            prediction_result = make_prediction(
                                prediction_model, 
                                record, 
                                st.session_state.training_config
                            )
                            if prediction_result is not None:
                                st.success(f"Result: {prediction_result}")
                            st.markdown("---")
                            
                    except json.JSONDecodeError as e:
                        st.error(f"Invalid JSON format: {str(e)}")
                        st.info("💡 **Common fixes:**\n- Remove trailing commas (,) before closing braces\n- Ensure all strings are quoted\n- Check for proper bracket/brace matching")
                    except Exception as e:
                        st.error(f"Error processing prediction: {str(e)}")
        
        else:
            st.error("No training results found. Please go back and train some models.")
            if st.button("← Back to Training"):
                st.session_state.training_step = 5
                st.rerun()
