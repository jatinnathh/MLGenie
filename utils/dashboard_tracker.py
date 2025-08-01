"""
Dashboard Statistics Tracker
Centralized tracking for all MLGenie operations
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Optional, Any

def init_dashboard_stats():
    """Initialize dashboard statistics in session state"""
    if 'dashboard_stats' not in st.session_state:
        st.session_state.dashboard_stats = {
            'datasets_count': 0,
            'feature_engineering_count': 0,
            'models_trained': 0,
            'ml_models': 0,
            'dl_models': 0,
            'best_model': None,
            'jobs_in_progress': 0
        }
    
    if 'model_leaderboard' not in st.session_state:
        st.session_state.model_leaderboard = []
    
    if 'recent_activities' not in st.session_state:
        st.session_state.recent_activities = []

def increment_dataset_count():
    """Increment dataset upload counter"""
    init_dashboard_stats()
    st.session_state.dashboard_stats['datasets_count'] += 1
    
    # Log activity
    log_activity(
        activity_type="dataset_upload",
        description="New dataset uploaded and processed",
        status="success"
    )

# Convenience function aliases for easier imports
def increment_datasets_count():
    """Alias for increment_dataset_count"""
    return increment_dataset_count()

def track_feature_engineering(operation_description: str = "Feature engineering operation"):
    """Alias for increment_feature_engineering_count"""
    return increment_feature_engineering_count(operation_description)

def track_model_training(model_type: str = "ML", model_data: Dict = None):
    """Alias for increment_model_count"""
    return increment_model_count(model_type, model_data)

def increment_feature_engineering_count(operation_description: str = "Feature engineering operation"):
    """Increment feature engineering operations counter"""
    init_dashboard_stats()
    st.session_state.dashboard_stats['feature_engineering_count'] += 1
    
    # Log activity
    log_activity(
        activity_type="feature_engineering",
        description=operation_description,
        status="success"
    )

def increment_model_count(model_type: str = "ML", model_data: Dict = None):
    """Increment model training counters"""
    init_dashboard_stats()
    
    # Increment total models
    st.session_state.dashboard_stats['models_trained'] += 1
    
    # Increment specific model type
    if model_type.upper() == "ML":
        st.session_state.dashboard_stats['ml_models'] += 1
    elif model_type.upper() == "DL":
        st.session_state.dashboard_stats['dl_models'] += 1
    
    # Add to leaderboard if model data provided
    if model_data:
        add_model_to_leaderboard(model_data)
        
        # Update best model if this one is better
        update_best_model(model_data)
    
    # Log activity
    model_name = model_data.get('name', 'Unknown Model') if model_data else 'New Model'
    log_activity(
        activity_type="model_training",
        description=f"{model_type} model '{model_name}' trained successfully",
        status="success"
    )

def add_model_to_leaderboard(model_data: Dict):
    """Add model to leaderboard and sort by score"""
    init_dashboard_stats()
    
    # Add timestamp if not present
    if 'timestamp' not in model_data:
        model_data['timestamp'] = datetime.now()
    
    # Add to leaderboard
    st.session_state.model_leaderboard.append(model_data)
    
    # Sort by score (descending)
    st.session_state.model_leaderboard.sort(
        key=lambda x: x.get('best_score', 0), 
        reverse=True
    )
    
    # Keep only top 50 models
    st.session_state.model_leaderboard = st.session_state.model_leaderboard[:50]

def update_best_model_data(model_data: Dict):
    """Update best model if this one is better"""
    init_dashboard_stats()
    
    current_best = st.session_state.dashboard_stats['best_model']
    new_score = model_data.get('best_score', 0)
    
    if current_best is None or new_score > current_best.get('best_score', 0):
        st.session_state.dashboard_stats['best_model'] = model_data.copy()

def update_best_model(model_name: str, metric_name: str, score: float):
    """Update best model with individual parameters"""
    model_data = {
        'model_name': model_name,
        'metric_name': metric_name,
        'best_score': score,
        'timestamp': datetime.now()
    }
    return update_best_model_data(model_data)

def increment_jobs_in_progress():
    """Increment jobs in progress counter"""
    init_dashboard_stats()
    st.session_state.dashboard_stats['jobs_in_progress'] += 1

def decrement_jobs_in_progress():
    """Decrement jobs in progress counter"""
    init_dashboard_stats()
    if st.session_state.dashboard_stats['jobs_in_progress'] > 0:
        st.session_state.dashboard_stats['jobs_in_progress'] -= 1

def log_activity(activity_type: str, description: str, status: str = "success", metadata: Dict = None):
    """Log activity for recent activities feed"""
    init_dashboard_stats()
    
    activity = {
        'type': activity_type,
        'description': description,
        'timestamp': datetime.now(),
        'status': status,
        'metadata': metadata or {}
    }
    
    # Add to recent activities
    st.session_state.recent_activities.append(activity)
    
    # Keep only last 100 activities
    st.session_state.recent_activities = st.session_state.recent_activities[-100:]

def get_dashboard_stats() -> Dict:
    """Get current dashboard statistics"""
    init_dashboard_stats()
    stats = st.session_state.dashboard_stats.copy()
    stats['recent_activities'] = st.session_state.recent_activities.copy()
    stats['model_leaderboard'] = st.session_state.model_leaderboard.copy()
    return stats

def reset_dashboard_stats():
    """Reset all dashboard statistics"""
    st.session_state.dashboard_stats = {
        'datasets_count': 0,
        'feature_engineering_count': 0,
        'models_trained': 0,
        'ml_models': 0,
        'dl_models': 0,
        'best_model': None,
        'jobs_in_progress': 0
    }
    st.session_state.model_leaderboard = []
    st.session_state.recent_activities = []
    
    # Log reset activity
    log_activity(
        activity_type="system",
        description="Dashboard statistics reset",
        status="success"
    )

def log_error(operation: str, error_message: str):
    """Log error activity"""
    log_activity(
        activity_type="error",
        description=f"Error in {operation}: {error_message}",
        status="failed"
    )
