"""
Global Persistent Dashboard Tracker
Tracks statistics across all users and sessions using file-based storage
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional, Any
import threading
import streamlit as st

# Global file path for persistent storage
STATS_FILE = "global_dashboard_stats.json"
ACTIVITIES_FILE = "global_activities.json"

# Thread lock for file operations
_file_lock = threading.Lock()

def _load_global_stats() -> Dict:
    """Load global statistics from file"""
    default_stats = {
        'datasets_count': 0,
        'feature_engineering_count': 0,
        'models_trained': 0,
        'ml_models': 0,
        'dl_models': 0,
        'best_model': None,
        'jobs_in_progress': 0,
        'last_updated': None
    }
    
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, 'r') as f:
                stats = json.load(f)
                return {**default_stats, **stats}
        return default_stats
    except Exception:
        return default_stats

def _save_global_stats(stats: Dict):
    """Save global statistics to file"""
    try:
        with _file_lock:
            stats['last_updated'] = datetime.now().isoformat()
            with open(STATS_FILE, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
    except Exception:
        pass

def _load_global_activities() -> list:
    """Load global activities from file"""
    try:
        if os.path.exists(ACTIVITIES_FILE):
            with open(ACTIVITIES_FILE, 'r') as f:
                return json.load(f)
        return []
    except Exception:
        return []

def _save_global_activities(activities: list):
    """Save global activities to file"""
    try:
        with _file_lock:
            # Keep only last 100 activities
            activities = activities[-100:]
            with open(ACTIVITIES_FILE, 'w') as f:
                json.dump(activities, f, indent=2, default=str)
    except Exception:
        pass

def increment_datasets_count():
    """Increment dataset upload counter globally"""
    stats = _load_global_stats()
    stats['datasets_count'] += 1
    _save_global_stats(stats)
    
    # Log activity
    log_activity(
        activity_type="dataset_upload",
        description="New dataset uploaded and processed",
        status="success"
    )

def track_feature_engineering(operation_description: str = "Feature engineering operation"):
    """Increment feature engineering operations counter globally"""
    stats = _load_global_stats()
    stats['feature_engineering_count'] += 1
    _save_global_stats(stats)
    
    # Log activity
    log_activity(
        activity_type="feature_engineering",
        description=operation_description,
        status="success"
    )

def track_model_training(model_type: str = "ML", model_data: Dict = None):
    """Increment model training counters globally"""
    stats = _load_global_stats()
    
    # Increment total models
    stats['models_trained'] += 1
    
    # Increment specific model type
    if model_type.upper() == "ML":
        stats['ml_models'] += 1
    elif model_type.upper() == "DL":
        stats['dl_models'] += 1
    
    _save_global_stats(stats)
    
    # Log activity
    model_name = model_data.get('model_name', 'New Model') if model_data else 'New Model'
    log_activity(
        activity_type="model_training",
        description=f"{model_type.upper()} model '{model_name}' trained successfully",
        status="success",
        metadata=model_data
    )

def update_best_model(model_name: str, metric_name: str, score: float):
    """Update best model with individual parameters globally"""
    stats = _load_global_stats()
    
    model_data = {
        'model_name': model_name,
        'metric_name': metric_name,
        'best_score': score,
        'timestamp': datetime.now().isoformat()
    }
    
    current_best = stats.get('best_model')
    
    if current_best is None or score > current_best.get('best_score', 0):
        stats['best_model'] = model_data
        _save_global_stats(stats)
        
        log_activity(
            activity_type="best_model_update",
            description=f"New best model: {model_name} ({metric_name}: {score:.4f})",
            status="success",
            metadata=model_data
        )

def log_activity(activity_type: str, description: str, status: str = "success", metadata: Dict = None):
    """Log activity globally"""
    activity = {
        'activity_type': activity_type,
        'description': description,
        'status': status,
        'timestamp': datetime.now().isoformat(),
        'metadata': metadata or {}
    }
    
    activities = _load_global_activities()
    activities.append(activity)
    _save_global_activities(activities)

def get_dashboard_stats() -> Dict:
    """Get current global dashboard statistics"""
    stats = _load_global_stats()
    activities = _load_global_activities()
    
    # Add recent activities to stats
    stats['recent_activities'] = activities[-20:]  # Last 20 activities
    stats['total_activities'] = len(activities)
    
    return stats

def reset_dashboard_stats():
    """Reset all global dashboard statistics"""
    default_stats = {
        'datasets_count': 0,
        'feature_engineering_count': 0,
        'models_trained': 0,
        'ml_models': 0,
        'dl_models': 0,
        'best_model': None,
        'jobs_in_progress': 0,
        'last_updated': datetime.now().isoformat()
    }
    
    _save_global_stats(default_stats)
    _save_global_activities([])
    
    log_activity(
        activity_type="system_reset",
        description="Dashboard statistics reset",
        status="success"
    )

def get_system_info() -> Dict:
    """Get system information for debugging"""
    try:
        stats_exists = os.path.exists(STATS_FILE)
        activities_exists = os.path.exists(ACTIVITIES_FILE)
        
        stats_size = os.path.getsize(STATS_FILE) if stats_exists else 0
        activities_size = os.path.getsize(ACTIVITIES_FILE) if activities_exists else 0
        
        return {
            'stats_file_exists': stats_exists,
            'activities_file_exists': activities_exists,
            'stats_file_size': stats_size,
            'activities_file_size': activities_size,
            'working_directory': os.getcwd(),
            'stats_file_path': os.path.abspath(STATS_FILE),
            'activities_file_path': os.path.abspath(ACTIVITIES_FILE)
        }
    except Exception as e:
        return {'error': str(e)}

# Initialize files if they don't exist
def initialize_global_tracking():
    """Initialize global tracking files"""
    if not os.path.exists(STATS_FILE):
        reset_dashboard_stats()
    
    # Add system startup activity
    log_activity(
        activity_type="system_startup",
        description="MLGenie dashboard tracking system initialized",
        status="success"
    )

# Auto-initialize when module is imported
initialize_global_tracking()
