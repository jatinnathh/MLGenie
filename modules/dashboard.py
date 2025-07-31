import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from typing import Dict, List

def init_session_stats():
    """Initialize session state statistics"""
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

def get_platform_css():
    """Get dark theme CSS styling"""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;

;
        color: white;
        min-height: 100vh;
    }
    
    /* Dark theme for main content */
    .main .block-container {
        background: transparent;
        padding-top: 2rem;
    }
    
    /* Modern Metric Cards with dark theme */
    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(64, 224, 255, 0.1), transparent);
        transition: left 0.6s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-12px) scale(1.03);
        box-shadow: 0 20px 60px rgba(64, 224, 255, 0.2);
        border-color: rgba(64, 224, 255, 0.4);
        background: rgba(255, 255, 255, 0.12);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #40e0ff, #64b5f6, #81c784);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 500;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    
    .metric-subtitle {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.6);
        margin-top: 0.25rem;
    }
    
    /* Activity Cards with dark theme */
    .activity-card {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        position: relative;
    }
    
    .activity-card:hover {
        transform: translateX(12px);
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(64, 224, 255, 0.3);
        box-shadow: 0 12px 35px rgba(64, 224, 255, 0.15);
    }
    
    /* Header Styling with dark theme */
    .dashboard-header {
        text-align: center;
        padding: 3rem 0;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(25px);
        border-radius: 20px;
        margin: 1rem 0 3rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .dashboard-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #40e0ff, #64b5f6, #81c784);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        text-shadow: 0 0 30px rgba(64, 224, 255, 0.3);
    }
    
    .dashboard-subtitle {
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.7);
        margin-top: 0.8rem;
        font-weight: 400;
    }
    
    /* Chart Container with dark theme */
    .chart-container {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        transition: all 0.4s ease;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    .chart-container:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 50px rgba(64, 224, 255, 0.15);
        border-color: rgba(64, 224, 255, 0.3);
        background: rgba(255, 255, 255, 0.08);
    }
    
    /* Status Indicators with dark theme colors */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 10px;
        animation: pulse 2s infinite;
    }
    
    .status-online { 
        background: #40e0ff;
        box-shadow: 0 0 10px rgba(64, 224, 255, 0.5);
    }
    .status-warning { 
        background: #ffa726;
        box-shadow: 0 0 10px rgba(255, 167, 38, 0.5);
    }
    .status-error { 
        background: #ef5350;
        box-shadow: 0 0 10px rgba(239, 83, 80, 0.5);
    }
    
    @keyframes pulse {
        0% { 
            box-shadow: 0 0 0 0 rgba(64, 224, 255, 0.7);
            transform: scale(1);
        }
        70% { 
            box-shadow: 0 0 0 12px rgba(64, 224, 255, 0);
            transform: scale(1.1);
        }
        100% { 
            box-shadow: 0 0 0 0 rgba(64, 224, 255, 0);
            transform: scale(1);
        }
    }
    
    /* Section Headers with dark theme */
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        background: linear-gradient(135deg, #40e0ff, #64b5f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 3rem 0 1.5rem 0;
        text-align: center;
        padding-bottom: 0.8rem;
        border-bottom: 2px solid rgba(64, 224, 255, 0.3);
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    /* Button Styling with dark theme */
    .stButton > button {
        background: linear-gradient(135deg, #40e0ff, #64b5f6);
        color: #0f0f23;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.4s ease;
        box-shadow: 0 6px 20px rgba(64, 224, 255, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) scale(1.05);
        box-shadow: 0 12px 30px rgba(64, 224, 255, 0.4);
        background: linear-gradient(135deg, #64b5f6, #40e0ff);
    }
    
    /* Streamlit specific dark theme overrides */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background: rgba(255, 255, 255, 0.02);
    }
    
    .stMarkdown h3 {
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* Plotly chart dark theme */
    .js-plotly-plot {
        background: transparent !important;
    }
    
    /* Remove Streamlit Branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: visible; }
    
    /* Ensure sidebar controls are visible */
    button[data-testid="collapsedControl"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: rgba(255, 255, 255, 0.8) !important;
        border-radius: 4px !important;
        padding: 4px 8px !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
        margin: 0.5rem !important;
        cursor: pointer !important;
    }
    
    button[data-testid="collapsedControl"]:hover {
        background: rgba(255, 255, 255, 0.2) !important;
        color: white !important;
        border-color: rgba(255, 255, 255, 0.4) !important;
    }
    
    /* Make sure sidebar toggle button is properly styled */
    section[data-testid="stSidebar"] button[data-testid="collapsedControl"] {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: rgba(255, 255, 255, 0.8) !important;
        border-radius: 4px !important;
        padding: 4px 8px !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
    }
    
    section[data-testid="stSidebar"] button[data-testid="collapsedControl"]:hover {
        background: rgba(255, 255, 255, 0.2) !important;
        color: white !important;
    }
    
    /* Ensure sidebar container doesn't hide controls */
    section[data-testid="stSidebar"] {
        position: relative !important;
    }
    
    /* Make sure sidebar button container is visible */
    .css-1vbkxwb, .css-1aehpvj {
        display: block !important;
        visibility: visible !important;
    }
    
    /* Custom scrollbar for dark theme */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(64, 224, 255, 0.3);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(64, 224, 255, 0.5);
    }
    </style>
    """

def create_metric_card(title: str, value: str, subtitle: str = ""):
    """Create a metric card with hover effects"""
    subtitle_html = f"<div class='metric-subtitle'>{subtitle}</div>" if subtitle else ""
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)

def create_activity_item(title: str, description: str, time_ago: str, status: str = "online"):
    """Create an activity item"""
    st.markdown(f"""
    <div class="activity-card">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span class="status-indicator status-{status}"></span>
            <strong style="color: rgba(255, 255, 255, 0.9); font-size: 1.1rem;">{title}</strong>
        </div>
        <div style="color: rgba(255, 255, 255, 0.7); margin-bottom: 0.5rem;">{description}</div>
        <div style="color: rgba(255, 255, 255, 0.5); font-size: 0.9rem;">{time_ago}</div>
    </div>
    """, unsafe_allow_html=True)

def get_real_model_performance_data():
    """Get real model performance data from session state"""
    if 'model_leaderboard' not in st.session_state or not st.session_state.model_leaderboard:
        return None
    
    # Extract algorithm usage from model leaderboard
    algorithms = {}
    for model in st.session_state.model_leaderboard:
        algo = model.get('algorithm', 'Unknown')
        algorithms[algo] = algorithms.get(algo, 0) + 1
    
    return algorithms

def create_algorithm_distribution_chart():
    """Create algorithm usage distribution chart from real data"""
    real_data = get_real_model_performance_data()
    
    if real_data and len(real_data) > 0:
        algorithms = list(real_data.keys())
        usage_counts = list(real_data.values())
    else:
        # Fallback to sample data if no real data
        algorithms = ['No Models', 'Train Some Models', 'To See Real Data']
        usage_counts = [1, 1, 1]
    
    fig = px.pie(
        values=usage_counts,
        names=algorithms,
        title="Algorithm Usage Distribution (Real Data)",
        color_discrete_sequence=['#40e0ff', '#64b5f6', '#81c784', '#ffa726', '#ef5350', '#ab47bc', '#26c6da', '#66bb6a']
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='rgba(255, 255, 255, 0.9)',
        title_font_size=18,
        title_x=0.5,
        title_font_color='rgba(255, 255, 255, 0.9)',
        legend=dict(
            font_color='rgba(255, 255, 255, 0.8)',
            bgcolor='rgba(0,0,0,0)'
        )
    )
    
    return fig

def create_performance_trend_chart():
    """Create model performance trend from real data"""
    if 'model_leaderboard' not in st.session_state or not st.session_state.model_leaderboard:
        # Show placeholder chart
        fig = go.Figure()
        fig.add_annotation(
            text="Train models to see performance trends",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font_size=16, font_color="rgba(255, 255, 255, 0.7)"
        )
        fig.update_layout(
            title='Model Performance Trends (Train Models to See Data)',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='rgba(255, 255, 255, 0.9)',
            title_font_size=18,
            title_x=0.5,
            title_font_color='rgba(255, 255, 255, 0.9)',
            xaxis=dict(showgrid=False, showticklabels=False, color='rgba(255, 255, 255, 0.7)'),
            yaxis=dict(showgrid=False, showticklabels=False, color='rgba(255, 255, 255, 0.7)')
        )
        return fig
    
    # Create trend from real model data
    models = st.session_state.model_leaderboard
    dates = []
    accuracies = []
    
    for i, model in enumerate(models[:10]):  # Show last 10 models
        dates.append(datetime.now() - timedelta(days=len(models)-i))
        accuracies.append(model.get('best_score', 0) * 100)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=accuracies,
        mode='lines+markers',
        name='Model Accuracy (%)',
        line=dict(color='#40e0ff', width=4),
        marker=dict(size=10, color='#40e0ff', line=dict(width=2, color='#64b5f6'))
    ))
    
    fig.update_layout(
        title='Model Performance Trends (Real Data)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='rgba(255, 255, 255, 0.9)',
        title_font_size=18,
        title_x=0.5,
        title_font_color='rgba(255, 255, 255, 0.9)',
        yaxis=dict(
            title='Accuracy (%)', 
            range=[0, 100],
            gridcolor='rgba(255, 255, 255, 0.1)',
            color='rgba(255, 255, 255, 0.7)'
        ),
        xaxis=dict(
            title='Training Date',
            gridcolor='rgba(255, 255, 255, 0.1)',
            color='rgba(255, 255, 255, 0.7)'
        ),
        legend=dict(
            font_color='rgba(255, 255, 255, 0.8)',
            bgcolor='rgba(0,0,0,0)'
        )
    )
    
    return fig

def create_system_health_indicators():
    """Create system health indicators"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="chart-container" style="text-align: center;">
            <span class="status-indicator status-online"></span>
            <div style="color: rgba(255, 255, 255, 0.9); font-weight: 600; margin-top: 0.5rem;">Data Pipeline</div>
            <div style="color: rgba(255, 255, 255, 0.6);">Ready</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="chart-container" style="text-align: center;">
            <span class="status-indicator status-online"></span>
            <div style="color: rgba(255, 255, 255, 0.9); font-weight: 600; margin-top: 0.5rem;">Model Training</div>
            <div style="color: rgba(255, 255, 255, 0.6);">Available</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="chart-container" style="text-align: center;">
            <span class="status-indicator status-online"></span>
            <div style="color: rgba(255, 255, 255, 0.9); font-weight: 600; margin-top: 0.5rem;">Storage</div>
            <div style="color: rgba(255, 255, 255, 0.6);">Unlimited</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="chart-container" style="text-align: center;">
            <span class="status-indicator status-online"></span>
            <div style="color: rgba(255, 255, 255, 0.9); font-weight: 600; margin-top: 0.5rem;">Processing</div>
            <div style="color: rgba(255, 255, 255, 0.6);">Online</div>
        </div>
        """, unsafe_allow_html=True)

def create_real_activities():
    """Create recent activities from real session state data"""
    activities = []
    
    # Get real activities from session state
    if 'recent_activities' in st.session_state and st.session_state.recent_activities:
        for activity in st.session_state.recent_activities[-6:]:  # Show last 6 activities
            activities.append((
                activity.get('type', 'Activity'),
                activity.get('description', 'No description'),
                activity.get('timestamp', 'Recently'),
                "online"
            ))
    
    # If no activities, show placeholder
    if not activities:
        activities = [
            ("Welcome to MLGenie", "Start by uploading a dataset", "Just now", "online"),
            ("Getting Started", "Use the sidebar to navigate between features", "Just now", "online"),
            ("Data Visualization", "Upload CSV files and create interactive charts", "Just now", "online"),
            ("Feature Engineering", "Transform and prepare your data", "Just now", "online"),
            ("Model Training", "Train ML models with various algorithms", "Just now", "online"),
            ("Real Data", "Your activities will appear here as you use the platform", "Just now", "online")
        ]
    
    for title, desc, time_ago, status in activities:
        create_activity_item(title, desc, time_ago, status)

def app():
    """Main dashboard application"""
    # Initialize session stats
    init_session_stats()
    
    # Apply platform CSS
    st.markdown(get_platform_css(), unsafe_allow_html=True)
    
    # Add JavaScript to ensure sidebar controls are working
    st.markdown("""
    <script>
    // Ensure sidebar controls are visible and functional
    setTimeout(function() {
        const sidebarControls = document.querySelectorAll('button[data-testid="collapsedControl"]');
        sidebarControls.forEach(control => {
            control.style.display = 'block';
            control.style.visibility = 'visible';
            control.style.opacity = '1';
        });
        
        // Force show sidebar section if hidden
        const sidebar = document.querySelector('section[data-testid="stSidebar"]');
        if (sidebar) {
            sidebar.style.display = 'block';
        }
    }, 100);
    </script>
    """, unsafe_allow_html=True)
    
    # Dashboard Header
    st.markdown("""
    <div class="dashboard-header">
        <h1 class="dashboard-title"> Dashboard</h1>
        <p class="dashboard-subtitle">Progress & Statistics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get real stats from session state
    stats = st.session_state.dashboard_stats
    
    # Key Metrics Section
    st.markdown('<div class="section-header">Your Progress</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card(
            "Datasets Processed", 
            str(stats.get('datasets_count', 0)),
            "Total datasets uploaded"
        )
    
    with col2:
        create_metric_card(
            "Feature Engineering", 
            str(stats.get('feature_engineering_count', 0)),
            "Operations performed"
        )
    
    with col3:
        create_metric_card(
            "Models Trained", 
            str(stats.get('models_trained', 0)),
            "Total models created"
        )
    
    with col4:
        ml_count = stats.get('ml_models', 0)
        dl_count = stats.get('dl_models', 0)
        create_metric_card(
            "Active Models", 
            f"{ml_count + dl_count}",
            f"ML: {ml_count}, DL: {dl_count}"
        )
    
    # Model Leaderboard Section
    if 'model_leaderboard' in st.session_state and st.session_state.model_leaderboard:
        st.markdown('<div class="section-header">Your Top Models</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            # Create leaderboard table
            leaderboard_data = []
            for i, model in enumerate(st.session_state.model_leaderboard[:5]):
                leaderboard_data.append({
                    'Rank': i + 1,
                    'Model': model.get('name', 'Unknown'),
                    'Algorithm': model.get('algorithm', 'Unknown'),
                    'Score': f"{model.get('best_score', 0):.3f}",
                    'Type': model.get('model_type', 'Unknown')
                })
            
            if leaderboard_data:
                df_leaderboard = pd.DataFrame(leaderboard_data)
                st.dataframe(df_leaderboard, use_container_width=True, hide_index=True)
            else:
                st.info("Train some models to see your leaderboard!")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            best_model = st.session_state.model_leaderboard[0] if st.session_state.model_leaderboard else None
            if best_model:
                st.markdown(f"""
                <h3 style="color: rgba(255, 255, 255, 0.9); text-align: center; margin-bottom: 1rem;">Best Model</h3>
                <div style="text-align: center; color: rgba(255, 255, 255, 0.7);">
                    <div style="margin: 1rem 0;">
                        <strong style="color: #40e0ff; font-size: 1.2rem;">{best_model.get('name', 'Unknown')}</strong>
                    </div>
                    <div style="margin: 1rem 0;">
                        <strong>Algorithm:</strong><br>
                        {best_model.get('algorithm', 'Unknown')}
                    </div>
                    <div style="margin: 1rem 0;">
                        <strong>Score:</strong><br>
                        <span style="color: #40e0ff; font-size: 1.5rem; font-weight: bold;">{best_model.get('best_score', 0):.3f}</span>
                    </div>
                    <div style="margin: 1rem 0;">
                        <strong>Type:</strong><br>
                        {best_model.get('model_type', 'Unknown')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Train models to see your best performer!")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts Section
    st.markdown('<div class="section-header">Analytics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(create_algorithm_distribution_chart(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(create_performance_trend_chart(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Recent Activity Section
    st.markdown('<div class="section-header">Recent Activity</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        create_real_activities()
    
    with col2:
        st.markdown("""
        <div class="chart-container">
            <h3 style="color: rgba(255, 255, 255, 0.9); text-align: center; margin-bottom: 1.5rem;">Platform Status</h3>
            <div style="text-align: center; color: rgba(255, 255, 255, 0.7);">
                <div style="margin: 1rem 0;">
                    <span class="status-indicator status-online"></span>
                    <strong>System Status:</strong><br>
                    All systems operational
                </div>
                <div style="margin: 1rem 0;">
                    <span class="status-indicator status-online"></span>
                    <strong>Data Processing:</strong><br>
                    Ready for upload
                </div>
                <div style="margin: 1rem 0;">
                    <span class="status-indicator status-online"></span>
                    <strong>Model Training:</strong><br>
                    Available
                </div>
                <div style="margin: 1rem 0;">
                    <span class="status-indicator status-online"></span>
                    <strong>Feature Engineering:</strong><br>
                    Ready to use
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Instructions for new users
    if (stats.get('datasets_count', 0) == 0 and 
        stats.get('models_trained', 0) == 0 and 
        stats.get('feature_engineering_count', 0) == 0):
        
        st.markdown('<div class="section-header">Getting Started</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="chart-container" style="text-align: center;">
                <h4 style="color: #40e0ff;">1. Upload Data</h4>
                <p style="color: rgba(255, 255, 255, 0.7);">Go to 'Visualize' and upload your CSV dataset to get started.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="chart-container" style="text-align: center;">
                <h4 style="color: #40e0ff;">2. Engineer Features</h4>
                <p style="color: rgba(255, 255, 255, 0.7);">Use 'Feature Engineering' to clean and transform your data.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="chart-container" style="text-align: center;">
                <h4 style="color: #40e0ff;">3. Train Models</h4>
                <p style="color: rgba(255, 255, 255, 0.7);">Use 'ML Training' or 'DL Training' to build predictive models.</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    app()