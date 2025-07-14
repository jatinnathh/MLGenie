import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Custom CSS for modern UI
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stButton > button {
        background: linear-gradient(45deg, #00c7b7, #0066cc);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 199, 183, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 199, 183, 0.4);
    }
    .stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(45deg, #ff6b6b, #ff8e8e);
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    .chart-container {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .category-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #00c7b7, #0066cc);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

def app():
    # Header with gradient background
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;'>
        <h1 style='color: white; margin: 0; font-size: 2.5rem;'>Advanced Data Visualization Studio</h1>
        <p style='color: white; margin: 0.5rem 0 0 0; font-size: 1.2rem;'>Interactive charts with modern design</p>
    </div>
    """, unsafe_allow_html=True)

    # Persistent upload section at the top
    st.markdown("### Upload or Replace Data")
    uploaded_file = st.file_uploader(
        "Upload your data file (CSV, Excel, JSON)", 
        type=["csv", "xlsx", "json"],
        help="Supported formats: CSV, Excel, JSON",
        key="visualize_uploader_top"
    )
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".json"):
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file format.")
            return
        if df.empty:
            st.warning("The uploaded file is empty. Please upload a valid file.")
            return
        st.session_state.df_visualize = df
        st.success("Data uploaded and saved for Visualize!")

    # Use persistent data if available
    if "df_visualize" in st.session_state:
        df = st.session_state.df_visualize
        st.success("Using previously uploaded data for Visualize.")
    else:
        st.info("Please upload a data file to begin.")
        return
    
    # Data overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><h3>{len(df)}</h3><p>Total Rows</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h3>{len(df.columns)}</h3><p>Columns</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h3>{df.isnull().sum().sum()}</h3><p>Missing Values</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><h3>{df.memory_usage(deep=True).sum() / 1024:.1f} KB</h3><p>Memory Usage</p></div>', unsafe_allow_html=True)
    # Data preview
    with st.expander("Data Preview", expanded=False):
        st.dataframe(df.head(), use_container_width=True)
    # Column analysis
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    # Try to detect date columns from object columns
    for col in cat_cols:
        try:
            pd.to_datetime(df[col].head())
            if col not in date_cols:
                date_cols.append(col)
        except Exception:
            pass
    # Initialize session state
    if 'selected_category' not in st.session_state:
        st.session_state.selected_category = None
    if 'selected_chart' not in st.session_state:
        st.session_state.selected_chart = None
    # Category selection with enhanced UI
    st.markdown("## Select Visualization Category")
    # Create category buttons in a row
    categories = [
        ("Distribution", "distribution", "Analyze data distributions"),
        ("Comparison", "comparison", "Compare different variables"),
        ("Categorical", "categorical", "Visualize categorical data"),
        ("Geospatial", "geospatial", "Maps and geographic data"),
        ("Time Series", "timeseries", "Time-based analysis"),
        ("Relationships", "relationships", "Correlations and patterns"),
        ("Animated", "animated", "Dynamic visualizations")
    ]
    cols = st.columns(len(categories))
    for i, (label, key, desc) in enumerate(categories):
        with cols[i]:
            if st.button(label, use_container_width=True, 
                        type="primary" if st.session_state.selected_category == key else "secondary",
                        help=desc):
                st.session_state.selected_category = key
                st.session_state.selected_chart = None
    # Display chart options based on selected category
    if st.session_state.selected_category == "distribution":
        st.markdown('<div class="category-header">Distribution Analysis</div>', unsafe_allow_html=True)
        chart_options = [
            ("Histogram", "histogram"),
            ("Violin Plot", "violin"),
            ("Density Plot", "density"),
            ("Box Plot", "box"),
            ("Ridge Plot", "ridge"),
            ("Q-Q Plot", "qq"),
            ("Normal Dist", "normal"),
            ("Distplot", "distplot"),
            ("ECDF", "ecdf")
        ]
        cols = st.columns(3)
        for i, (label, key) in enumerate(chart_options):
            with cols[i % 3]:
                if st.button(label, use_container_width=True, key=f"dist_{key}"):
                    st.session_state.selected_chart = key
        # Chart generation
        if st.session_state.selected_chart:
            generate_distribution_chart(df, numeric_cols, cat_cols)
    elif st.session_state.selected_category == "comparison":
        st.markdown('<div class="category-header">Comparison Analysis</div>', unsafe_allow_html=True)
        chart_options = [
            ("Bar Chart", "bar"),
            ("Line Chart", "line"),
            ("Heatmap", "heatmap"),
            ("Grouped Bar", "grouped_bar"),
            ("Stacked Bar", "stacked_bar"),
            ("Area Chart", "area"),
            ("Radar Chart", "radar"),
            ("Waterfall", "waterfall"),
            ("Parallel Coords", "parallel")
        ]
        cols = st.columns(3)
        for i, (label, key) in enumerate(chart_options):
            with cols[i % 3]:
                if st.button(label, use_container_width=True, key=f"comp_{key}"):
                    st.session_state.selected_chart = key
        if st.session_state.selected_chart:
            generate_comparison_chart(df, numeric_cols, cat_cols)
    elif st.session_state.selected_category == "categorical":
        st.markdown('<div class="category-header">Categorical Analysis</div>', unsafe_allow_html=True)
        chart_options = [
            ("Pie Chart", "pie"),
            ("Donut Chart", "donut"),
            ("Count Plot", "count"),
            ("Treemap", "treemap"),
            ("Sunburst", "sunburst"),
            ("Funnel Chart", "funnel"),
            ("Sankey", "sankey"),
            ("Word Cloud", "wordcloud"),
            ("Pareto Chart", "pareto")
        ]
        cols = st.columns(3)
        for i, (label, key) in enumerate(chart_options):
            with cols[i % 3]:
                if st.button(label, use_container_width=True, key=f"cat_{key}"):
                    st.session_state.selected_chart = key
        if st.session_state.selected_chart:
            generate_categorical_chart(df, cat_cols, numeric_cols)
    elif st.session_state.selected_category == "geospatial":
        st.markdown('<div class="category-header">Geospatial Analysis</div>', unsafe_allow_html=True)
        chart_options = [
            ("Scatter Map", "scatter_map"),
            ("Density Map", "density_map"),
            ("Choropleth", "choropleth"),
            ("Bubble Map", "bubble_map"),
            ("Satellite View", "satellite"),
            ("Heatmap Geo", "heatmap_geo")
        ]
        cols = st.columns(3)
        for i, (label, key) in enumerate(chart_options):
            with cols[i % 3]:
                if st.button(label, use_container_width=True, key=f"geo_{key}"):
                    st.session_state.selected_chart = key
        if st.session_state.selected_chart:
            generate_geospatial_chart(df, numeric_cols, cat_cols)
    elif st.session_state.selected_category == "timeseries":
        st.markdown('<div class="category-header">Time Series Analysis</div>', unsafe_allow_html=True)
        chart_options = [
            ("Time Series", "timeseries"),
            ("Seasonal Plot", "seasonal"),
            ("Decomposition", "decomposition"),
            ("Candlestick", "candlestick"),
            ("Forecast", "forecast"),
            ("Stock Chart", "stock")
        ]
        cols = st.columns(3)
        for i, (label, key) in enumerate(chart_options):
            with cols[i % 3]:
                if st.button(label, use_container_width=True, key=f"ts_{key}"):
                    st.session_state.selected_chart = key
        if st.session_state.selected_chart:
            generate_timeseries_chart(df, numeric_cols, date_cols)
    elif st.session_state.selected_category == "relationships":
        st.markdown('<div class="category-header">Relationship Analysis</div>', unsafe_allow_html=True)
        chart_options = [
            ("Scatter Plot", "scatter"),
            ("3D Scatter", "scatter_3d"),
            ("Correlation", "correlation"),
            ("Pair Plot", "pairplot"),
            ("Network Graph", "network"),
            ("Regression", "regression")
        ]
        cols = st.columns(3)
        for i, (label, key) in enumerate(chart_options):
            with cols[i % 3]:
                if st.button(label, use_container_width=True, key=f"rel_{key}"):
                    st.session_state.selected_chart = key
        if st.session_state.selected_chart:
            generate_relationship_chart(df, numeric_cols, cat_cols)
    elif st.session_state.selected_category == "animated":
        st.markdown('<div class="category-header">Animated Visualizations</div>', unsafe_allow_html=True)
        chart_options = [
            ("Animated Scatter", "animated_scatter"),
            ("Animated Bar", "animated_bar"),
            ("Animated Map", "animated_map"),
            ("Racing Chart", "racing"),
            ("Bubble Animation", "bubble_animated"),
            ("Time Evolution", "time_evolution")
        ]
        cols = st.columns(3)
        for i, (label, key) in enumerate(chart_options):
            with cols[i % 3]:
                if st.button(label, use_container_width=True, key=f"anim_{key}"):
                    st.session_state.selected_chart = key
        if st.session_state.selected_chart:
            generate_animated_chart(df, numeric_cols, cat_cols, date_cols)
    # Control buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reset Selection", use_container_width=True):
            st.session_state.selected_category = None
            st.session_state.selected_chart = None
            st.rerun()
    with col2:
        if st.button("Generate Dashboard", use_container_width=True):
            generate_dashboard(df, numeric_cols, cat_cols)
    
    # Sample data showcase
    st.markdown("## Try with Sample Data")
    if st.button("Load Sample Dataset", use_container_width=True):
        # Create sample data
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'sales': np.random.normal(1000, 200, 100),
            'category': np.random.choice(['A', 'B', 'C', 'D'], 100),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
            'profit': np.random.normal(100, 30, 100),
            'lat': np.random.uniform(25, 49, 100),
            'lon': np.random.uniform(-125, -66, 100)
        })
        st.session_state.sample_data = sample_data
        st.success("Sample data loaded! You can now explore visualizations.")

def generate_distribution_chart(df, numeric_cols, cat_cols):
    """Generate distribution charts"""
    chart_type = st.session_state.selected_chart
    
    if chart_type == "histogram":
        col = st.selectbox("Select column", numeric_cols, key="hist_col")
        col1, col2 = st.columns(2)
        with col1:
            bins = st.slider("Number of bins", 10, 100, 30)
        with col2:
            opacity = st.slider("Opacity", 0.1, 1.0, 0.8)
        fig = px.histogram(df, x=col, nbins=bins, opacity=opacity,
                          color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(title=f"Distribution of {col}", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "violin":
        col = st.selectbox("Select column", numeric_cols, key="violin_col")
        if cat_cols:
            group_by = st.selectbox("Group by (optional)", ["None"] + cat_cols, key="violin_group")
            if group_by != "None":
                fig = px.violin(df, y=col, x=group_by, box=True, points="all", color=group_by, color_discrete_sequence=px.colors.qualitative.Set2)
            else:
                fig = px.violin(df, y=col, box=True, points="all", color_discrete_sequence=px.colors.qualitative.Set2)
        else:
            fig = px.violin(df, y=col, box=True, points="all", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(title=f"Violin Plot: {col}", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "qq":
        col = st.selectbox("Select column", numeric_cols, key="qq_col")
        from scipy import stats
        
        fig = go.Figure()
        data = df[col].dropna()
        qq_data = stats.probplot(data, dist="norm")
        
        fig.add_trace(go.Scatter(
            x=qq_data[0][0],
            y=qq_data[0][1],
            mode='markers',
            name='Q-Q Plot',
            marker=dict(color='#00c7b7')
        ))
        
        # Add reference line
        fig.add_trace(go.Scatter(
            x=qq_data[0][0],
            y=qq_data[1][1] + qq_data[1][0] * qq_data[0][0],
            mode='lines',
            name='Reference Line',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(title=f"Q-Q Plot of {col}", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

def generate_comparison_chart(df, numeric_cols, cat_cols):
    """Generate comparison charts"""
    chart_type = st.session_state.selected_chart
    
    if chart_type == "bar":
        col1, col2 = st.columns(2)
        with col1:
            x = st.selectbox("X axis", cat_cols, key="bar_x")
        with col2:
            y = st.selectbox("Y axis", numeric_cols, key="bar_y")
        fig = px.bar(df, x=x, y=y, color=x, color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(title=f"Bar Chart: {y} by {x}", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "line":
        col1, col2 = st.columns(2)
        with col1:
            x = st.selectbox("X axis", cat_cols + numeric_cols, key="line_x")
        with col2:
            y = st.selectbox("Y axis", numeric_cols, key="line_y")
        fig = px.line(df, x=x, y=y, color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(title=f"Line Chart: {y} by {x}", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "heatmap":
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
            fig.update_layout(title="Correlation Heatmap", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric columns for correlation heatmap.")
    elif chart_type == "area":
        col1, col2 = st.columns(2)
        with col1:
            x = st.selectbox("X axis", cat_cols + numeric_cols, key="area_x")
        with col2:
            y = st.selectbox("Y axis", numeric_cols, key="area_y")
        fig = px.area(df, x=x, y=y, color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(title=f"Area Chart: {y} by {x}", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "grouped_bar":
        col1, col2, col3 = st.columns(3)
        with col1:
            x = st.selectbox("X axis (group)", cat_cols, key="groupedbar_x")
        with col2:
            y = st.selectbox("Y axis", numeric_cols, key="groupedbar_y")
        with col3:
            group = st.selectbox("Group by", cat_cols, key="groupedbar_group")
        fig = px.bar(df, x=x, y=y, color=group, barmode="group", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(title=f"Grouped Bar Chart: {y} by {x} and {group}", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "stacked_bar":
        col1, col2, col3 = st.columns(3)
        with col1:
            x = st.selectbox("X axis (stack)", cat_cols, key="stackedbar_x")
        with col2:
            y = st.selectbox("Y axis", numeric_cols, key="stackedbar_y")
        with col3:
            stack = st.selectbox("Stack by", cat_cols, key="stackedbar_stack")
        fig = px.bar(df, x=x, y=y, color=stack, barmode="stack", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(title=f"Stacked Bar Chart: {y} by {x} and {stack}", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "parallel":
        if len(numeric_cols) >= 3:
            color_col = st.selectbox("Color by (optional)", [None] + numeric_cols, key="parallel_color")
            fig = create_parallel_coordinates(df, numeric_cols, color_col=color_col if color_col else None)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 3 numeric columns for parallel coordinates plot.")
        else:
            st.warning("Need at least 3 numeric columns for parallel coordinates plot.")
        # Distribution overview
    if numeric_cols:
        st.subheader("Distribution Overview")
        cols = st.columns(min(3, len(numeric_cols)))
        for i, col in enumerate(numeric_cols[:3]):
            with cols[i]:
                fig = px.histogram(df, x=col, color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(title=f"Distribution of {col}", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "radar":
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                category = st.selectbox("Category", cat_cols, key="radar_cat")
            with col2:
                value = st.selectbox("Value", numeric_cols, key="radar_val")
            
            radar_data = df.groupby(category)[value].mean().reset_index()
            fig = px.line_polar(radar_data, r=value, theta=category, line_close=True, 
                               color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_traces(fill='toself')
            fig.update_layout(title="Radar Chart", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric columns for radar chart")

# Additional utility functions for advanced charts
def create_waterfall_chart(df, category_col, value_col):
    """Create a waterfall chart"""
    data = df.groupby(category_col)[value_col].sum().reset_index()
    data = data.sort_values(value_col, ascending=False)
    
    fig = go.Figure(go.Waterfall(
        name="Waterfall",
        orientation="v",
        x=data[category_col],
        y=data[value_col],
        textposition="outside",
        text=[f"{val:,.0f}" for val in data[value_col]],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(title="Waterfall Chart", template="plotly_white")
    return fig

def create_parallel_coordinates(df, numeric_cols, color_col=None):
    """Create parallel coordinates plot"""
    if len(numeric_cols) < 3:
        return None
    
    # Normalize data for better visualization
    df_norm = df[numeric_cols].copy()
    for col in numeric_cols:
        df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
    
    dimensions = []
    for col in numeric_cols:
        dimensions.append(dict(
            range=[0, 1],
            label=col,
            values=df_norm[col]
        ))
    
    fig = go.Figure(data=go.Parcoords(
        line=dict(color=df[color_col] if color_col else df_norm.iloc[:, 0],
                 colorscale='Viridis',
                 showscale=True),
        dimensions=dimensions
    ))
    
    fig.update_layout(title="Parallel Coordinates", template="plotly_white")
    return fig

def create_sankey_diagram(df, source_col, target_col, value_col=None):
    """Create Sankey diagram"""
    if value_col is None:
        # Count occurrences
        sankey_data = df.groupby([source_col, target_col]).size().reset_index(name='count')
        value_col = 'count'
    else:
        sankey_data = df.groupby([source_col, target_col])[value_col].sum().reset_index()
    
    # Create node lists
    source_nodes = sankey_data[source_col].unique()
    target_nodes = sankey_data[target_col].unique()
    all_nodes = list(set(list(source_nodes) + list(target_nodes)))
    
    # Create mappings
    node_map = {node: i for i, node in enumerate(all_nodes)}
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color="blue"
        ),
        link=dict(
            source=[node_map[src] for src in sankey_data[source_col]],
            target=[node_map[tgt] for tgt in sankey_data[target_col]],
            value=sankey_data[value_col]
        )
    )])
    
    fig.update_layout(title="Sankey Diagram", template="plotly_white")
    return fig

def create_bubble_chart(df, x_col, y_col, size_col, color_col=None):
    """Create bubble chart"""
    fig = px.scatter(df, x=x_col, y=y_col, size=size_col,
                    color=color_col, hover_name=color_col,
                    size_max=60)
    fig.update_layout(title="Bubble Chart", template="plotly_white")
    return fig

def create_ridge_plot(df, value_col, category_col):
    """Create ridge plot using plotly"""
    categories = df[category_col].unique()
    
    fig = go.Figure()
    
    for i, category in enumerate(categories):
        data = df[df[category_col] == category][value_col]
        
        fig.add_trace(go.Violin(
            y=data,
            name=str(category),
            side="positive",
            line_color="rgba(0,0,0,0.2)",
            fillcolor=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)],
            opacity=0.7,
            showlegend=True
        ))
    
    fig.update_layout(
        title=f"Ridge Plot: {value_col} by {category_col}",
        template="plotly_white",
        yaxis_title=value_col,
        xaxis_title="Density"
    )
    
    return fig

def create_calendar_heatmap(df, date_col, value_col):
    """Create calendar heatmap"""
    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Create day-wise aggregation
    daily_data = df.groupby(df[date_col].dt.date)[value_col].sum().reset_index()
    daily_data['day_of_week'] = pd.to_datetime(daily_data[date_col]).dt.day_name()
    daily_data['week'] = pd.to_datetime(daily_data[date_col]).dt.isocalendar().week
    
    # Create pivot table
    pivot_data = daily_data.pivot(index='day_of_week', columns='week', values=value_col)
    
    fig = px.imshow(pivot_data, aspect="auto", color_continuous_scale="Blues")
    fig.update_layout(title="Calendar Heatmap", template="plotly_white")
    return fig

def create_funnel_chart(df, stage_col, value_col):
    """Create funnel chart"""
    funnel_data = df.groupby(stage_col)[value_col].sum().reset_index()
    funnel_data = funnel_data.sort_values(value_col, ascending=False)
    
    fig = go.Figure(go.Funnel(
        y=funnel_data[stage_col],
        x=funnel_data[value_col],
        textinfo="value+percent previous",
        textposition="inside",
        marker=dict(color=px.colors.sequential.Blues_r)
    ))
    
    fig.update_layout(title="Funnel Chart", template="plotly_white")
    return fig

def create_gauge_chart(df, value_col, title="Gauge Chart"):
    """Create gauge chart"""
    value = df[value_col].mean()
    max_val = df[value_col].max()
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': max_val * 0.8},
        gauge={
            'axis': {'range': [None, max_val]},
            'bar': {'color': "#00c7b7"},
            'steps': [
                {'range': [0, max_val * 0.5], 'color': "lightgray"},
                {'range': [max_val * 0.5, max_val * 0.8], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_val * 0.9
            }
        }
    ))
    
    fig.update_layout(template="plotly_white")
    return fig

def create_polar_chart(df, category_col, value_col):
    """Create polar chart"""
    polar_data = df.groupby(category_col)[value_col].sum().reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=polar_data[value_col],
        theta=polar_data[category_col],
        fill='toself',
        name='Values',
        marker=dict(color='#00c7b7')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, polar_data[value_col].max()])
        ),
        title="Polar Chart",
        template="plotly_white"
    )
    
    return fig

def create_stream_graph(df, date_col, category_col, value_col):
    """Create stream graph"""
    # Prepare data
    stream_data = df.groupby([date_col, category_col])[value_col].sum().unstack(fill_value=0)
    
    fig = go.Figure()
    
    categories = stream_data.columns
    colors = px.colors.qualitative.Set1[:len(categories)]
    
    for i, category in enumerate(categories):
        fig.add_trace(go.Scatter(
            x=stream_data.index,
            y=stream_data[category],
            mode='lines',
            stackgroup='one',
            name=str(category),
            line=dict(width=0.5, color=colors[i % len(colors)]),
            fillcolor=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        title="Stream Graph",
        template="plotly_white",
        xaxis_title=date_col,
        yaxis_title=value_col
    )
    
    return fig

# Enhanced chart generation functions with the new charts
def generate_distribution_chart(df, numeric_cols, cat_cols):
    """Enhanced distribution chart generation"""
    chart_type = st.session_state.selected_chart
    
    if chart_type == "histogram":
        col = st.selectbox("Select column", numeric_cols, key="hist_col")
        col1, col2, col3 = st.columns(3)
        with col1:
            bins = st.slider("Number of bins", 10, 100, 30)
        with col2:
            opacity = st.slider("Opacity", 0.1, 1.0, 0.8)
        with col3:
            marginal = st.selectbox("Marginal plot", ["None", "rug", "box", "violin"], key="hist_marginal")
        
        fig = px.histogram(df, x=col, nbins=bins, opacity=opacity,
                          marginal=marginal if marginal != "None" else None,
                          color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(title=f"Distribution of {col}", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "violin":
        col = st.selectbox("Select column", numeric_cols, key="violin_col")
        if cat_cols:
            group_by = st.selectbox("Group by (optional)", ["None"] + cat_cols, key="violin_group")
            if group_by != "None":
                fig = px.violin(df, y=col, x=group_by, box=True, points="all", color=group_by, color_discrete_sequence=px.colors.qualitative.Set2)
            else:
                fig = px.violin(df, y=col, box=True, points="all", color_discrete_sequence=px.colors.qualitative.Set2)
        else:
            fig = px.violin(df, y=col, box=True, points="all", color_discrete_sequence=px.colors.qualitative.Set2)
        
        fig.update_layout(title=f"Violin Plot of {col}", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "density":
        col = st.selectbox("Select column", numeric_cols, key="density_col")
        fig = px.histogram(df, x=col, histnorm='probability density', 
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(title=f"Density Plot of {col}", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "box":
        col = st.selectbox("Select column", numeric_cols, key="box_col")
        if cat_cols:
            group_by = st.selectbox("Group by (optional)", ["None"] + cat_cols, key="box_group")
            if group_by != "None":
                fig = px.box(df, y=col, x=group_by, color=group_by, color_discrete_sequence=px.colors.qualitative.Set2)
            else:
                fig = px.box(df, y=col, color_discrete_sequence=px.colors.qualitative.Set2)
        else:
            fig = px.box(df, y=col, color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(title=f"Box Plot of {col}", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "ridge":
        if cat_cols:
            col1, col2 = st.columns(2)
            with col1:
                value_col = st.selectbox("Value column", numeric_cols, key="ridge_value")
            with col2:
                category_col = st.selectbox("Category column", cat_cols, key="ridge_cat")
            fig = create_ridge_plot(df, value_col, category_col)
            fig.update_layout(title=f"Ridge Plot: {value_col} by {category_col}", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need categorical columns for ridge plot")
    elif chart_type == "qq":
        col = st.selectbox("Select column", numeric_cols, key="qq_col")
        try:
            from scipy import stats
            fig = go.Figure()
            data = df[col].dropna()
            qq_data = stats.probplot(data, dist="norm")
            fig.add_trace(go.Scatter(
                x=qq_data[0][0],
                y=qq_data[0][1],
                mode='markers',
                name='Q-Q Plot',
                marker=dict(color='#00c7b7', size=8)
            ))
            # Add reference line
            slope, intercept = qq_data[1][0], qq_data[1][1]
            ref_line = slope * np.array(qq_data[0][0]) + intercept
            fig.add_trace(go.Scatter(
                x=qq_data[0][0],
                y=ref_line,
                mode='lines',
                name='Reference Line',
                line=dict(color='red', dash='dash', width=2)
            ))
            fig.update_layout(title=f"Q-Q Plot: {col}", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.error("scipy is required for Q-Q plots")
    elif chart_type == "ecdf":
        col = st.selectbox("Select column", numeric_cols, key="ecdf_col")
        fig = px.ecdf(df, x=col, color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(title=f"ECDF: {col}", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    app()

def generate_categorical_chart(df, cat_cols, numeric_cols):
    """Generate categorical charts"""
    chart_type = st.session_state.selected_chart
    
    if chart_type == "pie":
        col = st.selectbox("Select categorical column", cat_cols, key="pie_col")
        pie_df = df[col].value_counts().reset_index()
        pie_df.columns = [col, "count"]
        fig = px.pie(pie_df, names=col, values="count", 
                    color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(title=f"Pie Chart of {col}", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "sunburst":
        if len(cat_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                level1 = st.selectbox("First level", cat_cols, key="sun_level1")
            with col2:
                level2 = st.selectbox("Second level", [c for c in cat_cols if c != level1], key="sun_level2")
            sunburst_df = df.groupby([level1, level2]).size().reset_index(name='count')
            fig = px.sunburst(sunburst_df, path=[level1, level2], values='count',
                            color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(title="Sunburst Chart", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 categorical columns for sunburst chart")

def generate_geospatial_chart(df, numeric_cols, cat_cols):
    """Generate geospatial charts"""
    chart_type = st.session_state.selected_chart
    
    # Check for lat/lon columns
    lat_cols = [col for col in df.columns if 'lat' in col.lower() or 'latitude' in col.lower()]
    lon_cols = [col for col in df.columns if 'lon' in col.lower() or 'longitude' in col.lower()]
    
    if not lat_cols or not lon_cols:
        st.warning("⚠️ No latitude/longitude columns found. Please ensure your data contains columns with 'lat' and 'lon' in the names.")
        return
    
    if chart_type == "scatter_map":
        col1, col2, col3 = st.columns(3)
        with col1:
            lat_col = st.selectbox("Latitude column", lat_cols, key="map_lat")
        with col2:
            lon_col = st.selectbox("Longitude column", lon_cols, key="map_lon")
        with col3:
            if numeric_cols:
                size_col = st.selectbox("Size by", ["None"] + numeric_cols, key="map_size")
            else:
                size_col = "None"
        
        fig = px.scatter_mapbox(
            df, lat=lat_col, lon=lon_col,
            size=size_col if size_col != "None" else None,
            color_discrete_sequence=["#00c7b7"],
            zoom=3,
            mapbox_style="open-street-map"
        )
        fig.update_layout(title="Scatter Map", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

def generate_timeseries_chart(df, numeric_cols, date_cols):
    """Generate time series charts"""
    chart_type = st.session_state.selected_chart
    
    if not date_cols:
        st.warning("⚠️ No date columns found. Please ensure your data contains date/time columns.")
        return
    
    if chart_type == "timeseries":
        col1, col2 = st.columns(2)
        with col1:
            date_col = st.selectbox("Date column", date_cols, key="ts_date")
        with col2:
            value_col = st.selectbox("Value column", numeric_cols, key="ts_value")
        
        # Convert to datetime if needed
        if df[date_col].dtype == 'object':
            df[date_col] = pd.to_datetime(df[date_col])
        
        fig = px.line(df, x=date_col, y=value_col, 
                     color_discrete_sequence=["#00c7b7"])
        fig.update_layout(title=f"Time Series: {value_col}", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "seasonal":
        import statsmodels.api as sm
        date_col = st.selectbox("Date column", date_cols, key="seasonal_date")
        value_col = st.selectbox("Value column", numeric_cols, key="seasonal_value")
        df[date_col] = pd.to_datetime(df[date_col])
        df_sorted = df.sort_values(date_col)
        try:
            decomposition = sm.tsa.seasonal_decompose(df_sorted[value_col], model='additive', period=12)
            fig, ax = plt.subplots(4, 1, figsize=(10, 8))
            decomposition.observed.plot(ax=ax[0], title="Observed")
            decomposition.trend.plot(ax=ax[1], title="Trend")
            decomposition.seasonal.plot(ax=ax[2], title="Seasonal")
            decomposition.resid.plot(ax=ax[3], title="Residual")
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Decomposition failed: {e}")
    elif chart_type == "decomposition":
        import statsmodels.api as sm
        date_col = st.selectbox("Date column", date_cols, key="decomp_date")
        value_col = st.selectbox("Value column", numeric_cols, key="decomp_value")
        df[date_col] = pd.to_datetime(df[date_col])
        df_sorted = df.sort_values(date_col)
        try:
            decomposition = sm.tsa.seasonal_decompose(df_sorted[value_col], model='additive', period=12)
            fig, ax = plt.subplots(4, 1, figsize=(10, 8))
            decomposition.observed.plot(ax=ax[0], title="Observed")
            decomposition.trend.plot(ax=ax[1], title="Trend")
            decomposition.seasonal.plot(ax=ax[2], title="Seasonal")
            decomposition.resid.plot(ax=ax[3], title="Residual")
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Decomposition failed: {e}")
    elif chart_type == "candlestick":
        date_col = st.selectbox("Date column", date_cols, key="candle_date")
        open_col = st.selectbox("Open column", numeric_cols, key="candle_open")
        high_col = st.selectbox("High column", numeric_cols, key="candle_high")
        low_col = st.selectbox("Low column", numeric_cols, key="candle_low")
        close_col = st.selectbox("Close column", numeric_cols, key="candle_close")
        fig = go.Figure(data=[go.Candlestick(x=df[date_col], open=df[open_col], high=df[high_col], low=df[low_col], close=df[close_col])])
        fig.update_layout(title="Candlestick Chart", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "forecast":
        import statsmodels.api as sm
        date_col = st.selectbox("Date column", date_cols, key="forecast_date")
        value_col = st.selectbox("Value column", numeric_cols, key="forecast_value")
        df[date_col] = pd.to_datetime(df[date_col])
        df_sorted = df.sort_values(date_col)
        try:
            model = sm.tsa.ARIMA(df_sorted[value_col], order=(1,1,1))
            results = model.fit()
            forecast = results.get_forecast(steps=12)
            pred_ci = forecast.conf_int()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_sorted[date_col], y=df_sorted[value_col], mode='lines', name='Observed'))
            future_dates = pd.date_range(df_sorted[date_col].max(), periods=13, freq='D')[1:]
            fig.add_trace(go.Scatter(x=future_dates, y=forecast.predicted_mean, mode='lines', name='Forecast'))
            fig.add_trace(go.Scatter(x=future_dates, y=pred_ci.iloc[:,0], mode='lines', name='Lower CI', line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=future_dates, y=pred_ci.iloc[:,1], mode='lines', name='Upper CI', line=dict(dash='dot')))
            fig.update_layout(title="Forecast", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Forecast failed: {e}")
    elif chart_type == "stock":
        date_col = st.selectbox("Date column", date_cols, key="stock_date")
        value_col = st.selectbox("Value column", numeric_cols, key="stock_value")
        fig = px.line(df, x=date_col, y=value_col, color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(title="Stock Chart", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

def generate_animated_chart(df, numeric_cols, cat_cols, date_cols):
    """Generate animated charts"""
    chart_type = st.session_state.selected_chart
    
    if chart_type == "animated_scatter":
        if not date_cols:
            st.warning("Need date column for animation")
            return
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            x = st.selectbox("X axis", numeric_cols, key="anim_x")
        with col2:
            y = st.selectbox("Y axis", [col for col in numeric_cols if col != x], key="anim_y")
        with col3:
            animation_frame = st.selectbox("Animation frame", date_cols, key="anim_frame")
        with col4:
            color = st.selectbox("Color by", ["None"] + cat_cols, key="anim_color") if cat_cols else "None"
        if df[animation_frame].dtype == 'object':
            df[animation_frame] = pd.to_datetime(df[animation_frame])
        fig = px.scatter(df, x=x, y=y, animation_frame=animation_frame,
                        color=color if color != "None" else None,
                        size_max=55,
                        color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(title="Animated Scatter Plot", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "animated_bar":
        if not date_cols or not cat_cols or not numeric_cols:
            st.warning("Need date, categorical, and numeric columns for animated bar")
            return
        col1, col2, col3 = st.columns(3)
        with col1:
            animation_frame = st.selectbox("Animation frame", date_cols, key="animbar_frame")
        with col2:
            category = st.selectbox("Category", cat_cols, key="animbar_cat")
        with col3:
            value = st.selectbox("Value", numeric_cols, key="animbar_val")
        if df[animation_frame].dtype == 'object':
            df[animation_frame] = pd.to_datetime(df[animation_frame])
        fig = px.bar(df, x=category, y=value, animation_frame=animation_frame,
                    color=category, template="plotly_white", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(title=f"Animated Bar Chart: {value} by {category}", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

def generate_dashboard(df, numeric_cols, cat_cols):
    """Generate a comprehensive dashboard"""
    st.markdown("## Comprehensive Dashboard")
    
    # Summary statistics
    st.subheader("Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Multiple charts in columns
    col1, col2 = st.columns(2)
    
    with col1:
        if numeric_cols:
            # Correlation heatmap
            st.subheader("Correlation Matrix")
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto",
                           color_continuous_scale="RdBu_r")
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution of numeric variables
        st.subheader("Distribution of Numeric Variables")
        for col in numeric_cols[:3]:  # Top 3 numeric columns
            fig = px.histogram(df, x=col, color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(title=f"Distribution of {col}", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if cat_cols:
            # Bar chart of top categories
            st.subheader("Top Categories")
            for col in cat_cols[:2]:  # Top 2 categorical columns
                top_values = df[col].value_counts().head(5)
                fig = px.bar(x=top_values.index, y=top_values.values,
                           color=top_values.index, color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(title=f"Top 5 {col}", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
        
        # Time series of numeric variables
        st.subheader("Time Series of Numeric Variables")
        if len(numeric_cols) > 0 and len(date_cols) > 0:
            for col in numeric_cols[:3]:  # Top 3 numeric columns
                fig = px.line(df, x=date_cols[0], y=col, 
                             color_discrete_sequence=["#00c7b7"])
                fig.update_layout(title=f"Time Series of {col}", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
    
    # Key metrics
    st.subheader("Key Metrics")
    metrics = {
        "Total Sales": df["sales"].sum(),
        "Average Discount": df["discount"].mean(),
        "Total Orders": df["order_id"].nunique(),
        "Returning Customers": df[df["is_returning"] == 1]["customer_id"].nunique()
    }
    st.write(metrics)
    
    # Advanced charts section
    st.markdown("---")
    st.markdown("## Advanced Charts")
    
    # Distribution analysis
    st.subheader("Distribution Analysis")
    if st.button("Show Distribution Analysis", key="dist_analysis"):
        st.session_state.selected_category = "distribution"
        st.session_state.selected_chart = "histogram"
        st.experimental_rerun()
    
    # Comparison analysis
    st.subheader("Comparison Analysis")
    if st.button("Show Comparison Analysis", key="comp_analysis"):
        st.session_state.selected_category = "comparison"
        st.session_state.selected_chart = "bar"
        st.experimental_rerun()
    
    # Categorical analysis
    st.subheader("Categorical Analysis")
    if st.button("Show Categorical Analysis", key="cat_analysis"):
        st.session_state.selected_category = "categorical"
        st.session_state.selected_chart = "pie"
        st.experimental_rerun()
    
    # Geospatial analysis
    st.subheader("Geospatial Analysis")
    if st.button("Show Geospatial Analysis", key="geo_analysis"):
        st.session_state.selected_category = "geospatial"
        st.session_state.selected_chart = "scatter_map"
        st.experimental_rerun()
    
    # Time series analysis
    st.subheader("Time Series Analysis")
    if st.button("Show Time Series Analysis", key="ts_analysis"):
        st.session_state.selected_category = "timeseries"
        st.session_state.selected_chart = "timeseries"
        st.experimental_rerun()
    
    # Relationship analysis
    st.subheader("Relationship Analysis")
    if st.button("Show Relationship Analysis", key="rel_analysis"):
        st.session_state.selected_category = "relationships"
        st.session_state.selected_chart = "scatter"
        st.experimental_rerun()
    
    # Animated visualizations
    st.subheader("Animated Visualizations")
    if st.button("Show Animated Visualizations", key="anim_analysis"):
        st.session_state.selected_category = "animated"
        st.session_state.selected_chart = "animated_scatter"
        st.experimental_rerun()
    
    # Download report button
    st.markdown("---")
    st.subheader("Download Report")
    if st.button("Generate PDF Report", use_container_width=True):
        # Placeholder for report generation
        st.success("PDF Report generated! (Placeholder)")
        # Here you can integrate with a PDF generation library to create a report

    # Data download section
    st.markdown("---")
    st.subheader("Data Download")
    st.write("Download the processed data or any analysis results.")
    
    # Download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Download Processed Data", use_container_width=True):
            # Placeholder for data download
            st.success("Processed data download link (Placeholder)")
            # Here you can provide a link to download the processed data
    with col2:
        if st.button("Download Analysis Results", use_container_width=True):
            # Placeholder for analysis results download
            st.success("Analysis results download link (Placeholder)")
            # Here you can provide a link to download the analysis results

    # Session state management
    st.markdown("---")
    st.subheader("Session State Management")
    if st.button("Reset Session", use_container_width=True):
        st.session_state.clear()
        st.success("Session reset successful!")
        st.experimental_rerun()

    st.markdown("---")
    st.markdown("### Thank you for using the Advanced Data Visualization Studio!")
    st.markdown("#### Developed with ❤️ by Your Name")
    st.markdown("##### For feedback and suggestions, please contact: your.email@example.com")

    # Footer with social media links
    st.markdown("""
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            text-align: center;
            padding: 10px;
            font-size: 0.9rem;
            color: #333;
        }
    </style>
    <div class="footer">
        Connect with us on 
        <a href="https://twitter.com/yourprofile" target="_blank">Twitter</a> | 
        <a href="https://linkedin.com/in/yourprofile" target="_blank">LinkedIn</a> | 
        <a href="https://github.com/yourprofile" target="_blank">GitHub</a>
    </div>
    """, unsafe_allow_html=True)