import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt

st.markdown("""
<style>
    .stButton > button {
        background-color: #0066cc;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .metric-card {
        background: #f8f9fa;
        color: #333;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
        border: 1px solid #dee2e6;
    }
    .category-header {
        background: #f8f9fa;
        color: #333;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: bold;
        border: 1px solid #dee2e6;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
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
    st.markdown("""
    
        <h1 style='color: white; margin: 0; font-size: 2.5rem;'> Data Visualization Studio</h1>

    """, unsafe_allow_html=True)

    st.markdown("### Choose Data Source")
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your data file (CSV, Excel, JSON)", 
            type=["csv", "xlsx", "json"],
            help="Supported formats: CSV, Excel, JSON",
            key="visualize_uploader_top"
        )
    
    with col2:
        use_sample = st.button("Use Sample Data", 
                             help="Load sample data to explore all visualization capabilities",
                             use_container_width=True)

    if use_sample:
        df = create_sample_data()
        st.session_state.df_visualize = df
        st.success("Sample data loaded successfully! Explore various visualizations with our comprehensive demo dataset.")
        
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

    if "df_visualize" in st.session_state:
        df = st.session_state.df_visualize
        st.success("Using previously uploaded data for Visualize.")
    else:
        st.info("Please upload a data file to begin.")
        return
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
    if st.button("Reset Selection", use_container_width=True):
        st.session_state.selected_category = None
        st.session_state.selected_chart = None
        st.rerun()
    
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
    
    # Get custom colors for each category
    colors = get_chart_colors(len(categories), 
                            ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"], 
                            "ridge")
    
    fig = go.Figure()
    
    for i, category in enumerate(categories):
        data = df[df[category_col] == category][value_col]
        
        fig.add_trace(go.Violin(
            y=data,
            name=str(category),
            side="positive",
            line_color="rgba(0,0,0,0.2)",
            fillcolor=colors[i % len(colors)],
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

def create_funnel_chart(df, stage_col, value_col, colors=None):
    """Create funnel chart"""
    funnel_data = df.groupby(stage_col)[value_col].sum().reset_index()
    funnel_data = funnel_data.sort_values(value_col, ascending=False)
    
    # Default professional color palette if none provided
    if colors is None:
        colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"]
    
    fig = go.Figure(go.Funnel(
        y=funnel_data[stage_col],
        x=funnel_data[value_col],
        textinfo="value+percent previous",
        textposition="inside",
        marker=dict(color=colors[:len(funnel_data)])
    ))
    
    fig.update_layout(title="Funnel Chart", template="plotly_white")
    return fig

def create_gauge_chart(df, value_col, title="Gauge Chart"):
    """Create gauge chart"""
    value = df[value_col].mean()
    max_val = df[value_col].max()
    
    # Get custom colors
    colors = get_chart_colors(3, ["#00c7b7", "#E5E5E5", "#A9A9A9"], "gauge")
    bar_color = colors[0]
    step_colors = colors[1:]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': max_val * 0.8},
        gauge={
            'axis': {'range': [None, max_val]},
            'bar': {'color': bar_color},
            'steps': [
                {'range': [0, max_val * 0.5], 'color': step_colors[0]},
                {'range': [max_val * 0.5, max_val * 0.8], 'color': step_colors[1]}
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
    
    # Get custom color
    colors = get_chart_colors(1, ["#4C72B0"], "polar")
    color = colors[0]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=polar_data[value_col],
        theta=polar_data[category_col],
        fill='toself',
        name='Values',
        marker=dict(color=color)
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

# Color utility functions
def get_chart_colors(num_colors=1, default_colors=None, key_prefix=""):
    """Get custom colors for charts with color picker"""
    if default_colors is None:
        default_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"]
    
    custom_colors = []
    use_custom = st.checkbox("Customize colors", key=f"{key_prefix}_use_custom")
    
    if use_custom:
        col1, col2 = st.columns(2)
        with col1:
            for i in range(num_colors):
                color = st.color_picker(
                    f"Color {i+1}", 
                    default_colors[i % len(default_colors)], 
                    key=f"{key_prefix}_color_{i}"
                )
                custom_colors.append(color)
        return custom_colors
    return default_colors[:num_colors]

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
        
        # Get custom colors
        colors = get_chart_colors(1, ["#4C72B0"], "hist")
        
        fig = px.histogram(df, x=col, nbins=bins, opacity=opacity,
                          marginal=marginal if marginal != "None" else None,
                          color_discrete_sequence=colors)
        fig.update_layout(title=f"Distribution of {col}", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "violin":
        col = st.selectbox("Select column", numeric_cols, key="violin_col")
        if cat_cols:
            group_by = st.selectbox("Group by (optional)", ["None"] + cat_cols, key="violin_group")
            # Get custom colors for number of groups
            num_groups = len(df[group_by].unique()) if group_by != "None" else 1
            colors = get_chart_colors(num_groups, ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"], "violin")
            
            if group_by != "None":
                fig = px.violin(df, y=col, x=group_by, box=True, points="all", color=group_by, color_discrete_sequence=colors)
            else:
                fig = px.violin(df, y=col, box=True, points="all", color_discrete_sequence=colors)
        else:
            colors = get_chart_colors(1, ["#4C72B0"], "violin")
            fig = px.violin(df, y=col, box=True, points="all", color_discrete_sequence=colors)
        
        fig.update_layout(title=f"Violin Plot of {col}", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "density":
        col = st.selectbox("Select column", numeric_cols, key="density_col")
        try:
            import scipy.stats as stats
            from scipy.stats import gaussian_kde
            
            # Get custom colors
            colors = get_chart_colors(4, ["#4C72B0", "#C44E52", "#55A868", "#8172B3"], "density")
            
            data = df[col].dropna()
            density = gaussian_kde(data)
            xs = np.linspace(data.min(), data.max(), 200)
            ys = density(xs)
            
            fig = go.Figure()
            # Add histogram
            fig.add_trace(go.Histogram(
                x=data,
                name="Histogram",
                histnorm='probability density',
                marker_color=colors[0],
                opacity=0.5
            ))
            # Add KDE
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                name="KDE",
                line=dict(color=colors[1], width=2)
            ))
            
            # Add normal distribution fit
            mean, std = stats.norm.fit(data)
            ys_norm = stats.norm.pdf(xs, mean, std)
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys_norm,
                name="Normal Fit",
                line=dict(color=colors[2], dash='dash', width=2)
            ))
            
            # Add rug plot
            fig.add_trace(go.Scatter(
                x=data,
                y=np.zeros_like(data),
                mode='markers',
                marker=dict(symbol='line-ns', size=8, color=colors[3]),
                name='Data Points',
                opacity=0.5
            ))
            
            fig.update_layout(
                title=f"Density Plot of {col}",
                template="plotly_white",
                showlegend=True,
                xaxis_title=col,
                yaxis_title="Density"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add statistical tests
            if st.checkbox("Show distribution tests"):
                st.write("### Statistical Tests")
                # Normality test
                stat, p_value = stats.normaltest(data)
                st.write(f"Normal test p-value: {p_value:.4f}")
                # Skewness and Kurtosis
                st.write(f"Skewness: {stats.skew(data):.4f}")
                st.write(f"Kurtosis: {stats.kurtosis(data):.4f}")
                
        except ImportError:
            st.error("scipy is required for density plots")
    
    elif chart_type == "box":
        col = st.selectbox("Select column", numeric_cols, key="box_col")
        if cat_cols:
            group_by = st.selectbox("Group by (optional)", ["None"] + cat_cols, key="box_group")
            # Get custom colors for groups
            num_groups = len(df[group_by].unique()) if group_by != "None" else 1
            colors = get_chart_colors(num_groups, ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"], "box")
            
            if group_by != "None":
                fig = px.box(df, y=col, x=group_by, color=group_by, color_discrete_sequence=colors)
            else:
                fig = px.box(df, y=col, color_discrete_sequence=colors)
        else:
            colors = get_chart_colors(1, ["#4C72B0"], "box")
            fig = px.box(df, y=col, color_discrete_sequence=colors)
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
            # Get custom colors
            colors = get_chart_colors(2, ["#4C72B0", "#C44E52"], "qq")
            
            fig = go.Figure()
            data = df[col].dropna()
            qq_data = stats.probplot(data, dist="norm")
            fig.add_trace(go.Scatter(
                x=qq_data[0][0],
                y=qq_data[0][1],
                mode='markers',
                name='Q-Q Plot',
                marker=dict(color=colors[0], size=8)
            ))
            # Add reference line
            slope, intercept = qq_data[1][0], qq_data[1][1]
            ref_line = slope * np.array(qq_data[0][0]) + intercept
            fig.add_trace(go.Scatter(
                x=qq_data[0][0],
                y=ref_line,
                mode='lines',
                name='Reference Line',
                line=dict(color=colors[1], dash='dash', width=2)
            ))
            fig.update_layout(title=f"Q-Q Plot: {col}", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.error("scipy is required for Q-Q plots")
    elif chart_type == "ecdf":
        col = st.selectbox("Select column", numeric_cols, key="ecdf_col")
        colors = get_chart_colors(1, ["#4C72B0"], "ecdf")
        fig = px.ecdf(df, x=col, color_discrete_sequence=colors)
        fig.update_layout(title=f"ECDF: {col}", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    app()

def generate_categorical_chart(df, cat_cols, numeric_cols):
    """Generate categorical charts"""
    if not cat_cols:
        st.warning("No categorical columns found in the dataset")
        return
        
    chart_type = st.session_state.selected_chart
    
    # Professional color palette
    default_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"]
    
    if chart_type == "pie":
        col1, col2 = st.columns([3, 1])
        with col1:
            col = st.selectbox("Select categorical column", cat_cols, key="pie_col")
        with col2:
            use_custom_colors = st.checkbox("Custom Colors", key="pie_custom_colors")
        
        pie_df = df[col].value_counts().reset_index()
        pie_df.columns = [col, "count"]
        
        if use_custom_colors:
            unique_categories = pie_df[col].unique()
            custom_colors = []
            for i, cat in enumerate(unique_categories):
                color = st.color_picker(f"Color for {cat}", default_colors[i % len(default_colors)], key=f"pie_color_{i}")
                custom_colors.append(color)
            fig = px.pie(pie_df, names=col, values="count", 
                        color_discrete_sequence=custom_colors)
        else:
            fig = px.pie(pie_df, names=col, values="count", 
                        color_discrete_sequence=default_colors)
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
    
    if not (lat_cols or lon_cols or 'country' in df.columns or 'region' in df.columns):
        st.warning("No geographical data found. Please ensure your data contains latitude/longitude columns or country/region information.")
        return
    
    # Map settings    
    st.sidebar.markdown("### Map Settings")
    map_style = st.sidebar.selectbox(
        "Map Style",
        ["open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner"],
        key="map_style"
    )
    
    # Common settings
    zoom_level = st.sidebar.slider("Zoom Level", 1, 20, 3, key="map_zoom")
    show_labels = st.sidebar.checkbox("Show Labels", value=True, key="map_labels")
    
    if chart_type == "scatter_map":
        col1, col2, col3 = st.columns(3)
        with col1:
            lat_col = st.selectbox("Latitude column", lat_cols, key="map_lat")
        with col2:
            lon_col = st.selectbox("Longitude column", lon_cols, key="map_lon")
        with col3:
            color_col = st.selectbox("Color by", ["None"] + numeric_cols + cat_cols, key="map_color")
            
        size_col = st.selectbox("Size by", ["None"] + numeric_cols, key="map_size")
        
        fig = px.scatter_mapbox(
            df,
            lat=lat_col,
            lon=lon_col,
            color=color_col if color_col != "None" else None,
            size=size_col if size_col != "None" else None,
            hover_data=numeric_cols[:3],
            zoom=zoom_level,
            mapbox_style=map_style,
            color_continuous_scale="Viridis" if color_col in numeric_cols else None
        )
        
        if show_labels:
            fig.update_traces(textposition="top center")
            
        fig.update_layout(
            title="Scatter Map",
            mapbox=dict(zoom=zoom_level),
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "density_map":
        col1, col2 = st.columns(2)
        with col1:
            lat_col = st.selectbox("Latitude column", lat_cols, key="density_lat")
        with col2:
            lon_col = st.selectbox("Longitude column", lon_cols, key="density_lon")
            
        radius = st.slider("Radius", 5, 50, 20, key="density_radius")
        
        fig = px.density_mapbox(
            df,
            lat=lat_col,
            lon=lon_col,
            radius=radius,
            zoom=zoom_level,
            mapbox_style=map_style
        )
        
        fig.update_layout(
            title="Density Map",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "choropleth":
        if 'country' in df.columns or 'region' in df.columns:
            geo_col = 'country' if 'country' in df.columns else 'region'
            
            col1, col2 = st.columns(2)
            with col1:
                value_col = st.selectbox("Value column", numeric_cols, key="chloro_val")
            with col2:
                color_scale = st.selectbox("Color scale", 
                                       ["Viridis", "Plasma", "Inferno", "Magma", "RdBu"],
                                       key="chloro_scale")
            
            fig = px.choropleth(
                df,
                locations=geo_col,
                locationmode='country-names',
                color=value_col,
                color_continuous_scale=color_scale,
                title=f"{value_col} by {geo_col}"
            )
            
            fig.update_layout(
                geo=dict(showframe=False, showcoastlines=True),
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need country or region column for choropleth map")
            
    elif chart_type == "bubble_map":
        col1, col2, col3 = st.columns(3)
        with col1:
            lat_col = st.selectbox("Latitude column", lat_cols, key="bubble_lat")
        with col2:
            lon_col = st.selectbox("Longitude column", lon_cols, key="bubble_lon")
        with col3:
            size_col = st.selectbox("Bubble size", numeric_cols, key="bubble_size")
            
        color_col = st.selectbox("Color by", ["None"] + numeric_cols + cat_cols, key="bubble_color")
        
        fig = px.scatter_geo(
            df,
            lat=lat_col,
            lon=lon_col,
            size=size_col,
            color=color_col if color_col != "None" else None,
            projection="natural earth",
            color_continuous_scale="Viridis" if color_col in numeric_cols else None
        )
        
        if show_labels:
            fig.update_traces(textposition="top center")
            
        fig.update_layout(
            title="Bubble Map",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
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

def generate_relationship_chart(df, numeric_cols, cat_cols):
    """Generate relationship charts"""
    chart_type = st.session_state.selected_chart
    
    if chart_type == "scatter":
        # Professional color palette
        default_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            x = st.selectbox("X axis", numeric_cols, key="scatter_x")
        with col2:
            y = st.selectbox("Y axis", [col for col in numeric_cols if col != x], key="scatter_y")
        with col3:
            color = st.selectbox("Color by", ["None"] + cat_cols, key="scatter_color") if cat_cols else "None"
        
        col4, col5 = st.columns(2)
        with col4:
            point_color = st.color_picker("Point Color", default_colors[0], key="scatter_point_color")
            show_trendline = st.checkbox("Add trendline", key="scatter_trend")
        with col5:
            if show_trendline:
                trendline_color = st.color_picker("Trendline Color", "#FF6B6B", key="scatter_trendline_color")
        
        fig = px.scatter(df, x=x, y=y,
                        color=color if color != "None" else None,
                        trendline="ols" if show_trendline else None,
                        color_discrete_sequence=[point_color] if color == "None" else default_colors)
        
        if show_trendline:
            for trace in fig.data:
                if trace.mode == "lines":
                    trace.line.color = trendline_color
        fig.update_layout(title=f"Scatter Plot: {y} vs {x}", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "scatter_3d":
        if len(numeric_cols) >= 3:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                x = st.selectbox("X axis", numeric_cols, key="scatter3d_x")
            with col2:
                y = st.selectbox("Y axis", [col for col in numeric_cols if col != x], key="scatter3d_y")
            with col3:
                z = st.selectbox("Z axis", [col for col in numeric_cols if col not in [x, y]], key="scatter3d_z")
            with col4:
                color = st.selectbox("Color by", ["None"] + cat_cols, key="scatter3d_color") if cat_cols else "None"
            
            fig = px.scatter_3d(df, x=x, y=y, z=z,
                              color=color if color != "None" else None,
                              color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(title="3D Scatter Plot", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 3 numeric columns for 3D scatter plot")
    
    elif chart_type == "correlation":
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            
            fig = px.imshow(corr,
                          labels=dict(x="Features", y="Features", color="Correlation"),
                          x=corr.columns,
                          y=corr.columns,
                          color_continuous_scale="RdBu")
            
            fig.update_layout(title="Correlation Matrix",
                            template="plotly_white",
                            height=600)
            
            # Add correlation values
            annotations = []
            for i, row in enumerate(corr.values):
                for j, value in enumerate(row):
                    annotations.append(
                        dict(
                            x=j,
                            y=i,
                            text=f"{value:.2f}",
                            font=dict(color="white" if abs(value) > 0.5 else "black"),
                            showarrow=False
                        )
                    )
            fig.update_layout(annotations=annotations)
            
            st.plotly_chart(fig, use_container_width=True)
            
            if st.checkbox("Show detailed correlation analysis", key="corr_detail"):
                st.write("### Detailed Correlation Analysis")
                corr_df = corr.unstack()
                corr_df = corr_df[corr_df != 1.0]
                corr_df = corr_df.sort_values(ascending=False)
                st.dataframe(corr_df, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric columns for correlation analysis")
    
    elif chart_type == "pairplot":
        if len(numeric_cols) >= 2:
            n_cols = st.slider("Number of features", 2, min(6, len(numeric_cols)), 3, key="pair_n")
            selected_cols = st.multiselect("Select features", numeric_cols, default=numeric_cols[:n_cols], key="pair_cols")
            
            if len(selected_cols) >= 2:
                color = st.selectbox("Color by", ["None"] + cat_cols, key="pair_color") if cat_cols else "None"
                
                fig = px.scatter_matrix(df,
                                      dimensions=selected_cols,
                                      color=color if color != "None" else None,
                                      color_discrete_sequence=px.colors.qualitative.Set2)
                
                fig.update_layout(title="Pair Plot",
                                template="plotly_white",
                                height=800)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least 2 features")
        else:
            st.warning("Need at least 2 numeric columns for pair plot")
    
    elif chart_type == "network":
        if len(cat_cols) >= 2:
            col1, col2, col3 = st.columns(3)
            with col1:
                source = st.selectbox("Source", cat_cols, key="network_source")
            with col2:
                target = st.selectbox("Target", [col for col in cat_cols if col != source], key="network_target")
            with col3:
                value = st.selectbox("Value (optional)", ["None"] + numeric_cols, key="network_value")
            
            import networkx as nx
            G = nx.Graph()
            
            # Create edges
            if value != "None":
                edges = df.groupby([source, target])[value].sum().reset_index()
                for _, row in edges.iterrows():
                    G.add_edge(row[source], row[target], weight=row[value])
            else:
                edges = df.groupby([source, target]).size().reset_index(name='count')
                for _, row in edges.iterrows():
                    G.add_edge(row[source], row[target], weight=row['count'])
            
            # Calculate layout
            pos = nx.spring_layout(G)
            
            # Create traces for edges
            edge_trace = go.Scatter(
                x=[], y=[],
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines')
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace['x'] += (x0, x1, None)
                edge_trace['y'] += (y0, y1, None)
            
            # Create traces for nodes
            node_trace = go.Scatter(
                x=[], y=[],
                text=[],
                mode='markers+text',
                hoverinfo='text',
                marker=dict(
                    showscale=True,
                    colorscale='YlGnBu',
                    reversescale=True,
                    color=[],
                    size=10,
                    line=dict(width=2)))
            
            for node in G.nodes():
                x, y = pos[node]
                node_trace['x'] += (x,)
                node_trace['y'] += (y,)
                node_trace['text'] += (node,)
                node_trace['marker']['color'] += (G.degree(node),)
            
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              title="Network Graph",
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=20,l=5,r=5,t=40),
                              template="plotly_white"))
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 categorical columns for network graph")
    
    elif chart_type == "regression":
        col1, col2, col3 = st.columns(3)
        with col1:
            x = st.selectbox("Independent variable (X)", numeric_cols, key="reg_x")
        with col2:
            y = st.selectbox("Dependent variable (Y)", [col for col in numeric_cols if col != x], key="reg_y")
        with col3:
            reg_type = st.selectbox("Regression type", 
                                  ["Linear", "Lowess", "Polynomial"], 
                                  key="reg_type")
        
        if reg_type == "Polynomial":
            degree = st.slider("Polynomial degree", 1, 5, 2, key="poly_degree")
            fig = px.scatter(df, x=x, y=y, 
                           trendline="ols", 
                           trendline_options=dict(order=degree),
                           color_discrete_sequence=px.colors.qualitative.Set2)
        else:
            fig = px.scatter(df, x=x, y=y,
                           trendline=reg_type.lower(),
                           color_discrete_sequence=px.colors.qualitative.Set2)
        
        fig.update_layout(title=f"{reg_type} Regression: {y} vs {x}",
                         template="plotly_white")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show regression statistics
        if reg_type == "Linear":
            import statsmodels.api as sm
            X = sm.add_constant(df[x])
            model = sm.OLS(df[y], X).fit()
            st.write("### Regression Statistics")
            st.write(model.summary().tables[1])

def generate_timeseries_chart(df, numeric_cols, date_cols):
    """Generate time series charts"""
    chart_type = st.session_state.selected_chart
    
    # Professional color palette
    default_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"]
    
    if not date_cols:
        st.warning("No date columns found. Please ensure your data contains date/time columns.")
        return
    
    if chart_type == "timeseries":
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            date_col = st.selectbox("Date column", date_cols, key="ts_date")
        with col2:
            value_col = st.selectbox("Value column", numeric_cols, key="ts_value")
        with col3:
            color = st.color_picker("Line Color", default_colors[0], key="ts_color")
        
        # Convert to datetime if needed
        if df[date_col].dtype == 'object':
            df[date_col] = pd.to_datetime(df[date_col])
        
        fig = px.line(df, x=date_col, y=value_col, 
                     color_discrete_sequence=[color])
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
    
    if not date_cols:
        st.warning("Need date column for animation. Please ensure your data contains date/time columns.")
        return
        
    if chart_type == "animated_scatter":
        col1, col2 = st.columns(2)
        with col1:
            x = st.selectbox("X axis", numeric_cols, key="anim_x")
            y = st.selectbox("Y axis", [col for col in numeric_cols if col != x], key="anim_y")
            size = st.selectbox("Size by", ["None"] + numeric_cols, key="anim_size")
        with col2:
            animation_frame = st.selectbox("Animation frame", date_cols, key="anim_frame")
            color = st.selectbox("Color by", ["None"] + cat_cols + numeric_cols, key="anim_color")
            animation_group = st.selectbox("Animation group", ["None"] + cat_cols, key="anim_group")

        # Ensure datetime type
        if df[animation_frame].dtype == 'object':
            df[animation_frame] = pd.to_datetime(df[animation_frame])

        # Create animated scatter plot
        fig = px.scatter(
            df,
            x=x,
            y=y,
            animation_frame=animation_frame,
            animation_group=animation_group if animation_group != "None" else None,
            size=df[size] if size != "None" else None,
            color=color if color != "None" else None,
            range_x=[df[x].min(), df[x].max()],
            range_y=[df[y].min(), df[y].max()],
            color_continuous_scale="Viridis" if color in numeric_cols else None,
            color_discrete_sequence=px.colors.qualitative.Set2 if color in cat_cols else None
        )
        
        fig.update_layout(
            title="Animated Scatter Plot",
            template="plotly_white",
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=500, redraw=True),
                        fromcurrent=True,
                        mode='immediate'
                    )]
                )]
            )]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "animated_bar":
        col1, col2 = st.columns(2)
        with col1:
            cat_col = st.selectbox("Category", cat_cols, key="animbar_cat")
            value_col = st.selectbox("Value", numeric_cols, key="animbar_val")
        with col2:
            animation_col = st.selectbox("Animation frame", date_cols, key="animbar_frame")
            sort_order = st.selectbox("Sort order", ["Total", "Current"], key="animbar_sort")
        
        # Prepare data
        if sort_order == "Total":
            totals = df.groupby(cat_col)[value_col].sum().sort_values(ascending=False)
            category_order = totals.index.tolist()
        else:
            category_order = None
        
        fig = px.bar(
            df,
            x=cat_col,
            y=value_col,
            animation_frame=animation_col,
            category_orders={cat_col: category_order} if category_order else None,
            color=cat_col,
            range_y=[0, df[value_col].max()],
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        fig.update_layout(
            title="Animated Bar Chart",
            template="plotly_white",
            showlegend=False,
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=500, redraw=True),
                        fromcurrent=True,
                        mode='immediate'
                    )]
                )]
            )]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "bubble_animated":
        col1, col2 = st.columns(2)
        with col1:
            x = st.selectbox("X axis", numeric_cols, key="animbub_x")
            y = st.selectbox("Y axis", [col for col in numeric_cols if col != x], key="animbub_y")
            size = st.selectbox("Size", numeric_cols, key="animbub_size")
        with col2:
            animation_frame = st.selectbox("Animation frame", date_cols, key="animbub_frame")
            color = st.selectbox("Color by", ["None"] + cat_cols, key="animbub_color")
            
        fig = px.scatter(
            df,
            x=x,
            y=y,
            size=size,
            animation_frame=animation_frame,
            color=color if color != "None" else None,
            range_x=[df[x].min(), df[x].max()],
            range_y=[df[y].min(), df[y].max()],
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        fig.update_layout(
            title="Animated Bubble Chart",
            template="plotly_white",
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=500, redraw=True),
                        fromcurrent=True,
                        mode='immediate'
                    )]
                )]
            )]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "time_evolution":
        col1, col2 = st.columns(2)
        with col1:
            value_col = st.selectbox("Value", numeric_cols, key="evolution_val")
        with col2:
            time_col = st.selectbox("Time column", date_cols, key="evolution_time")
            
        color_col = st.selectbox("Group by", ["None"] + cat_cols, key="evolution_group")
        
        # Ensure datetime type
        if df[time_col].dtype == 'object':
            df[time_col] = pd.to_datetime(df[time_col])
            
        fig = px.line(
            df,
            x=time_col,
            y=value_col,
            color=color_col if color_col != "None" else None,
            animation_frame=df[time_col].dt.strftime('%Y-%m'),
            range_y=[df[value_col].min(), df[value_col].max()],
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        fig.update_layout(
            title="Time Evolution Chart",
            template="plotly_white",
            showlegend=True,
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True},
                                      "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "type": "buttons"
            }]
        )
        st.plotly_chart(fig, use_container_width=True)
    # Removed duplicate animated_bar section

def create_sample_data():
    """Create sample dataset with essential visualization types"""
    np.random.seed(42)
    n_samples = 100
    
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='D')
    
    data = {
        'date': dates,
        'sales': np.random.normal(1000, 200, n_samples) + np.sin(np.linspace(0, 4*np.pi, n_samples)) * 100,
        'revenue': np.random.normal(5000, 1000, n_samples),
        'satisfaction': np.random.beta(8, 2, n_samples) * 5,
        'visitors': np.random.randint(100, 1000, n_samples),
        
        'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], n_samples),
        'segment': np.random.choice(['New', 'Returning', 'VIP'], n_samples),
        
        'latitude': np.random.normal(40, 10, n_samples).clip(25, 50),
        'longitude': np.random.normal(-100, 20, n_samples).clip(-120, -70),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        
        'stock_open': np.random.normal(100, 10, n_samples).cumsum(),
        'stock_high': None,
        'stock_low': None,
        'stock_close': None
    }
    
    df = pd.DataFrame(data)
    
    # Process financial data
    vol = np.random.uniform(1, 3, n_samples)
    df['stock_high'] = df['stock_open'] + np.abs(np.random.normal(0, 1, n_samples)) * vol
    df['stock_low'] = df['stock_open'] - np.abs(np.random.normal(0, 1, n_samples)) * vol
    df['stock_close'] = df['stock_open'] + np.random.normal(0, vol, n_samples)
    
    # Add time features
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.day_name()
    
    # Add some missing values
    mask = np.random.choice([True, False], n_samples, p=[0.05, 0.95])
    df.loc[mask, 'satisfaction'] = np.nan
    
    return df    # Footer
    st.markdown("---")
    st.markdown("### Thank you for using the Advanced Data Visualization Studio!")
    st.markdown("#### Developed by MLGenie Team")

if __name__ == "__main__":
    app()
    df = create_sample_data()
    st.success("Using sample data to demonstrate visualization capabilities!")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()

    st.markdown("## Data Insights Dashboard")
    
    # Distribution Analysis
    st.markdown("### Distribution Analysis")
    if numeric_cols:
        col1, col2 = st.columns(2)
        with col1:
            selected_num = st.selectbox("Select numeric feature", numeric_cols, key="dash_dist")
            fig = px.histogram(df, x=selected_num, color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(title=f"Distribution of {selected_num}", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                mask = np.zeros_like(corr)
                mask[np.triu_indices_from(mask)] = True
                fig = px.imshow(corr, 
                              labels=dict(x="Features", y="Features", color="Correlation"),
                              color_continuous_scale="RdBu")
                fig.update_layout(title="Correlation Heatmap", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
    
    # Categorical Analysis
    if cat_cols:
        st.markdown("### Categorical Analysis")
        selected_cat = st.selectbox("Select categorical feature", cat_cols, key="dash_cat")
        fig = px.pie(df[selected_cat].value_counts().reset_index(), 
                    values=selected_cat, names="index",
                    color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(title=f"Distribution of {selected_cat}", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    # Time Series Analysis
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    if len(numeric_cols) > 0 and len(date_cols) > 0:
        st.markdown("### Time Series Analysis")
        for col in numeric_cols[:3]:  # Show first 3 numeric columns
            fig = px.line(df, x=date_cols[0], y=col,
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(title=f"Time Series: {col}", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
    
    # Statistical Summary
    st.markdown("### Statistical Summary")
    if numeric_cols:
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    # Missing Values Analysis
    st.markdown("### Missing Values Analysis")
    missing = pd.DataFrame({
        'Column': df.columns,
        'Missing Values': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
    }).sort_values('Missing %', ascending=False)
    
    if missing['Missing %'].sum() > 0:
        fig = px.bar(missing[missing['Missing %'] > 0],
                    x='Column', y='Missing %',
                    color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(title="Missing Values by Column",
                         template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values in the dataset!")
    
    # Outlier Analysis
    if numeric_cols:
        st.markdown("### Outlier Analysis")
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_num = st.selectbox("Select feature for outlier analysis", numeric_cols, key="dash_outlier")
        with col2:
            box_color = st.color_picker("Box Color", "#4C72B0", key="box_color")
            
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=df[selected_num], 
            name=selected_num,
            fillcolor=box_color,
            line=dict(color=box_color),
            marker=dict(color=box_color)
        ))
        fig.update_layout(title=f"Box Plot: {selected_num}",
                         template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate and show outlier statistics
        Q1 = df[selected_num].quantile(0.25)
        Q3 = df[selected_num].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[selected_num] < (Q1 - 1.5 * IQR)) | (df[selected_num] > (Q3 + 1.5 * IQR))][selected_num]
        
        st.write(f"Number of outliers detected: {len(outliers)}")
        if len(outliers) > 0:
            st.write("Outlier Statistics:")
            st.dataframe(outliers.describe(), use_container_width=True)
    
    # Recommendations
    st.markdown("### Recommendations")
    recommendations = []
    
    # Missing values recommendations
    if missing['Missing %'].sum() > 0:
        recommendations.append("Consider handling missing values in the following columns: " + 
                            ", ".join(missing[missing['Missing %'] > 0]['Column'].tolist()))
    
    # Correlation recommendations
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        high_corr = np.where(np.abs(corr) > 0.8)
        high_corr = [(corr.index[x], corr.columns[y], corr.iloc[x, y]) 
                    for x, y in zip(*high_corr) if x != y and x < y]
        if high_corr:
            recommendations.append("Consider feature selection due to high correlation between: " + 
                                ", ".join([f"{x}-{y} ({z:.2f})" for x, y, z in high_corr]))
    
    # Categorical recommendations
    if cat_cols:
        for col in cat_cols:
            if df[col].nunique() > 10:
                recommendations.append(f"Consider encoding or grouping {col} due to high cardinality")
    else:
        recommendations.append("No categorical columns found for analysis")
    
    # Display recommendations
    if recommendations:
        for rec in recommendations:
            st.info(rec)
    else:
        st.success("No immediate data quality issues detected!")

    # End of data analysis section
    
    # Reset session button

    # Session state management
    st.markdown("---")
    st.subheader("Session State Management")
    if st.button("Reset Session", use_container_width=True):
        st.session_state.clear()
        st.success("Session reset successful!")
        st.experimental_rerun()

    st.markdown("---")
    st.markdown("### Thank you for using the Advanced Data Visualization Studio!")
    st.markdown("#### Developed by MLGenie Team")
    st.markdown("##### For feedback and suggestions, please contact us")
