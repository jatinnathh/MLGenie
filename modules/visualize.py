import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from utils.shared import apply_common_settings

apply_common_settings()

def app():
    st.title("Data Visualization")
    
    uploaded_file = st.file_uploader("Upload your data", type=["csv", "xlsx"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        st.subheader(" Select Chart Category")
        
        # Initialize session state for category selection
        if 'selected_category' not in st.session_state:
            st.session_state.selected_category = None
        
        # Create category buttons in a row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(" Distribution", use_container_width=True, 
                        type="primary" if st.session_state.selected_category == "distribution" else "secondary"):
                st.session_state.selected_category = "distribution"
        
        with col2:
            if st.button("Comparison", use_container_width=True,
                        type="primary" if st.session_state.selected_category == "comparison" else "secondary"):
                st.session_state.selected_category = "comparison"
        
        with col3:
            if st.button("Categorical", use_container_width=True,
                        type="primary" if st.session_state.selected_category == "categorical" else "secondary"):
                st.session_state.selected_category = "categorical"
        
        # Display chart options based on selected category
        if st.session_state.selected_category == "distribution":
            st.markdown("###  Distribution Charts")
            
            # Chart selection grid
            chart_col1, chart_col2, chart_col3 = st.columns(3)
            
            with chart_col1:
                if st.button(" Histogram", use_container_width=True):
                    st.session_state.selected_chart = "histogram"
                
                if st.button(" Violin Plot", use_container_width=True):
                    st.session_state.selected_chart = "violin"
            
            with chart_col2:
                if st.button(" Density Plot", use_container_width=True):
                    st.session_state.selected_chart = "density"
                
                if st.button(" Box Plot", use_container_width=True):
                    st.session_state.selected_chart = "box"
            
            with chart_col3:
                if st.button(" Ridge Plot", use_container_width=True):
                    st.session_state.selected_chart = "ridge"
                
                if st.button(" Q-Q Plot", use_container_width=True):
                    st.session_state.selected_chart = "qq"
            
            # Chart generation based on selection
            if hasattr(st.session_state, 'selected_chart'):
                if st.session_state.selected_chart == "histogram":
                    col = st.selectbox("Select column", numeric_cols, key="hist_col")
                    bins = st.slider("Number of bins", 10, 100, 30)
                    fig = px.histogram(df, x=col, nbins=bins, color_discrete_sequence=["#00c7b7"])
                    fig.update_layout(title=f"Histogram of {col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif st.session_state.selected_chart == "violin":
                    col = st.selectbox("Select column", numeric_cols, key="violin_col")
                    fig = px.violin(df, y=col, box=True, points="all", color_discrete_sequence=["#00c7b7"])
                    fig.update_layout(title=f"Violin Plot of {col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif st.session_state.selected_chart == "density":
                    col = st.selectbox("Select column", numeric_cols, key="density_col")
                    fig = px.histogram(df, x=col, histnorm='probability density', 
                                     color_discrete_sequence=["#00c7b7"])
                    fig.update_layout(title=f"Density Plot of {col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif st.session_state.selected_chart == "box":
                    col = st.selectbox("Select column", numeric_cols, key="box_col")
                    fig = px.box(df, y=col, color_discrete_sequence=["#00c7b7"])
                    fig.update_layout(title=f"Box Plot of {col}")
                    st.plotly_chart(fig, use_container_width=True)
        
        elif st.session_state.selected_category == "comparison":
            st.markdown("### ⚖️ Comparison Charts")
            
            # Chart selection grid
            chart_col1, chart_col2, chart_col3 = st.columns(3)
            
            with chart_col1:
                if st.button("Bar Chart", use_container_width=True):
                    st.session_state.selected_chart = "bar"
                
                if st.button("Line Chart", use_container_width=True):
                    st.session_state.selected_chart = "line"
            
            with chart_col2:
                if st.button(" Heatmap", use_container_width=True):
                    st.session_state.selected_chart = "heatmap"
                
                if st.button(" Grouped Bar", use_container_width=True):
                    st.session_state.selected_chart = "grouped_bar"
            
            with chart_col3:
                if st.button("Stacked Bar", use_container_width=True):
                    st.session_state.selected_chart = "stacked_bar"
                
                if st.button("Area Chart", use_container_width=True):
                    st.session_state.selected_chart = "area"
            
            # Chart generation based on selection
            if hasattr(st.session_state, 'selected_chart'):
                if st.session_state.selected_chart == "bar":
                    x = st.selectbox("X axis", cat_cols, key="bar_x")
                    y = st.selectbox("Y axis", numeric_cols, key="bar_y")
                    fig = px.bar(df, x=x, y=y, color_discrete_sequence=["#00c7b7"])
                    fig.update_layout(title=f"Bar Chart: {y} by {x}")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif st.session_state.selected_chart == "line":
                    x = st.selectbox("X axis", df.columns, key="line_x")
                    y = st.selectbox("Y axis", numeric_cols, key="line_y")
                    fig = px.line(df, x=x, y=y, color_discrete_sequence=["#00c7b7"])
                    fig.update_layout(title=f"Line Chart: {y} over {x}")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif st.session_state.selected_chart == "heatmap":
                    st.write("Correlation Heatmap of Numeric Variables")
                    corr = df[numeric_cols].corr()
                    fig = px.imshow(corr, text_auto=True, aspect="auto",
                                   color_continuous_scale="RdBu_r")
                    fig.update_layout(title="Correlation Heatmap")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif st.session_state.selected_chart == "grouped_bar":
                    x = st.selectbox("X axis", cat_cols, key="grouped_x")
                    y = st.selectbox("Y axis", numeric_cols, key="grouped_y")
                    color = st.selectbox("Group by", cat_cols, key="grouped_color")
                    fig = px.bar(df, x=x, y=y, color=color, barmode='group')
                    fig.update_layout(title=f"Grouped Bar Chart: {y} by {x} and {color}")
                    st.plotly_chart(fig, use_container_width=True)
        
        elif st.session_state.selected_category == "categorical":
            st.markdown("### Categorical Charts")
            
            # Chart selection grid
            chart_col1, chart_col2, chart_col3 = st.columns(3)
            
            with chart_col1:
                if st.button("Pie Chart", use_container_width=True):
                    st.session_state.selected_chart = "pie"
                
                if st.button("Donut Chart", use_container_width=True):
                    st.session_state.selected_chart = "donut"
            
            with chart_col2:
                if st.button("Count Plot", use_container_width=True):
                    st.session_state.selected_chart = "count"
                
                if st.button(" Treemap", use_container_width=True):
                    st.session_state.selected_chart = "treemap"
            
            with chart_col3:
                if st.button(" Sunburst", use_container_width=True):
                    st.session_state.selected_chart = "sunburst"
                
                if st.button("📊 Funnel Chart", use_container_width=True):
                    st.session_state.selected_chart = "funnel"
            
            # Chart generation based on selection
            if hasattr(st.session_state, 'selected_chart'):
                if st.session_state.selected_chart == "pie":
                    col = st.selectbox("Categorical column", cat_cols, key="pie_col")
                    pie_df = df[col].value_counts().reset_index()
                    pie_df.columns = [col, "count"]
                    fig = px.pie(pie_df, names=col, values="count", 
                               color_discrete_sequence=px.colors.sequential.RdBu)
                    fig.update_layout(title=f"Pie Chart of {col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif st.session_state.selected_chart == "donut":
                    col = st.selectbox("Categorical column", cat_cols, key="donut_col")
                    pie_df = df[col].value_counts().reset_index()
                    pie_df.columns = [col, "count"]
                    fig = px.pie(pie_df, names=col, values="count", hole=0.4,
                               color_discrete_sequence=px.colors.sequential.RdBu)
                    fig.update_layout(title=f"Donut Chart of {col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif st.session_state.selected_chart == "count":
                    col = st.selectbox("Categorical column", cat_cols, key="count_col")
                    count_df = df[col].value_counts().reset_index()
                    count_df.columns = [col, "count"]
                    fig = px.bar(count_df, x=col, y="count", color_discrete_sequence=["#00c7b7"])
                    fig.update_layout(title=f"Count Plot of {col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif st.session_state.selected_chart == "treemap":
                    col = st.selectbox("Categorical column", cat_cols, key="treemap_col")
                    treemap_df = df[col].value_counts().reset_index()
                    treemap_df.columns = [col, "count"]
                    fig = px.treemap(treemap_df, path=[col], values="count",
                                   color_discrete_sequence=px.colors.sequential.RdBu)
                    fig.update_layout(title=f"Treemap of {col}")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Clear selection button
        if st.button(" Clear Selection"):
            if 'selected_category' in st.session_state:
                del st.session_state.selected_category
            if 'selected_chart' in st.session_state:
                del st.session_state.selected_chart
            st.rerun()

if __name__ == "__main__":
    app()