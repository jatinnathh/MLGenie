import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# --- Only set_page_config in main app, not in this module ---

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
    st.title("MLGenie Feature Engineering")

    ############## DATA UPLOAD ##################
    st.markdown('<div class="section-header"><h3>Data Input</h3></div>', unsafe_allow_html=True)
    if "df_feature_eng" in st.session_state:
        df = st.session_state.df_feature_eng.copy()
        st.success("Dataset loaded successfully")
        col1, col2 = st.columns([3,1])
        with col2:
            if st.button("Load New Dataset", use_container_width=True):
                del st.session_state.df_feature_eng
                st.rerun()
    else:
        uploaded_file = st.file_uploader("Select your dataset (CSV/Excel)", type=["csv", "xlsx", "xls"], key="feature_eng_uploader")
        if uploaded_file is None:
            st.warning("Please upload a CSV or Excel file.")
            return
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
        st.success("Data uploaded and saved for Feature Engineering!")

    ############## DATA OVERVIEW ##################
    st.markdown('<div class="section-header"><h3>Dataset Overview</h3></div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", df.shape[0], help="Total number of records")
    with col2:
        st.metric("Columns", df.shape[1], help="Total number of features")
    with col3:
        missing_count = df.isnull().sum().sum()
        st.metric("Missing Values", missing_count, help="Total null values in dataset")
    with col4:
        numeric_count = len(df.select_dtypes(include=[np.number]).columns)
        st.metric("Numeric Features", numeric_count, help="Count of numeric columns")

    ########## DATA PREVIEW #############
    tab_preview, tab_profile = st.tabs(["Data Preview", "Data Profile"])
    with tab_preview:
        st.dataframe(df.head(10), use_container_width=True)
    with tab_profile:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        grid1, grid2 = st.columns(2)
        with grid1:
            if num_cols:
                st.write("Numeric Columns Summary")
                st.dataframe(df[num_cols].describe(), use_container_width=True)
        with grid2:
            if cat_cols:
                st.write("Categorical Columns Summary")
                cat_summary = pd.DataFrame({
                    'Column': cat_cols,
                    'Unique Values': [df[col].nunique() for col in cat_cols],
                    'Missing Values': [df[col].isnull().sum() for col in cat_cols]
                })
                st.dataframe(cat_summary, use_container_width=True)

    ########## FEATURE ENGINEERING TABS ########
    st.markdown('<div class="section-header"><h3>Feature Engineering Tools</h3></div>', unsafe_allow_html=True)
    auto_tab, basic_tab, advanced_tab, quality_tab = st.tabs(["Auto-Engineering", "Basic Operations", "Advanced Features", "Data Quality"])

    with auto_tab:
        st.markdown("### Automated Feature Engineering")
        with st.form("auto_engineering_form"):
            auto_missing = st.checkbox("Handle Missing Values", value=True)
            auto_scaling = st.checkbox("Scale Numeric Features", value=True)
            auto_outliers = st.checkbox("Handle Outliers", value=True)
            auto_encoding = st.checkbox("Encode Categorical Variables", value=True)
            submitted = st.form_submit_button("Run Auto-Engineering", use_container_width=True, type="primary")
            if submitted:
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    if auto_missing:
                        status_text.text("Handling missing values...")
                        for col in df.columns:
                            if df[col].isnull().any():
                                if df[col].dtype in ['int64', 'float64']:
                                    df[col] = df[col].fillna(df[col].mean())
                                else:
                                    df[col] = df[col].fillna(df[col].mode()[0])
                        progress_bar.progress(25)
                    if auto_scaling:
                        status_text.text("Scaling numeric features...")
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            scaler = StandardScaler()
                            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                        progress_bar.progress(50)
                    if auto_outliers:
                        status_text.text("Handling outliers...")
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols:
                            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                            df[col] = df[col].mask(z_scores > 2.5, df[col].mean())
                        progress_bar.progress(75)
                    if auto_encoding:
                        status_text.text("Encoding categorical variables...")
                        cat_cols = df.select_dtypes(include=["object", "category"]).columns
                        for col in cat_cols:
                            if df[col].nunique() < 10:
                                df = pd.get_dummies(df, columns=[col], prefix=col)
                            else:
                                le = LabelEncoder()
                                df[f"{col}_encoded"] = le.fit_transform(df[col])
                        progress_bar.progress(100)
                    status_text.empty()
                    st.session_state.df_feature_eng = df.copy()
                    st.success("Auto-engineering completed successfully!")
                except Exception as e:
                    st.error(f"Auto-engineering failed: {str(e)}")

    ########## BASIC TAB ########
    with basic_tab:
        st.markdown("### Basic Operations")
        st.markdown("##### Select a basic feature engineering operation:")
        basic_operation = st.selectbox(
            "Basic Operations",
            [
                "Choose an operation...",
                "Handle Missing Values",
                "Convert Data Types",
                "Scale Features",
                "One-Hot Encoding",
                "Binning",
                "Remove Duplicates",
                "Add Missing Value Indicator Columns"
            ],
            key="basic_ops"
        )
        if basic_operation == "Handle Missing Values":
            missing_cols = [col for col in df.columns if df[col].isnull().any()]
            if not missing_cols:
                st.info("No columns with missing values.")
            else:
                col = st.selectbox("Select column with missing values", missing_cols)
                if col:
                    method = st.radio("Choose filling method", ["Average (mean)", "Middle value (median)", "Most frequent (mode)", "Custom value"])
                    if method == "Custom value":
                        custom = st.text_input("Enter custom value")
                    else:
                        custom = None
                    if st.button("Apply", use_container_width=True):
                        try:
                            if method == "Average (mean)":
                                df[col] = df[col].fillna(df[col].mean())
                            elif method == "Middle value (median)":
                                df[col] = df[col].fillna(df[col].median())
                            elif method == "Most frequent (mode)":
                                df[col] = df[col].fillna(df[col].mode()[0])
                            else:
                                df[col] = df[col].fillna(custom)
                            st.session_state.df_feature_eng = df.copy()
                            st.success("Missing values handled successfully")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        elif basic_operation == "Convert Data Types":
            col = st.selectbox("Select column to convert", df.columns)
            if col:
                new_type = st.radio("Convert to", ["Number", "Category", "Text"])
                if st.button("Convert", use_container_width=True):
                    try:
                        if new_type == "Number":
                            df[col] = pd.to_numeric(df[col])
                        elif new_type == "Category":
                            df[col] = df[col].astype('category')
                        else:
                            df[col] = df[col].astype(str)
                        st.session_state.df_feature_eng = df.copy()
                        st.success("Data type converted successfully")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        elif basic_operation == "Scale Features":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cols_to_scale = st.multiselect("Select columns to scale", numeric_cols)
            if cols_to_scale:
                scale_type = st.radio("Choose scaling method", ["Standard Scale (mean=0, std=1)", "Min-Max Scale (0 to 1)"])
                if st.button("Scale", use_container_width=True):
                    try:
                        if scale_type == "Standard Scale (mean=0, std=1)":
                            scaler = StandardScaler()
                        else:
                            scaler = MinMaxScaler()
                        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
                        st.session_state.df_feature_eng = df.copy()
                        st.success("Scaling completed successfully")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    ########## ADVANCED TAB #########
    with advanced_tab:
        st.markdown("### Advanced Features")
        st.markdown("##### Select an advanced feature engineering operation:")
        advanced_operation = st.selectbox(
            "Advanced Operations",
            [
                "Choose an operation...",
                "Target/Mean Encoding",
                "Frequency Encoding",
                "Binary Encoding",
                "Polynomial Features",
                "Group Aggregations",
                "Datetime Feature Extraction",
                "Text Feature Extraction",
                "Outlier Handling (Winsorization)",
                "Remove Constant/Quasi-Constant Features",
                "Remove Highly Correlated Features",
                "Feature Selection",
                "Dimensionality Reduction (PCA)",
                "Feature Interactions",
                "Conditional Features",
                "Quantile/Power/Robust Scaling"
            ],
            key="advanced_ops"
        )
        if advanced_operation == "Target/Mean Encoding":
            target_col = st.selectbox("Select Target Column", df.columns)
            categorical_cols = st.multiselect("Select Categorical Columns", df.select_dtypes(include=['object', 'category']).columns)
            if st.button("Apply Target Encoding", use_container_width=True):
                try:
                    for col in categorical_cols:
                        mean_dict = df.groupby(col)[target_col].mean().to_dict()
                        df[f"{col}_target_encoded"] = df[col].map(mean_dict)
                    st.session_state.df_feature_eng = df.copy()
                    st.success("Target encoding applied successfully!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        elif advanced_operation == "Frequency Encoding":
            categorical_cols = st.multiselect("Select Columns for Frequency Encoding", df.select_dtypes(include=['object', 'category']).columns)
            if st.button("Apply Frequency Encoding", use_container_width=True):
                try:
                    for col in categorical_cols:
                        freq_dict = df[col].value_counts(normalize=True).to_dict()
                        df[f"{col}_freq_encoded"] = df[col].map(freq_dict)
                    st.session_state.df_feature_eng = df.copy()
                    st.success("Frequency encoding applied successfully!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        elif advanced_operation == "Binary Encoding":
            categorical_cols = st.multiselect("Select Columns for Binary Encoding", df.select_dtypes(include=['object', 'category']).columns)
            if st.button("Apply Binary Encoding", use_container_width=True):
                try:
                    for col in categorical_cols:
                        df[f"{col}_binary"] = pd.factorize(df[col])[0]
                    st.session_state.df_feature_eng = df.copy()
                    st.success("Binary encoding applied successfully!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        elif advanced_operation == "Polynomial Features":
            numeric_cols = st.multiselect("Select Numeric Columns", df.select_dtypes(include=['int64', 'float64']).columns)
            degree = st.slider("Select Polynomial Degree", 2, 5, 2)
            if st.button("Generate Polynomial Features", use_container_width=True):
                try:
                    from sklearn.preprocessing import PolynomialFeatures
                    poly = PolynomialFeatures(degree=degree, include_bias=False)
                    poly_features = poly.fit_transform(df[numeric_cols])
                    feature_names = poly.get_feature_names_out(numeric_cols)
                    for i, name in enumerate(feature_names[len(numeric_cols):]):
                        df[name] = poly_features[:, len(numeric_cols) + i]
                    st.session_state.df_feature_eng = df.copy()
                    st.success("Polynomial features generated successfully!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        elif advanced_operation == "Group Aggregations":
            group_col = st.selectbox("Select Grouping Column", df.select_dtypes(include=['object', 'category']).columns)
            agg_cols = st.multiselect("Select Columns to Aggregate", df.select_dtypes(include=['int64', 'float64']).columns)
            agg_funcs = st.multiselect("Select Aggregation Functions", ['mean', 'sum', 'min', 'max', 'std'])
            if st.button("Create Aggregation Features", use_container_width=True):
                try:
                    for col in agg_cols:
                        for func in agg_funcs:
                            agg_dict = df.groupby(group_col)[col].agg(func).to_dict()
                            df[f"{group_col}_{col}_{func}"] = df[group_col].map(agg_dict)
                    st.session_state.df_feature_eng = df.copy()
                    st.success("Group aggregation features created successfully!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        elif advanced_operation == "Datetime Feature Extraction":
            date_cols = st.multiselect("Select DateTime Columns", df.select_dtypes(include=['datetime64']).columns)
            if st.button("Extract DateTime Features", use_container_width=True):
                try:
                    for col in date_cols:
                        df[f"{col}_year"] = df[col].dt.year
                        df[f"{col}_month"] = df[col].dt.month
                        df[f"{col}_day"] = df[col].dt.day
                        df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                        df[f"{col}_quarter"] = df[col].dt.quarter
                    st.session_state.df_feature_eng = df.copy()
                    st.success("DateTime features extracted successfully!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        elif advanced_operation == "Text Feature Extraction":
            text_cols = st.multiselect("Select Text Columns", df.select_dtypes(include=['object']).columns)
            if st.button("Extract Text Features", use_container_width=True):
                try:
                    for col in text_cols:
                        df[f"{col}_length"] = df[col].str.len()
                        df[f"{col}_word_count"] = df[col].str.split().str.len()
                        df[f"{col}_capital_ratio"] = df[col].str.count(r'[A-Z]') / df[col].str.len()
                    st.session_state.df_feature_eng = df.copy()
                    st.success("Text features extracted successfully!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        elif advanced_operation == "Outlier Handling (Winsorization)":
            numeric_cols = st.multiselect("Select Columns for Outlier Handling", df.select_dtypes(include=['int64', 'float64']).columns)
            percentile = st.slider("Select Winsorization Percentile", 1, 10, 5)
            if st.button("Handle Outliers", use_container_width=True):
                try:
                    from scipy import stats
                    for col in numeric_cols:
                        df[f"{col}_winsorized"] = stats.mstats.winsorize(df[col], limits=[percentile/100, percentile/100])
                    st.session_state.df_feature_eng = df.copy()
                    st.success("Outliers handled successfully!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        elif advanced_operation == "Remove Constant/Quasi-Constant Features":
            threshold = st.slider("Select Variance Threshold", 0.0, 1.0, 0.01)
            if st.button("Remove Low Variance Features", use_container_width=True):
                try:
                    from sklearn.feature_selection import VarianceThreshold
                    selector = VarianceThreshold(threshold=threshold)
                    numeric_data = df.select_dtypes(include=['int64', 'float64'])
                    selector.fit(numeric_data)
                    selected_features = numeric_data.columns[selector.get_support()].tolist()
                    df = df[selected_features + [col for col in df.columns if col not in numeric_data.columns]]
                    st.session_state.df_feature_eng = df.copy()
                    st.success("Low variance features removed successfully!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        elif advanced_operation == "Remove Highly Correlated Features":
            threshold = st.slider("Select Correlation Threshold", 0.5, 1.0, 0.8)
            if st.button("Remove Correlated Features", use_container_width=True):
                try:
                    numeric_data = df.select_dtypes(include=['int64', 'float64'])
                    corr_matrix = numeric_data.corr().abs()
                    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
                    df = df.drop(columns=to_drop)
                    st.session_state.df_feature_eng = df.copy()
                    st.success("Highly correlated features removed successfully!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        elif advanced_operation == "Feature Selection":
            target_col = st.selectbox("Select Target Column", df.columns)
            n_features = st.slider("Select Number of Features to Keep", 1, len(df.columns)-1, min(5, len(df.columns)-1))
            method = st.selectbox("Select Feature Selection Method", ["SelectKBest", "RFE", "SelectFromModel"])
            if st.button("Select Features", use_container_width=True):
                try:
                    from sklearn.feature_selection import SelectKBest, f_classif, RFE
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.feature_selection import SelectFromModel
                    
                    X = df.drop(columns=[target_col])
                    y = df[target_col]
                    
                    if method == "SelectKBest":
                        selector = SelectKBest(score_func=f_classif, k=n_features)
                    elif method == "RFE":
                        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
                        selector = RFE(estimator=estimator, n_features_to_select=n_features)
                    else:  # SelectFromModel
                        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
                        selector = SelectFromModel(estimator=estimator, max_features=n_features)
                        
                    selector.fit(X, y)
                    selected_features = X.columns[selector.get_support()].tolist()
                    df = df[selected_features + [target_col]]
                    st.session_state.df_feature_eng = df.copy()
                    st.success("Feature selection completed successfully!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        elif advanced_operation == "Dimensionality Reduction (PCA)":
            n_components = st.slider("Select Number of Components", 2, min(df.shape[1], 10), 2)
            if st.button("Apply PCA", use_container_width=True):
                try:
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.decomposition import PCA
                    
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(df[numeric_cols])
                    
                    pca = PCA(n_components=n_components)
                    pca_result = pca.fit_transform(scaled_data)
                    
                    for i in range(n_components):
                        df[f"PC{i+1}"] = pca_result[:, i]
                        
                    explained_variance = pca.explained_variance_ratio_
                    st.write(f"Explained variance ratios: {explained_variance}")
                    st.session_state.df_feature_eng = df.copy()
                    st.success("PCA transformation applied successfully!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        elif advanced_operation == "Feature Interactions":
            numeric_cols = st.multiselect("Select Numeric Columns for Interactions", df.select_dtypes(include=['int64', 'float64']).columns)
            if st.button("Generate Interactions", use_container_width=True):
                try:
                    for i in range(len(numeric_cols)):
                        for j in range(i+1, len(numeric_cols)):
                            col1, col2 = numeric_cols[i], numeric_cols[j]
                            df[f"{col1}_mul_{col2}"] = df[col1] * df[col2]
                            df[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-6)
                    st.session_state.df_feature_eng = df.copy()
                    st.success("Feature interactions generated successfully!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        elif advanced_operation == "Conditional Features":
            condition_col = st.selectbox("Select Condition Column", df.columns)
            target_col = st.selectbox("Select Target Column for Conditional Feature", df.columns)
            threshold = st.number_input("Enter Condition Threshold", value=0.0)
            if st.button("Create Conditional Feature", use_container_width=True):
                try:
                    df[f"{target_col}_when_{condition_col}_above_{threshold}"] = np.where(
                        df[condition_col] > threshold,
                        df[target_col],
                        0
                    )
                    st.session_state.df_feature_eng = df.copy()
                    st.success("Conditional feature created successfully!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        elif advanced_operation == "Quantile/Power/Robust Scaling":
            scaling_method = st.selectbox("Select Scaling Method", ["Quantile", "Power", "Robust"])
            numeric_cols = st.multiselect("Select Columns to Scale", df.select_dtypes(include=['int64', 'float64']).columns)
            if st.button("Apply Scaling", use_container_width=True):
                try:
                    if scaling_method == "Quantile":
                        from sklearn.preprocessing import QuantileTransformer
                        scaler = QuantileTransformer(output_distribution='normal')
                    elif scaling_method == "Power":
                        from sklearn.preprocessing import PowerTransformer
                        scaler = PowerTransformer(method='yeo-johnson')
                    else:  # Robust
                        from sklearn.preprocessing import RobustScaler
                        scaler = RobustScaler()
                        
                    scaled_data = scaler.fit_transform(df[numeric_cols])
                    for i, col in enumerate(numeric_cols):
                        df[f"{col}_{scaling_method.lower()}_scaled"] = scaled_data[:, i]
                    st.session_state.df_feature_eng = df.copy()
                    st.success(f"{scaling_method} scaling applied successfully!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    ########## QUALITY TAB ########
    with quality_tab:
        st.markdown("### Data Quality Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Missing Values Analysis")
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Values': df.isnull().sum(),
                'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
            }).sort_values('Missing %', ascending=False)
            st.dataframe(missing_df, use_container_width=True)
        with col2:
            st.write("Duplicate Rows")
            duplicate_count = df.duplicated().sum()
            st.metric("Duplicate Rows", duplicate_count)
            if duplicate_count > 0:
                if st.button("Remove Duplicates", use_container_width=True):
                    df = df.drop_duplicates()
                    st.session_state.df_feature_eng = df.copy()
                    st.success("Duplicates removed successfully")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            st.write("Correlation Analysis")
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, title="Feature Correlation Matrix", color_continuous_scale="RdBu", aspect="auto")
            st.plotly_chart(fig, use_container_width=True)

    ########## EXPORT ########
    st.markdown('<div class="section-header"><h3>Export Data</h3></div>', unsafe_allow_html=True)
    with st.expander("Final Data Preview"):
        st.dataframe(df.head(10), use_container_width=True)
        st.info(f"Final dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="engineered_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col2:
        import io
        try:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Engineered Data')
            excel_data = buffer.getvalue()
            st.download_button(
                label="Download as Excel",
                data=excel_data,
                file_name="engineered_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except ImportError:
            st.info("To export as Excel, please install xlsxwriter via pip install xlsxwriter.")
    with col3:
        if st.button("Start Over", use_container_width=True):
            del st.session_state.df_feature_eng
            st.rerun()

if __name__ == "_main_":
    app()