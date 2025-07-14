import streamlit as st
import pandas as pd 
import numpy as np 
from utils.shared import apply_common_settings

def app():
    st.title("Feature Engineering")
    # Persistent upload section
    if "df_feature_eng" in st.session_state:
        df = st.session_state.df_feature_eng.copy()
        st.success("Using previously uploaded data for Feature Engineering.")
    else:
        uploaded_file = st.file_uploader("Upload a CSV/Excel", type=["csv", "xlsx", "xls"], key="feature_eng_uploader")
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

    st.subheader("Data Preview")
    st.dataframe(df, use_container_width=True)

    st.subheader("Missing Values")
    missing_values = df.isnull().sum()
    st.write(missing_values[missing_values > 0])

    st.markdown("---")
    st.subheader("Feature Engineering Tools")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # 1. Auto Feature Creation
    with st.expander("Auto Feature Creation", expanded=False):
        poly_degree = st.slider("Polynomial Degree", 2, 4, 2)
        interaction = st.checkbox("Create Interaction Features")
        binning_col = st.selectbox("Column for Binning", numeric_cols, key="binning_col") if numeric_cols else None
        bin_count = st.slider("Number of Bins", 2, 10, 4)
        if st.button("Apply Auto Feature Creation"):
            from sklearn.preprocessing import PolynomialFeatures
            pf = PolynomialFeatures(degree=poly_degree, interaction_only=not interaction, include_bias=False)
            poly_features = pf.fit_transform(df[numeric_cols]) if numeric_cols else None
            if poly_features is not None:
                poly_df = pd.DataFrame(poly_features, columns=pf.get_feature_names_out(numeric_cols))
                df = pd.concat([df, poly_df], axis=1)
            if binning_col:
                df[f"{binning_col}_bin"] = pd.cut(df[binning_col], bins=bin_count, labels=[f"Bin_{i+1}" for i in range(bin_count)])
            st.session_state.df_feature_eng = df.copy()
            st.success("Auto feature creation applied!")

    # 2. Encoding Options
    with st.expander("Encoding Options", expanded=False):
        encoding_type = st.selectbox("Encoding Type", ["One-Hot", "Label"])
        encoding_col = st.selectbox("Column to Encode", cat_cols, key="encoding_col") if cat_cols else None
        if st.button("Apply Encoding") and encoding_col:
            if encoding_type == "One-Hot":
                df = pd.get_dummies(df, columns=[encoding_col])
            else:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df[encoding_col + "_label"] = le.fit_transform(df[encoding_col])
            st.session_state.df_feature_eng = df.copy()
            st.success(f"{encoding_type} encoding applied!")

    # 3. Scaling & Normalization
    with st.expander("Scaling & Normalization", expanded=False):
        scale_type = st.selectbox("Scaling Method", ["Standard", "Min-Max", "Robust", "Power"])
        scale_cols = st.multiselect("Columns to Scale", numeric_cols, key="scale_cols")
        if st.button("Apply Scaling") and scale_cols:
            if scale_type == "Standard":
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
            elif scale_type == "Min-Max":
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
            elif scale_type == "Robust":
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
            else:
                from sklearn.preprocessing import PowerTransformer
                scaler = PowerTransformer()
            df[scale_cols] = scaler.fit_transform(df[scale_cols])
            st.session_state.df_feature_eng = df.copy()
            st.success(f"{scale_type} scaling applied!")

    # 4. Outlier Detection & Handling
    with st.expander("Outlier Detection & Handling", expanded=False):
        outlier_col = st.selectbox("Column for Outlier Detection", numeric_cols, key="outlier_col") if numeric_cols else None
        outlier_method = st.selectbox("Method", ["IQR", "Z-Score"])
        outlier_action = st.selectbox("Action", ["Remove", "Cap", "Impute"])
        if st.button("Apply Outlier Handling") and outlier_col:
            col_data = df[outlier_col]
            if outlier_method == "IQR":
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                mask = (col_data < lower) | (col_data > upper)
            else:
                mean = col_data.mean()
                std = col_data.std()
                mask = (col_data < mean - 3*std) | (col_data > mean + 3*std)
            if outlier_action == "Remove":
                df = df[~mask]
            elif outlier_action == "Cap":
                df.loc[col_data < lower, outlier_col] = lower
                df.loc[col_data > upper, outlier_col] = upper
            else:
                df.loc[mask, outlier_col] = col_data.median()
            st.session_state.df_feature_eng = df.copy()
            st.success(f"Outlier handling ({outlier_action}) applied!")

    # 5. Feature Selection
    with st.expander("Feature Selection", expanded=False):
        selection_method = st.selectbox("Method", ["Correlation", "Variance", "Model-Based"])
        target_col = st.selectbox("Target Column", numeric_cols + cat_cols, key="target_col") if (numeric_cols or cat_cols) else None
        if st.button("Apply Feature Selection") and target_col:
            selected_features = []
            if selection_method == "Correlation":
                corr = df.corr(numeric_only=True)[target_col].abs().sort_values(ascending=False)
                selected_features = corr[corr > 0.1].index.tolist()
            elif selection_method == "Variance":
                from sklearn.feature_selection import VarianceThreshold
                selector = VarianceThreshold(threshold=0.1)
                selector.fit(df[numeric_cols])
                selected_features = [numeric_cols[i] for i in range(len(numeric_cols)) if selector.get_support()[i]]
            else:
                try:
                    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                    y = df[target_col]
                    X = df.drop(columns=[target_col])
                    if y.dtype.kind in 'biufc':
                        model = RandomForestRegressor()
                    else:
                        model = RandomForestClassifier()
                    model.fit(X.select_dtypes(include=[np.number]), y)
                    importances = model.feature_importances_
                    selected_features = X.select_dtypes(include=[np.number]).columns[np.argsort(importances)[::-1][:5]].tolist()
                except Exception as e:
                    st.error(f"Model-based selection failed: {e}")
            st.write(f"Selected Features: {selected_features}")

    # 6. Missing Value Imputation
    with st.expander("Missing Value Imputation", expanded=False):
        impute_method = st.selectbox("Imputation Method", ["Mean", "Median", "Mode", "Custom Value"])
        impute_col = st.selectbox("Column to Impute", df.columns.tolist(), key="impute_col")
        custom_value = st.text_input("Custom Value (if selected)") if impute_method == "Custom Value" else None
        if st.button("Apply Imputation") and impute_col:
            if impute_method == "Mean":
                df[impute_col] = df[impute_col].fillna(df[impute_col].mean())
            elif impute_method == "Median":
                df[impute_col] = df[impute_col].fillna(df[impute_col].median())
            elif impute_method == "Mode":
                df[impute_col] = df[impute_col].fillna(df[impute_col].mode()[0])
            else:
                df[impute_col] = df[impute_col].fillna(custom_value)
            st.session_state.df_feature_eng = df.copy()
            st.success(f"Imputation applied to {impute_col}!")

    # 7. Export Options
    st.markdown("---")
    st.subheader("Export Engineered Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download as CSV", csv, "engineered_data.csv", use_container_width=True)



