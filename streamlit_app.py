import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
from scipy import stats
from io import BytesIO
import base64
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
from statsmodels.tsa.seasonal import seasonal_decompose

# Function to create dummy air pollution data
def create_dummy_air_pollution_data():
    np.random.seed(42)
    dates = [datetime.today() - timedelta(days=i) for i in range(100)]
    data = {
        'Date': dates,
        'PM2.5': np.random.normal(50, 15, 100),
        'PM10': np.random.normal(100, 25, 100),
        'NO2': np.random.normal(40, 10, 100),
        'CO': np.random.normal(1.0, 0.2, 100),
        'Temperature': np.random.normal(20, 5, 100),
        'Humidity': np.random.normal(50, 10, 100)
    }
    df = pd.DataFrame(data)
    return df

# Function to convert DataFrame to CSV and download
def download_csv(df, filename="file.csv"):
    csv_data = df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv_data).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

# Function to download figures as PNG using Plotly
def download_figure(fig, filename="figure.png"):
    img_bytes = fig.to_image(format="png")
    b64 = base64.b64encode(img_bytes).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">Download Figure</a>'
    return href

# Function to display summary statistics
def display_summary_statistics(df):
    st.subheader("Summary Statistics")
    summary = df.describe()
    st.write(summary)
    st.markdown(download_csv(summary, "summary_statistics.csv"), unsafe_allow_html=True)

# Function to perform data quality checks
def data_quality_checks(df):
    st.subheader("Data Quality Checks")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    st.write("Missing Values:")
    st.write(missing_values)
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    st.write(f"Number of duplicate rows: {duplicates}")
    
    # Check data types
    st.write("Data Types:")
    st.write(df.dtypes)

# Function for interactive data filtering
def interactive_data_filter(df):
    st.subheader("Data Filtering")
    
    # Date range filter
    date_columns = df.select_dtypes(include=['datetime64']).columns
    if len(date_columns) > 0:
        date_col = st.selectbox("Select date column", date_columns, key="filter_date_col")
        start_date = st.date_input("Start date", df[date_col].min(), key="filter_start_date")
        end_date = st.date_input("End date", df[date_col].max(), key="filter_end_date")
        df = df[(df[date_col] >= pd.Timestamp(start_date)) & (df[date_col] <= pd.Timestamp(end_date))]
    else:
        st.warning("No date columns found in the dataset.")
    
    # Numeric column filters
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for i, col in enumerate(numeric_cols):
        min_val, max_val = st.slider(f"Filter {col}", 
                                     float(df[col].min()), 
                                     float(df[col].max()), 
                                     (float(df[col].min()), float(df[col].max())),
                                     key=f"filter_slider_{i}")
        df = df[(df[col] >= min_val) & (df[col] <= max_val)]
    
    st.write(f"Filtered data shape: {df.shape}")
    return df

# Function for enhanced machine learning section
def enhanced_ml_section(df, features, target, model_type):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if model_type == 'Linear Regression':
        model = LinearRegression()
    elif model_type == 'Ridge Regression':
        model = Ridge()
    elif model_type == 'Lasso Regression':
        model = Lasso()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model performance metrics
    st.subheader("Model Performance Metrics")
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse:.4f}")
    st.write(f"Mean Absolute Error: {mae:.4f}")
    st.write(f"R² Score: {r2:.4f}")

    # Feature importance
    st.subheader("Feature Importance")
    if model_type in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
        importance = pd.DataFrame({'feature': features, 'importance': abs(model.coef_)})
        importance = importance.sort_values('importance', ascending=False)
        fig = px.bar(importance, x='feature', y='importance', title="Feature Importance")
        st.plotly_chart(fig)

    # SHAP values for model interpretability
    st.subheader("SHAP Values for Model Interpretability")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(fig)

# Function for trends and seasonality analysis
def trends_seasonality_analysis(df):
    st.subheader("Trends and Seasonality Analysis")
    
    date_columns = df.select_dtypes(include=['datetime64']).columns
    if len(date_columns) > 0:
        date_col = st.selectbox("Select date column", date_columns, key="trends_date_col")
        variable = st.selectbox("Select variable for analysis", df.select_dtypes(include=['float64', 'int64']).columns, key="trends_variable")
        
        df_sorted = df.sort_values(by=date_col)
        df_sorted.set_index(date_col, inplace=True)
        
        # Perform seasonal decomposition
        result = seasonal_decompose(df_sorted[variable], model='additive', period=30)
        
        # Plot the decomposition
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
        result.observed.plot(ax=ax1)
        ax1.set_title('Observed')
        result.trend.plot(ax=ax2)
        ax2.set_title('Trend')
        result.seasonal.plot(ax=ax3)
        ax3.set_title('Seasonal')
        result.resid.plot(ax=ax4)
        ax4.set_title('Residual')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No date columns found in the dataset. Trends and seasonality analysis cannot be performed.")


# Title and Introduction
st.title('Interactive EDA for Air Pollution Data 🌍💨')
st.write("""
Welcome to the **Interactive EDA Application** for Air Pollution Data. This tool helps you:
- Upload your air pollution dataset (CSV)
- Explore, clean, and preprocess your data
- Visualize relationships, detect outliers, and generate correlation plots
- Train basic machine learning models on air pollution data
- Analyze trends and seasonality in your data
Use the sidebar to navigate through different features.
""")

# Sidebar - Action Selection
st.sidebar.title('📋 Actions')

# Sidebar - Step 1: Upload Data or Use Sample Dataset
st.sidebar.header("📂 Step 1: Upload Data")
use_sample = st.sidebar.checkbox("Use Sample Air Pollution Data")
data = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])

if use_sample:
    df = create_dummy_air_pollution_data()
    st.success("Sample air pollution dataset loaded.")
else:
    if data is not None:
        df = pd.read_csv(data)
        st.success("File uploaded successfully.")
    else:
        st.warning("Please upload a CSV file or use the sample data to proceed.")
        df = None

# Ensure Data is Available for Analysis
if df is not None:
    st.write("### Preview of the Dataset")
    st.write(df.head())

    # Interactive Data Filtering
    with st.sidebar:
        df_filtered = interactive_data_filter(df)

    # Use Tabs for Easy Navigation between Different Sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Data Overview", "Data Cleaning", "Visualizations", "Correlation Plot", "Outlier Detection", "Machine Learning"])

    # Tab 1: Data Overview
    with tab1:
        st.header("📊 Data Overview")
        display_summary_statistics(df_filtered)
        data_quality_checks(df_filtered)

    # Tab 2: Data Cleaning
    with tab2:
        st.header("🧹 Data Cleaning")

        # Drop Columns
        st.subheader("Drop Columns")
        columns_to_drop = st.multiselect("Select columns to drop", df_filtered.columns)
        if columns_to_drop:
            df_filtered.drop(columns=columns_to_drop, axis=1, inplace=True)
            st.write("### Data after Dropping Columns")
            st.write(df_filtered.head())

        # Rename Columns
        st.subheader("Rename Columns")
        columns_to_rename = st.multiselect("Select columns to rename", df_filtered.columns)
        new_names = st.text_input("Enter new names (comma-separated) for the selected columns")
        if columns_to_rename and new_names:
            new_names_list = new_names.split(',')
            df_filtered.rename(columns=dict(zip(columns_to_rename, new_names_list)), inplace=True)
            st.write("### Data after Renaming Columns")
            st.write(df_filtered.head())

        # Change Data Types
        st.subheader("Change Data Types")
        columns_to_change_type = st.multiselect("Select columns to change data type", df_filtered.columns)
        new_type = st.selectbox("Select the new data type", ('int64', 'float64', 'object'))
        if columns_to_change_type:
            df_filtered[columns_to_change_type] = df_filtered[columns_to_change_type].astype(new_type)
            st.write("### Data after Changing Data Type")
            st.write(df_filtered.head())

        # Handle Missing Values
        st.subheader("Handle Missing Values")
        columns_to_impute = st.multiselect("Select columns to impute missing values", df_filtered.columns)
        impute_method = st.selectbox('Select the imputation method', ['Linear Interpolation', 'Mean Imputation', 'Median Imputation'])
        if columns_to_impute:
            if impute_method == 'Linear Interpolation':
                df_filtered[columns_to_impute] = df_filtered[columns_to_impute].interpolate(method='linear')
            elif impute_method == 'Mean Imputation':
                imputer = SimpleImputer(strategy='mean')
                df_filtered[columns_to_impute] = imputer.fit_transform(df_filtered[columns_to_impute])
            elif impute_method == 'Median Imputation':
                imputer = SimpleImputer(strategy='median')
                df_filtered[columns_to_impute] = imputer.fit_transform(df_filtered[columns_to_impute])
            st.write("### Data after Imputation")
            st.write(df_filtered.head())

        st.markdown(download_csv(df_filtered, "cleaned_data.csv"), unsafe_allow_html=True)

    # Tab 3: Data Visualizations
    with tab3:
        st.header("📊 Data Visualizations")

        # Time-Series Plots for Air Pollution Data
        st.subheader("Time-Series Visualization")
        time_series_var = st.selectbox("Select variable for time-series plot", df_filtered.select_dtypes(include=['float64', 'int64']).columns)
        time_chart = px.line(df_filtered, x='Date', y=time_series_var, title=f"Time-Series of {time_series_var}")
        st.plotly_chart(time_chart)

        # Download Plot
        st.markdown(download_figure(time_chart, f"time_series_{time_series_var}.png"), unsafe_allow_html=True)

        # Other Visualization Options
        st.subheader("Generate Other Interactive Plots")
        x_axis = st.selectbox("Select X axis", df_filtered.columns)
        y_axis = st.selectbox("Select Y axis", df_filtered.columns)
        plot_type = st.selectbox("Select Plot Type", ['Scatter Plot', 'Line Chart', 'Bar Chart', 'Histogram'])
        
        if st.button("Generate Plot"):
            st.write(f"### {plot_type} for {x_axis} vs {y_axis}")
            if plot_type == 'Scatter Plot':
                fig = px.scatter(df_filtered, x=x_axis, y=y_axis)
            elif plot_type == 'Line Chart':
                fig = px.line(df_filtered, x=x_axis, y=y_axis)
            elif plot_type == 'Bar Chart':
                fig = px.bar(df_filtered, x=x_axis, y=y_axis)
            elif plot_type == 'Histogram':
                fig = px.histogram(df_filtered, x=x_axis, y=y_axis)
            st.plotly_chart(fig)

            # Download Plot
            st.markdown(download_figure(fig, f"{plot_type.lower()}_{x_axis}_vs_{y_axis}.png"), unsafe_allow_html=True)

    # Tab 4: Correlation Plot
    with tab4:
        st.header("📈 Correlation Plot")

        st.subheader("Generate Correlation Heatmap")
        corr = df_filtered.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Download correlation matrix
        st.markdown(download_csv(corr, "correlation_matrix.csv"), unsafe_allow_html=True)

    # Tab 5: Outlier Detection
    
    with tab5:
        st.header("🚨 Outlier Detection")

    # Info about Outlier Detection Methods
    st.info("""
    **Outlier Detection Methods**:
    - **Isolation Forest**: Detects outliers by isolating anomalies through random partitioning of data points.
    - **Z-score**: Measures how many standard deviations a data point is from the mean; values beyond a threshold (e.g., 3) are considered outliers.
    """)

    # Detect Outliers in Selected Columns
    columns_to_find_outliers = st.multiselect("Select columns for outlier detection", df_filtered.columns, key="outlier_columns")
    outlier_method = st.selectbox("Select Outlier Detection Method", ['Isolation Forest', 'Z-score'], key="outlier_method")
    
    if st.button("Find Outliers"):
        if len(columns_to_find_outliers) < 2:
            st.warning("Please select at least two columns for outlier detection.")
        else:
            if outlier_method == 'Isolation Forest':
                iso_forest = IsolationForest(contamination='auto')
                outliers = iso_forest.fit_predict(df_filtered[columns_to_find_outliers])
                df_filtered['Outlier'] = outliers
                st.write("### Detected Outliers")
                st.write(df_filtered[df_filtered['Outlier'] == -1])  # Display outliers

                # Visualize Outliers
                fig = px.scatter(df_filtered, x=columns_to_find_outliers[0], y=columns_to_find_outliers[1], color=df_filtered['Outlier'].astype(str))
                st.plotly_chart(fig)

                st.markdown(download_csv(df_filtered, "outliers_data.csv"), unsafe_allow_html=True)

            elif outlier_method == 'Z-score':
                z_scores = np.abs(stats.zscore(df_filtered[columns_to_find_outliers]))
                outliers = (z_scores > 3).any(axis=1)
                df_filtered['Outlier'] = outliers
                st.write("### Detected Outliers")
                st.write(df_filtered[outliers])

                # Visualize Outliers
                fig = px.scatter(df_filtered, x=columns_to_find_outliers[0], y=columns_to_find_outliers[1], color=df_filtered['Outlier'].astype(str))
                st.plotly_chart(fig)

                st.markdown(download_csv(df_filtered, "outliers_data.csv"), unsafe_allow_html=True)

    # Tab 6: Machine Learning
    with tab6:
        st.header("📈 Machine Learning")

        st.info("""
        **Machine Learning Models**:
        - **Linear Regression**: A linear approach to modeling the relationship between a dependent variable and one or more independent variables.
        - **Ridge Regression**: A variation of linear regression that introduces a regularization term to prevent overfitting by penalizing large coefficients.
        - **Lasso Regression**: Another regularization method that can shrink coefficients to zero, effectively performing feature selection.
        """)

        # Select Features and Target for Model Training
        features = st.multiselect("Select Features for Model", df_filtered.columns)
        target = st.selectbox("Select Target Variable", df_filtered.columns)
        model_type = st.selectbox("Select Model Type", ['Linear Regression', 'Ridge Regression', 'Lasso Regression'])
        
        if st.button("Train Model") and features and target:
            enhanced_ml_section(df_filtered, features, target, model_type)

    # Trends and Seasonality Analysis
    with st.expander("Trends and Seasonality Analysis"):
        trends_seasonality_analysis(df_filtered)

    # Save Cleaned Data
    st.sidebar.header("💾 Save Data")
    if st.sidebar.button("Save Data as CSV"):
        st.markdown(download_csv(df_filtered, "cleaned_air_pollution_data.csv"), unsafe_allow_html=True)
        st.success("Data saved as 'cleaned_air_pollution_data.csv'")

else:
    st.info("Please upload a dataset or select the sample dataset to begin analysis.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
