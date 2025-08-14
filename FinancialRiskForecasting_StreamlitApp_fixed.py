
# Auto-generated Streamlit app from your Jupyter Notebook

import sys, subprocess, importlib
from pathlib import Path

# Install missing packages
def ensure_packages(packages):
    for pkg, import_name in packages:
        try:
            importlib.import_module(import_name)
        except Exception:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            except Exception:
                pass

REQUIRED = [
    ("streamlit", "streamlit"),
    ("matplotlib", "matplotlib"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("seaborn", "seaborn"),
    ("plotly", "plotly"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("statsmodels", "statsmodels"),
    ("optuna", "optuna"),
    ("tensorflow", "tensorflow"),
]

ensure_packages(REQUIRED)

import streamlit as st

# Use a non-interactive backend for Matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Financial Risk Forecasting – EDA", layout="wide")
st.title("Financial Risk Forecasting – EDA & Preprocessing")

# Patch plt.show() for Streamlit
_orig_plt_show = getattr(plt, "show", None)
def _st_show(*args, **kwargs):
    st.pyplot(plt.gcf())
plt.show = _st_show

# Patch Plotly show for Streamlit
try:
    from plotly.graph_objs import Figure as _PlotlyFigure
    def _figure_show(self, *args, **kwargs):
        st.plotly_chart(self, use_container_width=True)
    _PlotlyFigure.show = _figure_show
except Exception:
    pass


#!/usr/bin/env python
# coding: utf-8

# In[ ]:




# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
from scipy import stats
from scipy.stats import jarque_bera, shapiro
import os

warnings.filterwarnings('ignore')


# In[ ]:


# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

st.write("=== FINANCIAL RISK FORECASTING - EDA ===")
st.write("Project: Integrating Exogenous News Sentiment and Macroeconomic Indicators")
st.write("="*70)


# ## EDA

# In[ ]:


# Financial Risk Forecasting - Comprehensive EDA
# Run this in Google Colab

# Install required packages
# !pip install plotly seaborn matplotlib pandas numpy scipy scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
from scipy import stats
from scipy.stats import jarque_bera, shapiro
import os

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

st.write("=== FINANCIAL RISK FORECASTING - EDA ===")
st.write("Project: Integrating Exogenous News Sentiment and Macroeconomic Indicators")
st.write("="*70)

# =============================================================================
# 1. DATA LOADING AND INITIAL INSPECTION
# =============================================================================

# 📁 FILE PATHS - MODIFY THESE PATHS AS NEEDED
FILE_PATHS = {
    'aapl': 'AAPL_timeseries.csv',  # Paste your AAPL_timeseries.csv path here
    'googl': 'GOOGL_timeseries.csv',  # Paste your GOOGL_timeseries.csv path here
    'msft': 'MSFT_timeseries.csv',  # Paste your MSFT_timeseries.csv path here
    'financial_cross': 'financial_risk_crosssectional.csv',  # Paste your financial_risk_crosssectional.csv path here
    'financial_ts': 'financial_risk_timeseries (1).csv',  # Paste your financial_risk_timeseries.csv path here
    'macro': 'macro_indicators.csv',  # Paste your macro_indicators.csv path here
    'sentiment': 'sentiment_timeseries.csv'  # Paste your sentiment_timeseries.csv path here
}

def load_datasets():
    """Load all datasets and perform initial inspection"""
    datasets = {}

    st.write("📊 LOADING DATASETS...")
    st.write("-" * 50)

    for key, filepath in FILE_PATHS.items():
        if filepath == '':
            st.write(f"⚠️  {key}: No file path provided - skipping")
            continue

        try:
            df = pd.read_csv(filepath)
            datasets[key] = df
            filename = filepath.split('/')[-1]  # Extract filename from path
            st.write(f"✅ {filename}: {df.shape[0]} rows × {df.shape[1]} columns")
        except FileNotFoundError:
            st.write(f"❌ {key}: File not found at path: {filepath}")
        except Exception as e:
            st.write(f"❌ {key}: Error - {str(e)}")

    return datasets

def inspect_dataset_structure(datasets):
    """Inspect structure of each dataset"""
    st.write("\n" + "="*70)
    st.write("📋 DATASET STRUCTURE INSPECTION")
    st.write("="*70)

    for name, df in datasets.items():
        st.write(f"\n🔍 {name.upper()} DATASET:")
        st.write(f"Shape: {df.shape}")
        st.write(f"Columns: {list(df.columns)}")
        st.write(f"Data types:\n{df.dtypes}")
        st.write(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # Check for date columns
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            st.write(f"Date columns found: {date_cols}")
            for date_col in date_cols:
                st.write(f"  {date_col} range: {df[date_col].min()} to {df[date_col].max()}")

        st.write("-" * 40)

# =============================================================================
# 2. DATA QUALITY ASSESSMENT
# =============================================================================

def assess_data_quality(datasets):
    """Comprehensive data quality assessment"""
    st.write("\n" + "="*70)
    st.write("🔍 DATA QUALITY ASSESSMENT")
    st.write("="*70)

    quality_report = {}

    for name, df in datasets.items():
        st.write(f"\n📊 {name.upper()} QUALITY REPORT:")

        # Missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100

        quality_metrics = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': missing.sum(),
            'missing_percentage': missing_pct.sum() / len(df.columns),
            'duplicates': df.duplicated().sum(),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns)
        }

        quality_report[name] = quality_metrics

        st.write(f"  Total rows: {quality_metrics['total_rows']:,}")
        st.write(f"  Total columns: {quality_metrics['total_columns']}")
        st.write(f"  Missing values: {quality_metrics['missing_values']:,}")
        st.write(f"  Missing percentage: {quality_metrics['missing_percentage']:.2f}%")
        st.write(f"  Duplicate rows: {quality_metrics['duplicates']:,}")
        st.write(f"  Numeric columns: {quality_metrics['numeric_columns']}")
        st.write(f"  Categorical columns: {quality_metrics['categorical_columns']}")

        # Show columns with missing values
        if missing.sum() > 0:
            st.write("  Columns with missing values:")
            for col, count in missing[missing > 0].items():
                st.write(f"    {col}: {count} ({missing_pct[col]:.2f}%)")

    return quality_report

def visualize_missing_data(datasets):
    """Visualize missing data patterns"""
    st.write("\n" + "="*70)
    st.write("📊 MISSING DATA VISUALIZATION")
    st.write("="*70)

    fig, axes = plt.subplots(len(datasets), 1, figsize=(15, 4*len(datasets)))
    if len(datasets) == 1:
        axes = [axes]

    for idx, (name, df) in enumerate(datasets.items()):
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            missing_data[missing_data > 0].plot(kind='bar', ax=axes[idx])
            axes[idx].set_title(f'{name.upper()} - Missing Values by Column')
            axes[idx].set_ylabel('Missing Count')
            axes[idx].tick_params(axis='x', rotation=45)
        else:
            axes[idx].text(0.5, 0.5, f'{name.upper()}\nNo Missing Values',
                          ha='center', va='center', transform=axes[idx].transAxes, fontsize=14)
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])

    plt.tight_layout()
    st.pyplot(plt.gcf())

# =============================================================================
# 3. DESCRIPTIVE STATISTICS
# =============================================================================

def generate_descriptive_stats(datasets):
    """Generate comprehensive descriptive statistics"""
    st.write("\n" + "="*70)
    st.write("📈 DESCRIPTIVE STATISTICS")
    st.write("="*70)

    for name, df in datasets.items():
        st.write(f"\n📊 {name.upper()} DESCRIPTIVE STATISTICS:")

        # Numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) > 0:
            desc_stats = numeric_df.describe()
            st.write(desc_stats.round(4))

            # Additional statistics
            st.write(f"\nAdditional Statistics:")
            st.write(f"Skewness:\n{numeric_df.skew().round(4)}")
            st.write(f"\nKurtosis:\n{numeric_df.kurtosis().round(4)}")
        else:
            st.write("No numeric columns found for statistical analysis")

        st.write("-" * 50)

# =============================================================================
# 4. TIME SERIES ANALYSIS
# =============================================================================

def analyze_time_series_patterns(datasets):
    """Analyze time series patterns in the data"""
    st.write("\n" + "="*70)
    st.write("📅 TIME SERIES PATTERN ANALYSIS")
    st.write("="*70)

    # Focus on time series datasets
    ts_datasets = ['aapl', 'googl', 'msft', 'financial_ts', 'macro', 'sentiment']

    for name in ts_datasets:
        if name in datasets:
            df = datasets[name]
            st.write(f"\n📊 {name.upper()} TIME SERIES ANALYSIS:")

            # Try to identify date column
            date_cols = [col for col in df.columns if any(x in col.lower() for x in ['date', 'time', 'period'])]

            if date_cols:
                date_col = date_cols[0]
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                    st.write(f"  Date range: {df[date_col].min()} to {df[date_col].max()}")
                    st.write(f"  Time span: {(df[date_col].max() - df[date_col].min()).days} days")
                    st.write(f"  Frequency: {pd.infer_freq(df[date_col].sort_values())}")
                except:
                    st.write(f"  Could not parse date column: {date_col}")
            else:
                st.write("  No clear date column identified")

def plot_time_series_overview(datasets):
    """Create time series plots for key variables"""
    st.write("\n" + "="*70)
    st.write("📊 TIME SERIES VISUALIZATION")
    st.write("="*70)

    # Individual stock plots
    stock_datasets = ['aapl', 'googl', 'msft']

    fig = make_subplots(
        rows=len(stock_datasets), cols=1,
        subplot_titles=[f'{name.upper()} Financial Metrics' for name in stock_datasets],
        vertical_spacing=0.08
    )

    colors = ['blue', 'green', 'red']

    for idx, name in enumerate(stock_datasets):
        if name in datasets:
            df = datasets[name]
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            # Plot first few numeric columns
            for i, col in enumerate(numeric_cols[:3]):  # Limit to first 3 columns
                fig.add_trace(
                    go.Scatter(
                        y=df[col],
                        name=f'{name.upper()}_{col}',
                        line=dict(color=colors[i], width=1.5)
                    ),
                    row=idx+1, col=1
                )

    fig.update_layout(height=300*len(stock_datasets), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 5. CORRELATION ANALYSIS
# =============================================================================

def correlation_analysis(datasets):
    """Perform correlation analysis across datasets"""
    st.write("\n" + "="*70)
    st.write("🔗 CORRELATION ANALYSIS")
    st.write("="*70)

    for name, df in datasets.items():
        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) > 1:
            st.write(f"\n📊 {name.upper()} CORRELATION MATRIX:")

            # Calculate correlation matrix
            corr_matrix = numeric_df.corr()

            # Plot heatmap
            plt.figure(figsize=(12, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f')
            plt.title(f'{name.upper()} - Correlation Matrix')
            plt.tight_layout()
            st.pyplot(plt.gcf())

            # Find highly correlated pairs
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr.append((
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_matrix.iloc[i, j]
                        ))

            if high_corr:
                st.write(f"  High correlations (|r| > 0.7):")
                for col1, col2, corr_val in high_corr:
                    st.write(f"    {col1} ↔ {col2}: {corr_val:.3f}")

# =============================================================================
# 6. DISTRIBUTION ANALYSIS
# =============================================================================

def analyze_distributions(datasets):
    """Analyze distributions of key variables"""
    st.write("\n" + "="*70)
    st.write("📊 DISTRIBUTION ANALYSIS")
    st.write("="*70)

    for name, df in datasets.items():
        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) > 0:
            st.write(f"\n📊 {name.upper()} DISTRIBUTION ANALYSIS:")

            # Plot distributions for first 4 numeric columns
            cols_to_plot = numeric_df.columns[:4]

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()

            for idx, col in enumerate(cols_to_plot):
                if idx < 4:  # Ensure we don't exceed subplot limit
                    # Histogram with KDE
                    axes[idx].hist(numeric_df[col].dropna(), bins=30, alpha=0.7, density=True)

                    # Add KDE line
                    try:
                        numeric_df[col].dropna().plot.kde(ax=axes[idx], color='red', linewidth=2)
                    except:
                        pass

                    axes[idx].set_title(f'{col} Distribution')
                    axes[idx].set_xlabel(col)
                    axes[idx].set_ylabel('Density')

                    # Normality test
                    try:
                        _, p_value = shapiro(numeric_df[col].dropna().sample(min(5000, len(numeric_df[col].dropna()))))
                        normal_text = "Normal" if p_value > 0.05 else "Non-normal"
                        axes[idx].text(0.05, 0.95, f'Shapiro-Wilk: {normal_text}\np-value: {p_value:.4f}',
                                     transform=axes[idx].transAxes, verticalalignment='top',
                                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                    except:
                        pass

            # Hide empty subplots
            for idx in range(len(cols_to_plot), 4):
                axes[idx].set_visible(False)

            plt.suptitle(f'{name.upper()} - Variable Distributions')
            plt.tight_layout()
            st.pyplot(plt.gcf())

# =============================================================================
# 7. OUTLIER DETECTION
# =============================================================================

def detect_outliers(datasets):
    """Detect outliers using multiple methods"""
    st.write("\n" + "="*70)
    st.write("🎯 OUTLIER DETECTION")
    st.write("="*70)

    outlier_summary = {}

    for name, df in datasets.items():
        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) > 0:
            st.write(f"\n📊 {name.upper()} OUTLIER ANALYSIS:")

            outliers_per_method = {}

            for col in numeric_df.columns:
                data = numeric_df[col].dropna()

                # IQR Method
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                iqr_outliers = len(data[(data < lower_bound) | (data > upper_bound)])

                # Z-score method
                z_scores = np.abs(stats.zscore(data))
                z_outliers = len(data[z_scores > 3])

                outliers_per_method[col] = {
                    'iqr_outliers': iqr_outliers,
                    'z_score_outliers': z_outliers,
                    'total_observations': len(data)
                }

                st.write(f"  {col}:")
                st.write(f"    IQR outliers: {iqr_outliers} ({iqr_outliers/len(data)*100:.2f}%)")
                st.write(f"    Z-score outliers: {z_outliers} ({z_outliers/len(data)*100:.2f}%)")

            outlier_summary[name] = outliers_per_method

    return outlier_summary

# =============================================================================
# 8. MAIN EDA EXECUTION
# =============================================================================

def run_comprehensive_eda():
    """Run complete EDA pipeline"""
    st.write("🚀 STARTING COMPREHENSIVE EDA...")

    # Load datasets
    datasets = load_datasets()

    if not datasets:
        st.write("❌ No datasets loaded successfully!")
        return

    # Run all EDA components
    inspect_dataset_structure(datasets)
    quality_report = assess_data_quality(datasets)
    visualize_missing_data(datasets)
    generate_descriptive_stats(datasets)
    analyze_time_series_patterns(datasets)
    plot_time_series_overview(datasets)
    correlation_analysis(datasets)
    analyze_distributions(datasets)
    outlier_summary = detect_outliers(datasets)

    st.write("\n" + "="*70)
    st.write("✅ EDA COMPLETED SUCCESSFULLY!")
    st.write("="*70)

    return datasets, quality_report, outlier_summary

# Run the EDA
if True:
    datasets, quality_report, outlier_summary = run_comprehensive_eda()

    # Save EDA summary
    st.write("\n📝 Saving EDA summary...")
    eda_summary = {
        'datasets_info': {name: {'shape': df.shape, 'columns': list(df.columns)}
                         for name, df in datasets.items()},
        'quality_report': quality_report,
        'outlier_summary': outlier_summary
    }

    # You can save this to a file if needed
    st.write("EDA Summary prepared for next steps!")


# ##Data Preprocessing & Feature Engineering

# In[ ]:


# FIXED Data Preprocessing and Feature Engineering for Financial Risk Forecasting
# For ARIMAX and LSTM Models - WITH STATIONARITY CONVERSION

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# For stationarity testing and conversion
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose

st.write("=== FIXED DATA PREPROCESSING & FEATURE ENGINEERING ===")
st.write("="*60)

# =============================================================================
# 1. STATIONARITY TESTING AND CONVERSION
# =============================================================================

def test_stationarity(timeseries, column_name):
    """
    Test if a time series is stationary using ADF and KPSS tests
    """
    st.write(f"\n📊 Testing stationarity for: {column_name}")

    # Remove NaN values
    ts_clean = timeseries.dropna()

    if len(ts_clean) < 10:
        st.write(f"❌ Not enough data points for {column_name}")
        return False

    # Augmented Dickey-Fuller test
    try:
        adf_result = adfuller(ts_clean, autolag='AIC')
        adf_pvalue = adf_result[1]

        # KPSS test
        kpss_result = kpss(ts_clean, regression='c')
        kpss_pvalue = kpss_result[1]

        st.write(f"   ADF p-value: {adf_pvalue:.4f} (H0: Non-stationary)")
        st.write(f"   KPSS p-value: {kpss_pvalue:.4f} (H0: Stationary)")

        # Interpretation
        is_stationary = (adf_pvalue < 0.05) and (kpss_pvalue > 0.05)

        if is_stationary:
            st.write(f"   ✅ {column_name} is STATIONARY")
        else:
            st.write(f"   ❌ {column_name} is NON-STATIONARY")

        return is_stationary

    except Exception as e:
        st.write(f"   ⚠️ Error testing {column_name}: {str(e)}")
        return False

def make_stationary(df, column, method='difference'):
    """
    Convert non-stationary series to stationary
    """
    st.write(f"\n🔄 Converting {column} to stationary using {method}...")

    original_data = df[column].copy()

    if method == 'difference':
        # First difference
        stationary_data = original_data.diff().dropna()

        # Test if first difference is enough
        if len(stationary_data) > 10:
            is_stat = test_stationarity(stationary_data, f"{column}_diff1")

            if not is_stat and len(stationary_data) > 20:
                # Try second difference
                st.write(f"   Trying second difference for {column}...")
                stationary_data = stationary_data.diff().dropna()
                is_stat = test_stationarity(stationary_data, f"{column}_diff2")

    elif method == 'log_difference':
        # Log transform then difference
        if (original_data > 0).all():
            log_data = np.log(original_data)
            stationary_data = log_data.diff().dropna()
        else:
            # If negative values, use regular difference
            stationary_data = original_data.diff().dropna()

    else:  # percentage change
        stationary_data = original_data.pct_change().dropna()
        # Replace inf values with 0
        stationary_data = stationary_data.replace([np.inf, -np.inf], 0)

    st.write(f"   ✅ {column} converted to stationary: {len(original_data)} → {len(stationary_data)} points")

    return stationary_data

def convert_to_stationary(df, feature_cols, target_col):
    """
    Convert all non-stationary columns to stationary
    """
    st.write(f"\n🔧 CONVERTING NON-STATIONARY DATA TO STATIONARY...")
    st.write("-" * 50)

    df_stationary = df.copy()
    stationary_info = {}

    # Test and convert target variable first
    if target_col in df.columns:
        st.write(f"\n🎯 Processing TARGET: {target_col}")
        is_target_stationary = test_stationarity(df[target_col], target_col)

        if not is_target_stationary:
            stationary_target = make_stationary(df, target_col, 'difference')
            # Align the dataframe
            start_idx = len(df) - len(stationary_target)
            df_stationary = df_stationary.iloc[start_idx:].reset_index(drop=True)
            df_stationary[target_col] = stationary_target.values
            stationary_info[target_col] = 'differenced'
        else:
            stationary_info[target_col] = 'already_stationary'

    # Test and convert features
    st.write(f"\n📊 Processing FEATURES...")
    features_to_remove = []

    for col in feature_cols:
        if col in df_stationary.columns:
            st.write(f"\n   Processing: {col}")
            is_stationary = test_stationarity(df_stationary[col], col)

            if not is_stationary:
                try:
                    # Choose method based on data characteristics
                    if (df_stationary[col] > 0).all():
                        method = 'log_difference' if df_stationary[col].var() > df_stationary[col].mean() else 'difference'
                    else:
                        method = 'difference'

                    stationary_series = make_stationary(df_stationary, col, method)

                    if len(stationary_series) > 0:
                        # Align with the current dataframe length
                        if len(stationary_series) < len(df_stationary):
                            start_idx = len(df_stationary) - len(stationary_series)
                            df_stationary = df_stationary.iloc[start_idx:].reset_index(drop=True)

                        df_stationary[col] = stationary_series.values[:len(df_stationary)]
                        stationary_info[col] = method
                    else:
                        features_to_remove.append(col)

                except Exception as e:
                    st.write(f"   ❌ Error converting {col}: {str(e)}")
                    features_to_remove.append(col)
            else:
                stationary_info[col] = 'already_stationary'

    # Remove problematic features
    for col in features_to_remove:
        if col in feature_cols:
            feature_cols.remove(col)
        st.write(f"   🗑️ Removed {col} (conversion failed)")

    # Remove any remaining NaN values
    df_stationary = df_stationary.dropna()

    st.write(f"\n✅ STATIONARITY CONVERSION COMPLETE!")
    st.write(f"   Final dataset shape: {df_stationary.shape}")
    st.write(f"   Remaining features: {len([col for col in feature_cols if col in df_stationary.columns])}")

    return df_stationary, feature_cols, stationary_info

# =============================================================================
# 2. FIXED DATA SELECTION & PREPARATION
# =============================================================================

def prepare_arimax_data(datasets):
    """
    FIXED: Prepare data specifically for ARIMAX model training
    """
    st.write("📊 PREPARING ARIMAX DATA...")
    st.write("-" * 40)

    # Load your actual dataset - REPLACE WITH YOUR FILE PATH
    try:
        df = pd.read_csv('financial_risk_timeseries.csv')  # REPLACE THIS PATH
    except FileNotFoundError:
        st.write("⚠️ File not found, using dataset from memory if available")
        if 'financial_ts' in datasets:
            df = datasets['financial_ts'].copy()
        else:
            st.write("❌ No dataset available!")
            return None, None, None

    st.write(f"✅ Original shape: {df.shape}")

    # Identify date column
    date_cols = [col for col in df.columns if any(x in col.lower() for x in ['date', 'time', 'period'])]
    if date_cols:
        date_col = date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        st.write(f"✅ Date column identified: {date_col}")
    else:
        st.write("⚠️  No date column found, assuming data is already sorted chronologically")
        date_col = None

    # Remove perfectly correlated features
    target_col = 'Risk_Score'
    high_corr_features_to_remove = ['Altman_Z']  # Perfect correlation with Risk_Score

    for col in high_corr_features_to_remove:
        if col in df.columns:
            df = df.drop(col, axis=1)
            st.write(f"🗑️  Removed {col} (high correlation)")

    # Identify numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = [date_col, target_col] if date_col else [target_col]
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

    # Define feature categories
    financial_features = [col for col in numeric_cols if
                         any(x in col for x in ['PE_Ratio', 'PB_Ratio', 'Debt_to_Equity', 'Current_Ratio',
                                               'ROE', 'ROA', 'Market_Cap', 'Total_Revenue', 'Net_Income'])]

    macro_features = [col for col in numeric_cols if
                     any(x in col for x in ['CPI', 'GDP', 'Fed_Funds_Rate', 'Unemployment_Rate'])]

    sentiment_features = [col for col in numeric_cols if
                         any(x in col for x in ['Sentiment'])]

    other_numeric_features = [col for col in numeric_cols if col not in
                             financial_features + macro_features + sentiment_features]

    feature_cols = financial_features + macro_features + sentiment_features + other_numeric_features

    if target_col not in df.columns:
        st.write(f"❌ Target column '{target_col}' not found!")
        return None, None, None

    # Create final dataset
    final_cols = ([date_col] if date_col else []) + feature_cols + [target_col]
    arimax_data = df[final_cols].copy()

    st.write(f"✅ ARIMAX dataset prepared: {arimax_data.shape}")
    st.write(f"📋 Features: {len(feature_cols)}")
    st.write(f"🎯 Target: {target_col}")

    return arimax_data, feature_cols, target_col

def prepare_lstm_data(datasets):
    """
    FIXED: Prepare data for LSTM model training
    """
    st.write("\n📊 PREPARING LSTM DATA...")
    st.write("-" * 40)

    # Try to load individual stock data or use combined dataset
    stock_dfs = []
    stock_names = ['aapl', 'googl', 'msft']

    for stock in stock_names:
        try:
            if stock in datasets:
                df = datasets[stock].copy()
                df['Company'] = stock.upper()
                stock_dfs.append(df)
                st.write(f"✅ {stock.upper()}: {df.shape}")
        except:
            st.write(f"⚠️ {stock} data not available")

    if stock_dfs:
        combined_stocks = pd.concat(stock_dfs, ignore_index=True)
    else:
        # Use the main dataset if individual stocks not available
        st.write("⚠️ Using main dataset instead of individual stocks")
        try:
            combined_stocks = pd.read_csv('financial_risk_timeseries.csv')  # REPLACE PATH
        except:
            if 'financial_ts' in datasets:
                combined_stocks = datasets['financial_ts'].copy()
            else:
                st.write("❌ No dataset available!")
                return None, None, None

    # Remove perfectly correlated features
    target_col = 'Risk_Score'
    high_corr_features_to_remove = ['Altman_Z']

    for col in high_corr_features_to_remove:
        if col in combined_stocks.columns:
            combined_stocks = combined_stocks.drop(col, axis=1)
            st.write(f"🗑️  Removed {col} (high correlation)")

    # Identify feature columns
    exclude_cols = ['Company', target_col]
    date_cols = [col for col in combined_stocks.columns if any(x in col.lower() for x in ['date', 'time', 'period'])]
    exclude_cols.extend(date_cols)

    numeric_cols = combined_stocks.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    st.write(f"✅ LSTM dataset prepared: {combined_stocks.shape}")
    st.write(f"📋 Features: {len(feature_cols)}")
    st.write(f"🎯 Target: {target_col}")

    return combined_stocks, feature_cols, target_col

# =============================================================================
# 3. FIXED DATA CLEANING & PREPROCESSING
# =============================================================================

def clean_and_preprocess(df, feature_cols, target_col, model_type='arimax'):
    """
    FIXED: Clean and preprocess data - preserve more data points
    """
    st.write(f"\n🧹 CLEANING & PREPROCESSING DATA FOR {model_type.upper()}...")
    st.write("-" * 50)

    df_clean = df.copy()

    # Ensure all feature columns and target are numeric
    st.write("0️⃣ Ensuring numeric data types...")
    for col in feature_cols + [target_col]:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # 1. Handle missing values - MORE CONSERVATIVE APPROACH
    st.write("1️⃣ Handling missing values...")
    missing_before = df_clean[feature_cols + [target_col]].isnull().sum().sum()

    # For time series, use interpolation first, then forward/backward fill
    for col in feature_cols + [target_col]:
        if col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                # Linear interpolation for small gaps
                df_clean[col] = df_clean[col].interpolate(method='linear', limit=3)
                # Forward fill for remaining
                df_clean[col] = df_clean[col].fillna(method='ffill', limit=5)
                # Backward fill
                df_clean[col] = df_clean[col].fillna(method='bfill', limit=5)
                # Final fallback to median
                if df_clean[col].isnull().sum() > 0:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    missing_after = df_clean[feature_cols + [target_col]].isnull().sum().sum()
    st.write(f"   Missing values: {missing_before} → {missing_after}")

    # 2. Handle outliers - MORE CONSERVATIVE (preserve more data)
    st.write("2️⃣ Handling outliers...")
    outliers_removed = 0

    for col in feature_cols:
        if col in df_clean.columns and df_clean[col].dtype in ['int64', 'float64']:
            if df_clean[col].nunique() > 1:
                Q1 = df_clean[col].quantile(0.10)  # More conservative
                Q3 = df_clean[col].quantile(0.90)  # More conservative
                IQR = Q3 - Q1

                if IQR > 0:
                    lower_bound = Q1 - 3 * IQR  # Even more conservative
                    upper_bound = Q3 + 3 * IQR

                    outlier_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                    outliers_in_col = outlier_mask.sum()

                    if outliers_in_col > 0:
                        # Cap outliers instead of removing
                        df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
                        outliers_removed += outliers_in_col

    st.write(f"   Outliers capped: {outliers_removed}")

    # 3. Remove duplicate rows
    duplicates_before = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    duplicates_after = df_clean.duplicated().sum()
    st.write(f"3️⃣ Duplicates removed: {duplicates_before} → {duplicates_after}")

    st.write(f"✅ Cleaned dataset shape: {df_clean.shape}")

    return df_clean

# =============================================================================
# 4. FIXED FEATURE ENGINEERING - PRESERVE MORE DATA
# =============================================================================

def engineer_features_arimax(df, feature_cols, target_col):
    """
    FIXED: Feature engineering for ARIMAX - preserve more data points
    """
    st.write("\n🔧 FEATURE ENGINEERING FOR ARIMAX...")
    st.write("-" * 40)

    df_engineered = df.copy()
    new_features = []

    available_features = [col for col in feature_cols if col in df_engineered.columns and
                         df_engineered[col].dtype in ['int64', 'float64']]

    st.write(f"   Working with {len(available_features)} numeric features")

    # 1. REDUCED Lag features to preserve data
    st.write("1️⃣ Creating lag features...")
    lag_periods = [1, 3, 6]  # REDUCED from [1, 3, 6, 12]

    for col in available_features:
        for lag in lag_periods:
            lag_col = f"{col}_lag_{lag}"
            df_engineered[lag_col] = df_engineered[col].shift(lag)
            new_features.append(lag_col)

    st.write(f"   Added {len([f for f in new_features if 'lag' in f])} lag features")

    # 2. REDUCED Rolling statistics
    st.write("2️⃣ Creating rolling statistics...")
    windows = [3, 6]  # REDUCED from [3, 6, 12]

    for col in available_features:
        for window in windows:
            # Rolling mean
            mean_col = f"{col}_rolling_mean_{window}"
            df_engineered[mean_col] = df_engineered[col].rolling(window=window, min_periods=1).mean()
            new_features.append(mean_col)

            # Rolling standard deviation
            std_col = f"{col}_rolling_std_{window}"
            df_engineered[std_col] = df_engineered[col].rolling(window=window, min_periods=1).std()
            # Fill NaN in std with 0
            df_engineered[std_col] = df_engineered[std_col].fillna(0)
            new_features.append(std_col)

    st.write(f"   Added {len([f for f in new_features if 'rolling' in f])} rolling features")

    # 3. Rate of change features
    st.write("3️⃣ Creating rate of change features...")
    for col in available_features:
        roc_col = f"{col}_roc"
        df_engineered[roc_col] = df_engineered[col].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        new_features.append(roc_col)

    st.write(f"   Added {len([f for f in new_features if 'roc' in f])} rate of change features")

    # 4. REDUCED Interaction features
    st.write("4️⃣ Creating interaction features...")
    interactions = [
        ('PE_Ratio', 'ROE'),
        ('Debt_to_Equity', 'Current_Ratio')  # REDUCED interactions
    ]

    for col1, col2 in interactions:
        if col1 in available_features and col2 in available_features:
            interaction_col = f"{col1}_x_{col2}"
            df_engineered[interaction_col] = df_engineered[col1] * df_engineered[col2]
            new_features.append(interaction_col)

    st.write(f"   Added {len([f for f in new_features if '_x_' in f])} interaction features")

    all_features = available_features + [f for f in new_features if f in df_engineered.columns]

    st.write(f"✅ Total features after engineering: {len(all_features)}")

    return df_engineered, all_features

def engineer_features_lstm(df, feature_cols, target_col):
    """
    FIXED: Feature engineering for LSTM - preserve more data points
    """
    st.write("\n🔧 FEATURE ENGINEERING FOR LSTM...")
    st.write("-" * 40)

    df_engineered = df.copy()
    new_features = []

    # 1. REDUCED Lag features
    st.write("1️⃣ Creating lag features...")
    lag_periods = [1, 3]  # REDUCED from [1, 3, 6]

    if 'Company' in df_engineered.columns:
        for col in feature_cols:
            if col in df_engineered.columns:
                for lag in lag_periods:
                    lag_col = f"{col}_lag_{lag}"
                    df_engineered[lag_col] = df_engineered.groupby('Company')[col].shift(lag)
                    new_features.append(lag_col)
    else:
        for col in feature_cols:
            if col in df_engineered.columns:
                for lag in lag_periods:
                    lag_col = f"{col}_lag_{lag}"
                    df_engineered[lag_col] = df_engineered[col].shift(lag)
                    new_features.append(lag_col)

    st.write(f"   Added {len([f for f in new_features if 'lag' in f])} lag features")

    # 2. REDUCED Technical indicators
    st.write("2️⃣ Creating technical indicators...")
    windows = [5, 10]  # REDUCED from [5, 10, 20]

    for col in feature_cols:
        if col in df_engineered.columns:
            for window in windows:
                ma_col = f"{col}_MA_{window}"
                if 'Company' in df_engineered.columns:
                    df_engineered[ma_col] = df_engineered.groupby('Company')[col].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
                else:
                    df_engineered[ma_col] = df_engineered[col].rolling(window=window, min_periods=1).mean()
                new_features.append(ma_col)

    st.write(f"   Added {len([f for f in new_features if 'MA_' in f])} technical indicators")

    # 3. Volatility features
    st.write("3️⃣ Creating volatility features...")
    for col in feature_cols[:10]:  # LIMIT to first 10 features
        if col in df_engineered.columns:
            vol_col = f"{col}_volatility"
            if 'Company' in df_engineered.columns:
                df_engineered[vol_col] = df_engineered.groupby('Company')[col].rolling(window=5, min_periods=1).std().reset_index(0, drop=True)
            else:
                df_engineered[vol_col] = df_engineered[col].rolling(window=5, min_periods=1).std()
            df_engineered[vol_col] = df_engineered[vol_col].fillna(0)
            new_features.append(vol_col)

    st.write(f"   Added {len([f for f in new_features if 'volatility' in f])} volatility features")

    all_features = feature_cols + new_features

    st.write(f"✅ Total features after engineering: {len(all_features)}")

    return df_engineered, all_features

# =============================================================================
# 5. FIXED FEATURE SELECTION
# =============================================================================

def select_features(df, feature_cols, target_col, method='mutual_info', k=20):
    """
    FIXED: Select most important features - handle edge cases
    """
    st.write(f"\n🎯 FEATURE SELECTION (top {k} features)...")
    st.write("-" * 40)

    available_features = [col for col in feature_cols if col in df.columns and
                         df[col].dtype in ['int64', 'float64']]

    if len(available_features) == 0:
        st.write("❌ No numeric features available for selection!")
        return []

    # Prepare data - MORE ROBUST handling
    X = df[available_features].copy()
    y = df[target_col].copy()

    # Handle inf and extreme values more carefully
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)

    # Fill NaN with median instead of 0 to preserve relationships
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            X[col] = X[col].fillna(X[col].median())

    if y.isnull().sum() > 0:
        y = y.fillna(y.median())

    # Remove constant features
    constant_features = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_features:
        X = X.drop(columns=constant_features)
        st.write(f"   Removed {len(constant_features)} constant features")

    # Limit k to available features
    k = min(k, len(X.columns))

    if k == 0:
        st.write("❌ No features left after cleaning!")
        return []

    try:
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
        else:
            selector = SelectKBest(score_func=f_regression, k=k)

        X_selected = selector.fit_transform(X, y)
        selected_features = [X.columns[i] for i in selector.get_support(indices=True)]

        st.write(f"✅ Selected {len(selected_features)} features using {method}")
        st.write("Top 10 selected features:")
        for i, feature in enumerate(selected_features[:10]):
            st.write(f"   {i+1}. {feature}")

        return selected_features

    except Exception as e:
        st.write(f"⚠️ Feature selection failed: {str(e)}")
        # Return top features by variance if selection fails
        feature_vars = X.var().sort_values(ascending=False)
        selected_features = feature_vars.head(k).index.tolist()
        st.write(f"✅ Selected {len(selected_features)} features by variance instead")
        return selected_features

# =============================================================================
# 6. FIXED DATA SCALING
# =============================================================================

def scale_data(df, feature_cols, target_col, method='robust'):
    """
    FIXED: Scale features and target variable - handle edge cases
    """
    st.write(f"\n📏 SCALING DATA using {method} scaler...")
    st.write("-" * 40)

    df_scaled = df.copy()

    available_features = [col for col in feature_cols if col in df_scaled.columns and
                         df_scaled[col].dtype in ['int64', 'float64']]

    if len(available_features) == 0:
        st.write("❌ No features available for scaling!")
        return df_scaled, None, None

    # Choose scaler
    if method == 'standard':
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
    elif method == 'minmax':
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
    else:  # robust
        feature_scaler = RobustScaler()
        target_scaler = RobustScaler()

    # Scale features - ROBUST handling
    feature_data = df_scaled[available_features].copy()
    feature_data = feature_data.replace([np.inf, -np.inf], np.nan)

    # Fill NaN with median
    for col in feature_data.columns:
        if feature_data[col].isnull().sum() > 0:
            feature_data[col] = feature_data[col].fillna(feature_data[col].median())

    df_scaled[available_features] = feature_scaler.fit_transform(feature_data)

    # Scale target
    if target_col in df_scaled.columns:
        target_values = df_scaled[target_col].replace([np.inf, -np.inf], np.nan)
        if target_values.isnull().sum() > 0:
            target_values = target_values.fillna(target_values.median())

        target_values = target_values.values.reshape(-1, 1)
        df_scaled[target_col] = target_scaler.fit_transform(target_values).flatten()
    else:
        st.write(f"⚠️  Target column {target_col} not found!")
        target_scaler = None

    st.write(f"✅ Scaled {len(available_features)} features and target variable")

    return df_scaled, feature_scaler, target_scaler

# =============================================================================
# 7. FIXED MAIN PREPROCESSING PIPELINE
# =============================================================================

def preprocess_for_arimax(datasets):
    """FIXED: Complete preprocessing pipeline for ARIMAX"""
    st.write("\n" + "="*60)
    st.write("🎯 FIXED ARIMAX PREPROCESSING PIPELINE")
    st.write("="*60)

    # 1. Prepare data
    arimax_data, feature_cols, target_col = prepare_arimax_data(datasets)
    if arimax_data is None:
        return None

    # 2. Clean data
    clean_data = clean_and_preprocess(arimax_data, feature_cols, target_col, 'arimax')

    # 3. CONVERT TO STATIONARY (NEW!)
    stationary_data, stationary_features, stationary_info = convert_to_stationary(clean_data, feature_cols, target_col)

    # 4. Feature engineering (on stationary data)
    engineered_data, all_features = engineer_features_arimax(stationary_data, stationary_features, target_col)

    # 5. Feature selection (limit features for ARIMAX)
    selected_features = select_features(engineered_data, all_features, target_col, k=15)

    # 6. Scale data
    final_data, feature_scaler, target_scaler = scale_data(engineered_data, selected_features, target_col)

    # 7. CONSERVATIVE removal of NaN (preserve more data)
    st.write(f"\n🧹 Final data cleaning...")
    initial_rows = len(final_data)

    # Only remove rows where target is NaN
    final_data = final_data.dropna(subset=[target_col])

    # For features, fill remaining NaN with 0 instead of dropping rows
    feature_cols_in_data = [col for col in selected_features if col in final_data.columns]
    final_data[feature_cols_in_data] = final_data[feature_cols_in_data].fillna(0)

    final_rows = len(final_data)
    st.write(f"   Preserved {final_rows}/{initial_rows} rows ({final_rows/initial_rows*100:.1f}%)")

    st.write(f"\n✅ ARIMAX PREPROCESSING COMPLETE!")
    st.write(f"   Final dataset shape: {final_data.shape}")
    st.write(f"   Features: {len(selected_features)}")
    st.write(f"   Time periods: {len(final_data)}")
    st.write(f"   Stationarity conversions applied: {len([k for k, v in stationary_info.items() if v != 'already_stationary'])}")

    return {
        'data': final_data,
        'features': selected_features,
        'target': target_col,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'stationary_info': stationary_info
    }

def preprocess_for_lstm(datasets):
    """FIXED: Complete preprocessing pipeline for LSTM"""
    st.write("\n" + "="*60)
    st.write("🎯 FIXED LSTM PREPROCESSING PIPELINE")
    st.write("="*60)

    # 1. Prepare data
    lstm_data, feature_cols, target_col = prepare_lstm_data(datasets)
    if lstm_data is None:
        return None

    # 2. Clean data
    clean_data = clean_and_preprocess(lstm_data, feature_cols, target_col, 'lstm')

    # 3. CONVERT TO STATIONARY (NEW!)
    stationary_data, stationary_features, stationary_info = convert_to_stationary(clean_data, feature_cols, target_col)

    # 4. Feature engineering (on stationary data)
    engineered_data, all_features = engineer_features_lstm(stationary_data, stationary_features, target_col)

    # 5. Feature selection (more features for LSTM)
    selected_features = select_features(engineered_data, all_features, target_col, k=25)

    # 6. Scale data
    final_data, feature_scaler, target_scaler = scale_data(engineered_data, selected_features, target_col)

    # 7. CONSERVATIVE removal of NaN (preserve more data)
    st.write(f"\n🧹 Final data cleaning...")
    initial_rows = len(final_data)

    # Only remove rows where target is NaN
    final_data = final_data.dropna(subset=[target_col])

    # For features, fill remaining NaN with 0 instead of dropping rows
    feature_cols_in_data = [col for col in selected_features if col in final_data.columns]
    final_data[feature_cols_in_data] = final_data[feature_cols_in_data].fillna(0)

    final_rows = len(final_data)
    st.write(f"   Preserved {final_rows}/{initial_rows} rows ({final_rows/initial_rows*100:.1f}%)")

    st.write(f"\n✅ LSTM PREPROCESSING COMPLETE!")
    st.write(f"   Final dataset shape: {final_data.shape}")
    st.write(f"   Features: {len(selected_features)}")
    st.write(f"   Samples: {len(final_data)}")
    st.write(f"   Stationarity conversions applied: {len([k for k, v in stationary_info.items() if v != 'already_stationary'])}")

    return {
        'data': final_data,
        'features': selected_features,
        'target': target_col,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'stationary_info': stationary_info
    }

# =============================================================================
# 8. FIXED EXECUTION
# =============================================================================

def run_fixed_preprocessing(datasets=None):
    """FIXED: Run complete preprocessing for both models"""
    st.write("🚀 STARTING FIXED PREPROCESSING FOR BOTH MODELS...")

    # If datasets not provided, create the dictionary with your file paths
    if datasets is None:
        datasets = {
            'financial_ts': None,  # Will be loaded directly in functions
            'aapl': None,
            'googl': None,
            'msft': None,
            'macro': None,
            'sentiment': None
        }

    try:
        # Preprocess for ARIMAX
        arimax_processed = preprocess_for_arimax(datasets)

        # Preprocess for LSTM
        lstm_processed = preprocess_for_lstm(datasets)

        # Summary
        st.write(f"\n" + "="*60)
        st.write("📋 PREPROCESSING SUMMARY")
        st.write("="*60)

        if arimax_processed:
            st.write(f"✅ ARIMAX: {arimax_processed['data'].shape[0]} samples, {len(arimax_processed['features'])} features")
        else:
            st.write("❌ ARIMAX: Failed")

        if lstm_processed:
            st.write(f"✅ LSTM: {lstm_processed['data'].shape[0]} samples, {len(lstm_processed['features'])} features")
        else:
            st.write("❌ LSTM: Failed")

        return arimax_processed, lstm_processed

    except Exception as e:
        st.write(f"❌ Error in preprocessing: {str(e)}")
        return None, None

# =============================================================================
# 9. UTILITY FUNCTIONS
# =============================================================================

def check_stationarity_summary(processed_data):
    """Check what stationarity transformations were applied"""
    if processed_data and 'stationary_info' in processed_data:
        st.write("\n📊 STATIONARITY TRANSFORMATIONS APPLIED:")
        st.write("-" * 50)
        for var, method in processed_data['stationary_info'].items():
            if method == 'already_stationary':
                st.write(f"✅ {var}: Already stationary")
            else:
                st.write(f"🔄 {var}: {method}")

def save_processed_data(arimax_processed, lstm_processed, output_dir='processed_data'):
    """Save processed data to files"""
    import os

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if arimax_processed:
        arimax_processed['data'].to_csv(f'{output_dir}/arimax_processed_data.csv', index=False)
        st.write(f"💾 ARIMAX processed data saved to {output_dir}/arimax_processed_data.csv")

    if lstm_processed:
        lstm_processed['data'].to_csv(f'{output_dir}/lstm_processed_data.csv', index=False)
        st.write(f"💾 LSTM processed data saved to {output_dir}/lstm_processed_data.csv")

# =============================================================================
# 10. USAGE EXAMPLE
# =============================================================================

if True:
    st.write("🔧 FIXED PREPROCESSING PIPELINE READY!")
    st.write("\nTo use this fixed pipeline:")
    st.write("1. Replace file paths in prepare_arimax_data() and prepare_lstm_data()")
    st.write("2. Run: arimax_processed, lstm_processed = run_fixed_preprocessing()")
    st.write("3. Check results with: check_stationarity_summary(arimax_processed)")
    st.write("\n" + "="*60)


# In[ ]:


# Your existing call should now work:
arimax_processed, lstm_processed = run_fixed_preprocessing()

# Check what stationarity transformations were applied:
check_stationarity_summary(arimax_processed)
check_stationarity_summary(lstm_processed)


# ## ARIMAX Model training

# In[ ]:


# ARIMAX Model Training Pipeline - FIXED FOR YOUR PROCESSED DATA
# Financial Risk Forecasting - Uses arimax_processed data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Statistical and ML libraries
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import pickle
import os

# Time series libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

st.write("=== ARIMAX MODEL TRAINING WITH YOUR DATA ===")
st.write("="*60)

# Create directories for saving models
os.makedirs('saved_models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# =============================================================================
# 1. DATA PREPARATION FOR ARIMAX TRAINING
# =============================================================================

def prepare_arimax_train_test_split(processed_data, test_size=0.2, val_size=0.1):
    """
    Prepare train/validation/test splits for ARIMAX time series data
    """
    st.write("\n📊 PREPARING ARIMAX TRAIN/VALIDATION/TEST SPLITS...")
    st.write("-" * 50)

    if processed_data is None:
        st.write("❌ No processed data provided!")
        return None

    data = processed_data['data']
    features = processed_data['features']
    target = processed_data['target']

    st.write(f"✅ Input data shape: {data.shape}")
    st.write(f"✅ Features: {len(features)}")
    st.write(f"✅ Target: {target}")

    # Time series split (no random shuffling)
    n_samples = len(data)
    test_idx = int(n_samples * (1 - test_size))
    val_idx = int(test_idx * (1 - val_size))

    # Split data
    train_data = data[:val_idx].copy()
    val_data = data[val_idx:test_idx].copy()
    test_data = data[test_idx:].copy()

    st.write(f"✅ Training set: {len(train_data)} samples ({len(train_data)/len(data)*100:.1f}%)")
    st.write(f"✅ Validation set: {len(val_data)} samples ({len(val_data)/len(data)*100:.1f}%)")
    st.write(f"✅ Test set: {len(test_data)} samples ({len(test_data)/len(data)*100:.1f}%)")

    return {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'features': features,
        'target': target,
        'scalers': {
            'feature_scaler': processed_data.get('feature_scaler'),
            'target_scaler': processed_data.get('target_scaler')
        },
        'stationary_info': processed_data.get('stationary_info', {})
    }

# =============================================================================
# 2. SIMPLIFIED ARIMAX MODEL CLASS
# =============================================================================

class ARIMAXModel:
    def __init__(self):
        self.model = None
        self.fitted_model = None
        self.best_params = None
        self.is_fitted = False

    def check_stationarity(self, series, significance_level=0.05):
        """Check if series is stationary using ADF test"""
        try:
            # Remove NaN values and ensure we have enough data
            clean_series = series.dropna()
            if len(clean_series) < 10:
                return False, None

            result = adfuller(clean_series, maxlag=min(12, len(clean_series)//4))
            p_value = result[1]
            is_stationary = p_value <= significance_level

            st.write(f"   ADF Test p-value: {p_value:.6f}")
            st.write(f"   Series is {'stationary' if is_stationary else 'non-stationary'}")

            return is_stationary, result
        except Exception as e:
            st.write(f"   ⚠️  Error in stationarity test: {str(e)}")
            return False, None

    def grid_search_arimax(self, y_train, X_train, max_p=3, max_d=2, max_q=3):
        """Simple grid search for ARIMAX parameters"""
        st.write("   🔍 Performing grid search for ARIMAX parameters...")

        best_aic = float('inf')
        best_order = (1, 1, 1)  # Default
        best_model = None

        # Grid search with error handling
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    if p == 0 and q == 0:  # Skip (0,d,0) models
                        continue

                    if p + q > 5:  # Limit complexity
                        continue

                    try:
                        # Fit model with different solvers
                        for method in ['lbfgs', 'nm']:
                            try:
                                model = SARIMAX(y_train, exog=X_train, order=(p, d, q), trend='c')
                                fitted = model.fit(disp=False, maxiter=100, method=method)

                                if fitted.aic < best_aic:
                                    best_aic = fitted.aic
                                    best_order = (p, d, q)
                                    best_model = fitted
                                break  # If successful, don't try other methods
                            except:
                                continue
                    except:
                        continue

        st.write(f"   ✅ Best parameters found: {best_order} (AIC: {best_aic:.4f})")
        return {
            'p': best_order[0],
            'd': best_order[1],
            'q': best_order[2],
            'fitted_model': best_model
        }

    def tune_hyperparameters(self, train_data, features, target):
        """Simplified hyperparameter tuning"""
        st.write("\n🎯 TUNING ARIMAX HYPERPARAMETERS...")
        st.write("-" * 40)

        # Extract data properly
        y_train = train_data[target].values
        X_train = train_data[features].values

        st.write(f"   📊 Training data: {len(y_train)} samples, {X_train.shape[1]} features")

        # Check stationarity of target variable
        st.write("   📊 Checking target variable stationarity...")
        is_stationary, _ = self.check_stationarity(pd.Series(y_train))

        # Grid search for best parameters
        result = self.grid_search_arimax(y_train, X_train, max_p=3, max_d=2, max_q=3)

        self.best_params = {
            'p': result['p'],
            'd': result['d'],
            'q': result['q']
        }

        return self.best_params

    def fit(self, train_data, val_data, features, target):
        """Fit ARIMAX model with best parameters"""
        st.write("\n🏋️‍♂️ TRAINING ARIMAX MODEL...")
        st.write("-" * 40)

        # Extract training data
        y_train = train_data[target].values
        X_train = train_data[features].values

        st.write(f"   📊 Training on {len(y_train)} samples with {X_train.shape[1]} features")

        # Get best parameters if not already done
        if self.best_params is None:
            self.tune_hyperparameters(train_data, features, target)

        p, d, q = self.best_params['p'], self.best_params['d'], self.best_params['q']

        try:
            # Try different methods to fit the model
            methods = ['lbfgs', 'nm', 'bfgs']

            for method in methods:
                try:
                    st.write(f"   🔄 Trying {method} optimization method...")
                    self.model = SARIMAX(y_train, exog=X_train, order=(p, d, q), trend='c')
                    self.fitted_model = self.model.fit(
                        disp=False,
                        maxiter=200,
                        method=method,
                        warn_convergence=False
                    )

                    self.is_fitted = True
                    st.write(f"   ✅ Model fitted successfully with {method}")
                    break
                except Exception as e:
                    st.write(f"   ⚠️  {method} failed: {str(e)}")
                    continue

            if not self.is_fitted:
                # Try simplest model as fallback
                st.write("   🔄 Trying simplest model ARIMAX(1,1,1)...")
                self.model = SARIMAX(y_train, exog=X_train, order=(1, 1, 1), trend='c')
                self.fitted_model = self.model.fit(disp=False, maxiter=100)
                self.is_fitted = True
                self.best_params = {'p': 1, 'd': 1, 'q': 1}
                st.write("   ✅ Fallback model fitted successfully")

            if self.is_fitted:
                st.write(f"   📊 Final Parameters: ARIMAX({p},{d},{q})")
                st.write(f"   📊 AIC: {self.fitted_model.aic:.4f}")
                st.write(f"   📊 BIC: {self.fitted_model.bic:.4f}")
                st.write(f"   📊 Log Likelihood: {self.fitted_model.llf:.4f}")

                # Basic diagnostics
                self.print_diagnostics()

            return self.fitted_model

        except Exception as e:
            st.write(f"   ❌ All fitting attempts failed: {str(e)}")
            return None

    def print_diagnostics(self):
        """Print basic model diagnostics"""
        try:
            residuals = self.fitted_model.resid

            st.write(f"\n   📋 Model Diagnostics:")
            st.write(f"   📊 Residual mean: {np.mean(residuals):.6f}")
            st.write(f"   📊 Residual std: {np.std(residuals):.6f}")

            # Simple autocorrelation check
            try:
                lb_test = acorr_ljungbox(residuals, lags=min(10, len(residuals)//4), return_df=True)
                if not lb_test.empty:
                    significant_lags = (lb_test['lb_pvalue'] < 0.05).sum()
                    if significant_lags == 0:
                        st.write("   ✅ No significant residual autocorrelation detected")
                    else:
                        st.write(f"   ⚠️  {significant_lags} lags show autocorrelation")
            except:
                st.write("   ⚠️  Could not perform autocorrelation test")

        except Exception as e:
            st.write(f"   ⚠️  Error in diagnostics: {str(e)}")

    def predict(self, test_data, features):
        """Make predictions on test data"""
        if not self.is_fitted:
            st.write("❌ Model not fitted yet!")
            return None, None

        try:
            # Extract test features
            if isinstance(test_data, pd.DataFrame):
                X_test = test_data[features].values
            else:
                X_test = np.array(test_data)

            # Ensure X_test is 2D
            if len(X_test.shape) == 1:
                X_test = X_test.reshape(1, -1)

            st.write(f"   📊 Making predictions for {len(X_test)} samples...")

            # Get forecasts
            forecast_result = self.fitted_model.get_forecast(steps=len(X_test), exog=X_test)

            # Extract predictions
            predictions = forecast_result.predicted_mean
            if hasattr(predictions, 'values'):
                predictions = predictions.values

            # Get confidence intervals
            try:
                conf_int = forecast_result.conf_int()
                if hasattr(conf_int, 'values'):
                    conf_int = conf_int.values
            except:
                conf_int = None

            st.write(f"   ✅ Predictions generated successfully")
            return predictions, conf_int

        except Exception as e:
            st.write(f"❌ Error making predictions: {str(e)}")
            return None, None

    def save_model(self, filepath):
        """Save ARIMAX model"""
        try:
            model_data = {
                'fitted_model': self.fitted_model,
                'best_params': self.best_params,
                'is_fitted': self.is_fitted,
                'model_summary': {
                    'aic': float(self.fitted_model.aic),
                    'bic': float(self.fitted_model.bic),
                    'llf': float(self.fitted_model.llf)
                }
            }

            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

            st.write(f"✅ ARIMAX model saved to {filepath}")
        except Exception as e:
            st.write(f"❌ Error saving ARIMAX model: {str(e)}")

# =============================================================================
# 3. MODEL EVALUATION
# =============================================================================

def calculate_metrics(y_true, y_pred, model_name="ARIMAX"):
    """Calculate comprehensive evaluation metrics"""

    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        st.write(f"⚠️  No valid predictions for {model_name}")
        return {}

    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)

    # R-squared
    try:
        r2 = r2_score(y_true, y_pred)
    except:
        r2 = np.nan

    # MAPE
    mask_nonzero = y_true != 0
    if np.sum(mask_nonzero) > 0:
        mape = np.mean(np.abs((y_true[mask_nonzero] - y_pred[mask_nonzero]) / y_true[mask_nonzero])) * 100
    else:
        mape = np.nan

    # Directional accuracy
    if len(y_true) > 1:
        actual_changes = np.diff(y_true)
        pred_changes = np.diff(y_pred)

        # Binary directional accuracy (up/down)
        actual_directions = actual_changes > 0
        pred_directions = pred_changes > 0
        directional_accuracy = (np.sum(actual_directions == pred_directions) / len(actual_directions)) * 100
    else:
        directional_accuracy = np.nan

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MSE': mse,
        'R²': r2,
        'MAPE': mape,
        'Directional_Accuracy': directional_accuracy,
        'N_Predictions': len(y_true)
    }

    return metrics

def evaluate_arimax_model(model, test_split, target_scaler=None):
    """Evaluate ARIMAX model"""
    st.write(f"\n📊 EVALUATING ARIMAX MODEL...")
    st.write("-" * 50)

    test_data = test_split['test']
    features = test_split['features']
    target = test_split['target']

    # Get predictions
    predictions, conf_int = model.predict(test_data, features)

    if predictions is None:
        st.write(f"❌ Failed to get predictions from ARIMAX")
        return None

    # Get actual values
    y_true = test_data[target].values

    # Ensure same length
    min_len = min(len(y_true), len(predictions))
    y_true = y_true[:min_len]
    predictions = predictions[:min_len]

    # Inverse transform if scaler is available
    if target_scaler is not None:
        try:
            y_true = target_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
            predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        except Exception as e:
            st.write(f"   ⚠️  Could not inverse transform: {str(e)}")

    # Calculate metrics
    metrics = calculate_metrics(y_true, predictions, "ARIMAX")

    # Print results
    st.write(f"📈 ARIMAX Performance Metrics:")
    for metric, value in metrics.items():
        if not np.isnan(value):
            if metric in ['MAE', 'RMSE', 'MSE']:
                st.write(f"   {metric}: {value:.6f}")
            elif metric in ['R²']:
                st.write(f"   {metric}: {value:.4f}")
            elif metric in ['MAPE', 'Directional_Accuracy']:
                st.write(f"   {metric}: {value:.2f}%")
            else:
                st.write(f"   {metric}: {value}")

    return {
        'predictions': predictions,
        'actual': y_true,
        'metrics': metrics,
        'confidence_intervals': conf_int
    }

# =============================================================================
# 4. VISUALIZATION
# =============================================================================

def plot_arimax_results(arimax_results, save_plot=True):
    """Plot ARIMAX model results"""
    st.write("\n📊 Creating ARIMAX prediction plots...")

    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Predictions vs Actual
        ax1 = axes[0, 0]
        ax1.plot(arimax_results['actual'], label='Actual', color='blue', alpha=0.8, linewidth=2)
        ax1.plot(arimax_results['predictions'], label='ARIMAX Predictions', color='red', alpha=0.8, linewidth=2)

        # Add confidence intervals if available
        if arimax_results['confidence_intervals'] is not None:
            conf_int = arimax_results['confidence_intervals']
            ax1.fill_between(range(len(arimax_results['predictions'])),
                           conf_int[:, 0], conf_int[:, 1],
                           alpha=0.3, color='red', label='95% Confidence')

        ax1.set_title('ARIMAX: Predictions vs Actual', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Risk Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Scatter plot
        ax2 = axes[0, 1]
        ax2.scatter(arimax_results['actual'], arimax_results['predictions'], alpha=0.6, color='red', s=50)

        # Perfect prediction line
        min_val = min(arimax_results['actual'].min(), arimax_results['predictions'].min())
        max_val = max(arimax_results['actual'].max(), arimax_results['predictions'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.8)

        ax2.set_xlabel('Actual Values')
        ax2.set_ylabel('Predicted Values')
        ax2.set_title('ARIMAX: Actual vs Predicted Scatter', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Add R² to scatter plot
        r2 = arimax_results['metrics'].get('R²', np.nan)
        if not np.isnan(r2):
            ax2.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax2.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=12)

        # Plot 3: Residuals
        ax3 = axes[1, 0]
        residuals = arimax_results['actual'] - arimax_results['predictions']
        ax3.scatter(arimax_results['predictions'], residuals, alpha=0.6, color='green', s=50)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax3.set_xlabel('Predicted Values')
        ax3.set_ylabel('Residuals')
        ax3.set_title('ARIMAX: Residual Plot', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Performance metrics bar chart
        ax4 = axes[1, 1]
        metrics = arimax_results['metrics']
        metric_names = []
        metric_values = []

        display_metrics = ['MAE', 'RMSE', 'R²', 'MAPE', 'Directional_Accuracy']
        for name in display_metrics:
            if name in metrics and not np.isnan(metrics[name]):
                metric_names.append(name)
                metric_values.append(metrics[name])

        if metric_names:
            bars = ax4.bar(metric_names, metric_values,
                          color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'][:len(metric_names)])
            ax4.set_title('ARIMAX: Performance Metrics', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Value')
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(metric_values),
                        f'{value:.3f}', ha='center', va='bottom', fontsize=10)

        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plot:
            plt.savefig('results/arimax_results.png', dpi=300, bbox_inches='tight')
            st.write("✅ ARIMAX plots saved to 'results/arimax_results.png'")

        st.pyplot(plt.gcf())

    except Exception as e:
        st.write(f"⚠️  Could not create ARIMAX plots: {str(e)}")

# =============================================================================
# 5. MAIN PIPELINE FUNCTION
# =============================================================================

def run_arimax_pipeline(arimax_processed):
    """
    Complete pipeline for ARIMAX model training using YOUR processed data

    Args:
        arimax_processed: Output from your preprocessing pipeline
    """
    st.write("\n" + "="*60)
    st.write("🚀 STARTING ARIMAX MODEL PIPELINE WITH YOUR DATA")
    st.write("="*60)

    if arimax_processed is None:
        st.write("❌ No processed data provided!")
        return None

    try:
        # Prepare data splits
        st.write("\n📊 PREPARING DATA SPLITS...")
        arimax_splits = prepare_arimax_train_test_split(arimax_processed)

        if arimax_splits is None:
            st.write("❌ Failed to create data splits!")
            return None

        # Initialize and train ARIMAX model
        st.write("\n🏋️‍♂️ INITIALIZING ARIMAX MODEL...")
        arimax_model = ARIMAXModel()

        # Train model
        fitted_arimax = arimax_model.fit(
            arimax_splits['train'],
            arimax_splits['val'],
            arimax_splits['features'],
            arimax_splits['target']
        )

        if fitted_arimax is not None:
            # Evaluate model
            arimax_results = evaluate_arimax_model(
                arimax_model,
                arimax_splits,
                arimax_splits['scalers']['target_scaler']
            )

            if arimax_results is not None:
                # Plot results
                plot_arimax_results(arimax_results)

                # Save model
                arimax_model.save_model('saved_models/arimax_model.pkl')

                # Save detailed results
                results_data = {
                    'timestamp': datetime.now().isoformat(),
                    'best_params': arimax_model.best_params,
                    'metrics': arimax_results['metrics'],
                    'model_summary': {
                        'AIC': float(fitted_arimax.aic),
                        'BIC': float(fitted_arimax.bic),
                        'Log_Likelihood': float(fitted_arimax.llf)
                    },
                    'data_info': {
                        'n_features': len(arimax_splits['features']),
                        'n_train': len(arimax_splits['train']),
                        'n_test': len(arimax_splits['test']),
                        'features_used': arimax_splits['features']
                    },
                    'stationarity_info': arimax_splits.get('stationary_info', {})
                }

                # Save results to JSON
                import json

                def convert_numpy(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif pd.isna(obj):
                        return None
                    return obj

                with open('results/arimax_results.json', 'w') as f:
                    clean_results = json.loads(json.dumps(results_data, default=convert_numpy))
                    json.dump(clean_results, f, indent=2)

                st.write("\n" + "="*60)
                st.write("🎉 ARIMAX PIPELINE COMPLETED SUCCESSFULLY!")
                st.write("="*60)
                st.write("📊 What was accomplished:")
                st.write("   ✅ ARIMAX model trained with your processed data")
                st.write("   ✅ Hyperparameters optimized using grid search")
                st.write("   ✅ Model evaluated on test data")
                st.write("   ✅ Results visualized and saved")
                st.write("   ✅ Model saved for future use")

                st.write("\n📁 Files created:")
                st.write("   💾 saved_models/arimax_model.pkl - Trained model")
                st.write("   📊 results/arimax_results.json - Performance metrics")
                st.write("   📈 results/arimax_results.png - Visualization plots")

                return {
                    'model': arimax_model,
                    'results': arimax_results,
                    'splits': arimax_splits
                }
            else:
                st.write("❌ ARIMAX model evaluation failed")
                return None
        else:
            st.write("❌ ARIMAX model training failed")
            return None

    except Exception as e:
        st.write(f"❌ ARIMAX pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# 6. UTILITY FUNCTION TO LOAD SAVED MODEL
# =============================================================================

def load_arimax_model(filepath='saved_models/arimax_model.pkl'):
    """Load a saved ARIMAX model"""
    try:
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        # Create new ARIMAXModel instance
        model = ARIMAXModel()
        model.fitted_model = model_data['fitted_model']
        model.best_params = model_data['best_params']
        model.is_fitted = model_data['is_fitted']

        st.write(f"✅ ARIMAX model loaded from {filepath}")
        st.write(f"   Best parameters: {model.best_params}")

        return model
    except Exception as e:
        st.write(f"❌ Error loading ARIMAX model: {str(e)}")
        return None

# =============================================================================
# 7. USAGE INSTRUCTIONS
# =============================================================================

def main():
    """
    Main function showing how to use this pipeline
    """
    st.write("="*70)
    st.write("🚀 ARIMAX MODEL TRAINING PIPELINE")
    st.write("   Fixed to work with your processed data")
    st.write("="*70)

    st.write("\n📋 USAGE INSTRUCTIONS:")
    st.write("1. First run your preprocessing pipeline:")
    st.write("   arimax_processed, lstm_processed = run_fixed_preprocessing()")
    st.write()
    st.write("2. Then train ARIMAX model:")
    st.write("   result = run_arimax_pipeline(arimax_processed)")
    st.write()
    st.write("3. To load saved model later:")
    st.write("   model = load_arimax_model('saved_models/arimax_model.pkl')")
    st.write()

    st.write("🔧 This pipeline will:")
    st.write("   ✅ Use your stationary processed data")
    st.write("   ✅ Handle all data types properly")
    st.write("   ✅ Optimize ARIMAX parameters")
    st.write("   ✅ Evaluate model performance")
    st.write("   ✅ Save model and results")
    st.write("   ✅ Create visualizations")

if True:
    main()


# In[ ]:


# Then train ARIMAX with your processed data
result = run_arimax_pipeline(arimax_processed)


# In[ ]:


# LSTM Model Training Pipeline for Financial Risk Forecasting
# Uses actual preprocessed data from your preprocessing pipeline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import os
import json
import pickle

# ML and evaluation libraries
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Deep learning libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import regularizers, optimizers

# Hyperparameter tuning
import optuna

# Configure GPU if available
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        st.write(f"✅ GPU configured: {len(gpus)} GPU(s)")
    else:
        st.write("📊 Running on CPU")
except:
    st.write("📊 Running on CPU")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create directories
os.makedirs('saved_models', exist_ok=True)
os.makedirs('results', exist_ok=True)

st.write("=== LSTM MODEL TRAINING WITH ACTUAL DATA ===")
st.write("="*60)

# =============================================================================
# 1. DATA PREPARATION FOR LSTM
# =============================================================================

def create_lstm_sequences(data, features, target, sequence_length=12):
    """Create sequences for LSTM training from your processed data"""
    st.write(f"📊 Creating LSTM sequences with length {sequence_length}...")

    # Extract feature and target data
    X_data = data[features].values
    y_data = data[target].values

    # Handle any remaining NaN values
    X_data = np.nan_to_num(X_data, nan=0.0)
    y_data = np.nan_to_num(y_data, nan=0.0)

    X_sequences = []
    y_sequences = []

    # Create sequences
    for i in range(sequence_length, len(data)):
        X_sequences.append(X_data[i-sequence_length:i])
        y_sequences.append(y_data[i])

    X_sequences = np.array(X_sequences, dtype=np.float32)
    y_sequences = np.array(y_sequences, dtype=np.float32)

    st.write(f"✅ Created {len(X_sequences)} sequences")
    st.write(f"📊 Sequence shape: {X_sequences.shape}")
    st.write(f"📊 Target shape: {y_sequences.shape}")

    return X_sequences, y_sequences

def prepare_lstm_data_splits(lstm_processed, sequence_length=12):
    """Prepare train/validation/test splits using your preprocessed data"""
    st.write("\n📊 PREPARING LSTM DATA SPLITS...")
    st.write("-" * 50)

    data = lstm_processed['data']
    features = lstm_processed['features']
    target = lstm_processed['target']

    st.write(f"📋 Using {len(features)} features: {features[:5]}..." if len(features) > 5 else f"📋 Features: {features}")
    st.write(f"🎯 Target: {target}")
    st.write(f"📊 Total data points: {len(data)}")

    # Create sequences
    X_sequences, y_sequences = create_lstm_sequences(data, features, target, sequence_length)

    if len(X_sequences) < 50:
        st.write("⚠️ Warning: Very few sequences available. Consider reducing sequence_length.")

    # Time series split (chronological order preserved)
    n_samples = len(X_sequences)
    train_size = int(n_samples * 0.7)
    val_size = int(n_samples * 0.15)
    test_size = n_samples - train_size - val_size

    # Split data
    X_train = X_sequences[:train_size]
    y_train = y_sequences[:train_size]

    X_val = X_sequences[train_size:train_size+val_size]
    y_val = y_sequences[train_size:train_size+val_size]

    X_test = X_sequences[train_size+val_size:]
    y_test = y_sequences[train_size+val_size:]

    st.write(f"✅ Training set: {len(X_train)} sequences ({len(X_train)/n_samples*100:.1f}%)")
    st.write(f"✅ Validation set: {len(X_val)} sequences ({len(X_val)/n_samples*100:.1f}%)")
    st.write(f"✅ Test set: {len(X_test)} sequences ({len(X_test)/n_samples*100:.1f}%)")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'features': features,
        'target': target,
        'sequence_length': sequence_length,
        'scalers': {
            'feature_scaler': lstm_processed.get('feature_scaler'),
            'target_scaler': lstm_processed.get('target_scaler')
        }
    }

# =============================================================================
# 2. ENHANCED LSTM MODEL CLASS
# =============================================================================

class EnhancedLSTMModel:
    def __init__(self):
        self.model = None
        self.best_params = None
        self.is_fitted = False
        self.history = None

    def build_model(self, input_shape, hyperparams):
        """Build optimized LSTM model"""
        model = Sequential(name='Financial_Risk_LSTM')

        # First LSTM layer
        model.add(LSTM(
            units=hyperparams['lstm_units_1'],
            return_sequences=True,
            input_shape=input_shape,
            dropout=hyperparams['dropout_rate'],
            recurrent_dropout=hyperparams['recurrent_dropout'],
            name='LSTM_1'
        ))
        model.add(BatchNormalization())

        # Second LSTM layer
        model.add(LSTM(
            units=hyperparams['lstm_units_2'],
            return_sequences=False,
            dropout=hyperparams['dropout_rate'],
            recurrent_dropout=hyperparams['recurrent_dropout'],
            name='LSTM_2'
        ))
        model.add(BatchNormalization())
        model.add(Dropout(hyperparams['dropout_rate']))

        # Dense layers
        model.add(Dense(
            hyperparams['dense_units'],
            activation='relu',
            kernel_regularizer=regularizers.l2(hyperparams['l2_reg']),
            name='Dense_1'
        ))
        model.add(Dropout(hyperparams['dropout_rate']))

        # Second dense layer (smaller)
        model.add(Dense(
            hyperparams['dense_units'] // 2,
            activation='relu',
            kernel_regularizer=regularizers.l2(hyperparams['l2_reg']),
            name='Dense_2'
        ))
        model.add(Dropout(hyperparams['dropout_rate']))

        # Output layer
        model.add(Dense(1, activation='linear', name='Output'))

        # Compile with optimized settings
        optimizer = optimizers.Adam(
            learning_rate=hyperparams['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )

        model.compile(
            optimizer=optimizer,
            loss='huber',  # More robust to outliers than MSE
            metrics=['mae', 'mse']
        )

        return model

    def tune_hyperparameters(self, X_train, y_train, X_val, y_val, n_trials=25):
        """Optimized hyperparameter tuning"""
        st.write("\n🎯 TUNING LSTM HYPERPARAMETERS...")
        st.write("-" * 40)

        def objective(trial):
            try:
                # Suggest hyperparameters with better ranges
                hyperparams = {
                    'lstm_units_1': trial.suggest_int('lstm_units_1', 64, 128, step=16),
                    'lstm_units_2': trial.suggest_int('lstm_units_2', 32, 96, step=16),
                    'dense_units': trial.suggest_int('dense_units', 32, 128, step=16),
                    'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.5),
                    'recurrent_dropout': trial.suggest_float('recurrent_dropout', 0.1, 0.3),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                    'l2_reg': trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True),
                    'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
                }

                # Build model
                model = self.build_model(
                    input_shape=(X_train.shape[1], X_train.shape[2]),
                    hyperparams=hyperparams
                )

                # Callbacks
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=8,
                    restore_best_weights=True,
                    verbose=0
                )

                reduce_lr = ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=4,
                    min_lr=1e-7,
                    verbose=0
                )

                # Train model
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=40,
                    batch_size=hyperparams['batch_size'],
                    callbacks=[early_stopping, reduce_lr],
                    verbose=0
                )

                # Return best validation loss
                val_loss = min(history.history['val_loss'])

                # Clean up
                del model
                tf.keras.backend.clear_session()

                return val_loss

            except Exception as e:
                st.write(f"   ⚠️ Trial failed: {str(e)}")
                tf.keras.backend.clear_session()
                return float('inf')

        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.best_params = study.best_params

        st.write(f"✅ Best hyperparameters found:")
        for param, value in self.best_params.items():
            st.write(f"   📊 {param}: {value}")
        st.write(f"   📊 Best validation loss: {study.best_value:.6f}")

        return self.best_params

    def fit(self, X_train, y_train, X_val, y_val, epochs=100):
        """Train LSTM model with your data"""
        st.write("\n🏋️‍♂️ TRAINING LSTM MODEL...")
        st.write("-" * 40)

        if self.best_params is None:
            self.tune_hyperparameters(X_train, y_train, X_val, y_val)

        # Build final model
        self.model = self.build_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            hyperparams=self.best_params
        )

        st.write(f"📊 Model Summary:")
        st.write(f"   LSTM: {self.best_params['lstm_units_1']} → {self.best_params['lstm_units_2']} units")
        st.write(f"   Dense: {self.best_params['dense_units']} → {self.best_params['dense_units']//2} units")
        st.write(f"   Dropout: {self.best_params['dropout_rate']:.3f}")
        st.write(f"   Learning Rate: {self.best_params['learning_rate']:.6f}")
        st.write(f"   Batch Size: {self.best_params['batch_size']}")

        # Enhanced callbacks
        callbacks_list = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'saved_models/lstm_best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]

        # Train model
        st.write("🚀 Starting training...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=self.best_params['batch_size'],
            callbacks=callbacks_list,
            verbose=1
        )

        self.is_fitted = True

        # Training summary
        best_epoch = np.argmin(self.history.history['val_loss']) + 1
        best_val_loss = min(self.history.history['val_loss'])
        best_train_loss = self.history.history['loss'][best_epoch - 1]

        st.write(f"\n✅ Training completed!")
        st.write(f"   📊 Best epoch: {best_epoch}")
        st.write(f"   📊 Final training loss: {best_train_loss:.6f}")
        st.write(f"   📊 Final validation loss: {best_val_loss:.6f}")

        return self.history

    def predict(self, X_test):
        """Make predictions"""
        if not self.is_fitted or self.model is None:
            st.write("❌ Model not fitted yet!")
            return None

        try:
            predictions = self.model.predict(X_test, verbose=0)
            return predictions.flatten()
        except Exception as e:
            st.write(f"❌ Error making predictions: {str(e)}")
            return None

    def save_model(self, base_path='saved_models/lstm_financial_model'):
        """Save complete model"""
        try:
            # Save Keras model
            keras_path = f"{base_path}.keras"
            self.model.save(keras_path)

            # Save metadata
            metadata = {
                'best_params': self.best_params,
                'is_fitted': self.is_fitted,
                'history': self.history.history if self.history else None,
                'model_type': 'LSTM_Financial_Risk',
                'timestamp': datetime.now().isoformat()
            }

            metadata_path = f"{base_path}_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)

            st.write(f"✅ Model saved:")
            st.write(f"   🧠 Keras model: {keras_path}")
            st.write(f"   📋 Metadata: {metadata_path}")

        except Exception as e:
            st.write(f"❌ Error saving model: {str(e)}")

# =============================================================================
# 3. MODEL EVALUATION
# =============================================================================

def calculate_comprehensive_metrics(y_true, y_pred, target_scaler=None):
    """Calculate comprehensive evaluation metrics"""

    # Inverse transform if scaler available
    if target_scaler is not None:
        try:
            y_true_orig = target_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
            y_pred_orig = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        except:
            y_true_orig = y_true
            y_pred_orig = y_pred
    else:
        y_true_orig = y_true
        y_pred_orig = y_pred

    # Remove NaN values
    mask = ~(np.isnan(y_true_orig) | np.isnan(y_pred_orig))
    y_true_clean = y_true_orig[mask]
    y_pred_clean = y_pred_orig[mask]

    if len(y_true_clean) == 0:
        st.write("⚠️ No valid predictions for evaluation")
        return {}

    # Calculate metrics
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mse = mean_squared_error(y_true_clean, y_pred_clean)

    try:
        r2 = r2_score(y_true_clean, y_pred_clean)
    except:
        r2 = np.nan

    # MAPE
    nonzero_mask = y_true_clean != 0
    if np.sum(nonzero_mask) > 0:
        mape = np.mean(np.abs((y_true_clean[nonzero_mask] - y_pred_clean[nonzero_mask])
                             / y_true_clean[nonzero_mask])) * 100
    else:
        mape = np.nan

    # Directional accuracy
    if len(y_true_clean) > 1:
        actual_changes = np.diff(y_true_clean)
        pred_changes = np.diff(y_pred_clean)

        # Binary directional accuracy
        actual_up = actual_changes > 0
        pred_up = pred_changes > 0
        directional_accuracy = np.mean(actual_up == pred_up) * 100
    else:
        directional_accuracy = np.nan

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MSE': mse,
        'R²': r2,
        'MAPE': mape,
        'Directional_Accuracy': directional_accuracy,
        'N_Predictions': len(y_true_clean)
    }

    return metrics

def evaluate_lstm_model(model, test_data):
    """Evaluate LSTM model comprehensively"""
    st.write("\n📊 EVALUATING LSTM MODEL...")
    st.write("-" * 50)

    X_test = test_data['X_test']
    y_true = test_data['y_test']
    target_scaler = test_data['scalers']['target_scaler']

    # Get predictions
    predictions = model.predict(X_test)

    if predictions is None:
        return None

    # Calculate metrics
    metrics = calculate_comprehensive_metrics(y_true, predictions, target_scaler)

    # Print results
    st.write("📈 LSTM Performance Metrics:")
    for metric, value in metrics.items():
        if not np.isnan(value) and metric != 'N_Predictions':
            if metric in ['MAE', 'RMSE', 'MSE']:
                st.write(f"   {metric}: {value:.6f}")
            elif metric in ['R²']:
                st.write(f"   {metric}: {value:.4f}")
            elif metric in ['MAPE', 'Directional_Accuracy']:
                st.write(f"   {metric}: {value:.2f}%")

    st.write(f"   📊 Total predictions: {metrics['N_Predictions']}")

    # Prepare return data
    if target_scaler is not None:
        try:
            y_true_display = target_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
            predictions_display = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        except:
            y_true_display = y_true
            predictions_display = predictions
    else:
        y_true_display = y_true
        predictions_display = predictions

    return {
        'predictions': predictions_display,
        'actual': y_true_display,
        'metrics': metrics
    }

# =============================================================================
# 4. VISUALIZATION
# =============================================================================

def create_lstm_plots(lstm_results, model_history=None, save_path='results/lstm_results.png'):
    """Create comprehensive LSTM result plots"""
    st.write("\n📊 Creating LSTM result visualizations...")

    try:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('LSTM Model Results - Financial Risk Forecasting', fontsize=16, fontweight='bold')

        # Plot 1: Predictions vs Actual
        ax1 = axes[0, 0]
        ax1.plot(lstm_results['actual'], label='Actual', color='blue', linewidth=2, alpha=0.8)
        ax1.plot(lstm_results['predictions'], label='Predicted', color='red', linewidth=2, alpha=0.8)
        ax1.set_title('Predictions vs Actual Values', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Risk Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Scatter Plot
        ax2 = axes[0, 1]
        ax2.scatter(lstm_results['actual'], lstm_results['predictions'],
                   alpha=0.6, color='red', s=30)

        min_val = min(lstm_results['actual'].min(), lstm_results['predictions'].min())
        max_val = max(lstm_results['actual'].max(), lstm_results['predictions'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.8)

        ax2.set_xlabel('Actual Values')
        ax2.set_ylabel('Predicted Values')
        ax2.set_title('Actual vs Predicted Scatter Plot', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Add R² to scatter plot
        r2 = lstm_results['metrics'].get('R²', np.nan)
        if not np.isnan(r2):
            ax2.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax2.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Plot 3: Residuals
        ax3 = axes[0, 2]
        residuals = lstm_results['actual'] - lstm_results['predictions']
        ax3.scatter(lstm_results['predictions'], residuals, alpha=0.6, color='green', s=30)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax3.set_xlabel('Predicted Values')
        ax3.set_ylabel('Residuals')
        ax3.set_title('Residual Plot', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Metrics Bar Chart
        ax4 = axes[1, 0]
        metrics = lstm_results['metrics']
        metric_names = []
        metric_values = []

        for name, value in metrics.items():
            if name not in ['N_Predictions'] and not np.isnan(value):
                metric_names.append(name)
                metric_values.append(value)

        if metric_names:
            bars = ax4.bar(metric_names, metric_values,
                          color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
            ax4.set_title('Performance Metrics', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Value')
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9)

        ax4.grid(True, alpha=0.3)

        # Plot 5: Training History
        if model_history is not None:
            ax5 = axes[1, 1]
            epochs = range(1, len(model_history.history['loss']) + 1)
            ax5.plot(epochs, model_history.history['loss'], 'b-', label='Training Loss', linewidth=2)
            ax5.plot(epochs, model_history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            ax5.set_title('Training History', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Loss')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

            # Plot 6: MAE History
            ax6 = axes[1, 2]
            ax6.plot(epochs, model_history.history['mae'], 'g-', label='Training MAE', linewidth=2)
            ax6.plot(epochs, model_history.history['val_mae'], 'orange', label='Validation MAE', linewidth=2)
            ax6.set_title('MAE Learning Curve', fontsize=12, fontweight='bold')
            ax6.set_xlabel('Epoch')
            ax6.set_ylabel('MAE')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        else:
            # If no history, create prediction error distribution
            ax5 = axes[1, 1]
            errors = lstm_results['actual'] - lstm_results['predictions']
            ax5.hist(errors, bins=30, alpha=0.7, color='purple', edgecolor='black')
            ax5.set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Prediction Error')
            ax5.set_ylabel('Frequency')
            ax5.grid(True, alpha=0.3)

            # Summary stats
            ax6 = axes[1, 2]
            ax6.axis('off')
            stats_text = f"""
Model Performance Summary

Total Predictions: {len(lstm_results['actual'])}
MAE: {lstm_results['metrics'].get('MAE', 0):.6f}
RMSE: {lstm_results['metrics'].get('RMSE', 0):.6f}
R²: {lstm_results['metrics'].get('R²', 0):.4f}
MAPE: {lstm_results['metrics'].get('MAPE', 0):.2f}%
Directional Accuracy: {lstm_results['metrics'].get('Directional_Accuracy', 0):.2f}%
            """
            ax6.text(0.1, 0.5, stats_text, transform=ax6.transAxes, fontsize=11,
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        st.pyplot(plt.gcf())

        st.write(f"✅ Plots saved to {save_path}")

    except Exception as e:
        st.write(f"⚠️ Could not create plots: {str(e)}")

# =============================================================================
# 5. MAIN LSTM PIPELINE
# =============================================================================

def run_lstm_training_pipeline(lstm_processed, sequence_length=12, epochs=100):
    """Main pipeline for LSTM training using your preprocessed data"""
    st.write("\n" + "="*60)
    st.write("🚀 LSTM TRAINING PIPELINE WITH YOUR DATA")
    st.write("="*60)

    try:
        # Step 1: Prepare data splits
        st.write("📊 Step 1: Preparing data splits...")
        lstm_splits = prepare_lstm_data_splits(lstm_processed, sequence_length)

        # Step 2: Initialize and train model
        st.write("\n🧠 Step 2: Training LSTM model...")
        lstm_model = EnhancedLSTMModel()

        history = lstm_model.fit(
            lstm_splits['X_train'], lstm_splits['y_train'],
            lstm_splits['X_val'], lstm_splits['y_val'],
            epochs=epochs
        )

        # Step 3: Evaluate model
        st.write("\n📊 Step 3: Evaluating model...")
        lstm_results = evaluate_lstm_model(lstm_model, lstm_splits)

        if lstm_results is None:
            st.write("❌ Model evaluation failed")
            return None

        # Step 4: Create visualizations
        st.write("\n📈 Step 4: Creating visualizations...")
        create_lstm_plots(lstm_results, history)

        # Step 5: Save model
        st.write("\n💾 Step 5: Saving model...")
        lstm_model.save_model()

        # Step 6: Save results
        st.write("\n📋 Step 6: Saving results...")
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'Enhanced_LSTM',
            'sequence_length': sequence_length,
            'total_features': len(lstm_splits['features']),
            'training_samples': len(lstm_splits['X_train']),
            'test_samples': len(lstm_splits['X_test']),
            'best_hyperparameters': lstm_model.best_params,
            'performance_metrics': lstm_results['metrics'],
            'training_epochs_completed': len(history.history['loss'])
        }

        # Save results
        with open('results/lstm_training_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)

        st.write("\n" + "="*60)
        st.write("🎉 LSTM TRAINING PIPELINE COMPLETED!")
        st.write("="*60)
        st.write("✅ Model trained and saved successfully")
        st.write("✅ Evaluation metrics calculated")
        st.write("✅ Visualizations created")
        st.write("✅ Results saved")

        st.write("\n📁 Generated Files:")
        st.write("   🧠 saved_models/lstm_financial_model.keras - Trained model")
        st.write("   📋 saved_models/lstm_financial_model_metadata.pkl - Model metadata")
        st.write("   📊 results/lstm_training_results.json - Performance metrics")
        st.write("   📈 results/lstm_results.png - Visualization plots")

        # Print key performance metrics
        st.write("\n📊 Key Performance Metrics:")
        metrics = lstm_results['metrics']
        for metric, value in metrics.items():
            if not np.isnan(value) and metric != 'N_Predictions':
                if metric in ['MAE', 'RMSE', 'MSE']:
                    st.write(f"   {metric}: {value:.6f}")
                elif metric in ['R²']:
                    st.write(f"   {metric}: {value:.4f}")
                elif metric in ['MAPE', 'Directional_Accuracy']:
                    st.write(f"   {metric}: {value:.2f}%")

        return {
            'model': lstm_model,
            'results': lstm_results,
            'history': history,
            'splits': lstm_splits,
            'summary': results_summary
        }

    except Exception as e:
        st.write(f"❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# 6. UTILITY FUNCTIONS
# =============================================================================

def load_lstm_model(model_path='saved_models/lstm_financial_model'):
    """Load a saved LSTM model"""
    try:
        # Load Keras model
        keras_path = f"{model_path}.keras"
        model_obj = load_model(keras_path)

        # Load metadata
        metadata_path = f"{model_path}_metadata.pkl"
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        # Create model instance
        lstm_model = EnhancedLSTMModel()
        lstm_model.model = model_obj
        lstm_model.best_params = metadata['best_params']
        lstm_model.is_fitted = metadata['is_fitted']

        st.write(f"✅ LSTM model loaded successfully from {model_path}")
        return lstm_model

    except Exception as e:
        st.write(f"❌ Failed to load model: {str(e)}")
        return None

def predict_with_lstm(model, X_new, target_scaler=None):
    """Make predictions with loaded LSTM model"""
    try:
        predictions_scaled = model.predict(X_new)

        if target_scaler is not None:
            predictions = target_scaler.inverse_transform(
                predictions_scaled.reshape(-1, 1)
            ).flatten()
        else:
            predictions = predictions_scaled

        return predictions

    except Exception as e:
        st.write(f"❌ Prediction failed: {str(e)}")
        return None

# =============================================================================
# 7. EXECUTION FUNCTION WITH YOUR PREPROCESSED DATA
# =============================================================================

def main_lstm_training(lstm_processed):
    """
    Main function to train LSTM with your preprocessed data

    Parameters:
    lstm_processed: The output from your preprocessing pipeline
                   (lstm_processed from run_fixed_preprocessing())
    """
    st.write("="*70)
    st.write("🚀 LSTM FINANCIAL RISK FORECASTING")
    st.write("   Training with your preprocessed data")
    st.write("="*70)

    if lstm_processed is None:
        st.write("❌ Error: lstm_processed data is None!")
        st.write("   Please run your preprocessing pipeline first:")
        st.write("   arimax_processed, lstm_processed = run_fixed_preprocessing()")
        return None

    # Validate preprocessed data
    required_keys = ['data', 'features', 'target']
    missing_keys = [key for key in required_keys if key not in lstm_processed]
    if missing_keys:
        st.write(f"❌ Error: Missing keys in lstm_processed: {missing_keys}")
        return None

    st.write(f"✅ Received preprocessed data:")
    st.write(f"   📊 Data shape: {lstm_processed['data'].shape}")
    st.write(f"   📋 Features: {len(lstm_processed['features'])}")
    st.write(f"   🎯 Target: {lstm_processed['target']}")

    # Run the complete pipeline
    result = run_lstm_training_pipeline(
        lstm_processed,
        sequence_length=12,  # Adjusted for better performance
        epochs=80  # Good balance of training time and performance
    )

    if result is not None:
        st.write("\n🎯 TRAINING COMPLETED SUCCESSFULLY!")
        st.write(f"   Final validation loss: {min(result['history'].history['val_loss']):.6f}")
        st.write(f"   Test R²: {result['results']['metrics'].get('R²', 0):.4f}")
        st.write(f"   Test RMSE: {result['results']['metrics'].get('RMSE', 0):.6f}")
        st.write(f"   Directional Accuracy: {result['results']['metrics'].get('Directional_Accuracy', 0):.2f}%")

    return result

# =============================================================================
# 8. EXAMPLE USAGE
# =============================================================================

if True:
    st.write("🔧 LSTM Training Pipeline Ready!")
    st.write("\nTo use with your preprocessed data:")
    st.write("1. First run your preprocessing: arimax_processed, lstm_processed = run_fixed_preprocessing()")
    st.write("2. Then run: result = main_lstm_training(lstm_processed)")
    st.write("\nThe pipeline will:")
    st.write("   ✅ Use your actual preprocessed and feature-engineered data")
    st.write("   ✅ Create LSTM sequences for time series forecasting")
    st.write("   ✅ Perform hyperparameter tuning for optimal performance")
    st.write("   ✅ Train an enhanced LSTM model")
    st.write("   ✅ Evaluate performance with comprehensive metrics")
    st.write("   ✅ Generate visualization plots")
    st.write("   ✅ Save the trained model and results")

    # Example of how to run (uncomment when you have your data ready):
    """
    # Run your preprocessing pipeline first
    # External import removed for Streamlit deployment
    arimax_processed, lstm_processed = run_fixed_preprocessing()

    # Train LSTM with your data
    if lstm_processed is not None:
        lstm_result = main_lstm_training(lstm_processed)
    """


# In[ ]:


# 2. Then train LSTM with your actual data
lstm_result = main_lstm_training(lstm_processed)


# ##Perfomance Comparison

# In[ ]:


# Model Performance Comparison: ARIMAX vs LSTM
# Comprehensive visualization comparison

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def compare_model_performance(arimax_result, lstm_result, save_path='results/model_comparison.png'):
    """
    Comprehensive comparison of ARIMAX vs LSTM model performance

    Parameters:
    arimax_result: Result from run_arimax_pipeline()
    lstm_result: Result from main_lstm_training()
    save_path: Path to save the comparison plot
    """

    st.write("📊 Creating comprehensive model comparison...")
    st.write("-" * 60)

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('ARIMAX vs LSTM Model Performance Comparison\nFinancial Risk Forecasting',
                 fontsize=18, fontweight='bold', y=0.98)

    # Extract data from results
    try:
        # ARIMAX data
        arimax_actual = arimax_result['results']['actual']
        arimax_pred = arimax_result['results']['predictions']
        arimax_metrics = arimax_result['results']['metrics']

        # LSTM data
        lstm_actual = lstm_result['results']['actual']
        lstm_pred = lstm_result['results']['predictions']
        lstm_metrics = lstm_result['results']['metrics']

        st.write(f"✅ ARIMAX: {len(arimax_actual)} predictions")
        st.write(f"✅ LSTM: {len(lstm_actual)} predictions")

    except Exception as e:
        st.write(f"❌ Error extracting data: {str(e)}")
        return None

    # 1. Time Series Predictions Comparison (Top row, spanning 2 columns)
    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=4)

    # Align lengths for fair comparison
    min_len = min(len(arimax_actual), len(lstm_actual))
    arimax_actual_plot = arimax_actual[-min_len:]
    arimax_pred_plot = arimax_pred[-min_len:]
    lstm_actual_plot = lstm_actual[-min_len:]
    lstm_pred_plot = lstm_pred[-min_len:]

    time_steps = np.arange(min_len)

    ax1.plot(time_steps, arimax_actual_plot, 'b-', label='Actual Values', linewidth=2.5, alpha=0.8)
    ax1.plot(time_steps, arimax_pred_plot, 'r--', label='ARIMAX Predictions', linewidth=2, alpha=0.9)
    ax1.plot(time_steps, lstm_pred_plot, 'g:', label='LSTM Predictions', linewidth=2, alpha=0.9)

    ax1.set_title('Model Predictions Comparison Over Time', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Time Steps', fontsize=12)
    ax1.set_ylabel('Risk Score', fontsize=12)
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Add R² values as text
    arimax_r2 = arimax_metrics.get('R²', np.nan)
    lstm_r2 = lstm_metrics.get('R²', np.nan)

    ax1.text(0.02, 0.98, f'ARIMAX R²: {arimax_r2:.4f}', transform=ax1.transAxes,
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.3), fontsize=10, verticalalignment='top')
    ax1.text(0.02, 0.90, f'LSTM R²: {lstm_r2:.4f}', transform=ax1.transAxes,
            bbox=dict(boxstyle='round', facecolor='green', alpha=0.3), fontsize=10, verticalalignment='top')

    # 2. Scatter Plots Comparison
    ax2 = plt.subplot2grid((4, 4), (1, 0), colspan=2)
    ax2.scatter(arimax_actual, arimax_pred, alpha=0.6, color='red', s=40, label='ARIMAX')

    # Perfect prediction line
    min_val = min(arimax_actual.min(), arimax_pred.min())
    max_val = max(arimax_actual.max(), arimax_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.8)

    ax2.set_xlabel('Actual Values', fontsize=11)
    ax2.set_ylabel('Predicted Values', fontsize=11)
    ax2.set_title('ARIMAX: Actual vs Predicted', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = plt.subplot2grid((4, 4), (1, 2), colspan=2)
    ax3.scatter(lstm_actual, lstm_pred, alpha=0.6, color='green', s=40, label='LSTM')

    min_val = min(lstm_actual.min(), lstm_pred.min())
    max_val = max(lstm_actual.max(), lstm_pred.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.8)

    ax3.set_xlabel('Actual Values', fontsize=11)
    ax3.set_ylabel('Predicted Values', fontsize=11)
    ax3.set_title('LSTM: Actual vs Predicted', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 3. Performance Metrics Comparison
    ax4 = plt.subplot2grid((4, 4), (2, 0), colspan=2)

    # Collect common metrics
    metrics_comparison = {}
    common_metrics = ['MAE', 'RMSE', 'R²', 'MAPE']

    arimax_values = []
    lstm_values = []
    metric_names = []

    for metric in common_metrics:
        if metric in arimax_metrics and metric in lstm_metrics:
            arimax_val = arimax_metrics[metric]
            lstm_val = lstm_metrics[metric]

            if not (np.isnan(arimax_val) or np.isnan(lstm_val)):
                arimax_values.append(arimax_val)
                lstm_values.append(lstm_val)
                metric_names.append(metric)

    if metric_names:
        x = np.arange(len(metric_names))
        width = 0.35

        bars1 = ax4.bar(x - width/2, arimax_values, width, label='ARIMAX', color='red', alpha=0.7)
        bars2 = ax4.bar(x + width/2, lstm_values, width, label='LSTM', color='green', alpha=0.7)

        ax4.set_xlabel('Metrics', fontsize=11)
        ax4.set_ylabel('Values', fontsize=11)
        ax4.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metric_names)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, value in zip(bars1, arimax_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(arimax_values + lstm_values),
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)

        for bar, value in zip(bars2, lstm_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(arimax_values + lstm_values),
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)

    # 4. Residuals Comparison
    ax5 = plt.subplot2grid((4, 4), (2, 2), colspan=2)

    arimax_residuals = arimax_actual - arimax_pred
    lstm_residuals = lstm_actual - lstm_pred

    ax5.scatter(arimax_pred, arimax_residuals, alpha=0.6, color='red', s=30, label='ARIMAX Residuals')
    ax5.scatter(lstm_pred, lstm_residuals, alpha=0.6, color='green', s=30, label='LSTM Residuals')
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.8)

    ax5.set_xlabel('Predicted Values', fontsize=11)
    ax5.set_ylabel('Residuals', fontsize=11)
    ax5.set_title('Residuals Comparison', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 5. Error Distribution Comparison
    ax6 = plt.subplot2grid((4, 4), (3, 0), colspan=2)

    ax6.hist(arimax_residuals, bins=30, alpha=0.6, color='red', label='ARIMAX Errors', density=True)
    ax6.hist(lstm_residuals, bins=30, alpha=0.6, color='green', label='LSTM Errors', density=True)

    ax6.set_xlabel('Prediction Errors', fontsize=11)
    ax6.set_ylabel('Density', fontsize=11)
    ax6.set_title('Error Distribution Comparison', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 6. Model Summary Table
    ax7 = plt.subplot2grid((4, 4), (3, 2), colspan=2)
    ax7.axis('off')

    # Create summary table data
    summary_data = []

    # Compare key metrics
    for metric in ['MAE', 'RMSE', 'R²', 'MAPE', 'Directional_Accuracy']:
        arimax_val = arimax_metrics.get(metric, np.nan)
        lstm_val = lstm_metrics.get(metric, np.nan)

        if not np.isnan(arimax_val) and not np.isnan(lstm_val):
            # Determine winner (lower is better for MAE, RMSE, MAPE; higher is better for R², Directional_Accuracy)
            if metric in ['R²', 'Directional_Accuracy']:
                winner = 'ARIMAX' if arimax_val > lstm_val else 'LSTM'
                diff = abs(arimax_val - lstm_val)
            else:
                winner = 'ARIMAX' if arimax_val < lstm_val else 'LSTM'
                diff = abs(arimax_val - lstm_val)

            if metric in ['MAE', 'RMSE']:
                summary_data.append([metric, f'{arimax_val:.6f}', f'{lstm_val:.6f}', winner])
            elif metric == 'R²':
                summary_data.append([metric, f'{arimax_val:.4f}', f'{lstm_val:.4f}', winner])
            else:
                summary_data.append([metric, f'{arimax_val:.2f}%', f'{lstm_val:.2f}%', winner])

    # Add sample counts
    summary_data.append(['Predictions', f'{len(arimax_actual)}', f'{len(lstm_actual)}', '-'])

    # Create table
    table_text = "Model Performance Summary\n" + "="*50 + "\n"
    table_text += f"{'Metric':<20} {'ARIMAX':<12} {'LSTM':<12} {'Better':<8}\n"
    table_text += "-"*50 + "\n"

    for row in summary_data:
        table_text += f"{row[0]:<20} {row[1]:<12} {row[2]:<12} {row[3]:<8}\n"

    # Count wins
    arimax_wins = sum(1 for row in summary_data if row[3] == 'ARIMAX')
    lstm_wins = sum(1 for row in summary_data if row[3] == 'LSTM')

    table_text += "\n" + "="*50 + "\n"
    table_text += f"Overall Performance:\n"
    table_text += f"ARIMAX wins: {arimax_wins} metrics\n"
    table_text += f"LSTM wins: {lstm_wins} metrics\n"

    if arimax_wins > lstm_wins:
        table_text += f"\n🏆 ARIMAX performs better overall!"
    elif lstm_wins > arimax_wins:
        table_text += f"\n🏆 LSTM performs better overall!"
    else:
        table_text += f"\n🤝 Models perform similarly overall!"

    ax7.text(0.05, 0.95, table_text, transform=ax7.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.3)

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    st.write(f"✅ Comprehensive comparison saved to: {save_path}")

    st.pyplot(plt.gcf())

    # Print summary to console
    st.write("\n" + "="*60)
    st.write("📊 MODEL COMPARISON SUMMARY")
    st.write("="*60)

    st.write("\n📈 ARIMAX Model Performance:")
    for metric, value in arimax_metrics.items():
        if not np.isnan(value) and metric != 'N_Predictions':
            if metric in ['MAE', 'RMSE', 'MSE']:
                st.write(f"   {metric}: {value:.6f}")
            elif metric in ['R²']:
                st.write(f"   {metric}: {value:.4f}")
            elif metric in ['MAPE', 'Directional_Accuracy']:
                st.write(f"   {metric}: {value:.2f}%")

    st.write("\n📈 LSTM Model Performance:")
    for metric, value in lstm_metrics.items():
        if not np.isnan(value) and metric != 'N_Predictions':
            if metric in ['MAE', 'RMSE', 'MSE']:
                st.write(f"   {metric}: {value:.6f}")
            elif metric in ['R²']:
                st.write(f"   {metric}: {value:.4f}")
            elif metric in ['MAPE', 'Directional_Accuracy']:
                st.write(f"   {metric}: {value:.2f}%")

    st.write(f"\n🏆 Winner: ARIMAX performs better in {arimax_wins} metrics")
    st.write(f"🥈 LSTM performs better in {lstm_wins} metrics")

    return {
        'arimax_metrics': arimax_metrics,
        'lstm_metrics': lstm_metrics,
        'arimax_wins': arimax_wins,
        'lstm_wins': lstm_wins,
        'comparison_saved': save_path
    }

def quick_comparison_summary(arimax_result, lstm_result):
    """
    Quick text summary of model comparison
    """
    st.write("\n" + "🔥"*60)
    st.write("⚡ QUICK MODEL COMPARISON SUMMARY")
    st.write("🔥"*60)

    try:
        arimax_r2 = arimax_result['results']['metrics'].get('R²', 0)
        lstm_r2 = lstm_result['results']['metrics'].get('R²', 0)

        arimax_mae = arimax_result['results']['metrics'].get('MAE', float('inf'))
        lstm_mae = lstm_result['results']['metrics'].get('MAE', float('inf'))

        arimax_rmse = arimax_result['results']['metrics'].get('RMSE', float('inf'))
        lstm_rmse = lstm_result['results']['metrics'].get('RMSE', float('inf'))

        st.write(f"\n📊 R² Score (Higher = Better):")
        st.write(f"   🔴 ARIMAX: {arimax_r2:.4f}")
        st.write(f"   🟢 LSTM:   {lstm_r2:.4f}")
        st.write(f"   🏆 Winner: {'ARIMAX' if arimax_r2 > lstm_r2 else 'LSTM'}")

        st.write(f"\n📊 MAE (Lower = Better):")
        st.write(f"   🔴 ARIMAX: {arimax_mae:.6f}")
        st.write(f"   🟢 LSTM:   {lstm_mae:.6f}")
        st.write(f"   🏆 Winner: {'ARIMAX' if arimax_mae < lstm_mae else 'LSTM'}")

        st.write(f"\n📊 RMSE (Lower = Better):")
        st.write(f"   🔴 ARIMAX: {arimax_rmse:.6f}")
        st.write(f"   🟢 LSTM:   {lstm_rmse:.6f}")
        st.write(f"   🏆 Winner: {'ARIMAX' if arimax_rmse < lstm_rmse else 'LSTM'}")

        # Overall assessment
        arimax_score = (1 if arimax_r2 > lstm_r2 else 0) + (1 if arimax_mae < lstm_mae else 0) + (1 if arimax_rmse < lstm_rmse else 0)

        st.write(f"\n🏆 OVERALL WINNER:")
        if arimax_score >= 2:
            st.write("   🔴 ARIMAX performs better overall! 🎉")
        else:
            st.write("   🟢 LSTM performs better overall! 🎉")

    except Exception as e:
        st.write(f"❌ Error in quick comparison: {str(e)}")

# Main execution function
def run_model_comparison(arimax_result, lstm_result):
    """
    Run complete model comparison

    Usage:
    comparison_result = run_model_comparison(result, lstm_result)
    """

    st.write("🚀 Starting comprehensive model comparison...")

    # Quick summary first
    quick_comparison_summary(arimax_result, lstm_result)

    # Detailed comparison with visualization
    comparison_result = compare_model_performance(
        arimax_result,
        lstm_result,
        save_path='results/arimax_vs_lstm_comparison.png'
    )

    st.write("\n✅ Model comparison completed!")
    st.write("📁 Check 'results/arimax_vs_lstm_comparison.png' for detailed visualization")

    return comparison_result

# Example usage (uncomment when ready to use):
if True:
    st.write("🔧 Model Comparison Tool Ready!")
    st.write("\nTo compare your models:")
    st.write("1. Make sure you have both results:")
    st.write("   result = run_arimax_pipeline(arimax_processed)")
    st.write("   lstm_result = main_lstm_training(lstm_processed)")
    st.write("2. Run comparison:")
    st.write("   comparison = run_model_comparison(result, lstm_result)")

    """
    # Example usage:
    comparison = run_model_comparison(result, lstm_result)
    """


# In[ ]:


comparison = run_model_comparison(result, lstm_result)


# In[ ]:







try:
    for name in ["df", "datasets"]:
        if name in globals():
            st.subheader(f"Preview: {name}")
            obj = globals()[name]
            import pandas as pd
            if isinstance(obj, pd.DataFrame):
                st.dataframe(obj.head(50), use_container_width=True)
except Exception:
    pass
