import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def perform_eda(df, target_col=None, figsize=(20, 15)):
    """
    Comprehensive Exploratory Data Analysis function

    Parameters:
    -----------
    df : pandas DataFrame
        The dataset to analyze
    target_col : str, optional
        Name of target column for additional analysis
    figsize : tuple, optional
        Figure size for plots

    Returns:
    --------
    dict : Dictionary containing various EDA results
    """

    print("=" * 80)
    print("COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
    print("=" * 80)

    results = {}

    # 1. Basic Information
    print("\n1. BASIC INFORMATION")
    print("-" * 50)
    print(f"Dataset Shape: {df.shape}")
    print(f"Number of Rows: {df.shape[0]}")
    print(f"Number of Columns: {df.shape[1]}")
    print(f"Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB")

    # 2. Data Types
    print("\n2. DATA TYPES")
    print("-" * 50)
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"{dtype}: {count} columns")
    results['dtypes'] = df.dtypes

    # 3. Missing Values
    print("\n3. MISSING VALUES")
    print("-" * 50)
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_data,
        'Percentage': missing_percent
    }).sort_values('Missing Values', ascending=False)
    missing_df = missing_df[missing_df['Missing Values'] > 0]

    if len(missing_df) > 0:
        print(missing_df)
        results['missing_values'] = missing_df
    else:
        print("No missing values found!")

    # 4. Duplicates
    print("\n4. DUPLICATE ROWS")
    print("-" * 50)
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")
    print(f"Percentage of duplicates: {(duplicates / len(df)) * 100:.2f}%")
    results['duplicates'] = duplicates

    # 5. Statistical Summary
    print("\n5. STATISTICAL SUMMARY")
    print("-" * 50)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"\nNumeric Columns ({len(numeric_cols)}):")
    if numeric_cols:
        print(df[numeric_cols].describe().round(2))
        results['numeric_summary'] = df[numeric_cols].describe()

    print(f"\nCategorical Columns ({len(categorical_cols)}):")
    if categorical_cols:
        for col in categorical_cols[:3]:  # Show first 3 categorical columns
            print(f"\n{col}:")
            print(df[col].value_counts().head())
        results['categorical_summary'] = {col: df[col].value_counts() for col in categorical_cols}

    # 6. Visualizations
    print("\n6. GENERATING VISUALIZATIONS...")

    # Create subplots
    fig = plt.figure(figsize=figsize)

    # Plot 1: Distribution of numeric features
    if numeric_cols:
        n_numeric = len(numeric_cols)
        n_rows = (n_numeric + 2) // 3
        for i, col in enumerate(numeric_cols[:9]):  # Limit to 9 columns
            plt.subplot(n_rows, 3, i + 1)
            df[col].hist(bins=30, edgecolor='black', alpha=0.7)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.suptitle('Distribution of Numeric Features', y=1.02, fontsize=16)
        plt.show()

    # Plot 2: Correlation Matrix
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 8))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    fmt='.2f', square=True, linewidths=1)
        plt.title('Correlation Matrix', fontsize=16)
        plt.tight_layout()
        plt.show()
        results['correlation_matrix'] = correlation_matrix

    # Plot 3: Box plots for outliers
    if numeric_cols:
        n_numeric = len(numeric_cols)
        n_cols = 3
        n_rows = (min(n_numeric, 9) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for i, col in enumerate(numeric_cols[:9]):
            axes[i].boxplot(df[col].dropna())
            axes[i].set_title(f'Boxplot of {col}')
            axes[i].set_ylabel(col)
            axes[i].grid(True, alpha=0.3)

        # Hide empty subplots
        for i in range(len(numeric_cols[:9]), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle('Outlier Detection', fontsize=16)
        plt.tight_layout()
        plt.show()

    # Plot 4: Categorical features distribution
    if categorical_cols:
        n_categorical = len(categorical_cols)
        n_rows = (min(n_categorical, 6) + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(categorical_cols[:6]):
            value_counts = df[col].value_counts().head(10)
            axes[i].bar(range(len(value_counts)), value_counts.values)
            axes[i].set_title(f'Top 10 values in {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Count')
            axes[i].set_xticks(range(len(value_counts)))
            axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')
            axes[i].grid(True, alpha=0.3)

        # Hide empty subplots
        for i in range(len(categorical_cols[:6]), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle('Categorical Features Distribution', fontsize=16)
        plt.tight_layout()
        plt.show()

    # 7. Outlier Detection
    print("\n7. OUTLIER DETECTION")
    print("-" * 50)
    if numeric_cols:
        outlier_summary = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
            outlier_percentage = (outliers / len(df)) * 100
            outlier_summary[col] = {
                'outliers': outliers,
                'percentage': outlier_percentage
            }
            if outliers > 0:
                print(f"{col}: {outliers} outliers ({outlier_percentage:.2f}%)")
        results['outliers'] = outlier_summary

    # 8. Target Variable Analysis (if provided)
    if target_col and target_col in df.columns:
        print(f"\n8. TARGET VARIABLE ANALYSIS: {target_col}")
        print("-" * 50)

        if df[target_col].dtype in ['int64', 'float64']:
            # Continuous target
            print(f"Target Statistics:")
            print(df[target_col].describe())

            plt.figure(figsize=(15, 5))

            plt.subplot(1, 2, 1)
            df[target_col].hist(bins=30, edgecolor='black', alpha=0.7)
            plt.title(f'Distribution of {target_col}')
            plt.xlabel(target_col)
            plt.ylabel('Frequency')

            plt.subplot(1, 2, 2)
            stats.probplot(df[target_col].dropna(), dist="norm", plot=plt)
            plt.title('Q-Q Plot')

            plt.tight_layout()
            plt.show()

            results['target_summary'] = df[target_col].describe()

        else:
            # Categorical target
            print(f"Target Distribution:")
            target_dist = df[target_col].value_counts()
            print(target_dist)
            print(f"\nTarget Percentage:")
            print((target_dist / len(df) * 100).round(2))

            plt.figure(figsize=(10, 5))
            target_dist.plot(kind='bar')
            plt.title(f'Distribution of {target_col}')
            plt.xlabel(target_col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            results['target_summary'] = target_dist

    # 9. Data Quality Report
    print("\n9. DATA QUALITY REPORT")
    print("-" * 50)
    quality_report = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.values,
        'Missing %': [df[col].isnull().sum() / len(df) * 100 for col in df.columns],
        'Unique Values': [df[col].nunique() for col in df.columns],
        'Zero Variance': [df[col].nunique() == 1 for col in df.columns]
    })
    print(quality_report.to_string())
    results['quality_report'] = quality_report

    # 10. Recommendations
    print("\n10. RECOMMENDATIONS")
    print("-" * 50)

    # Check for columns that might need attention
    if missing_df.empty:
        print("✓ No missing values found")
    else:
        print(f"⚠️  Columns with missing values: {len(missing_df)}")
        print("  → Consider imputation or removal")

    if duplicates > 0:
        print(f"⚠️  Found {duplicates} duplicate rows")
        print("  → Consider removing duplicates")

    high_cardinality = [col for col in categorical_cols if df[col].nunique() > 50]
    if high_cardinality:
        print(f"⚠️  High cardinality categorical columns: {high_cardinality}")
        print("  → Consider encoding techniques like target encoding")

    zero_variance = quality_report[quality_report['Zero Variance']]['Column'].tolist()
    if zero_variance:
        print(f"⚠️  Zero variance columns: {zero_variance}")
        print("  → Consider removing these columns")

    if target_col:
        print(f"\n✓ Target variable '{target_col}' included in analysis")

    print("\n" + "=" * 80)
    print("EDA COMPLETED")
    print("=" * 80)

    return results
