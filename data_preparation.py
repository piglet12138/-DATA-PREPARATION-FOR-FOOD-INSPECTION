import re
import zipcodes
import requests
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz, process
import sqlite3
import missingno as msno
from difflib import get_close_matches, SequenceMatcher
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
from collections import defaultdict
from itertools import combinations
import sys
from tqdm import tqdm
from tane import TANE, PPattern, read_db, tostr
tane_imported = True
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
from collections import defaultdict



##### DATA PROFILING #####

def profile_name(df, dba_col='dba_name', aka_col='aka_name', 
                        similarity_threshold=0.8, fill_missing=True):
    """
    Analyzes similarity between two name columns and optionally fills missing values.
    
    Parameters:
    - df: DataFrame containing the columns
    - dba_col: Formal name column (default 'dba_name')
    - aka_col: Alternate name column (default 'aka_name')
    - similarity_threshold: Score above which names are considered similar (0-1)
    - fill_missing: Whether to fill missing aka_names with dba_names (default True)
    
    Returns:
    - DataFrame with similarity analysis and optionally filled values
    - Similarity statistics
    """
    
    # Create working copy
    df = df.copy()
    
    # 1. Handle missing values
    missing_mask = df[aka_col].isna()
    if fill_missing:
        df.loc[missing_mask, aka_col] = df.loc[missing_mask, dba_col]
    
    # 2. Calculate similarity metrics
    def get_similarity(a, b):
        if pd.isna(a) or pd.isna(b):
            return 0
        return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()
    
    df['name_similarity'] = df.apply(
        lambda x: get_similarity(x[dba_col], x[aka_col]), axis=1
    )
    
    # 3. Categorize relationships
    conditions = [
        df[aka_col].isna(),
        df[dba_col] == df[aka_col],
        df['name_similarity'] >= similarity_threshold,
        df['name_similarity'] > 0
    ]
    choices = [
        'missing',
        'exact_match',
        'similar',
        'different'
    ]
    df['name_relationship'] = pd.cut(
        df['name_similarity'],
        bins=[-1, 0, 0.1, similarity_threshold, 1],
        labels=['missing', 'different', 'some_similarity', 'similar']
    )
    
    # 4. Generate statistics
    stats = {
        'missing_initial': missing_mask.sum(),
        'missing_filled': fill_missing * missing_mask.sum(),
        'exact_matches': (df[dba_col] == df[aka_col]).sum(),
        'similar_count': (df['name_similarity'] >= similarity_threshold).sum(),
        'avg_similarity': df['name_similarity'].mean()
    }
    
    # display
    def display_similarity_stats(stats):
        """
        Displays the similarity statistics in a simple DataFrame format.
        
        Parameters:
            stats (dict): Dictionary containing the similarity statistics
        """
        # Convert stats to DataFrame and format
        stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
        
        # Format numeric values
        stats_df['Value'] = stats_df['Value'].apply(
            lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) and x == int(x) 
            else f"{x:,.2f}" if isinstance(x, float) 
            else x
        )
        
        # Add description for each metric
        descriptions = {
            'missing_initial': 'Initial missing aka_name values',
            'missing_filled': 'aka_name values filled with dba_name',
            'exact_matches': 'Exact matches between names',
            'similar_count': 'Similar names (above threshold)',
            'avg_similarity': 'Average similarity score'
        }
        
        stats_df['Description'] = stats_df.index.map(descriptions)
        stats_df = stats_df[['Description', 'Value']]  # Reorder columns
        
        # Display with basic formatting
        print("Name Similarity Statistics")
        print("=" * 40)
        print(stats_df)
        print("=" * 40)

    display_similarity_stats(stats)
    return df, stats

def profile_risk_column(df, risk_col='risk'):
    """
    Profiles a risk column with 'High', 'Medium', 'Low', and 'All' categories
    using the Set3 color palette without warnings.
    """
    # Create a clean copy and handle missing values
    df_clean = df.copy()
    df_clean[risk_col] = df_clean[risk_col].fillna('All')
    
    # Define allowed categories and Set3 colors
    valid_categories = ['High', 'Medium', 'Low', 'All']
    set3_colors = sns.color_palette("Set3", len(valid_categories)).as_hex()  # Convert to hex list
    
    # Initialize profile results
    profile = {
        'value_counts': None,
        'invalid_values': None,
        'pre_missing_count': df[risk_col].isna().sum()
    }
    
    # Value counts with percentages
    value_counts = df_clean[risk_col].value_counts()
    percentages = df_clean[risk_col].value_counts(normalize=True) * 100
    profile['value_counts'] = pd.DataFrame({
        'Count': value_counts,
        'Percentage': percentages.round(2)
    }).reindex(valid_categories)
    
    # Data quality checks
    invalid_mask = ~df_clean[risk_col].isin(valid_categories)
    profile['invalid_values'] = df_clean.loc[invalid_mask, risk_col].unique()
    profile['invalid_count'] = invalid_mask.sum()
    
    # Visualization with Set3 palette
    plt.figure(figsize=(14, 6))
    
    # 1. Count plot (corrected to avoid warnings)
    plt.subplot(1, 2, 1)
    ax = sns.countplot(
        x=risk_col, 
        data=df_clean, 
        order=valid_categories,
        hue=risk_col,  # Added to fix warning
        palette=set3_colors,
        legend=False   # Added to avoid duplicate legend
    )
    plt.title('Risk Level Distribution (Missing → All)', pad=20)
    
    # Add percentage labels
    total = len(df_clean)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.01*total,
                f'{height/total:.1%}',
                ha='center', fontsize=10)
    
    # 2. Pie chart
    plt.subplot(1, 2, 2)
    df_clean[risk_col].value_counts().reindex(valid_categories).plot.pie(
        autopct='%1.1f%%',
        colors=set3_colors,
        startangle=90,
        textprops={'fontsize': 12}
    )
    plt.title('Risk Level Proportion', pad=20)
    plt.ylabel('')
    
    plt.tight_layout()
    plt.show()
    
    return profile

def profile_inspection_date(df):
    """Profiles the inspection_date column with temporal analysis"""
    # Convert to datetime and extract features
    df['inspection_date'] = pd.to_datetime(df['inspection_date'])
    df['inspection_year'] = df['inspection_date'].dt.year
    df['inspection_month'] = df['inspection_date'].dt.month_name()
    
    # Plot temporal distribution
    plt.figure(figsize=(15, 5))
    
    # Yearly trend
    plt.subplot(1, 2, 1)
    yearly_counts = df['inspection_year'].value_counts().sort_index()
    sns.lineplot(x=yearly_counts.index, y=yearly_counts.values)
    plt.title('Inspections by Year')
    plt.xlabel('Year')
    plt.ylabel('Count')
    
    # Monthly distribution
    plt.subplot(1, 2, 2)
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_counts = df['inspection_month'].value_counts().reindex(month_order)
    sns.barplot(x=monthly_counts.index, y=monthly_counts.values)
    plt.title('Inspections by Month')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Date statistics
    print(f"Date Range: {df['inspection_date'].min().date()} to {df['inspection_date'].max().date()}")
    print(f"Most Active Year: {yearly_counts.idxmax()} ({yearly_counts.max():,} inspections)")
    print(f"Most Active Month: {monthly_counts.idxmax()} ({monthly_counts.max():,} inspections)")

def standardize_inspection_types(df):
    """
    Standardizes inspection types into 6 official categories with re-inspection flags:
    ['canvass', 'consultation', 'complaint', 'license', 
    'suspect food poisoning', 'task-force inspection']
    Preserves re-inspection status when present.
    """
    # Primary category mapping with regex patterns
    category_map = {
        'canvass': r'(canvass|routine|periodic|standard|surveillance)',
        'consultation': r'(consult|pre.?open|pre.?operat)',
        'complaint': r'(complaint|consumer|public\s*health)',
        'license': r'(license|permit|new\s*establishment|initial)',
        'suspect food poisoning': r'(food\s*poison|outbreak|illness|contaminat)',
        'task-force inspection': r'(task\s*force|bar|tavern|special|targeted)'
    }
    
    # Standardize text format
    df = df.copy()
    df['inspection_type'] = df['inspection_type'].str.strip().str.lower()
    
    # Function to categorize with re-inspection detection
    def categorize(insp_type):
        if pd.isna(insp_type):
            return None
        
        # Check for re-inspection designation first
        is_reinspection = 're-inspect' in insp_type or 'reinspect' in insp_type or 'follow.up' in insp_type
        
        # Determine primary category
        primary_category = 'other'
        for category, pattern in category_map.items():
            if re.search(pattern, insp_type, flags=re.IGNORECASE):
                primary_category = category
                break
        
        # Append re-inspection designation if found
        if is_reinspection and primary_category != 'other':
            return f"{primary_category} (re-inspection)"
        return primary_category
    
    # Apply standardization
    df['standardized_type'] = df['inspection_type'].apply(categorize)
    
    return df

def profile_inspection_type(df):
    """Profiles the standardized inspection types"""
    
    # standardize inspection_type options

    df_std = standardize_inspection_types(df)
    
    # Calculate distribution
    type_counts = df_std['standardized_type'].value_counts()
    type_pct = (df_std['standardized_type'].value_counts(normalize=True) * 100).round(1)
    
    # Get ordered list of categories for consistent plotting
    ordered_categories = type_counts.index.tolist()
    
    # Visualization
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x=type_counts.values,
        y=type_counts.index,
        order=ordered_categories,
        palette="husl"
    )
    
    # Add value labels - FIXED VERSION
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        category = ordered_categories[i]  # Get category by position
        ax.text(
            width + max(type_counts)*0.01,
            p.get_y() + p.get_height()/2,
            f'{width:,} ({type_pct[category]}%)',
            va='center'
        )
    
    plt.title('Standardized Inspection Types', pad=20)
    plt.xlabel('Count', labelpad=10)
    plt.ylabel('')
    plt.xlim(0, max(type_counts)*1.2)
    
    # Print stats
    print("Standardized Inspection Type Distribution:")
    print("="*55)
    for typ, cnt in type_counts.items():
        print(f"• {typ.title():<30}: {cnt:>7,} ({type_pct[typ]}%)")
    print("="*55)

def profile_results(df):
    """
    Profiles the results column with outcome analysis
    using a 7-color palette optimized for readability.
    """
    # Clean and standardize results
    df = df.copy()  # Avoid modifying original DataFrame
    df['results'] = df['results'].str.strip().str.title()
    
    # Calculate distribution
    result_counts = df['results'].value_counts()
    result_pct = (df['results'].value_counts(normalize=True) * 100).round(1)
    
    # Visualization with Set2 palette
    plt.figure(figsize=(10, 6))
    
    # Get 7 colors from Set2 palette
    palette = sns.color_palette("Set2", 7)
    
    # Create ordered list of results (pass-like outcomes first)
    ordered_results = sorted(
        result_counts.index,
        key=lambda x: (
            0 if 'Pass' in x else 
            1 if 'Fail' in x else 
            2 if 'Condition' in x else 3
        )
    )
    
    # Plot with corrected syntax
    ax = sns.barplot(
        x=result_counts.loc[ordered_results].values,
        y=ordered_results,
        palette=palette,
        hue=ordered_results if len(ordered_results) > 1 else None,
        legend=False,
        dodge=False
    )
    
    # Add value labels
    for p in ax.patches:
        width = p.get_width()
        ax.text(
            width + max(result_counts)*0.01,
            p.get_y() + p.get_height()/2,
            f'{width:,} ({result_pct[ordered_results[int(p.get_y() + 0.5)]]}%)',
            va='center',
            fontsize=10
        )
    
    plt.title('Inspection Outcomes Distribution', pad=20)
    plt.xlabel('Count', labelpad=10)
    plt.ylabel('Result', labelpad=10)
    plt.xlim(0, max(result_counts)*1.15)
    
    # Print key stats
    print("\nOutcome Distribution Summary:")
    print("="*40)
    for result in ordered_results:
        print(f"• {result:<20}: {result_counts[result]:>8,} ({result_pct[result]}%)")
    print("="*40)
    
def profile_zip(df):
    '''
    Function to profile ZIP codes in the dataset.
    Args:
        df: DataFrame containing ZIP codes
        
    Example usage:
    profile_zip(updated_food_dataset)
    '''
    
    # show most frequent zip codes
    plt.figure(figsize=(12, 4))
    sns.countplot(data=df, x='zip', order=df['zip'].value_counts().iloc[:5].index) 
    plt.title("Top 5 Most Frequent ZIP Codes")
    plt.xticks(rotation=45)
    plt.show()
    
    consistent_zip_length = df.copy()
    consistent_zip_length['zip_length'] = consistent_zip_length['zip'].astype(str).str.len()
    sns.countplot(data=consistent_zip_length, x='zip_length')
    plt.title("ZIP Code Length Distribution")
    
def profile_state(df):
    '''
    Function to profile states in the dataset.
    Args:
        df: DataFrame containing state data
        
    Example usage:
    profile_state(updated_food_dataset)
    '''
    
        
    state_profile = (
        df['state']
        .value_counts(dropna=False)
        .rename("Count")
        .to_frame()
        .assign(Percentage=lambda x: (x['Count'] / len(df)).round(2))
        .rename_axis('State')
        .reset_index()
    )
    print(state_profile)
    
    # Create bar chart for state counts
    plt.figure(figsize=(10, 6))
    state_counts = df['state'].value_counts()
    sns.barplot(x=state_counts.index, y=state_counts.values)
    plt.title("Distribution of States in Dataset")
    plt.xlabel("State")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def check_zip_state_city_mapping(df):
    """
    Check ZIP code to state/city mapping consistency.
    
    Args:
        df: DataFrame containing ZIP, state, and city columns
    """
    # Find ZIPs mapped to multiple states (invalid)
    zip_state_violations = df.groupby('zip')['state'].nunique()
    anomalous_zips = zip_state_violations[zip_state_violations > 1]
    print(f"ZIPs with conflicting states: {len(anomalous_zips)}")

    # Show examples
    print(df[df['zip'].isin(anomalous_zips.index)]
            .sort_values('zip')[['zip', 'state', 'city']].head(10))

def check_city_state_spelling(df, sample_size=5):
    '''
    Function to check city and state spelling in the dataset.
    Args:
        df: DataFrame containing city and state data
        
    Example usage:
    check_city_state_spelling(updated_food_dataset)
    '''
    
    
    sample_cities = df['city'].dropna().unique()
    for city in sample_cities[:sample_size]:  # Check first sample_size for demo
        matches = get_close_matches(city, sample_cities, n=sample_size, cutoff=0.8)
        if len(matches) > 1:
            print(f"Potential duplicates for '{city}': {matches}")
    
    sample_states = df['state'].dropna().unique()
    for state in sample_states[:sample_size]:  # Check first sample_size for demo
        matches = get_close_matches(city, sample_states, n=sample_size, cutoff=0.8)
        if len(matches) > 1:
            print(f"Potential duplicates for '{state}': {matches}")
            

def profile_violations(df, sample_size=100):
    """
    Analyze the structure of the 'violations' column in the dataframe.
    
    Parameters:
    sample_size (int): Number of non-empty violation entries to analyze
    
    Returns:
    dict: Analysis results including patterns found, separator counts, etc.
    
    Example usage: 
    profile_violations(updated_food_dataset, sample_size=1000)
    """
    # Get sample of non-empty violations
    df['violations'] = df['violations'].fillna('')
    violations_sample = df.loc[df['violations'] != '', 'violations'].head(sample_size)
    
    # Initialize analysis results
    analysis = {
        'total_samples': len(violations_sample),
        'avg_length': violations_sample.str.len().mean(),
        'separator_counts': {},
        'violation_counts': [],
        'common_patterns': [],
        'sample_entries': []
    }
    
    # Count separators
    for sep in ['|', '.', '-', ':', ';']:
        analysis['separator_counts'][sep] = violations_sample.str.count(sep).mean()
    
    # Count violations per entry
    violations_per_entry = violations_sample.apply(lambda x: len([v for v in x.split('|') if v.strip()]))
    analysis['violations_per_entry'] = {
        'min': violations_per_entry.min(),
        'max': violations_per_entry.max(),
        'avg': violations_per_entry.mean(),
        'distribution': violations_per_entry.value_counts().to_dict()
    }
    
    # Check for common patterns
    # Look for digit followed by period at the start of each violation
    starts_with_digit = violations_sample.apply(
        lambda x: all(re.match(r'^\s*\d+\.', v.strip()) for v in x.split('|') if v.strip())
    ).mean()
    analysis['common_patterns'].append(f"{starts_with_digit*100:.1f}% start with digits followed by period")
    
    # Check for "Comments:" pattern
    has_comments = violations_sample.apply(
        lambda x: any("Comments:" in v for v in x.split('|'))
    ).mean()
    analysis['common_patterns'].append(f"{has_comments*100:.1f}% contain 'Comments:' text")
    
    # Sample entries
    analysis['sample_entries'] = violations_sample.sample(min(5, len(violations_sample))).tolist()
    
    print("violations Column Structure Analysis:")
    for key, value in analysis.items():
        if key != 'sample_entries':
            print(f"\n{key.replace('_', ' ').title()}:")
            if isinstance(value, dict):
                for k, v in value.items():
                    print(f"  - {k}: {v}")
            elif isinstance(value, list):
                for item in value:
                    print(f"  - {item}")
            else:
                print(f"  {value}")
    
    return analysis

def verify_violations_structure(df, structure=r"^\s*\d+\.\s+.+?(\s+-\s+Comments:\s*.+)?$",sample_size=None):
    """
    Verify that all values in the 'violations' column follow the structure:
    "Code. Category - Comments: Comment text" with parts separated by "|"
    the Comments part is optional.
    
    Parameters:
    df (DataFrame): DataFrame containing 'violations' column
    sample_size (int, optional): Number of non-empty violations to check. If None, check all.
    
    Returns:
    dict: Results of verification including:
        - total_checked: Number of entries checked
        - valid_entries: Number of entries following the expected pattern
        - invalid_entries: Number of entries not following the expected pattern
        - validity_percentage: Percentage of valid entries
        - sample_invalid: Examples of invalid entries (up to 5)
        
    Example usage:
    verify_violations_structure(updated_food_dataset)
    """
    # Fill NaN values with empty string
    violations = df['violations'].fillna('')
    
    # Filter out empty violations
    non_empty_violations = violations[violations != '']
    
    # Take a sample if specified
    if sample_size is not None:
        violations_to_check = non_empty_violations.sample(min(sample_size, len(non_empty_violations)))
    else:
        violations_to_check = non_empty_violations
    
    # Define the expected pattern for each violation
    pattern = structure
    
    # Check each violation entry
    results = {
        'total_checked': 0,
        'valid_entries': 0,
        'invalid_entries': 0,
        'validity_percentage': 0,
        'sample_invalid': []
    }
    
    invalid_entries = []
    
    for violation_text in violations_to_check:
        results['total_checked'] += 1
        parts = [part.strip() for part in violation_text.split('|') if part.strip()]
        
        # If there are no parts, count as invalid
        if not parts:
            results['invalid_entries'] += 1
            if len(results['sample_invalid']) < 5:
                results['sample_invalid'].append(violation_text if len(violation_text) > 100 else violation_text)
            continue
        
        # Check each part against the pattern
        all_parts_valid = all(re.match(pattern, part) for part in parts)
        
        if all_parts_valid:
            results['valid_entries'] += 1
        else:
            results['invalid_entries'] += 1
            invalid_entries.append(violation_text)
            if len(results['sample_invalid']) < 5:
                results['sample_invalid'].append(violation_text if len(violation_text) > 100 else violation_text)
    
    # Calculate percentage of valid entries
    if results['total_checked'] > 0:
        results['validity_percentage'] = round(results['valid_entries'] / results['total_checked'] * 100, 2)
        
    print("Violations Structure Verification Results:")
    print(results)
    
    return results




def profile_violations(violations_df):
    '''
    Function to profile the violations data.
    '''
    # Get counts of each code_category and sort
    violations_df['code_category'] = violations_df['violation_code'].astype('str') + '-' + violations_df['category'].str[:40]
    code_category_counts = violations_df['code_category'].value_counts().reset_index()
    code_category_counts.columns = ['code_category', 'count']
    code_category_counts = code_category_counts.head(20)  # Top 20

    # Create figure with larger size for readability
    plt.figure(figsize=(14, 10))

    # Create horizontal bar chart
    sns.barplot(x='count', y='code_category', data=code_category_counts, palette='viridis')

    # Add title and labels
    plt.title('Top 20 Most Common Violation Code-Categories', fontsize=16)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Violation Code-Category', fontsize=12)
    plt.tight_layout()

    # Add count values at the end of each bar
    for i, v in enumerate(code_category_counts['count']):
        plt.text(v + 0.1, i, f"{v:,}", va='center')

    plt.show()


def extract_violation_codes(text):
    if pd.isna(text):
        return []
    # extract violation code(e.g. 32, 10, 47 )
    codes = re.findall(r"\b(\d+)\.(?=\s+[A-Z])", text)
    return codes


def preprocess_data(df, violation_col=None, max_categories=20):
    """
    Preprocess the data for association rule mining

    Parameters:
    -----------
    df : pandas DataFrame
        The input dataset
    violation_col : string, optional
        The column name containing violation text data
    max_categories : int, optional
        Maximum number of categories to keep for each categorical column

    Returns:
    --------
    trans_df : pandas DataFrame
        Transformed dataset ready for association rule mining
    """
    transactions = []

    # 限制每个分类列的唯一值数量
    category_maps = {}
    for col in df.columns:
        if col != violation_col and pd.api.types.is_object_dtype(df[col]):
            # 获取前N个最常见的分类
            value_counts = df[col].value_counts()
            top_categories = value_counts.index[:max_categories].tolist()
            # 创建映射
            category_maps[col] = {cat: f"{col}_{cat}" for cat in top_categories}

    for idx, row in df.iterrows():
        transaction = {}

        # 添加分类列
        for col in df.columns:
            if col != violation_col and pd.api.types.is_object_dtype(df[col]):
                val = row[col]
                if val in category_maps[col]:
                    transaction[category_maps[col][val]] = 1

        # 添加违规代码
        if violation_col is not None and not pd.isna(row[violation_col]):
            violation_codes = extract_violation_codes(row[violation_col])
            for code in violation_codes:
                transaction[f'Violation_{code}'] = 1

        transactions.append(transaction)

    # 从交易创建DataFrame
    trans_df = pd.DataFrame(transactions)

    # 用0填充缺失值
    trans_df = trans_df.fillna(0).astype(int)

    return trans_df


def mine_association_rules(df, columns_to_analyze, min_support=0.01,
                           min_threshold=1.0, metric="lift", max_results=10):
    """
    Mine association rules from the specified columns

    Parameters:
    -----------
    df : pandas DataFrame
        The preprocessed dataset
    columns_to_analyze : list
        List of column prefixes to include in the analysis
    min_support : float, optional
        Minimum support for apriori algorithm
    min_threshold : float, optional
        Minimum threshold for the specified metric
    metric : string, optional
        Metric to use for association rules
    max_results : int, optional
        Maximum number of results to return

    Returns:
    --------
    tuple : (frequent_itemsets, rules, formatted_rules)
        frequent_itemsets: DataFrame with frequent itemsets
        rules: DataFrame with association rules
        formatted_rules: list of formatted rule strings
    """
    # Select columns to analyze
    selected_cols = []
    for prefix in columns_to_analyze:
        selected_cols.extend([col for col in df.columns if col.startswith(prefix)])

    if not selected_cols:
        return None, None, ["No columns matching the specified prefixes were found."]

    selected_df = df[selected_cols]

    # Apply Apriori algorithm
    frequent_itemsets = apriori(selected_df, min_support=min_support, use_colnames=True)

    if len(frequent_itemsets) == 0:
        return frequent_itemsets, None, ["No frequent itemsets found with the given minimum support."]

    # Sort frequent itemsets by support
    frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)

    if len(rules) == 0:
        return frequent_itemsets, rules, ["No association rules found with the given parameters."]

    # Sort rules by the specified metric
    sorted_rules = rules.sort_values(metric, ascending=False)

    # Format the top rules
    formatted_rules = []
    for i, rule in sorted_rules.head(max_results).iterrows():
        antecedent = ", ".join([item for item in list(rule['antecedents'])])
        consequent = ", ".join([item for item in list(rule['consequents'])])
        formatted_rule = (f"{antecedent} => {consequent} "
                          f"(support: {rule['support']:.3f}, "
                          f"confidence: {rule['confidence']:.3f}, "
                          f"lift: {rule['lift']:.3f})")
        formatted_rules.append(formatted_rule)

    return frequent_itemsets, sorted_rules, formatted_rules


def analyze_between_categories(df, antecedent_prefix, consequent_prefix,
                               min_support=0.01, min_threshold=0.5,
                               metric="confidence", max_results=10):
    """
    Mine association rules between two specific categories

    Parameters:
    -----------
    df : pandas DataFrame
        The preprocessed dataset
    antecedent_prefix : string
        Prefix for columns to be used as antecedents
    consequent_prefix : string
        Prefix for columns to be used as consequents
    min_support : float, optional
        Minimum support for apriori algorithm
    min_threshold : float, optional
        Minimum threshold for the specified metric
    metric : string, optional
        Metric to use for association rules
    max_results : int, optional
        Maximum number of results to return

    Returns:
    --------
    tuple : (frequent_itemsets, rules, formatted_rules)
        frequent_itemsets: DataFrame with frequent itemsets
        rules: DataFrame with association rules
        formatted_rules: list of formatted rule strings
    """
    # Select columns for both antecedent and consequent
    all_prefixes = [antecedent_prefix, consequent_prefix]
    selected_cols = []
    for prefix in all_prefixes:
        selected_cols.extend([col for col in df.columns if col.startswith(prefix)])

    if not selected_cols:
        return None, None, ["No columns matching the specified prefixes were found."]

    selected_df = df[selected_cols]

    # Apply Apriori algorithm
    frequent_itemsets = apriori(selected_df, min_support=min_support, use_colnames=True)

    if len(frequent_itemsets) == 0:
        return frequent_itemsets, None, ["No frequent itemsets found with the given minimum support."]

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)

    if len(rules) == 0:
        return frequent_itemsets, rules, ["No association rules found with the given parameters."]

    # Filter rules: antecedent contains only items from antecedent_prefix and
    # consequent contains only items from consequent_prefix
    filtered_rules = rules[
        rules['antecedents'].apply(lambda x: all(antecedent_prefix in item for item in x)) &
        rules['consequents'].apply(lambda x: all(consequent_prefix in item for item in x))
        ]

    if len(filtered_rules) == 0:
        return frequent_itemsets, rules, ["No rules found matching the specified pattern."]

    # Sort rules by lift
    sorted_rules = filtered_rules.sort_values('lift', ascending=False)

    # Format the top rules
    formatted_rules = []
    for i, rule in sorted_rules.head(max_results).iterrows():
        antecedent = ", ".join([item.replace(f'{antecedent_prefix}_', f'{antecedent_prefix}:')
                                for item in list(rule['antecedents'])])
        consequent = ", ".join([item.replace(f'{consequent_prefix}_', f'{consequent_prefix}:')
                                for item in list(rule['consequents'])])
        formatted_rule = (f"{antecedent} => {consequent} "
                          f"(support: {rule['support']:.3f}, "
                          f"confidence: {rule['confidence']:.3f}, "
                          f"lift: {rule['lift']:.3f})")
        formatted_rules.append(formatted_rule)

    return frequent_itemsets, sorted_rules, formatted_rules


def association_rule_mining(dataset, categorical_columns=None, violation_column=None, analysis_type=None):
    """
    Main function to perform association rule mining on a dataset

    Parameters:
    -----------
    dataset : pandas DataFrame or path to CSV
        The dataset to analyze
    categorical_columns : list, optional
        List of categorical columns to include in the analysis
    violation_column : string, optional
        Column containing violation text data
    analysis_type : string, optional
        Type of analysis to perform:
        - 'violations': analyze relationships between violations
        - 'categories': analyze relationships between categorical attributes
        - 'category_violations': analyze relationships between categories and violations
        If None, all analyses are performed

    Returns:
    --------
    dict : Results of the association rule mining
    """
    # Load dataset if provided as path
    if isinstance(dataset, str):
        df = pd.read_csv(dataset)
    else:
        df = dataset.copy()

    # Define columns if not provided
    if categorical_columns is None:
        categorical_columns = [col for col in df.columns if pd.api.types.is_object_dtype(df[col])]

    # Preprocess data
    trans_df = preprocess_data(df, violation_column)

    results = {}

    # Analysis of violations
    if analysis_type is None or analysis_type == 'violations':
        violation_cols = [col for col in trans_df.columns if col.startswith('Violation_')]

        if violation_cols:
            print("Analyzing association rules between violations...")
            _, _, formatted_rules = mine_association_rules(
                trans_df, ['Violation_'], min_support=0.1, min_threshold=1.0
            )
            results['violations'] = formatted_rules
            print("\n".join(formatted_rules))
        else:
            results['violations'] = ["No violation data found."]
            print("No violation data found.")

    # Analysis of categorical attributes
    if analysis_type is None or analysis_type == 'categories':
        category_prefixes = []
        for col in categorical_columns:
            if col != violation_column:
                category_prefixes.append(f"{col}_")

        if category_prefixes:
            print("\nAnalyzing association rules between categorical attributes...")
            _, _, formatted_rules = mine_association_rules(
                trans_df, category_prefixes, min_support=0.05, min_threshold=1.0
            )
            results['categories'] = formatted_rules
            print("\n".join(formatted_rules))
        else:
            results['categories'] = ["No categorical data specified."]
            print("No categorical data specified.")

    # Analysis between facility type and violations
    if analysis_type is None or analysis_type == 'category_violations':
        violation_cols = [col for col in trans_df.columns if col.startswith('Violation_')]

        if violation_cols and categorical_columns:
            print("\nAnalyzing association rules between categories and violations...")
            for col in categorical_columns:
                if col != violation_column:
                    print(f"\nRules from {col} to violations:")
                    _, _, formatted_rules = analyze_between_categories(
                        trans_df, f"{col}_", "Violation_",
                        min_support=0.02, min_threshold=0.05
                    )
                    results[f'{col}_to_violations'] = formatted_rules
                    print("\n".join(formatted_rules))
        else:
            results['category_violations'] = ["Either no violation data or categorical data found."]
            print("Either no violation data or categorical data found.")

    return results



##### DATA PROCESSING ######
"""
try to normalize data cleaning and evaluate data quility
1. imputation of missing values
2. fix city names
"""
# Column name changer function
def col_name_changer(df):
    # 1. Change to lower case
    df.columns = df.columns.str.lower()
    # 2. Change space to _
    df.columns = df.columns.str.replace(" ","_")
    # 3. Remove special characters: change # on License # to "num"
    df.columns = df.columns.str.replace("#",'num')
    # 4. Perform strip to ensure no extra white spaces
    df.columns = df.columns.str.strip()
    return df


def process_license_numbers(df, license_col='license_num'):
    """
    Safely converts license numbers to strings and cleans formatting.

    Returns:
        DataFrame with processed license numbers
        Dictionary of processing statistics
    """
    # Create a copy
    df = df.copy()
    stats = {
        'initial_null_count': df[license_col].isna().sum(),
        'initial_dtype': str(df[license_col].dtype)
    }

    try:
        # Convert to string, handling nulls and floats
        df[license_col] = (
            df[license_col]
            .astype('string')
            .str.replace(r'\.0$', '', regex=True)  # Remove trailing .0 from floats
            .str.strip()  # Remove whitespace
            .replace('nan', pd.NA)  # Restore actual nulls
        )

        # Validation stats
        stats.update({
            'final_null_count': df[license_col].isna().sum(),
            'final_dtype': str(df[license_col].dtype),
            'sample_values': df[license_col].dropna().sample(3).tolist()
        })

    except Exception as e:
        stats['error'] = f"Processing failed: {str(e)}"
        raise ValueError(f"License number processing error: {e}") from e

    return df


# define the zip data cleaning function
def clean_zip_data(df, zip_col='zip'):
    """
    Enhanced ZIP code cleaner that:
    1. Handles floats/integers (e.g., 60601.0 → "60601")
    2. Preserves leading zeros (e.g., "07001" → "07001")
    3. Removes ZIP+4 extensions (e.g., "60601-1234" → "60601")
    4. Flags invalid ZIPs (non-numeric, wrong length)
    5. Compatible with the `zipcodes` package

    Returns:
        DataFrame with:
        - Original column (renamed to zip_raw)
        - Cleaned column (zip_clean)
        - Validation flag (zip_valid)
    """
    df = df.copy()

    # Convert to string and clean
    df['zip_clean'] = (
        df[zip_col]
        .astype(str)
        .str.strip()
        .str.replace(r'\.0$', '', regex=True)  # Remove .0 from floats
        .str.extract(r'(\d{5})')[0]  # Extract first 5 digits only
    )

    # Standardize to 5-digit strings
    df['zip_clean'] = (
        df['zip_clean']
        .str.zfill(5)  # Pad with leading zeros
        .where(df['zip_clean'].str.len() == 5)  # Only keep 5-digit codes
        .replace('00000', np.nan)  # Handle all-zero cases
    )

    # Validation (matches USPS 5-digit format)
    df['zip_valid'] = (
        df['zip_clean']
        .notna()
        .astype('boolean')
    )
    df['zip_clean'] = df['zip_clean'].astype('string')  # Pandas' StringDtype
    return df


def get_zip_info(zip_code, field='state'):
    """Get state/city from ZIP code with proper error handling.
    Args:
        zip_code: Input ZIP (str/int/float)
        field: 'state' or 'city'
    Returns:
        str or None
    """
    if pd.isna(zip_code):
        return None
    try:
        # Convert to string and pad with zeros
        zip_str = str(int(zip_code)).zfill(5) if str(zip_code).isdigit() else str(zip_code)
        matched = zipcodes.matching(zip_str)

        if not matched:
            return None
        return matched[0].get(field)  # access the needed field

    except Exception as e:
        print(f"Error processing ZIP {zip_code}: {str(e)}")
        return None


# formatting address
def remove_nested_brackets(text):
    """Removes all content inside brackets, including nested brackets, using a stack-based approach."""
    if pd.isna(text):
        return text

    text = str(text)
    stack = []
    result = []

    for char in text:
        if char == '(':
            stack.append(len(result))
        elif char == ')' and stack:
            result = result[:stack.pop()]
        elif not stack:
            result.append(char)

    return ''.join(result)


def format_address(address):
    if pd.isna(address):
        return address

    # Step 1: Remove all nested brackets
    address = remove_nested_brackets(str(address))

    # Step 2: Normalize whitespace
    address = re.sub(r'\s+', ' ', address).strip()

    # Step 3: Convert to title case (except abbreviations)
    address = address.title()

    # Step 4: Fix common postal abbreviations
    abbreviations = {
        r'\bSt\b': 'ST',
        r'\bAve\b': 'AVE',
        r'\bBlvd\b': 'BLVD',
        r'\bDr\b': 'DR',
        r'\bLn\b': 'LN',
        r'\bRd\b': 'RD',
        r'\bHwy\b': 'HWY',
        r'\bUs\b': 'US',
    }
    for pattern, replacement in abbreviations.items():
        address = re.sub(pattern, replacement, address)

    return address


def geocode_missing_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Geocode missing latitude/longitude data using Census API
    Follows exact workflow from your code example

    Parameters:
        df: DataFrame containing address components and missing coordinates

    Returns:
        DataFrame with updated latitude/longitude for previously missing rows
    """
    # --- 1. Prepare Missing Data ---
    # Get rows where coordinates are missing
    missing_mask = df['latitude'].isna() | df['longitude'].isna()
    missing_data = df[missing_mask][['address', 'city', 'state', 'latitude', 'longitude', 'location', 'zip']].copy()

    if len(missing_data) == 0:
        print("No missing coordinates to geocode")
        return df

    # Format addresses
    missing_data['address'] = missing_data['address'].apply(format_address)

    # --- 2. Prepare API Input ---
    geocoder_df = pd.DataFrame({
        'street': missing_data['address'],
        'city': missing_data['city'],
        'state': missing_data['state'],
        'zip': missing_data['zip']
    }).reset_index(drop=True)

    # Save to CSV
    with open('my_geocoder_input.csv', 'w', encoding='utf-8') as f:
        geocoder_df.to_csv(f, index=True, header=False)

    # --- 3. Call Census API ---
    url = 'https://geocoding.geo.census.gov/geocoder/geographies/addressbatch'
    try:
        with open('my_geocoder_input.csv', 'rb') as f:
            files = {'addressFile': ('my_geocoder_input.csv', f, 'text/csv')}
            payload = {
                'benchmark': 'Public_AR_Current',
                'vintage': 'Census2010_Current',
                'format': 'json'
            }
            response = requests.post(url, files=files, data=payload, timeout=30)

        if response.status_code != 200:
            raise Exception(f"API Error {response.status_code}: {response.text[:200]}")

        # Save raw response
        with open("geocoded_results.csv", "wb") as f:
            f.write(response.content)

    except Exception as e:
        print(f"Geocoding failed: {str(e)}")
        return df

    # --- 4. Process Results ---
    try:
        # Load and parse results
        result_df = pd.read_csv(
            'geocoded_results.csv',
            header=None,
            quotechar='"',
            names=[
                "ID", "input_address", "match_status", "match_type",
                "matched_address", "location", "tigerline_id", "side",
                "statefips", "countyfips", "tractcode", "blockcode"
            ]
        )

        # Filter successful matches
        matched = result_df[result_df['match_status'] == "Match"].copy()
        matched[['longitude', 'latitude']] = matched['location'].str.split(',', expand=True).astype(float)

    except Exception as e:
        print(f"Failed to process results: {str(e)}")
        return df

    # --- 5. Merge Results Back ---
    # Create matching keys
    missing_data['full_address'] = (
            missing_data['address'] + ', ' +
            missing_data['city'] + ', ' +
            missing_data['state'] + ', ' +
            missing_data['zip'].astype(str)
    )

    # Create coordinate mapping
    coord_map = {
        row['input_address']: (row['latitude'], row['longitude'], row['location'])
        for _, row in matched.iterrows()
    }

    # Update coordinates in missing data
    missing_data['latitude'] = missing_data['full_address'].map(
        lambda x: coord_map.get(x, (None, None, None))[0]
    )
    missing_data['longitude'] = missing_data['full_address'].map(
        lambda x: coord_map.get(x, (None, None, None))[1]
    )
    missing_data['location'] = missing_data['full_address'].map(
        lambda x: coord_map.get(x, (None, None, None))[2]
    )

    # --- 6. Update Original DataFrame ---
    # Update only the rows that were missing coordinates
    df.update(missing_data[['latitude', 'longitude', 'location']])

    return df


# final function form
def risk_column_transformation(df):
    """
    Transforms the 'risk' column in-place, standardizing to:
    - 'High' (from 'Risk 1 (High)')
    - 'Medium' (from 'Risk 2 (Medium)')
    - 'Low' (from 'Risk 3 (Low)')
    - 'All' (for NaN or any other input)
    """
    # Strict mapping - only these exact inputs will convert
    allowed_mappings = {
        'Risk 1 (High)': 'High',
        'Risk 2 (Medium)': 'Medium',
        'Risk 3 (Low)': 'Low'
    }

    # Create a mask for values that need to be changed to 'All'
    to_replace = ~df['risk'].isin(allowed_mappings.keys()) | df['risk'].isna()

    # First map the allowed values
    df['risk'] = df['risk'].map(allowed_mappings)

    # Then replace everything else with 'All'
    df.loc[to_replace, 'risk'] = 'All'

    return df


def standardize_name_columns(df, dba_col='dba_name', aka_col='aka_name'):
    """
    Standardizes text format for name columns by:
    - Converting to lowercase
    - Stripping leading/trailing whitespace
    - Handling NaN values

    Parameters:
        df: Input DataFrame
        dba_col: Name of formal business name column (default 'dba_name')
        aka_col: Name of alternate name column (default 'aka_name')

    Returns:
        DataFrame with standardized name columns
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Standardize dba_name
    if dba_col in df.columns:
        df[dba_col] = (
            df[dba_col]
            .astype(str)  # Convert all to string first
            .str.lower()  # Convert to lowercase
            .str.strip()  # Remove whitespace
            .replace('nan', pd.NA)  # Restore actual NaN values
        )

    # Standardize aka_name
    if aka_col in df.columns:
        df[aka_col] = (
            df[aka_col]
            .astype(str)
            .str.lower()
            .str.strip()
            .replace('nan', pd.NA)
        )

    return df


def fix_city_name(df, city_col='city', threshold=85):
    """
    fix city name，normalize city names similar to "CHICAGO"
    
    parameters:
        df: dataset
        city_col:  default as 'city'
        threshold: treshold for fixing default as 85
        
    return:
        updated dataset
    """
    df = df.copy()
    
    # make sure the city name to be uppercase
    df[city_col] = df[city_col].astype(str).str.upper().replace('NAN', pd.NA)
    valid_cities = ['CHICAGO']

    def correct_to_chicago(city):
        if pd.isna(city):
            return city
        

        if city == 'CHICAGO':
            return city
        
        #get match score
        match, score = process.extractOne(city, valid_cities, scorer=fuzz.ratio)
        

        return match if score >= threshold else city
    
    # apply correction
    df[city_col] = df[city_col].apply(correct_to_chicago)
    
    return df



def parse_comments(text):
    """
    Parse the comments from the 'violations' column.
    Each comment is expected to be in the format:
    "Code. Category - Comments: Comment text"
    The function returns a list of dictionaries with keys:
    'violation_code', 'category', and 'comment'.
    """
    parts = [part.strip() for part in text.split('|') if part.strip()]
    rows = []

    for part in parts:
        match = re.match(r"(?P<code>\d+)\.\s+(?P<category>.+?)\s+-\s+Comments:\s*(?P<comment>.+)", part)
        if match:
            rows.append({
                'violation_code': int(match.group('code')),
                'category': match.group('category').strip(),
                'comment': match.group('comment').strip()
            })
    return rows

def parse_violations(df):
    '''
    Parse the 'violations' column in the dataframe.
    Each entry in the 'violations' column is expected to contain multiple violations
    separated by '|'. Each violation is in the format:
    "Code. Category - Comments: Comment text"
    The function returns a new dataframe with the following columns:
    - inspection_id
    - violation_code
    - category
    - comment

    Example usage:
    normalized_violations = parse_violations(df)
    '''
    # Fill NA values with empty string to avoid errors
    df['violations'] = df['violations'].fillna('')
    
    # Parse each violation entry
    parsed_rows = df['violations'].apply(parse_comments)
    
    # Create a new dataframe with the parsed data
    violations_df = pd.DataFrame({
        'inspection_id': df['inspection_id'].repeat(parsed_rows.apply(len)),
        # 'dba_name': df['dba_name'].repeat(parsed_rows.apply(len)),
        # 'inspection_date': df['inspection_date'].repeat(parsed_rows.apply(len)),
        # 'inspection_type': df['inspection_type'].repeat(parsed_rows.apply(len)),
        # 'results': df['results'].repeat(parsed_rows.apply(len))
    })
    
    # Extract the parsed data into separate columns
    violations_data = [item for sublist in parsed_rows for item in sublist]
    
    # Create a dataframe from the parsed violations
    violations_details = pd.DataFrame(violations_data)
    
    # Combine the inspection info with violation details
    result_df = pd.concat([violations_df.reset_index(drop=True), 
                          violations_details.reset_index(drop=True)], axis=1)
    
    return result_df

def create_normalized_tables(df):
    """
    Split the food inspection dataframe into two normalized tables:
    1. facility - containing facility information
    2. inspection - containing inspection information
    
    Returns:
    tuple: (facility_df, inspection_df)
    """
    # Create the facility table with unique facility information
    # Use license_num as the primary key
    facility_df = df[['license_num', 'dba_name', 'aka_name', 'facility_type', 
                      'risk', 'address', 'city', 'state', 'zip', 
                      'latitude', 'longitude', 'location']].drop_duplicates(subset=['license_num'])
    
    # Create inspection table with inspection information and foreign key to facility table
    inspection_df = df[['inspection_id', 'license_num', 'inspection_date', 
                        'inspection_type', 'results', 'violations']]
    
    return facility_df, inspection_df


def df_to_sqlite(df, db_name, table_name, if_exists='replace', index=False, add_timestamp=True):
    """
    Save a pandas DataFrame to a SQLite database table with customization options.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to save to SQLite
    db_name : str
        Name of the SQLite database file (will be created if it doesn't exist)
    table_name : str
        Name of the table to create/update in the database
    if_exists : str, optional (default='replace')
        How to behave if the table already exists:
        - 'fail': Raise a ValueError
        - 'replace': Drop the table before inserting new values
        - 'append': Insert new values to the existing table
    index : bool, optional (default=False)
        Whether to include the DataFrame's index as a column
    add_timestamp : bool, optional (default=True)
        Whether to add a metadata table with creation timestamp
        
    Returns:
    --------
    bool
        True if successful
        
    Example:
    --------
    df_to_sqlite(my_dataframe, 'my_database.db', 'my_table', if_exists='append')
    """
    try:
        # Create a database connection
        conn = sqlite3.connect(db_name)
        
        # Save the DataFrame to SQLite
        df.to_sql(table_name, conn, if_exists=if_exists, index=index)
        
        # Add timestamp metadata if requested
        if add_timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor = conn.cursor()
            
            # Create metadata table if it doesn't exist
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                table_name TEXT,
                created_at TEXT,
                rows INTEGER,
                columns INTEGER
            )
            """)
            
            # Add or update metadata for this table
            cursor.execute("""
            INSERT OR REPLACE INTO metadata (table_name, created_at, rows, columns)
            VALUES (?, ?, ?, ?)
            """, (table_name, timestamp, len(df), len(df.columns)))
            
            conn.commit()
        
        # Close the connection
        conn.close()
        
        print(f"Successfully saved DataFrame to {db_name} as table '{table_name}'")
        print(f"Table contains {len(df)} rows and {len(df.columns)} columns")
        
        return True
    
    except Exception as e:
        print(f"Error saving DataFrame to SQLite: {e}")
        return False
    
def query_to_df(db_name, query):
    """
    Run a SQL query on the SQLite database and return the result as a DataFrame.
    
    Parameters:
    -----------
    db_name : str
        Name of the SQLite database file
    query : str
        SQL query to execute
        
    Returns:
    --------
    pandas.DataFrame
        Result of the query as a DataFrame
        
    Example:
    --------
    df = run_query('my_database.db', 'SELECT * FROM my_table')
    """
    try:
        # Create a database connection
        conn = sqlite3.connect(db_name)
        
        # Execute the query and return the result as a DataFrame
        df = pd.read_sql_query(query, conn)
        
        # Close the connection
        conn.close()
        
        return df
    
    except Exception as e:
        print(f"Error running query: {e}")
        return None

def convert_to_tane_input(df):
    """
    Convert pandas DataFrame to TANE input format (list of stripped partitions)
    """
    partitions = []
    for col_idx in range(df.shape[1]):
        # Group records by column values
        value_dict = {}
        for row_idx, value in enumerate(df.iloc[:, col_idx]):
            if value not in value_dict:
                value_dict[value] = set()
            value_dict[value].add(row_idx)

        # Keep only partitions with more than one row
        partition = [indices for indices in value_dict.values() if len(indices) > 1]

        # If PPattern is available, use its fix_desc method
        if 'PPattern' in globals():
            partition = PPattern.fix_desc(partition)

        partitions.append(partition)

    return partitions

def run_tane_via_module(df):
    """
    Run TANE algorithm by directly using the imported module
    """
    # Convert data to TANE format
    tane_input = convert_to_tane_input(df)

    # Create TANE instance and run
    tane_instance = TANE(tane_input)
    start_time = time.time()
    tane_instance.run()
    end_time = time.time()

    # Get results and convert to tuple format with column names
    dependencies = []
    for lhs, rhs in tane_instance.rules:
        # Convert attribute indices to column names
        if isinstance(lhs, tuple):
            lhs_cols = tuple(df.columns[i] for i in lhs)
        else:
            lhs_cols = df.columns[lhs]
        rhs_col = df.columns[rhs]

        # Calculate dependency strength (always 1.0 for TANE as it finds exact dependencies)
        dependencies.append((lhs_cols, rhs_col, 1.0))

    return dependencies, end_time - start_time

def visualize_dependencies(deps, title):
    """
    Visualize discovered functional dependencies as a network graph
    """
    print(f"\nVisualizing {len(deps)} functional dependencies...")

    G = nx.DiGraph()

    # Create nodes and edges
    for lhs, rhs, strength in deps:
        # Handle both single attribute and multi-attribute LHS
        if isinstance(lhs, tuple):
            # For multiple attributes on LHS, create a combined node
            lhs_label = " + ".join(str(attr) for attr in lhs)
            G.add_node(lhs_label, shape='box', style='filled', fillcolor='lightblue')

            # Also add individual attribute nodes and connect them to the combined node
            for attr in lhs:
                if attr not in G:
                    G.add_node(attr, shape='ellipse')
                G.add_edge(attr, lhs_label, style='dashed', weight=0.5)

            # Connect the combined node to RHS
            G.add_edge(lhs_label, rhs, width=strength*3, weight=strength, label=f"{strength:.2f}")
        else:
            # For single attribute, connect directly
            if lhs not in G:
                G.add_node(lhs, shape='ellipse')
            if rhs not in G:
                G.add_node(rhs, shape='ellipse')
            G.add_edge(lhs, rhs, width=strength*3, weight=strength, label=f"{strength:.2f}")

    # Draw the graph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.8, iterations=100)

    # Draw nodes
    node_shapes = nx.get_node_attributes(G, 'shape')
    for shape in set(node_shapes.values()):
        nodes = [node for node, s in node_shapes.items() if s == shape]
        if shape == 'box':
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color='lightblue',
                                   node_size=3000, node_shape='s')
        else:
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color='lightgreen',
                                   node_size=2000, node_shape='o')

    # Draw edges
    edge_widths = [G[u][v].get('width', 1) for u, v in G.edges()]
    edge_styles = [G[u][v].get('style', 'solid') for u, v in G.edges()]

    solid_edges = [(u, v) for (u, v), style in zip(G.edges(), edge_styles) if style == 'solid']
    dashed_edges = [(u, v) for (u, v), style in zip(G.edges(), edge_styles) if style == 'dashed']

    if solid_edges:
        nx.draw_networkx_edges(G, pos, edgelist=solid_edges, width=edge_widths[:len(solid_edges)])
    if dashed_edges:
        nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, width=1, style='dashed')

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='SimHei')

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

    # Save figure
    plt.savefig(f'{title.replace(" ", "_")}.png', dpi=300)
    plt.show()

def analyze_important_columns(deps, important_columns=None):
    """
    Analyze dependencies for important columns
    """
    print("\n========= Important Attribute Dependency Analysis =========")

    if important_columns is None:
        # If not specified, analyze the 5 most active columns
        column_involvement = defaultdict(int)
        for X, Y, _ in deps:
            if isinstance(X, tuple):
                for x in X:
                    column_involvement[x] += 1
            else:
                column_involvement[X] += 1
            column_involvement[Y] += 1

        important_columns = [col for col, _ in sorted(column_involvement.items(),
                                                      key=lambda x: x[1], reverse=True)[:5]]

    for col in important_columns:
        # As determining attribute
        determined = []
        for X, Y, strength in deps:
            if isinstance(X, tuple) and col in X:
                determined.append((X, Y, strength))
            elif X == col:
                determined.append((X, Y, strength))

        if determined:
            print(f"\nAttribute '{col}' helps determine these attributes:")
            for X, Y, strength in sorted(determined, key=lambda x: x[2], reverse=True):
                print(f"  {X} → {Y} (strength: {strength:.4f})")

        # As determined attribute
        determining = [(X, Y, strength) for X, Y, strength in deps if Y == col]
        if determining:
            print(f"\nThese attributes determine '{col}':")
            for X, Y, strength in sorted(determining, key=lambda x: x[2], reverse=True):
                print(f"  {X} → {Y} (strength: {strength:.4f})")

def find_inclusion_dependencies(df, min_confidence=0.95):
    """
    Find inclusion dependencies between columns in a DataFrame
    
    Parameters:
    -----------
    df : pandas DataFrame
        The data to analyze
    min_confidence : float, default=0.95
        Minimum confidence threshold for inclusion dependencies
        
    Returns:
    --------
    list of tuples (A, B, confidence)
        A ⊆ B with confidence score
    """
    print("Finding inclusion dependencies...")
    start_time = time.time()

    # Get unique values for each column
    unique_values = {}
    for col in df.columns:
        unique_values[col] = set(df[col].dropna().unique())

    dependencies = []
    total_comparisons = len(df.columns) * (len(df.columns) - 1)

    # Check each pair of columns
    for i, col_a in enumerate(df.columns):
        for col_b in df.columns:
            if col_a != col_b:
                # Skip if col_a has no values
                if not unique_values[col_a]:
                    continue

                # Check if values in col_a are subset of values in col_b
                overlap = unique_values[col_a].intersection(unique_values[col_b])
                confidence = len(overlap) / len(unique_values[col_a])

                # If confidence meets threshold, record dependency
                if confidence >= min_confidence:
                    dependencies.append((col_a, col_b, confidence))

        # Progress update
        if (i+1) % 5 == 0 or (i+1) == len(df.columns):
            progress = (i+1) * (len(df.columns) - 1) / total_comparisons * 100
            print(f"Progress: {progress:.1f}% ({i+1}/{len(df.columns)} columns processed)")

    print(f"Found {len(dependencies)} inclusion dependencies in {time.time() - start_time:.2f} seconds")
    return dependencies

def visualize_inclusion_dependencies(deps, title):
    """
    Visualize discovered inclusion dependencies as a network graph
    """
    print(f"\nVisualizing {len(deps)} inclusion dependencies...")

    G = nx.DiGraph()

    # Create nodes and edges
    for col_a, col_b, confidence in deps:
        if col_a not in G:
            G.add_node(col_a, shape='ellipse')
        if col_b not in G:
            G.add_node(col_b, shape='ellipse')

        # Edge from A to B means A ⊆ B
        G.add_edge(col_a, col_b, width=confidence*3, weight=confidence,
                   label=f"{confidence:.2f}")

    # Draw the graph
    plt.figure(figsize=(12, 10))

    # Use hierarchical layout for better visualization of inclusion relationships
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except:
        # Fallback to spring layout if graphviz is not available
        pos = nx.spring_layout(G, k=0.8, iterations=100)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightgreen', node_size=2000, node_shape='o')

    # Draw edges with varying width based on confidence
    edge_widths = [G[u][v].get('width', 1) for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths,
                           edge_color='blue', alpha=0.7,
                           arrowsize=20, connectionstyle='arc3,rad=0.1')

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='SimHei')

    # Edge labels (confidences)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

    # Save figure
    plt.savefig(f'{title.replace(" ", "_")}.png', dpi=300)
    plt.show()

def analyze_inclusion_dependencies(deps):
    """
    Analyze inclusion dependencies to extract insights
    """
    print("\n========= Inclusion Dependency Analysis =========")

    # Find columns that are subsets of many other columns
    subset_counts = defaultdict(int)
    for col_a, _, _ in deps:
        subset_counts[col_a] += 1

    # Find columns that contain many other columns
    superset_counts = defaultdict(int)
    for _, col_b, _ in deps:
        superset_counts[col_b] += 1

    # Print top columns that are subsets of others
    print("\nTop columns that are subsets of others (potential foreign keys):")
    for col, count in sorted(subset_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {col}: contained in {count} other columns")

    # Print top columns that contain others
    print("\nTop columns that contain other columns (potential reference tables):")
    for col, count in sorted(superset_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {col}: contains {count} other columns")

    # Find chains of inclusion dependencies
    print("\nChains of inclusion dependencies (transitive relationships):")
    for a, b, conf_ab in deps:
        for c, d, conf_cd in deps:
            if b == c and a != d:
                print(f"  {a} ⊆ {b} ⊆ {d} (confidence: {conf_ab:.2f}, {conf_cd:.2f})")

    # Find bidirectional (equivalent) dependencies
    print("\nBidirectional dependencies (equivalent columns):")
    bidirectional = []
    for a, b, conf_ab in deps:
        for c, d, conf_cd in deps:
            if a == d and b == c and a != b and (a, b) not in bidirectional and (b, a) not in bidirectional:
                bidirectional.append((a, b))
                print(f"  {a} ≡ {b} (confidence: {conf_ab:.2f}, {conf_cd:.2f})")



if __name__ == '__main__':
    food_dataset = pd.read_csv("Food_Inspections_20250216.csv",  dtype={'License #': str, 'Zip': str})
    updated_food_dataset = food_dataset.copy()
    # Apply function
    updated_food_dataset = col_name_changer(updated_food_dataset)

    ##### DATA PROCESSING #####
    updated_food_dataset = process_license_numbers(updated_food_dataset)
    updated_food_dataset = clean_zip_data(updated_food_dataset).drop(columns=['zip','zip_valid'],axis=1).rename(columns={'zip_clean':'zip'})
    updated_food_dataset['state'] = updated_food_dataset.apply(
        lambda row: row['state'] if pd.notna(row['state']) else get_zip_info(row['zip'],'state'),
        axis=1)
    updated_food_dataset['city'] = updated_food_dataset.apply(
        lambda row: row['city'] if pd.notna(row['city']) else get_zip_info(row['zip'],'city'),
        axis=1)
    updated_food_dataset = geocode_missing_coordinates(updated_food_dataset)
    updated_food_dataset = risk_column_transformation(updated_food_dataset)
    updated_food_dataset = standardize_name_columns(updated_food_dataset)
    updated_food_dataset['aka_name'] = updated_food_dataset['aka_name'].fillna(updated_food_dataset['dba_name'])
    updated_food_dataset = fix_city_name(updated_food_dataset)
    updated_food_dataset.to_csv("cleaned_dataset_for_FD.csv", index=False)

    ##### DATA PROFILING #####
    profile_name(updated_food_dataset)
    profile_risk_column(updated_food_dataset)
    profile_inspection_date(updated_food_dataset)
    profile_inspection_type(updated_food_dataset)
    profile_results(updated_food_dataset)
    profile_zip(updated_food_dataset)
    profile_state(updated_food_dataset)
    check_zip_state_city_mapping(updated_food_dataset)
    check_city_state_spelling(updated_food_dataset)
    profile_violations(updated_food_dataset, sample_size=1000)
    verify_violations_structure(updated_food_dataset)
    
    violations_df = parse_violations(updated_food_dataset) # parse the violations, output a separate dataframe. read the docstring for more details
    profile_violations(violations_df)
    result = association_rule_mining(
        updated_food_dataset,
        categorical_columns=['facility_type', 'risk', 'results', 'inspection_type'],
        violation_column='violations'
    )
    
    ##### INGESTING TO SQL DATABASE #####
    facility_df, inspection_df = create_normalized_tables(updated_food_dataset) # Create the normalized tables
    
    # Save tables to SQLite database
    df_to_sqlite(violations_df, 'food_inspections.db', 'violations', if_exists='replace', index=False)
    df_to_sqlite(facility_df, 'food_inspections.db', 'facility', if_exists='replace', index=False)
    df_to_sqlite(inspection_df, 'food_inspections.db', 'inspection', if_exists='replace', index=False)

    #TANE set up
    df = updated_food_dataset
    columns_to_exclude = ['violations']
    # Select subset of columns to speed up algorithm
    df_analysis = df.drop(columns=columns_to_exclude)
    df_analysis = df_analysis.dropna()
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype(str)

    #TANE
    print("Running TANE via direct module import...")
    start = time.time()
    dependencies, runtime = run_tane_via_module(df_analysis)
    print(f"\nTANE algorithm runtime: {runtime:.2f} seconds")
    print(f"Total functional dependencies discovered: {len(dependencies)}")

    if dependencies:
      print("\nSome functional dependencies discovered by TANE algorithm:")
      for dep in sorted(dependencies, key=lambda x: x[2], reverse=True)[:15]:
          X_str = str(dep[0])
          print(f"{X_str} → {dep[1]} (strength: {dep[2]:.4f})")
        
    if dependencies:
        important_columns = ['Inspection ID', 'License #', 'Facility Type', 'Risk', 'Results']
        analyze_important_columns(dependencies, important_columns)
    else:
        print("No functional dependencies discovered or unable to parse results.")

    #IND
    columns_to_exclude = ['Violations' ]
    columns_to_include = ['Inspection ID', 'DBA Name', 'AKA Name', 'License #',
                        'Facility Type', 'Risk', 'City', 'State', 'Zip',
                        'Inspection Type', 'Results','Location']
    df_analysis = df.drop(columns=columns_to_exclude)[columns_to_include]
    df_analysis = df_analysis.fillna('__NULL__')

    inclusion_deps = find_inclusion_dependencies(df_analysis, min_confidence=0.95)
  
    if inclusion_deps:
        print("\nSome inclusion dependencies discovered:")
        for dep in sorted(inclusion_deps, key=lambda x: x[2], reverse=True)[:10]:
            print(f"{dep[0]} ⊆ {dep[1]} (confidence: {dep[2]:.4f})")
