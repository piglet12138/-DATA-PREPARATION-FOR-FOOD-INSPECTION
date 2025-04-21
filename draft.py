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
from mlxtend.frequent_patterns import apriori, association_rules


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

updated_food_dataset = pd.read_csv("cleaned_dataset_for_FD.csv",  dtype={'license_num': str, 'zip': str})
result = association_rule_mining(
    updated_food_dataset,
    categorical_columns=['facility_type', 'risk', 'results', 'inspection_type'],
    violation_column='violations'
)