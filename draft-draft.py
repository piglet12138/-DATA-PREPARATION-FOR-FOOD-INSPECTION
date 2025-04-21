import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
import re
from datetime import datetime
import seaborn as sns
import networkx as nx
from sklearn.preprocessing import OneHotEncoder
df = pd.read_csv('Food_Inspections_20250216.csv')
def extract_violation_codes(text):
    if pd.isna(text):
        return []
    # 提取违规代码 (例如: 32, 10, 47 等)
    codes = re.findall(r"\b(\d+)\.(?=\s+[A-Z])", text)
    return codes
transactions = []
for idx, row in df.iterrows():
    transaction = {
        'Facility_Type': row['Facility Type'],
        'Risk': row['Risk'],
        'Results': row['Results'],
        'Inspection_Type': row['Inspection Type']
    }

    # 添加违规代码
    if not pd.isna(row['Violations']):
        violation_codes = extract_violation_codes(row['Violations'])
        for code in violation_codes:
            transaction[f'Violation_{code}'] = 1

    transactions.append(transaction)

trans_df = pd.DataFrame(transactions)
for col in trans_df.columns:
    if col.startswith('Violation_'):
        trans_df[col] = trans_df[col].fillna(0)
categorical_cols = ['Facility_Type', 'Risk', 'Results', 'Inspection_Type']
for col in categorical_cols:
    dummies = pd.get_dummies(trans_df[col], prefix=col)
    trans_df = pd.concat([trans_df, dummies], axis=1)
    trans_df.drop(col, axis=1, inplace=True)
trans_df = trans_df.astype(int)

violation_cols = [col for col in trans_df.columns if col.startswith('Violation_')]
violation_df = trans_df[violation_cols]
frequent_itemsets = apriori(violation_df, min_support=0.1, use_colnames=True)
frequent_itemsets.sort_values(ascending=False, by= 'support')
# 从频繁项集生成关联规则
violation_rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
print("association rules between violations:")
if len(violation_rules) > 0:
    sorted_rules = violation_rules.sort_values('lift', ascending=False)
    for i, rule in sorted_rules.head(10).iterrows():
        antecedent = ", ".join([item.replace('Violation_', 'violation') for item in list(rule['antecedents'])])
        consequent = ", ".join([item.replace('Violation_', 'violation') for item in list(rule['consequents'])])
        print(f"{antecedent} => {consequent} (support: {rule['support']:.3f}, confidence: {rule['confidence']:.3f}, lift: {rule['lift']:.3f})")
else:
    print("no association rules found")