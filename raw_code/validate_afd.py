import pandas as pd
import numpy as np
from collections import defaultdict
import json
import datetime

# Load AFD results
afd_results = pd.read_csv('AFD_result.csv')

# Load main dataset (adjust path as needed)
data = pd.read_csv('cleaned_dataset_for_FD.csv')

# Confidence threshold - can be adjusted as needed
CONFIDENCE_THRESHOLD = 95.0

# Filter high confidence functional dependencies
high_confidence_fds = afd_results[afd_results['confidence'] >= CONFIDENCE_THRESHOLD]

# Specify which FDs to fix (list of strings in format "X → Y")
FDS_TO_FIX = [
    # Add your FDs to fix here
    "address,inspection_date → facility_type",
    "dba_name → facility_type",
    "address → zip",
    "location → zip",
    "address → city",
    "aka_name,inspection_date → location"
]

print(f"Analyzed {len(afd_results)} potential functional dependencies")
print(f"Found {len(high_confidence_fds)} high confidence FDs (confidence >= {CONFIDENCE_THRESHOLD}%)")

# Create validation report
validation_report = {
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "total_fds_analyzed": len(high_confidence_fds),
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "fds_to_fix": FDS_TO_FIX,
    "fd_details": []
}

# Analyze each high confidence functional dependency
for _, row in high_confidence_fds.iterrows():
    x_attr = row['X']
    y_attr = row['Y']
    confidence = row['confidence']
    
    # Format FD string for comparison
    fd_string = f"{x_attr} → {y_attr}"
    
    # Handle compound attributes (e.g., "dba_name,risk")
    x_attrs = [x.strip() for x in x_attr.replace('"', '').split(',')]
    
    print(f"\nAnalyzing FD: {x_attr} → {y_attr} (confidence: {confidence:.2f}%)")
    
    fd_report = {
        "fd": fd_string,
        "confidence": float(confidence),
        "selected_for_repair": fd_string in FDS_TO_FIX,
        "violations": []
    }
    
    # Check if these columns exist in the dataset
    if not all(attr in data.columns for attr in x_attrs) or y_attr not in data.columns:
        print(f"  Warning: Some attributes don't exist in the dataset, skipping")
        fd_report["error"] = "Missing attributes in dataset"
        validation_report["fd_details"].append(fd_report)
        continue
    
    # Find rows that violate the functional dependency
    violations = defaultdict(list)
    groups = data.groupby(x_attrs)
    
    for name, group in groups:
        # If one X value corresponds to multiple Y values, there's a violation
        unique_y_values = group[y_attr].unique()
        if len(unique_y_values) > 1:
            key = tuple(name) if isinstance(name, tuple) else (name,)
            # Store all non-nan y values for this X
            non_nan_y_values = [y for y in group[y_attr].tolist() if not pd.isna(y)]
            violations[key] = non_nan_y_values if non_nan_y_values else group[y_attr].tolist()
    
    # Report violations
    if violations:
        total_violations = sum(len(group) - 1 for group in violations.values())
        print(f"  Found {len(violations)} distinct X values with violations, total {total_violations} violations")
        fd_report["violation_count"] = len(violations)
        fd_report["total_violations"] = total_violations
        
        # For a small number of violations, provide detailed information
        violation_limit = 5
        for i, (x_value, y_values) in enumerate(list(violations.items())[:violation_limit]):
            print(f"  X value {x_value} corresponds to multiple Y values: {y_values}")
            
            violation_detail = {
                "x_value": x_value if isinstance(x_value, str) else list(x_value),
                "y_values": y_values,
                "most_common_y": max(set(y_values), key=y_values.count) if y_values else None,
                "will_fix": fd_string in FDS_TO_FIX
            }
            fd_report["violations"].append(violation_detail)
            
        if len(violations) > violation_limit:
            print(f"  Too many violations, only showing the first {violation_limit}...")
    else:
        print(f"  Great! This functional dependency holds perfectly in the dataset")
        fd_report["violation_count"] = 0
        fd_report["total_violations"] = 0
    
    validation_report["fd_details"].append(fd_report)

# Save validation report
with open('fd_validation_report.json', 'w') as f:
    json.dump(validation_report, f, indent=2)
print("\nValidation report saved as 'fd_validation_report.json'")
print("Run repair_based_on_afd.py to apply repairs based on this validation.") 