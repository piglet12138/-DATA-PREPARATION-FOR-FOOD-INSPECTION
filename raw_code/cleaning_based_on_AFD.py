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
# Example: ["dba_name → facility_type", "address → zip"]
FDS_TO_FIX = [
    # Add your FDs to fix here(fix nan)
    "dba_name → facility_type",
    "address → zip",
    "location → zip"
    "address → city"
    "address,inspection_date → facility_type",
    "aka_name,inspection_date → location"

]

print(f"Analyzed {len(afd_results)} potential functional dependencies")
print(f"Found {len(high_confidence_fds)} high confidence FDs (confidence >= {CONFIDENCE_THRESHOLD}%)")

# Store repair suggestions and validation info
repair_suggestions = []
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
            
            # Only suggest repairs for FDs that are in the fix list
            if fd_string in FDS_TO_FIX:
                # Get most frequent non-NaN value if available
                if y_values:  # Make sure we have some non-NaN values
                    most_common_y = max(set(y_values), key=y_values.count)
                    repair_suggestions.append({
                        'fd': fd_string,
                        'x_value': x_value,
                        'current_y_values': y_values,
                        'suggested_y': most_common_y
                    })
            
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

# Generate repaired dataset
if repair_suggestions:
    print("\nCreating repaired dataset...")
    repaired_data = data.copy()
    repairs_applied = 0
    skipped_repairs = 0
    
    for suggestion in repair_suggestions:
        fd = suggestion['fd']
        x_value = suggestion['x_value']
        suggested_y = suggestion['suggested_y']
        
        x_attrs = [x.strip() for x in suggestion['fd'].split('→')[0].replace('"', '').split(',')]
        y_attr = suggestion['fd'].split('→')[1].strip()
        
        # Skip if suggested value is NaN
        if pd.isna(suggested_y):
            print(f"  Skipped: {fd} where X={x_value} - no non-NaN values available as suggestion")
            skipped_repairs += 1
            continue
            
        # Check for license_number rule
        skip_due_to_license_rule = False
        for i, attr in enumerate(x_attrs):
            if attr == 'license_num':
                # For single attribute case
                if not isinstance(x_value, tuple) and (x_value == 0 or x_value == 0.0):
                    print(f"  Skipped: {fd} where license_num={x_value} - zero license numbers are excluded from repairs")
                    skip_due_to_license_rule = True
                    break
                # For compound attributes case
                elif isinstance(x_value, tuple) and (x_value[i] == 0 or x_value[i] == 0.0):
                    print(f"  Skipped: {fd} where license_num={x_value[i]} - zero license numbers are excluded from repairs")
                    skip_due_to_license_rule = True
                    break
        
        if skip_due_to_license_rule:
            skipped_repairs += 1
            continue
        
        # Create filter condition
        filter_condition = True
        for i, attr in enumerate(x_attrs):
            filter_condition = filter_condition & (repaired_data[attr] == x_value[i] if len(x_attrs) > 1 else repaired_data[attr] == x_value)
        
        # Only fix NaN values in Y
        filter_condition = filter_condition & repaired_data[y_attr].isna()
        
        # Apply fix
        rows_affected = sum(filter_condition)
        if rows_affected > 0:
            repaired_data.loc[filter_condition, y_attr] = suggested_y
            repairs_applied += 1
            print(f"  Fixed: {fd} where X={x_value}, set NaN Y values to {suggested_y} (affected {rows_affected} rows)")
        else:
            print(f"  No NaN values to fix for {fd} where X={x_value}")
    
    # Save repaired dataset
    repaired_data.to_csv('repaired_dataset.csv', index=False)
    print(f"Repaired dataset saved as 'repaired_dataset.csv' with {repairs_applied} repairs applied ({skipped_repairs} skipped)")
    validation_report["repairs_applied"] = repairs_applied
    validation_report["repairs_skipped"] = skipped_repairs
else:
    print("\nNo violations to fix for the specified FDs")
    validation_report["repairs_applied"] = 0
    validation_report["repairs_skipped"] = 0

# Save validation report
with open('fd_validation_report.json', 'w') as f:
    json.dump(validation_report, f, indent=2)
print("Validation report saved as 'fd_validation_report.json'")