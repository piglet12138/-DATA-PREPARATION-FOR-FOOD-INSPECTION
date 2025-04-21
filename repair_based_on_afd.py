import pandas as pd
import json
import os

# Check if validation report exists
validation_report_path = 'fd_validation_report.json'
if not os.path.exists(validation_report_path):
    print("Error: Validation report not found. Please run validate_afd.py first.")
    exit(1)

# Load validation report
print("Loading validation report...")
with open(validation_report_path, 'r') as f:
    validation_report = json.load(f)

# Load dataset
dataset_path = 'cleaned_dataset_for_FD.csv'
print(f"Loading dataset from {dataset_path}...")
data = pd.read_csv(dataset_path)

# Get FDs to fix from validation report
FDS_TO_FIX = validation_report["fds_to_fix"]
print(f"Found {len(FDS_TO_FIX)} FDs to fix in validation report:")
for fd in FDS_TO_FIX:
    print(f"  - {fd}")

# Prepare repair suggestions
repair_suggestions = []

# Process FD details from validation report
for fd_detail in validation_report["fd_details"]:
    fd_string = fd_detail["fd"]
    
    # Skip if not selected for repair or has no violations
    if not fd_detail["selected_for_repair"] or fd_detail.get("violation_count", 0) == 0:
        continue
    
    # Skip if there was an error with this FD
    if "error" in fd_detail:
        print(f"Skipping {fd_string} due to error: {fd_detail['error']}")
        continue
    
    print(f"\nProcessing repairs for FD: {fd_string}")
    
    # Process violations to create repair suggestions
    for violation in fd_detail["violations"]:
        x_value = violation["x_value"]
        y_values = violation["y_values"]
        most_common_y = violation.get("most_common_y")
        
        # Skip if no most common value (all NaN)
        if most_common_y is None:
            print(f"  Skipping X={x_value} - no valid most common Y value")
            continue
        
        # Create repair suggestion
        repair_suggestions.append({
            'fd': fd_string,
            'x_value': x_value if isinstance(x_value, str) else tuple(x_value),
            'current_y_values': y_values,
            'suggested_y': most_common_y
        })

# Apply repairs
if repair_suggestions:
    print(f"\nFound {len(repair_suggestions)} potential repairs to apply")
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
            filter_condition = filter_condition & (repaired_data[attr] == x_value[i] if isinstance(x_value, tuple) else repaired_data[attr] == x_value)
        
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
    output_path = 'repaired_dataset.csv'
    repaired_data.to_csv(output_path, index=False)
    print(f"\nRepaired dataset saved as '{output_path}' with {repairs_applied} repairs applied ({skipped_repairs} skipped)")
    
    # Update validation report with repair info
    validation_report["repairs_applied"] = repairs_applied
    validation_report["repairs_skipped"] = skipped_repairs
    
    # Save updated validation report
    with open('fd_repair_report.json', 'w') as f:
        json.dump(validation_report, f, indent=2)
    print("Repair report saved as 'fd_repair_report.json'")
else:
    print("\nNo repairs to apply for the specified FDs") 