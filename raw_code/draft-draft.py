import pandas as pd
import numpy as np
from collections import defaultdict


def repair_dataset_based_on_fd(df, fds_to_fix):
    """
    Repair missing values in dataset based on functional dependencies (FD)

    Parameters:
        df: pandas DataFrame, dataset to be repaired
        fds_to_fix: list, functional dependencies to repair, format "X → Y"

    Returns:
        repaired_df: pandas DataFrame, repaired dataset
        repair_stats: dict, repair statistics
    """

    print("Starting data repair based on functional dependencies...")

    # Initialize repair suggestions list and statistics
    repair_suggestions = []
    repair_stats = {
        "total_fds": len(fds_to_fix),
        "repairs_applied": 0,
        "repairs_skipped": 0,
        "rows_affected": 0,
        "fd_details": {}
    }

    # Process each specified functional dependency
    for fd_string in fds_to_fix:
        parts = fd_string.split('→')
        if len(parts) != 2:
            print(f"  Warning: Invalid FD format {fd_string}, skipping")
            continue

        x_attr = parts[0].strip()
        y_attr = parts[1].strip()

        print(f"\nAnalyzing functional dependency: {x_attr} → {y_attr}")

        # Handle compound attributes (e.g., "dba_name,risk")
        x_attrs = [x.strip() for x in x_attr.replace('"', '').split(',')]

        # Check if these columns exist
        if not all(attr in df.columns for attr in x_attrs) or y_attr not in df.columns:
            print(f"  Warning: Some attributes missing in dataset, skipping")
            continue

        # Find rows that violate the functional dependency
        violations = defaultdict(list)
        groups = df.groupby(x_attrs)

        for name, group in groups:
            # If one X value corresponds to multiple Y values, there's a violation
            unique_y_values = group[y_attr].dropna().unique()
            if len(unique_y_values) > 1:
                key = tuple(name) if isinstance(name, tuple) else (name,)
                # Store all non-nan y values for this X
                violations[key] = unique_y_values.tolist()
            elif len(unique_y_values) == 1 and group[y_attr].isna().any():
                # Has null values and one unique non-null value - can be repaired
                key = tuple(name) if isinstance(name, tuple) else (name,)
                violations[key] = unique_y_values.tolist()

        # Generate repair suggestions for each violation
        fd_repairs = 0
        for x_value, y_values in violations.items():
            if y_values:  # Ensure we have non-NaN values
                most_common_y = max(set(y_values), key=y_values.count)
                repair_suggestions.append({
                    'fd': fd_string,
                    'x_value': x_value,
                    'current_y_values': y_values,
                    'suggested_y': most_common_y
                })
                fd_repairs += 1

        repair_stats["fd_details"][fd_string] = fd_repairs
        print(f"  Found {fd_repairs} potential repairs")

    # Generate repaired dataset
    repaired_df = df.copy()

    if repair_suggestions:
        print("\nCreating repaired dataset...")
        repairs_applied = 0
        skipped_repairs = 0
        total_rows_affected = 0

        for suggestion in repair_suggestions:
            fd = suggestion['fd']
            x_value = suggestion['x_value']
            suggested_y = suggestion['suggested_y']

            x_attrs = [x.strip() for x in fd.split('→')[0].replace('"', '').split(',')]
            y_attr = fd.split('→')[1].strip()

            # Skip if suggested value is NaN
            if pd.isna(suggested_y):
                print(f"  Skipped: {fd} X={x_value} - no non-null value suggestion available")
                skipped_repairs += 1
                continue

            # Check for license_number rule
            skip_due_to_license_rule = False
            for i, attr in enumerate(x_attrs):
                if attr == 'license_num':
                    # For single attribute case
                    if not isinstance(x_value, tuple) and (x_value == 0 or x_value == 0.0):
                        print(f"  Skipped: {fd} license_num={x_value} - zero license numbers excluded from repairs")
                        skip_due_to_license_rule = True
                        break
                    # For compound attributes case
                    elif isinstance(x_value, tuple) and (x_value[i] == 0 or x_value[i] == 0.0):
                        print(f"  Skipped: {fd} license_num={x_value[i]} - zero license numbers excluded from repairs")
                        skip_due_to_license_rule = True
                        break

            if skip_due_to_license_rule:
                skipped_repairs += 1
                continue

            # Create filter condition
            filter_condition = True
            for i, attr in enumerate(x_attrs):
                if len(x_attrs) > 1:
                    filter_condition = filter_condition & (repaired_df[attr] == x_value[i])
                else:
                    filter_condition = filter_condition & (repaired_df[attr] == x_value)

            # Only fix NaN values in Y
            filter_condition = filter_condition & repaired_df[y_attr].isna()

            # Apply fix
            rows_affected = sum(filter_condition)
            if rows_affected > 0:
                repaired_df.loc[filter_condition, y_attr] = suggested_y
                repairs_applied += 1
                total_rows_affected += rows_affected
                print(f"  Fixed: {fd} X={x_value}, set NaN values to {suggested_y} (affected {rows_affected} rows)")
            else:
                continue

        repair_stats["repairs_applied"] = repairs_applied
        repair_stats["repairs_skipped"] = skipped_repairs
        repair_stats["rows_affected"] = total_rows_affected
        print(
            f"\nRepair complete: Applied {repairs_applied} repairs (skipped {skipped_repairs}), affected {total_rows_affected} rows")
    else:
        print("\nNo violations to fix for the specified FDs")

    return repaired_df, repair_stats


def main():
    # Example usage
    try:
        # Load dataset
        print("Loading dataset...")
        fd = pd.read_csv('cleaned_dataset_for_FD.csv')

        # Define functional dependencies to fix
        fds_to_fix = [
            "dba_name → facility_type",
            "address → zip",
            "location → zip",
            "address → city",
            "address,inspection_date → facility_type",
            "aka_name,inspection_date → location"
        ]

        # Execute repair
        repaired_df, stats = repair_dataset_based_on_fd(fd, fds_to_fix)

        # Save repaired dataset
        repaired_df.to_csv('repaired_dataset.csv', index=False)
        print("Repaired dataset saved as 'repaired_dataset.csv'")

    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()