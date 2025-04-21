import pandas as pd
import itertools
from collections import defaultdict
import os

def calculate_g3_metric(df, X, Y):
    """
    Calculate g3 metric for approximate functional dependency X→Y
    
    Parameters:
        df: Dataset
        X: Determinant attribute set (list of column names)
        Y: Determined attribute (column name)
        
    Returns:
        g3 metric value (number between 0-1, smaller indicates stronger dependency)
    """
    # Convert X to list if it's not already
    if not isinstance(X, list):
        X = [X]
    
    # Create a copy containing X and Y columns to avoid warnings
    df_copy = df[X + [Y]].copy() if Y not in X else df[X].copy()
    
    # Group by X, check if Y is unique
    violations = 0
    total_groups = 0
    
    # Group by X
    for _, group in df_copy.groupby(X):
        total_groups += 1
        # If Y is not unique in this group, dependency is violated
        if len(group[Y].unique()) > 1:
            # Calculate violations (number of rows in group - 1)
            violations += len(group) - 1
    
    # Calculate g3 metric
    total_rows = len(df)
    if total_rows == 0:
        return 1.0  # Avoid division by zero
    
    g3 = violations / total_rows
    return g3

def is_redundant(current_fd, discovered_fds):
    """
    Check if a functional dependency is redundant based on previously discovered FDs
    
    Parameters:
        current_fd: Tuple (X, Y, g3) representing current FD
        discovered_fds: List of already discovered FDs
        
    Returns:
        Boolean indicating if the current FD is redundant
    """
    X, Y, g3 = current_fd
    X_set = set(X)
    
    # Check if there's a previously discovered FD with smaller X that determines the same Y
    for prev_X, prev_Y, prev_g3 in discovered_fds:
        # If determining the same attribute and previous g3 is similar or better
        if prev_Y == Y and set(prev_X).issubset(X_set) and len(prev_X) < len(X):
            # If previous FD has similar or better confidence
            if prev_g3 <= g3 * 1.1:  # Allow 10% tolerance
                return True
    
    return False

def tane_algorithm(df, columns=None, max_level=2, threshold=0.1, remove_redundant=True):
    """
    TANE algorithm implementation for mining AFDs
    
    Parameters:
        df: Dataset
        columns: Set of columns to consider
        max_level: Maximum level (maximum size of LHS attribute set)
        threshold: g3 metric threshold
        remove_redundant: Whether to remove redundant FDs
        
    Returns:
        List of discovered approximate functional dependencies
    """
    # Default to all columns
    if columns is None:
        columns = df.columns.tolist()
    
    # Initialize results
    discovered_fds = []
    
    # Calculate partitions for each attribute (set of row indices for each attribute value)
    partitions = {}
    for col in columns:
        col_partition = defaultdict(list)
        for idx, val in enumerate(df[col]):
            col_partition[val].append(idx)
        partitions[col] = list(col_partition.values())
    
    # Level-wise traversal
    current_level = [[col] for col in columns]
    
    for level in range(1, max_level + 1):
        print(f"Analyzing level {level} attribute sets...")
        
        # For each attribute set in current level
        for X in current_level:
            X_set = set(X)
            
            # Check all possibilities for X→Y
            for Y in columns:
                if Y in X_set:
                    continue
                
                # Calculate g3 metric
                g3 = calculate_g3_metric(df, X, Y)
                
                # If g3 is below threshold, report the AFD
                if g3 <= threshold:
                    fd = (X, Y, g3)
                    
                    # Check if this FD is redundant
                    if remove_redundant and level > 1 and is_redundant(fd, discovered_fds):
                        continue
                    
                    discovered_fds.append(fd)
        
        # Stop if max level reached
        if level == max_level:
            break
        
        # Generate next level
        next_level = []
        for i in range(len(current_level)):
            for j in range(i+1, len(current_level)):
                # Merge when prefixes are the same
                if current_level[i][:-1] == current_level[j][:-1]:
                    new_set = sorted(set(current_level[i]) | set(current_level[j]))
                    if len(new_set) == level + 1:
                        next_level.append(new_set)
        
        current_level = next_level
    
    # Sort results by g3 value
    discovered_fds.sort(key=lambda x: x[2])
    return discovered_fds

def filter_redundant_afds(afds):
    """
    Filter out redundant AFDs from the discovered set
    
    Parameters:
        afds: List of discovered AFDs in format (X, Y, g3)
        
    Returns:
        Filtered list of non-redundant AFDs
    """
    # Sort AFDs by X attribute size (ascending) and then by g3 (ascending)
    sorted_afds = sorted(afds, key=lambda x: (len(x[0]), x[2]))
    
    non_redundant_afds = []
    
    # For each AFD, check if it's redundant
    for afd in sorted_afds:
        if not is_redundant(afd, non_redundant_afds):
            non_redundant_afds.append(afd)
    
    return non_redundant_afds

def load_dataset(file_path):
    """Load CSV dataset"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    print(f"Loading dataset: {file_path}")
    
    # Try to automatically detect encoding
    try:
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        # If default encoding fails, try other common encodings
        for encoding in ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"Successfully read data using {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Unable to read file with common encodings, please specify correct encoding")
    
    print(f"Dataset loaded, {len(df)} rows, {len(df.columns)} columns")
    print(f"Column names: {', '.join(df.columns.tolist())}")
    return df

def sample_data(df, sample_size=1000, random_state=42):
    """
    Sample data to reduce computation time
    
    Parameters:
        df: Original dataset
        sample_size: Number of rows to sample
        random_state: Seed for reproducibility
        
    Returns:
        Sampled dataframe
    """
    if len(df) <= sample_size:
        print(f"Dataset has {len(df)} rows, using entire dataset")
        return df
    
    print(f"Sampling {sample_size} rows from {len(df)} total rows")
    return df.sample(n=sample_size, random_state=random_state)

def main():
    # In-script configuration
    config = {
        'csv_file': 'cleaned_dataset_for_FD.csv',  # Input CSV file path
        'columns': None,         # Columns to analyze (None for auto-selection)
        'max_lhs': 2,            # Maximum LHS attribute size
        'threshold': 0.05,       # g3 metric threshold
        'output': 'AFD_result.csv',  # Output file path
        'sample_size': 5000,     # Sample size for large datasets
        'remove_redundant': True # Whether to remove redundant AFDs
    }
    
    # Specific columns to analyze (set to None to auto-select)
    selected_columns = [
    "inspection_id", "license_num", "dba_name", "facility_type",
    "risk", "city", "state", "results"
]
    # Example: selected_columns = ["column1", "column2", "column3"]
    
    # Load dataset
    dataset = load_dataset(config['csv_file'])
    
    # Sample data if needed
    if config['sample_size'] > 0:
        dataset = sample_data(dataset, config['sample_size'])
    
    # Handle column selection
    if selected_columns is not None:
        config['columns'] = selected_columns
    
    if config['columns']:
        selected_columns = config['columns']
        # Validate that all specified columns exist
        for col in selected_columns:
            if col not in dataset.columns:
                raise ValueError(f"Column '{col}' not in dataset")
    else:
        # If no columns specified, use all columns, but limit to 10 to avoid combinatorial explosion
        if len(dataset.columns) > 10:
            print(f"Warning: Dataset contains {len(dataset.columns)} columns, selecting first 10 columns for efficiency")
            selected_columns = dataset.columns[:10].tolist()
        else:
            selected_columns = dataset.columns.tolist()
    
    print(f"Analyzing the following columns: {', '.join(selected_columns)}")
    
    # Remove rows with missing values
    valid_rows = dataset[selected_columns].dropna()
    print(f"After removing rows with missing values, {len(valid_rows)} valid rows remain")
    
    # Run TANE algorithm
    print("\nMining AFDs using TANE algorithm...")
    afds = tane_algorithm(
        valid_rows,
        columns=selected_columns,
        max_level=config['max_lhs'],
        threshold=config['threshold'],
        remove_redundant=config['remove_redundant']
    )
    
    # Apply additional redundancy filter if requested
    if config['remove_redundant']:
        original_count = len(afds)
        afds = filter_redundant_afds(afds)
        print(f"Removed {original_count - len(afds)} redundant AFDs")
    
    # Display results
    if not afds:
        print("No approximate functional dependencies discovered.")
    else:
        print(f"Discovered {len(afds)} approximate functional dependencies:")
        for X, Y, g3 in afds:
            confidence = (1 - g3) * 100
            X_str = ", ".join(X)
            print(f"{X_str} → {Y} (confidence: {confidence:.2f}%, g3: {g3:.4f})")
    
    # Save results if output file specified
    if config['output'] and afds:
        result_rows = []
        for X, Y, g3 in afds:
            confidence = (1 - g3) * 100
            result_rows.append({
                'X': ','.join(X),
                'Y': Y,
                'g3': g3,
                'confidence': confidence
            })
        
        result_df = pd.DataFrame(result_rows)
        result_df.to_csv(config['output'], index=False)
        print(f"\nResults saved to {config['output']}")

if __name__ == "__main__":
    main()
