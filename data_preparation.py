import re
import zipcodes
import requests
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz, process
import sqlite3
import missingno as msno
from difflib import get_close_matches
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

##### DATA PROFILING #####
def profiling_zip(df):
    '''
    Function to profile ZIP codes in the dataset.
    Args:
        df: DataFrame containing ZIP codes
        
    Example usage:
    profiling_zip(updated_food_dataset)
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
    
def profiling_state(df):
    '''
    Function to profile states in the dataset.
    Args:
        df: DataFrame containing state data
        
    Example usage:
    profiling_state(updated_food_dataset)
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
            

def analyze_violations_structure(df, sample_size=100):
    """
    Analyze the structure of the 'violations' column in the dataframe.
    
    Parameters:
    sample_size (int): Number of non-empty violation entries to analyze
    
    Returns:
    dict: Analysis results including patterns found, separator counts, etc.
    
    Example usage: 
    analyze_violations_structure(updated_food_dataset, sample_size=1000)
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
    analyze_violations_structure(updated_food_dataset, sample_size=1000)
    verify_violations_structure(updated_food_dataset)
    profiling_zip(updated_food_dataset)
    profiling_state(updated_food_dataset)
    check_zip_state_city_mapping(updated_food_dataset)
    check_city_state_spelling(updated_food_dataset)
    
    
    ##### INGESTING TO SQL DATABASE #####
    violations_df = parse_violations(updated_food_dataset) # parse the violations, output a separate dataframe. read the docstring for more details
    facility_df, inspection_df = create_normalized_tables(updated_food_dataset) # Create the normalized tables
    
    # Save tables to SQLite database
    df_to_sqlite(violations_df, 'food_inspections.db', 'violations', if_exists='replace', index=False)
    df_to_sqlite(facility_df, 'food_inspections.db', 'facility', if_exists='replace', index=False)
    df_to_sqlite(inspection_df, 'food_inspections.db', 'inspection', if_exists='replace', index=False)