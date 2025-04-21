# Chicago Food Inspections Data Pipeline

This repo contains a complete data pipeline for processing, cleaning, analyzing, and storing Chicago Food Inspections data. The pipeline consists of several steps:

## Data Loading and Pre-processing
- Loads food inspection data from CSV format
- Transforms column names for consistency
- Cleans and standardizes data fields:
    - License numbers
    - ZIP codes
    - Geographic information (city/state)
    - Missing coordinates (geocoding)
    - Risk categories
    - Business names
- Saves cleaned dataset to CSV for future use

## Data Profiling and Quality Checks
The pipeline performs various data quality checks:
- Profile business names for consistency
- Analyze risk levels distribution
- Validate inspection dates
- Examine inspection types
- Review inspection results
- Verify location data (ZIP codes, states)
- Check for proper mapping between ZIP/state/city
- Inspect violation data structure and content

## Database Normalization and Storage
The data is normalized into three main tables:
1. **Facility** - Information about food establishments
2. **Inspection** - Details about each inspection
3. **Violations** - Individual violations found during inspections

These tables are then stored in a SQLite database called `food_inspections.db`

## How to Run
1. Ensure you have the required dependencies installed (`pandas`, etc.)
2. Place the input CSV file (`Food_Inspections_20250216.csv`) in the working directory
3. Make sure the `data_preparation.py` module is available in your path
4. Execute the script in the command line 
    ```bash
    python data_preparation.py
    ```
5. Review the profiling results to understand data quality
6. Access the normalized data in the SQLite database

The final database structure follows a star schema design that allows for efficient querying across inspection records, facilities, and violations.