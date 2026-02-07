import pandas as pd
import sys

try:
    # Read the Excel file with different encoding options
    file_path = '总数据集_已合并_含碳排放_new.xlsx'
    df = pd.read_excel(file_path, engine='openpyxl')
    
    print(f"Successfully read file: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    # Print column names as they appear
    print("Column names in file:")
    print(df.columns.tolist())
    print("\n")
    
    # Filter for Dalian - try to find the correct encoding
    # The city name column shows garbled text, let's check unique values
    print("Unique city_name values (first 10):")
    print(df['city_name'].unique()[:10])
    print("\n")
    
    # Try to find Dalian by checking each city
    for idx, row in df.head(20).iterrows():
        if '2007' in str(row.get('year', '')):
            print(f"Sample row - city_name column: '{row.get('city_name')}' year: {row.get('year')}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
