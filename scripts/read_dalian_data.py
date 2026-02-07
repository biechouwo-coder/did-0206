import pandas as pd
import sys

try:
    # Read the Excel file
    file_path = '总数据集_已合并_含碳排放_new.xlsx'
    df = pd.read_excel(file_path, engine='openpyxl')
    
    print(f"Successfully read file: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    # Filter for Dalian
    df_dalian = df[df['city_name'] == '大连市'].copy()
    
    if len(df_dalian) == 0:
        print("No data found for 大连市 (Dalian)")
        print("\nAvailable city_names:")
        print(df['city_name'].unique())
        sys.exit(0)
    
    # Select columns of interest
    columns_to_show = [
        'Year',
        'treat', 'post', 'did',
        '碳排放量_吨', 'ln_碳排放量_吨',
        '碳排放强度_名义GDP', 'ln_碳排放强度_名义GDP',
        'ln_real_gdp', 'ln_人口密度', '第二产业占GDP比重'
    ]
    
    # Filter to only available columns
    available_cols = [col for col in columns_to_show if col in df_dalian.columns]
    missing_cols = [col for col in columns_to_show if col not in df_dalian.columns]
    
    if missing_cols:
        print(f"Missing columns: {missing_cols}\n")
    
    # Select and sort by year
    result = df_dalian[available_cols].sort_values('Year')
    
    # Format for display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    print('=' * 140)
    print('大连市 (Dalian City) - DID Analysis Variables (2007-2023)')
    print('=' * 140)
    print(result.to_string(index=False))
    print('=' * 140)
    print(f'\nTotal records: {len(result)} years')
    print(f'Year range: {result["Year"].min()} - {result["Year"].max()}')
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
