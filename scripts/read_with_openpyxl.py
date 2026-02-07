import openpyxl
import sys

try:
    # Load workbook
    wb = openpyxl.load_workbook('总数据集_已合并_含碳排放_new.xlsx')
    sheet = wb.active
    
    # Get header row
    headers = [cell.value for cell in sheet[1]]
    print("Headers (row 1):")
    for i, h in enumerate(headers, 1):
        print(f"{i:2d}. {h}")
    
    print("\nSearching for Dalian (大连市)...")
    print("\nSample data rows (rows 2-10):")
    for row_idx in range(2, min(15, sheet.max_row + 1)):
        row_values = [cell.value for cell in sheet[row_idx]]
        city = row_values[1] if len(row_values) > 1 else None
        year = row_values[2] if len(row_values) > 2 else None
        
        # Print to see what cities we have
        print(f"Row {row_idx}: city='{city}', year={year}")
        
        # Try to identify which row might be Dalian
        if city and '连' in str(city) or '连' in str(city) or 'dalian' in str(city).lower():
            print(f"  --> Found potential Dalian match!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
