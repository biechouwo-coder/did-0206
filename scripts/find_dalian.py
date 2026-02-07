import openpyxl
import sys

# Redirect output to file with UTF-8 encoding
sys.stdout = open('dalian_data_output.txt', 'w', encoding='utf-8')

try:
    # Load workbook
    wb = openpyxl.load_workbook('总数据集_已合并_含碳排放_new.xlsx')
    sheet = wb.active
    
    # Get header row
    headers = [str(cell.value) for cell in sheet[1]]
    
    print("=" * 140)
    print("大连市 - DID Analysis Variables")
    print("=" * 140)
    
    # Print header
    header_str = f"{'Row':<6} {'Province':<15} {'City':<15} {'Year':<6} {'Treat':<6} {'Post':<6} {'DID':<6} {'ln_碳排放量_吨':<15} {'ln_碳排放强度_名义GDP':<20} {'ln_real_gdp':<12} {'ln_人口密度':<12} {'第二产业占GDP比重':<18}"
    print(header_str)
    print("-" * 140)
    
    # Search for all rows and find Dalian
    dalian_rows = []
    for row_idx in range(2, sheet.max_row + 1):
        row_values = [cell.value for cell in sheet[row_idx]]
        city = str(row_values[1]) if len(row_values) > 1 else ""
        
        # Check if this is Dalian (大连市)
        if '大连' in city or 'Dalian' in city or 'dalian' in city.lower():
            year = row_values[2] if len(row_values) > 2 else ""
            treat = row_values[16] if len(row_values) > 16 else ""
            post = row_values[17] if len(row_values) > 17 else ""
            did = row_values[18] if len(row_values) > 18 else ""
            ln_carbon = row_values[22] if len(row_values) > 22 else ""  # ln_碳排放量_吨
            ln_intensity = row_values[23] if len(row_values) > 23 else ""  # ln_碳排放强度_名义GDP
            ln_gdp = row_values[19] if len(row_values) > 19 else ""  # ln_real_gdp
            ln_pop = row_values[20] if len(row_values) > 20 else ""  # ln_人口密度
            industry = row_values[9] if len(row_values) > 9 else ""  # 第二产业占GDP比重
            
            dalian_rows.append((row_idx, year, treat, post, did, ln_carbon, ln_intensity, ln_gdp, ln_pop, industry))
    
    # Sort by year and print
    dalian_rows.sort(key=lambda x: x[1] if x[1] is not None else 0)
    
    for row_data in dalian_rows:
        row_idx, year, treat, post, did, ln_carbon, ln_intensity, ln_gdp, ln_pop, industry = row_data
        province = row_values[0] if len(row_values) > 0 else ""
        city = "大连市"
        
        row_str = f"{row_idx:<6} {province:<15} {city:<15} {year if year else '':<6} {str(treat) if treat else '':<6} {str(post) if post else '':<6} {str(did) if did else '':<6} {str(ln_carbon) if ln_carbon else '':<15} {str(ln_intensity) if ln_intensity else '':<20} {str(ln_gdp) if ln_gdp else '':<12} {str(ln_pop) if ln_pop else '':<12} {str(industry) if industry else '':<18}"
        print(row_str)
    
    print("=" * 140)
    print(f"\nTotal records found: {len(dalian_rows)}")
    
    if dalian_rows:
        years = [r[1] for r in dalian_rows if r[1] is not None]
        if years:
            print(f"Year range: {min(years)} - {max(years)}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

sys.stdout.close()
