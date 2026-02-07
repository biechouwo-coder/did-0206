import openpyxl

wb = openpyxl.load_workbook('总数据集_已合并_含碳排放_new.xlsx')
sheet = wb.active

# Collect all data
all_data = []
for row_idx in range(2, sheet.max_row + 1):
    row = sheet[row_idx]
    city = str(row[1].value) if row[1].value else ""
    
    # Check for Dalian
    if len(city) >= 2:
        # Dalian in Chinese is 大连市, let's check the character codes
        city_chars = list(city)
        # Just collect city names to see what we have
        if row[2].value == 2007:  # First year
            all_data.append((row_idx, city, row[0].value))

# Find unique cities for 2007
print("Cities in dataset (year 2007):")
for row_idx, city, province in sorted(all_data)[:30]:
    try:
        print(f"{province} - {city}")
    except:
        print(f"Row {row_idx}: {repr(city)}")
