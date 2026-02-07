import csv

# Read CSV
with open('data_export.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Filter for Dalian
dalian_data = [row for row in rows if row['city_name'] == '大连市']

# Sort by year
dalian_data.sort(key=lambda x: int(x['year']))

# Get all column names to find correct indices
cols = rows[0].keys()
print("Available columns:")
for i, col in enumerate(cols):
    print(f"{i:2d}. {col}")

print("\n" + "="*180)
print("Dalian City (大连市) - DID Analysis Data (2007-2023)")
print("="*180)

# Map column names
for row in dalian_data:
    year = row['year']
    treat = row['treat']
    post = row['post'] 
    did = row['did']
    
    # Get the carbon emissions column (should be column 13: 碳排放量_吨)
    carbon_emissions = row.get('碳排放量_吨', 'N/A')
    
    # Find ln columns
    ln_carbon = None
    ln_intensity = None
    ln_gdp = row.get('ln_real_gdp', 'N/A')
    industry = row.get('第二产业占GDP比重', 'N/A')
    
    # Try to find the ln columns by looking for them
    for key in row.keys():
        if 'ln_' in key.lower() and '碳' in key and '强度' not in key:
            ln_carbon = row[key]
        elif 'ln_' in key.lower() and '碳' in key and '强度' in key:
            ln_intensity = row[key]
        elif 'ln_' in key.lower() and '人口' in key:
            ln_pop = row[key]
    
    intensity = row.get('碳排放强度_名义GDP', 'N/A')
    
    print(f"Year: {year}, Treat: {treat}, Post: {post}, DID: {did}")
    print(f"  Carbon Emissions (碳排放量_吨): {carbon_emissions}")
    print(f"  ln_Carbon Emissions (ln_碳排放量_吨): {ln_carbon}")
    print(f"  Carbon Intensity (碳排放强度_名义GDP): {intensity}")
    print(f"  ln_Carbon Intensity (ln_碳排放强度_名义GDP): {ln_intensity}")
    print(f"  ln_real_gdp: {ln_gdp}")
    print(f"  ln_人口密度: {ln_pop}")
    print(f"  第二产业占GDP比重: {industry}")
    print()

print("="*180)
print(f"Total years: {len(dalian_data)}")
