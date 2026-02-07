import csv

# Read CSV
with open('data_export.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Filter for Dalian
dalian_data = [row for row in rows if row['city_name'] == '大连市']

# Sort by year
dalian_data.sort(key=lambda x: int(x['year']))

# Print header
print("=" * 180)
print("大连市 DID Analysis Data (2007-2023)")
print("=" * 180)
print(f"{'Year':<6} {'Treat':<6} {'Post':<6} {'DID':<6} {'碳排放量_吨':<15} {'ln_碳排放量_吨':<15} {'碳排放强度_名义GDP':<20} {'ln_碳排放强度':<18} {'ln_real_gdp':<12} {'ln_人口密度':<12} {'第二产业占比':<12}")
print("-" * 180)

# Print data
for row in dalian_data:
    year = row['year']
    treat = row['treat']
    post = row['post']
    did = row['did']
    
    # Column 13 is 碳排放量_吨 (index 13)
    carbon = row.get('碳排放量_吨', row.get(list(row.keys())[13], 'N/A'))
    ln_carbon = row.get('ln_碳排放量_吨', row.get(list(row.keys())[22], 'N/A'))
    intensity = row.get('碳排放强度_名义GDP', row.get(list(row.keys())[14], 'N/A'))
    ln_intensity = row.get('ln_碳排放强度_名义GDP', row.get(list(row.keys())[23], 'N/A'))
    ln_gdp = row.get('ln_real_gdp', 'N/A')
    ln_pop = row.get('ln_人口密度', row.get(list(row.keys())[20], 'N/A'))
    industry = row.get('第二产业占GDP比重', row.get(list(row.keys())[9], 'N/A'))
    
    print(f"{year:<6} {treat:<6} {post:<6} {did:<6} {carbon:<15} {ln_carbon:<15} {intensity:<20} {ln_intensity:<18} {ln_gdp:<12} {ln_pop:<12} {industry:<12}")

print("=" * 180)
print(f"Total years: {len(dalian_data)}")
print("=" * 180)
