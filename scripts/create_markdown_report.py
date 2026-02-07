import csv

# Read CSV
with open('data_export.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Filter for Dalian
dalian_data = [row for row in rows if row['city_name'] == '大连市']

# Sort by year
dalian_data.sort(key=lambda x: int(x['year']))

# Create markdown report
with open('Dalian_DID_Report.md', 'w', encoding='utf-8') as f:
    f.write("# 大连市 DID Analysis Data (2007-2023)\n\n")
    f.write("## Summary\n")
    f.write(f"- **City**: 大连市\n")
    f.write(f"- **Province**: 辽宁省\n")
    f.write(f"- **Total Years**: {len(dalian_data)} (2007-2023)\n")
    f.write(f"- **Treatment Status**: treat={dalian_data[0]['treat']}, post={dalian_data[0]['post']}, did={dalian_data[0]['did']}\n\n")
    
    f.write("## Data Table\n\n")
    f.write("| Year | Treat | Post | DID | 碳排放量_吨 | ln_碳排放量_吨 | 碳排放强度_名义GDP | ln_碳排放强度_名义GDP | ln_real_gdp | ln_人口密度 | 第二产业占GDP比重 |\n")
    f.write("|------|-------|------|-----|-------------|----------------|-------------------|----------------------|------------|-------------|------------------|\n")
    
    for row in dalian_data:
        year = row['year']
        treat = row['treat']
        post = row['post']
        did = row['did']
        
        # Map columns by index based on what we saw
        carbon = row.get('碳排放量_吨', 'N/A')
        ln_carbon = row.get('ln_碳排放量_吨', 'N/A')
        intensity = row.get('碳排放强度_名义GDP', 'N/A')
        ln_intensity = row.get('ln_碳排放强度_名义GDP', 'N/A')
        ln_gdp = row.get('ln_real_gdp', 'N/A')
        ln_pop = row.get('ln_人口密度', 'N/A')
        industry = row.get('第二产业占GDP比重', 'N/A')
        
        f.write(f"| {year} | {treat} | {post} | {did} | {carbon} | {ln_carbon} | {intensity} | {ln_intensity} | {ln_gdp} | {ln_pop} | {industry} |\n")
    
    f.write("\n## Key Observations\n\n")
    
    # Calculate some statistics
    carbon_vals = [float(row.get('碳排放量_吨', 0)) for row in dalian_data if row.get('碳排放量_吨')]
    intensity_vals = [float(row.get('碳排放强度_名义GDP', 0)) for row in dalian_data if row.get('碳排放强度_名义GDP')]
    
    if carbon_vals:
        f.write(f"### Carbon Emissions Trend\n")
        f.write(f"- **2007**: {carbon_vals[0]:,.0f} tons\n")
        f.write(f"- **2023**: {carbon_vals[-1]:,.0f} tons\n")
        f.write(f"- **Change**: {((carbon_vals[-1] - carbon_vals[0]) / carbon_vals[0] * 100):.2f}%\n\n")
    
    if intensity_vals:
        f.write(f"### Carbon Intensity Trend\n")
        f.write(f"- **2007**: {intensity_vals[0]:.2f}\n")
        f.write(f"- **2023**: {intensity_vals[-1]:.2f}\n")
        f.write(f"- **Change**: {((intensity_vals[-1] - intensity_vals[0]) / intensity_vals[0] * 100):.2f}%\n\n")
        f.write(f"Carbon intensity decreased by {((intensity_vals[0] - intensity_vals[-1]) / intensity_vals[0] * 100):.1f}% from 2007 to 2023\n")

print("Report created: Dalian_DID_Report.md")
