import pandas as pd

# Read Excel
df = pd.read_excel('总数据集_已合并_含碳排放_new.xlsx', engine='openpyxl')

# Save to CSV with UTF-8 BOM
df.to_csv('data_export.csv', index=False, encoding='utf-8-sig')

print(f"Exported {len(df)} rows to data_export.csv")
print(f"Columns: {len(df.columns)}")
