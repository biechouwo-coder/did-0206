import pandas as pd

df = pd.read_excel('总数据集_已合并_含碳排放_new.xlsx', engine='openpyxl')

print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
print("All columns:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")
print("\nFirst 3 rows:")
print(df.head(3))
