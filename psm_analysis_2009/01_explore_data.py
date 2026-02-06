import pandas as pd
import numpy as np

# 读取数据
print("=" * 80)
print("步骤1: 读取并探索数据")
print("=" * 80)

file_path = '../总数据集_已合并_含碳排放_new.xlsx'
df = pd.read_excel(file_path)

print(f"\n数据集形状: {df.shape}")
print(f"变量列表 (共 {len(df.columns)} 个):\n")

# 查找可能的政策变量
policy_vars = []
for col in df.columns:
    print(f"  - {col}")

    # 查找包含treat、policy、treatment等关键词的变量
    col_lower = col.lower()
    if any(keyword in col_lower for keyword in ['treat', 'policy', 'treatment', 'experiment', 'pilot']):
        policy_vars.append(col)

print("\n" + "=" * 80)
print("可能的政策变量:")
print("=" * 80)
if policy_vars:
    for var in policy_vars:
        print(f"\n变量: {var}")
        print(f"  唯一值: {df[var].unique()}")
        print(f"  处理组数量: {(df[var] == 1).sum()}")
        print(f"  对照组数量: {(df[var] == 0).sum()}")
else:
    print("未找到明确的政策变量名称")
    print("\n请手动指定处理组变量")

# 检查年份范围
print("\n" + "=" * 80)
print("年份信息:")
print("=" * 80)
print(f"年份范围: {df['year'].min()} - {df['year'].max()}")
print(f"年份列表: {sorted(df['year'].unique())}")

# 检查2009年的数据
print("\n" + "=" * 80)
print("2009年数据:")
print("=" * 80)
df_2009 = df[df['year'] == 2009]
print(f"2009年观测数量: {len(df_2009)}")
print(f"2009年城市数量: {df_2009['city_name'].nunique()}")

# 检查所需的匹配变量
print("\n" + "=" * 80)
print("匹配变量检查:")
print("=" * 80)
match_vars = ['ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重']
for var in match_vars:
    if var in df.columns:
        missing = df[var].isnull().sum()
        print(f"[OK] {var}: 存在，缺失值 {missing} 个")
    else:
        print(f"[X] {var}: 不存在")

print("\n" + "=" * 80)
