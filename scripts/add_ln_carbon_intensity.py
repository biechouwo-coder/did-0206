"""
在总数据集中生成新变量：ln_碳排放强度_名义GDP
"""

import pandas as pd
import numpy as np
import shutil
from datetime import datetime

print("=" * 100)
print("在总数据集中生成新变量：ln_碳排放强度_名义GDP")
print("=" * 100)

# 文件路径
input_file = '总数据集_已合并_含碳排放_new.xlsx'
backup_file = f'总数据集_备份_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
temp_file = '总数据集_已合并_含碳排放_new_temp.xlsx'

# 读取数据
print(f"\n读取文件: {input_file}")
try:
    df = pd.read_excel(input_file)
    print(f"[OK] 文件读取成功")
except Exception as e:
    print(f"[错误] 无法读取文件: {e}")
    exit(1)

print(f"原始数据形状: {df.shape}")
print(f"原始列数: {len(df.columns)}")

# 检查碳排放强度列是否存在
target_var = '碳排放强度_名义GDP'
new_var = 'ln_碳排放强度_名义GDP'

if target_var not in df.columns:
    print(f"\n[错误] 数据集中不存在 '{target_var}' 列")
    print(f"可用列: {list(df.columns)}")
    exit(1)

# 如果ln_碳排放强度_名义GDP已存在，询问是否覆盖
if new_var in df.columns:
    print(f"\n[注意] 变量 '{new_var}' 已存在，将被覆盖")

# 统计信息
print(f"\n原始{target_var}统计:")
print(df[target_var].describe())

# 检查缺失值、零值、负值
missing = df[target_var].isna().sum()
zero_values = (df[target_var] == 0).sum()
negative_values = (df[target_var] < 0).sum()

print(f"\n数据质量检查:")
print(f"  缺失值: {missing}")
print(f"  零值: {zero_values}")
print(f"  负值: {negative_values}")

# 创建ln_碳排放强度_名义GDP变量
print(f"\n生成新变量: {new_var}")
df[new_var] = np.log(df[target_var])

# 统计新变量
print(f"\n新变量统计:")
print(df[new_var].describe())

# 检查是否有无穷值（由零值或负值取对数产生）
inf_values = np.isinf(df[new_var]).sum()
print(f"\n无穷值数量: {inf_values}")

if inf_values > 0:
    print(f"[警告] 存在 {inf_values} 个无穷值（由零值或负值取对数产生）")

# 1. 先创建备份
print(f"\n创建备份: {backup_file}")
try:
    shutil.copy2(input_file, backup_file)
    print(f"[OK] 备份创建成功")
except Exception as e:
    print(f"[错误] 无法创建备份: {e}")
    exit(1)

# 2. 保存到临时文件
print(f"\n保存到临时文件: {temp_file}")
try:
    df.to_excel(temp_file, index=False, engine='openpyxl')
    print(f"[OK] 临时文件保存成功")
except Exception as e:
    print(f"[错误] 无法保存临时文件: {e}")
    exit(1)

# 3. 尝试替换原文件
print(f"\n尝试替换原文件...")
try:
    import os
    if os.path.exists(input_file):
        os.remove(input_file)
    os.rename(temp_file, input_file)
    print(f"[OK] 原文件已更新")
except Exception as e:
    print(f"[警告] 无法自动替换原文件: {e}")
    print(f"\n[手动操作] 请按以下步骤操作:")
    print(f"  1. 关闭Excel程序（如果打开了 {input_file}）")
    print(f"  2. 手动删除 {input_file}")
    print(f"  3. 将 {temp_file} 重命名为 {input_file}")
    print(f"\n或者直接使用临时文件: {temp_file}")
    exit(1)

print(f"\n更新后数据形状: {df.shape}")
print(f"更新后列数: {len(df.columns)}")

# 显示新增的列
print(f"\n新增列:")
print(f"  {new_var}")

# 显示所有列名
print(f"\n所有列名:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

print("\n" + "=" * 100)
print("完成!")
print("=" * 100)
print(f"\n备份文件: {backup_file}")
print(f"更新文件: {input_file}")
