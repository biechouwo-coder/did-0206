import pandas as pd
import numpy as np

# 读取数据
file_path = '总数据集_已合并_含碳排放_new.xlsx'
df = pd.read_excel(file_path)

print("=" * 80)
print("第二产业占GDP比重 - 数据清洗")
print("=" * 80)

col_name = '第二产业占GDP比重'

# 1. 查看原始缺失值情况
print("\n【步骤1: 原始数据缺失值统计】")
print("-" * 80)
total_missing = df[col_name].isnull().sum()
print(f"总缺失值数量: {total_missing}")
print(f"缺失比例: {total_missing / len(df) * 100:.2f}%")

# 按年份统计缺失值
missing_by_year = df.groupby('year')[col_name].apply(lambda x: x.isnull().sum())
print("\n按年份统计缺失值:")
for year, count in missing_by_year.items():
    if count > 0:
        print(f"  {year}年: {count} 个缺失值")

# 2. 线性插值（按城市分组，按时间排序）
print("\n【步骤2: 线性插值处理】")
print("-" * 80)
print("按城市分组进行时间序列线性插值...")

# 记录插值前的缺失情况
missing_before = df[col_name].isnull().sum()

# 按城市分组插值
df_sorted = df.sort_values(['city_name', 'year'])
df_sorted[col_name] = df_sorted.groupby('city_name')[col_name].transform(
    lambda x: x.interpolate(method='linear', limit_direction='both')
)

# 记录插值后的情况
missing_after = df_sorted[col_name].isnull().sum()
imputed_count = missing_before - missing_after

print(f"插值前缺失值: {missing_before}")
print(f"插值后缺失值: {missing_after}")
print(f"成功插值: {imputed_count} 个")

# 如果还有缺失值，使用城市均值填充
if missing_after > 0:
    print(f"\n仍有 {missing_after} 个缺失值无法插值（可能是首尾年份数据）")
    print("使用城市均值填充...")
    city_mean = df_sorted.groupby('city_name')[col_name].transform('mean')
    df_sorted[col_name] = df_sorted[col_name].fillna(city_mean)
    missing_final = df_sorted[col_name].isnull().sum()
    print(f"填充后缺失值: {missing_final}")

    # 如果还有缺失，使用总体均值
    if missing_final > 0:
        print(f"仍有 {missing_final} 个缺失值，使用总体均值填充...")
        df_sorted[col_name] = df_sorted[col_name].fillna(df_sorted[col_name].mean())
        print(f"最终缺失值: {df_sorted[col_name].isnull().sum()}")

df = df_sorted

# 3. 查看1%缩尾的边界值
print("\n【步骤3: 计算1%缩尾边界】")
print("-" * 80)
lower_bound = df[col_name].quantile(0.01)
upper_bound = df[col_name].quantile(0.99)
print(f"1%分位数（下界）: {lower_bound:.6f} ({lower_bound*100:.2f}%)")
print(f"99%分位数（上界）: {upper_bound:.6f} ({upper_bound*100:.2f}%)")

# 统计超出边界的值
below_lower = (df[col_name] < lower_bound).sum()
above_upper = (df[col_name] > upper_bound).sum()
print(f"\n低于下界的值: {below_lower} 个")
print(f"高于上界的值: {above_upper} 个")
print(f"总需缩尾的值: {below_lower + above_upper} 个")

# 4. 执行缩尾处理
print("\n【步骤4: 执行缩尾处理】")
print("-" * 80)
df[col_name + '_winsorized'] = df[col_name].clip(lower=lower_bound, upper=upper_bound)
print(f"已创建新变量: {col_name + '_winsorized'}")

# 用缩尾后的值替换原变量
df[col_name] = df[col_name + '_winsorized']
df = df.drop(columns=[col_name + '_winsorized'])
print("已用缩尾后的值替换原变量")

# 5. 验证处理结果
print("\n【步骤5: 处理后数据验证】")
print("-" * 80)
print(f"最终缺失值: {df[col_name].isnull().sum()}")
print(f"最小值: {df[col_name].min():.6f} ({df[col_name].min()*100:.2f}%)")
print(f"最大值: {df[col_name].max():.6f} ({df[col_name].max()*100:.2f}%)")
print(f"均值: {df[col_name].mean():.6f} ({df[col_name].mean()*100:.2f}%)")
print(f"标准差: {df[col_name].std():.6f}")

# 6. 对比处理前后统计量
print("\n【步骤6: 处理前后对比】")
print("-" * 80)
# 读取原始数据对比
df_original = pd.read_excel(file_path)
print(f"{'统计量':<15} {'处理前':<15} {'处理后':<15}")
print("-" * 45)
print(f"{'缺失值':<15} {df_original[col_name].isnull().sum():<15} {df[col_name].isnull().sum():<15}")
print(f"{'均值':<15} {df_original[col_name].mean():<15.4f} {df[col_name].mean():<15.4f}")
print(f"{'标准差':<15} {df_original[col_name].std():<15.4f} {df[col_name].std():<15.4f}")
print(f"{'最小值':<15} {df_original[col_name].min():<15.4f} {df[col_name].min():<15.4f}")
print(f"{'最大值':<15} {df_original[col_name].max():<15.4f} {df[col_name].max():<15.4f}")

# 7. 保存清洗后的数据
output_file = '总数据集_已清洗_含碳排放.xlsx'
df.to_excel(output_file, index=False)

print(f"\n【步骤7: 保存数据】")
print("-" * 80)
print(f"清洗后的数据已保存到: {output_file}")
print(f"数据集形状: {df.shape}")

# 同时更新原文件
df.to_excel(file_path, index=False)
print(f"原文件也已更新: {file_path}")

print("\n" + "=" * 80)
print("数据清洗完成！")
print("=" * 80)
print("\n处理摘要:")
print(f"✅ 线性插值填充了 {imputed_count} 个缺失值")
print(f"✅ 1%缩尾处理了 {below_lower + above_upper} 个极端值")
print(f"✅ 数据质量显著提升，无缺失值")
