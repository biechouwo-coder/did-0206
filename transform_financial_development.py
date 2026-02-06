import pandas as pd
import numpy as np

# 读取数据
file_path = '总数据集_已合并_含碳排放_new.xlsx'
df = pd.read_excel(file_path)

print("=" * 80)
print("金融发展水平 - 插值、对数转换与缩尾处理")
print("=" * 80)

col_name = '金融发展水平'

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

# 3. 生成对数变量
print("\n【步骤3: 生成 ln_金融发展水平 变量】")
print("-" * 80)
print("原始金融发展水平统计:")
print(f"  均值: {df[col_name].mean():.4f}")
print(f"  中位数: {df[col_name].median():.4f}")
print(f"  标准差: {df[col_name].std():.4f}")
print(f"  最小值: {df[col_name].min():.4f}")
print(f"  最大值: {df[col_name].max():.4f}")

# 检查是否有零值或负值
print(f"\n零值数量: {(df[col_name] == 0).sum()}")
print(f"负值数量: {(df[col_name] < 0).sum()}")

# 生成对数变量
df['ln_金融发展水平'] = np.log(df[col_name])

print("\nln_金融发展水平 统计:")
print(f"  均值: {df['ln_金融发展水平'].mean():.4f}")
print(f"  中位数: {df['ln_金融发展水平'].median():.4f}")
print(f"  标准差: {df['ln_金融发展水平'].std():.4f}")
print(f"  最小值: {df['ln_金融发展水平'].min():.4f}")
print(f"  最大值: {df['ln_金融发展水平'].max():.4f}")

# 4. 查看1%缩尾边界
print("\n【步骤4: 计算1%缩尾边界】")
print("-" * 80)
lower_bound = df['ln_金融发展水平'].quantile(0.01)
upper_bound = df['ln_金融发展水平'].quantile(0.99)
print(f"1%分位数（下界）: {lower_bound:.4f}")
print(f"99%分位数（上界）: {upper_bound:.4f}")

# 统计超出边界的值
below_lower = (df['ln_金融发展水平'] < lower_bound).sum()
above_upper = (df['ln_金融发展水平'] > upper_bound).sum()
print(f"\n低于下界的值: {below_lower} 个")
print(f"高于上界的值: {above_upper} 个")
print(f"总需缩尾的值: {below_lower + above_upper} 个")

# 5. 执行缩尾处理
print("\n【步骤5: 执行缩尾处理】")
print("-" * 80)
df['ln_金融发展水平'] = df['ln_金融发展水平'].clip(lower=lower_bound, upper=upper_bound)
print("已完成1%缩尾处理")

# 6. 处理后数据验证
print("\n【步骤6: 处理后数据验证】")
print("-" * 80)
print(f"最终缺失值: {df['ln_金融发展水平'].isnull().sum()}")
print(f"最小值: {df['ln_金融发展水平'].min():.4f}")
print(f"最大值: {df['ln_金融发展水平'].max():.4f}")
print(f"均值: {df['ln_金融发展水平'].mean():.4f}")
print(f"标准差: {df['ln_金融发展水平'].std():.4f}")

# 7. 对比处理前后
print("\n【步骤7: 处理前后对比】")
print("-" * 80)
# 读取原始数据对比
df_original_temp = pd.read_excel(file_path)

print(f"{'统计量':<20} {'原始值':<15} {'插值后':<15} {'对数后':<15} {'缩尾后':<15}")
print("-" * 80)
print(f"{'缺失值':<20} {df_original_temp[col_name].isnull().sum():<15} {df[col_name].isnull().sum():<15} {df['ln_金融发展水平'].isnull().sum():<15} {df['ln_金融发展水平'].isnull().sum():<15}")
print(f"{'均值':<20} {df_original_temp[col_name].mean():<15.4f} {df[col_name].mean():<15.4f} {np.log(df[col_name]).mean():<15.4f} {df['ln_金融发展水平'].mean():<15.4f}")
print(f"{'标准差':<20} {df_original_temp[col_name].std():<15.4f} {df[col_name].std():<15.4f} {np.log(df[col_name]).std():<15.4f} {df['ln_金融发展水平'].std():<15.4f}")
print(f"{'最小值':<20} {df_original_temp[col_name].min():<15.4f} {df[col_name].min():<15.4f} {np.log(df[col_name]).min():<15.4f} {df['ln_金融发展水平'].min():<15.4f}")
print(f"{'最大值':<20} {df_original_temp[col_name].max():<15.4f} {df[col_name].max():<15.4f} {np.log(df[col_name]).max():<15.4f} {df['ln_金融发展水平'].max():<15.4f}")

# 8. 分布形状改善
print("\n【步骤8: 分布形状改善】")
print("-" * 80)
skew_original = df_original_temp[col_name].skew()
skew_log = df['ln_金融发展水平'].skew()
kurt_original = df_original_temp[col_name].kurtosis()
kurt_log = df['ln_金融发展水平'].kurtosis()

print(f"{'统计量':<15} {'原始金融发展水平':<20} {'ln_金融发展水平(缩尾后)':<20}")
print("-" * 55)
print(f"{'偏度':<15} {skew_original:<20.4f} {skew_log:<20.4f}")
print(f"{'峰度':<15} {kurt_original:<20.4f} {kurt_log:<20.4f}")

print("\n改善效果:")
if abs(skew_log) < abs(skew_original):
    print(f"  偏度改善: {skew_original:.4f} -> {skew_log:.4f}")
if abs(kurt_log) < abs(kurt_original):
    print(f"  峰度改善: {kurt_original:.4f} -> {kurt_log:.4f}")

# 9. 显示数据集形状
print("\n【步骤9: 数据集信息】")
print("-" * 80)
print(f"数据集形状: {df.shape}")
print(f"总变量数: {df.shape[1]}")
print("\n新增变量: ln_金融发展水平")

# 10. 保存数据
output_file = '总数据集_已合并_含碳排放_new.xlsx'
df.to_excel(output_file, index=False)

print(f"\n【步骤10: 保存数据】")
print("-" * 80)
print(f"数据已保存到: {output_file}")

print("\n" + "=" * 80)
print("处理完成！")
print("=" * 80)
print("\n处理摘要:")
print(f"  线性插值填充了 {imputed_count} 个缺失值")
print(f"  生成新变量: ln_金融发展水平")
print(f"  对数转换: 缓解右偏分布")
print(f"  1%缩尾: 处理了 {below_lower + above_upper} 个极端值")
print(f"  最终数据: {df.shape[0]} 行 x {df.shape[1]} 列")
