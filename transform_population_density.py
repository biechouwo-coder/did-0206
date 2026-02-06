import pandas as pd
import numpy as np

# 读取数据
file_path = '总数据集_已合并_含碳排放_new.xlsx'
df = pd.read_excel(file_path)

print("=" * 80)
print("人口密度 - 对数转换与缩尾处理")
print("=" * 80)

col_name = '人口密度'

# 1. 查看原始数据统计
print("\n【步骤1: 原始人口密度统计】")
print("-" * 80)
print(f"均值: {df[col_name].mean():.2f}")
print(f"中位数: {df[col_name].median():.2f}")
print(f"标准差: {df[col_name].std():.2f}")
print(f"最小值: {df[col_name].min():.2f}")
print(f"最大值: {df[col_name].max():.2f}")
print(f"缺失值: {df[col_name].isnull().sum()}")

# 检查是否有零值或负值
print(f"\n零值数量: {(df[col_name] == 0).sum()}")
print(f"负值数量: {(df[col_name] < 0).sum()}")

# 2. 生成对数变量
print("\n【步骤2: 生成 ln_人口密度 变量】")
print("-" * 80)
df['ln_人口密度'] = np.log(df[col_name])

print("ln_人口密度 统计:")
print(f"  均值: {df['ln_人口密度'].mean():.4f}")
print(f"  中位数: {df['ln_人口密度'].median():.4f}")
print(f"  标准差: {df['ln_人口密度'].std():.4f}")
print(f"  最小值: {df['ln_人口密度'].min():.4f}")
print(f"  最大值: {df['ln_人口密度'].max():.4f}")

# 3. 查看1%缩尾边界
print("\n【步骤3: 计算1%缩尾边界】")
print("-" * 80)
lower_bound = df['ln_人口密度'].quantile(0.01)
upper_bound = df['ln_人口密度'].quantile(0.99)
print(f"1%分位数（下界）: {lower_bound:.4f}")
print(f"99%分位数（上界）: {upper_bound:.4f}")

# 统计超出边界的值
below_lower = (df['ln_人口密度'] < lower_bound).sum()
above_upper = (df['ln_人口密度'] > upper_bound).sum()
print(f"\n低于下界的值: {below_lower} 个")
print(f"高于上界的值: {above_upper} 个")
print(f"总需缩尾的值: {below_lower + above_upper} 个")

# 4. 执行缩尾处理
print("\n【步骤4: 执行缩尾处理】")
print("-" * 80)
df['ln_人口密度'] = df['ln_人口密度'].clip(lower=lower_bound, upper=upper_bound)
print("已完成1%缩尾处理")

# 5. 处理后数据验证
print("\n【步骤5: 处理后数据验证】")
print("-" * 80)
print(f"最终缺失值: {df['ln_人口密度'].isnull().sum()}")
print(f"最小值: {df['ln_人口密度'].min():.4f}")
print(f"最大值: {df['ln_人口密度'].max():.4f}")
print(f"均值: {df['ln_人口密度'].mean():.4f}")
print(f"标准差: {df['ln_人口密度'].std():.4f}")

# 6. 对比处理前后
print("\n【步骤6: 处理前后对比】")
print("-" * 80)
print(f"{'统计量':<15} {'原始值':<20} {'对数转换后':<20} {'缩尾后':<20}")
print("-" * 75)
print(f"{'均值':<15} {df[col_name].mean():<20.2f} {np.log(df[col_name]).mean():<20.4f} {df['ln_人口密度'].mean():<20.4f}")
print(f"{'标准差':<15} {df[col_name].std():<20.2f} {np.log(df[col_name]).std():<20.4f} {df['ln_人口密度'].std():<20.4f}")
print(f"{'最小值':<15} {df[col_name].min():<20.2f} {np.log(df[col_name]).min():<20.4f} {df['ln_人口密度'].min():<20.4f}")
print(f"{'最大值':<15} {df[col_name].max():<20.2f} {np.log(df[col_name]).max():<20.4f} {df['ln_人口密度'].max():<20.4f}")

# 7. 分布形状改善
print("\n【步骤7: 分布形状改善】")
print("-" * 80)
skew_original = df[col_name].skew()
skew_log = df['ln_人口密度'].skew()
kurt_original = df[col_name].kurtosis()
kurt_log = df['ln_人口密度'].kurtosis()

print(f"{'统计量':<15} {'原始人口密度':<20} {'ln_人口密度(缩尾后)':<20}")
print("-" * 55)
print(f"{'偏度':<15} {skew_original:<20.4f} {skew_log:<20.4f}")
print(f"{'峰度':<15} {kurt_original:<20.4f} {kurt_log:<20.4f}")

print("\n改善效果:")
if abs(skew_log) < abs(skew_original):
    print(f"  偏度改善: {skew_original:.4f} -> {skew_log:.4f}")
if abs(kurt_log) < abs(kurt_original):
    print(f"  峰度改善: {kurt_original:.4f} -> {kurt_log:.4f}")

# 8. 显示数据集形状
print("\n【步骤8: 数据集信息】")
print("-" * 80)
print(f"数据集形状: {df.shape}")
print(f"总变量数: {df.shape[1]}")
print("\n新增变量: ln_人口密度")

# 9. 保存数据
output_file = '总数据集_已合并_含碳排放_new.xlsx'
df.to_excel(output_file, index=False)

print(f"\n【步骤9: 保存数据】")
print("-" * 80)
print(f"数据已保存到: {output_file}")

print("\n" + "=" * 80)
print("处理完成！")
print("=" * 80)
print("\n处理摘要:")
print(f"  生成新变量: ln_人口密度")
print(f"  对数转换: 缓解右偏分布")
print(f"  1%缩尾: 处理了 {below_lower + above_upper} 个极端值")
print(f"  最终数据: {df.shape[0]} 行 x {df.shape[1]} 列")
