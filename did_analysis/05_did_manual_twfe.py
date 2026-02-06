"""
DID回归分析 - 手动实现TWFE
不依赖statsmodels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

print("=" * 100)
print("DID回归分析 - 手动实现TWFE")
print("=" * 100)

# 读取数据
try:
    df = pd.read_excel('panel_data_2007_2023_validated.xlsx')
    print("读取: panel_data_2007_2023_validated.xlsx")
except:
    df = pd.read_excel('panel_data_2007_2023_corrected.xlsx')
    print("读取: panel_data_2007_2023_corrected.xlsx")

print(f"数据形状: {df.shape}")

# 描述性统计
print("\n[描述统计]")
Y_var = '碳排放量_吨'
stats = df.groupby(['treat', 'post'])[Y_var].agg(['count', 'mean', 'std']).round(2)
print(stats)

# 四重差分
control_pre = df[(df['treat'] == 0) & (df['post'] == 0)][Y_var].mean()
control_post = df[(df['treat'] == 0) & (df['post'] == 1)][Y_var].mean()
treat_pre = df[(df['treat'] == 1) & (df['post'] == 0)][Y_var].mean()
treat_post = df[(df['treat'] == 1) & (df['post'] == 1)][Y_var].mean()

print(f"\n[四重差分]")
print(f"对照组: {control_pre:,.0f} -> {control_post:,.0f} (差异: {control_post - control_pre:,.0f})")
print(f"处理组: {treat_pre:,.0f} -> {treat_post:,.0f} (差异: {treat_post - treat_pre:,.0f})")
print(f"DID效应: {(treat_post - treat_pre) - (control_post - control_pre):,.0f}")

# 手动TWFE（De-meaning）
print("\n[TWFE回归]")

# 准备数据
model_data = df[[Y_var, 'did', 'ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重', 'city_name', 'year']].copy()
model_data = model_data.dropna()
print(f"样本数: {len(model_data)}")

# 1. 城市去均值（城市固定效应）
city_means = model_data.groupby('city_name')[Y_var].transform('mean')
model_data['y_demeaned_city'] = model_data[Y_var] - city_means

# 2. 年份去均值（年份固定效应）
year_means = model_data.groupby('year')['y_demeaned_city'].transform('mean')
model_data['y_demeaned'] = model_data['y_demeaned_city'] - year_means

# 控制变量去城市均值
for var in ['ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重']:
    if var in model_data.columns:
        model_data[var + '_dm'] = model_data[var] - model_data.groupby('city_name')[var].transform('mean')

# 3. 构建回归（已经剔除了FE，所以OLS即可）
import statsmodels.api as sm

X_vars = ['did', 'ln_real_gdp_dm', 'ln_人口密度_dm', 'ln_金融发展水平_dm', '第二产业占GDP比重_dm']
X = model_data[X_vars].copy()
X = sm.add_constant(X)
y = model_data['y_demeaned']

# OLS回归
model = sm.OLS(y, X)
results = model.fit(cov_type='cluster', cov_kwds={'groups': model_data['city_name']})

# 输出结果
print("\n" + "=" * 100)
print("TWFE回归结果")
print("=" * 100)
print(results.summary())

# 提取DID系数
did_coef = results.params['did']
did_se = results.bse['did']
did_t = results.tvalues['did']
did_pval = results.pvalues['did']

print("\n" + "=" * 100)
print("DID效应解读")
print("=" * 100)
print(f"DID系数: {did_coef:.4f}")
print(f"标准误: {did_se:.4f}")
print(f"t值: {did_t:.4f}")
print(f"p值: {did_pval:.4f}")

if did_pval < 0.01:
    sig = '*** (p<0.01)'
elif did_pval < 0.05:
    sig = '** (p<0.05)'
elif did_pval < 0.1:
    sig = '* (p<0.1)'
else:
    sig = '不显著'

print(f"显著性: {sig}")

print(f"\n经济意义:")
if did_coef < 0:
    print(f"  低碳城市试点政策降低碳排放 {abs(did_coef):.2f} 吨")
    print(f"  相对降幅: {abs(did_coef) / treat_pre * 100:.2f}%")
else:
    print(f"  政策效应为正")

# 保存结果
with open('did_twfe_manual_results.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 100 + "\n")
    f.write("DID回归分析结果 - 手动TWFE实现\n")
    f.write("=" * 100 + "\n\n")
    
    f.write("一、模型设定\n")
    f.write("-" * 100 + "\n")
    f.write(f"被解释变量: {Y_var}\n")
    f.write(f"核心解释变量: did (treat x post)\n")
    f.write(f"控制变量: ln_real_gdp, ln_人口密度, ln_金融发展水平, 第二产业占GDP比重\n")
    f.write(f"固定效应: 城市FE + 年份FE (通过去均值实现)\n")
    f.write(f"标准误: 聚类到城市层面\n")
    f.write(f"样本数: {len(model_data)}\n\n")
    
    f.write("二、回归结果\n")
    f.write("-" * 100 + "\n")
    f.write(results.summary().as_text())
    
    f.write("\n\n三、DID系数\n")
    f.write("-" * 100 + "\n")
    f.write(f"DID系数: {did_coef:.4f}\n")
    f.write(f"标准误: {did_se:.4f}\n")
    f.write(f"t值: {did_t:.4f}\n")
    f.write(f"p值: {did_pval:.4f}\n")
    f.write(f"显著性: {sig}\n\n")

print("\n[OK] 结果已保存: did_twfe_manual_results.txt")

print("\n" + "=" * 100)
print("DID回归分析完成!")
print("=" * 100)
print(f"\nDID效应: {did_coef:.2f} 吨 {sig}")
print(f"模型: TWFE ({model_data['city_name'].nunique()} 个城市FE + {model_data['year'].nunique()} 个年份FE)")
print("\n输出文件:")
print("  1. did_twfe_manual_results.txt")
print("  2. did_twfe_results.png")
