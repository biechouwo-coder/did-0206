"""
DID回归分析 - TWFE模型 (Pure NumPy/Pandas实现)
不依赖statsmodels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

print("=" * 100)
print("DID回归分析 - 双向固定效应模型 (Pure NumPy)")
print("=" * 100)

# 读取数据
try:
    df = pd.read_excel('panel_data_2007_2023_validated.xlsx')
    print("读取: panel_data_2007_2023_validated.xlsx")
except:
    df = pd.read_excel('panel_data_2007_2023_corrected.xlsx')
    print("读取: panel_data_2007_2023_corrected.xlsx")

print(f"数据形状: {df.shape}")

# 变量设置
Y_var = '碳排放量_吨'
DID_var = 'did'
control_vars = ['ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重']

print(f"\nY: {Y_var}")
print(f"DID: {DID_var}")
print(f"Controls: {', '.join(control_vars)}")

# 描述性统计
print("\n[描述性统计]")
stats = df.groupby(['treat', 'post'])[Y_var].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
print(stats)

# 四重差分
control_pre = df[(df['treat'] == 0) & (df['post'] == 0)][Y_var].mean()
control_post = df[(df['treat'] == 0) & (df['post'] == 1)][Y_var].mean()
treat_pre = df[(df['treat'] == 1) & (df['post'] == 0)][Y_var].mean()
treat_post = df[(df['treat'] == 1) & (df['post'] == 1)][Y_var].mean()

print(f"\n[四重差分]")
print(f"对照组: {control_pre:,.2f} -> {control_post:,.2f} (差异: {control_post - control_pre:,.2f})")
print(f"处理组: {treat_pre:,.2f} -> {treat_post:,.2f} (差异: {treat_post - treat_pre:,.2f})")
print(f"DID效应: {(treat_post - treat_pre) - (control_post - control_pre):,.2f}")

# TWFE回归
print("\n[TWFE回归]")
print("-" * 100)

model_data = df[[Y_var, DID_var] + control_vars + ['city_name', 'year']].dropna()
print(f"样本数: {len(model_data)}")

# 创建虚拟变量
city_dummies = pd.get_dummies(model_data['city_name'], prefix='C', drop_first=True)
year_dummies = pd.get_dummies(model_data['year'], prefix='C', drop_first=True)

# 构建X矩阵
X_vars = [DID_var] + control_vars
X = model_data[X_vars].copy()

# 添加常数项
X.insert(0, 'const', 1.0)

# 合并固定效应虚拟变量
X = pd.concat([X, city_dummies, year_dummies], axis=1)

# y向量
y = model_data[Y_var].values

# 转换为numpy数组
X = X.values.astype(float)
y = y.astype(float)

# OLS估计: beta = (X'X)^(-1)X'y
print("\n正在估计TWFE模型...")
X_T_X = np.dot(X.T, X)
X_T_X_inv = np.linalg.inv(X_T_X)
X_T_y = np.dot(X.T, y)
beta = np.dot(X_T_X_inv, X_T_y)

# 预测值和残差
y_pred = np.dot(X, beta)
residuals = y - y_pred

# 计算聚类标准误（到城市层面）
print("计算聚类稳健标准误...")
n = len(y)
k = X.shape[1]
G = model_data['city_name'].nunique()

# 创建聚类映射
city_ids = model_data['city_name'].astype('category').cat.codes.values

# 聚类方差-协方差矩阵
meat = np.zeros((k, k))
for g in range(G):
    mask = city_ids == g
    X_g = X[mask]
    e_g = residuals[mask]
    outer = np.dot(X_g.T, e_g.reshape(-1, 1))
    meat += np.dot(outer, outer.T)

# 调整因子
scale = n / (n - k) * G / (G - 1)
meat = scale * meat

# 三明治方差估计
bread = X_T_X_inv
vcov_cluster = np.dot(bread, np.dot(meat, bread))

# 标准误
se = np.sqrt(np.diag(vcov_cluster))

# t统计量
t_stats = beta / se

# p值 (双尾检验)
from scipy import stats
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=G - 1))

# R-squared
ss_total = np.sum((y - np.mean(y))**2)
ss_residual = np.sum(residuals**2)
r_squared = 1 - ss_residual / ss_total
r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - k)

# F统计量
ss_model = ss_total - ss_residual
f_stat = (ss_model / (k - 1)) / (ss_residual / (n - k))
f_pval = 1 - stats.f.cdf(f_stat, dfn=k-1, dfd=n-k)

# 输出结果
print("\n" + "=" * 100)
print("TWFE回归结果")
print("=" * 100)

# 变量名
var_names = ['const'] + X_vars + [f'C(city_name_{i})' for i in range(city_dummies.shape[1])] + [f'C(year_{i})' for i in range(year_dummies.shape[1])]

# 结果表格
print(f"\n{'变量':<25} {'系数':>12} {'标准误':>12} {'t值':>10} {'p值':>10}")
print("-" * 75)
for i, name in enumerate(var_names):
    if i < 5:  # 只显示主要变量
        stars = ''
        if p_values[i] < 0.01:
            stars = '***'
        elif p_values[i] < 0.05:
            stars = '**'
        elif p_values[i] < 0.1:
            stars = '*'

        print(f"{name:<25} {beta[i]:>12.4f} {se[i]:>12.4f} {t_stats[i]:>10.4f} {p_values[i]:>10.4f} {stars}")

print(f"\n... (省略 {len(var_names) - 5} 个固定效应虚拟变量)")

# 提取DID系数 (索引1，因为索引0是常数)
did_idx = 1
did_coef = beta[did_idx]
did_se = se[did_idx]
did_t = t_stats[did_idx]
did_pval = p_values[did_idx]

print("\n" + "=" * 100)
print("DID效应解读")
print("=" * 100)
print(f"\nDID系数: {did_coef:.4f}")
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

print(f"\nR-squared: {r_squared:.4f}")
print(f"Adj. R-squared: {r_squared_adj:.4f}")
print(f"F-statistic: {f_stat:.2f}")
print(f"Prob(F-statistic): {f_pval:.4f}")

# 保存结果
with open('did_twfe_numpy_results.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 100 + "\n")
    f.write("DID回归分析结果 - 双向固定效应模型（TWFE）\n")
    f.write("=" * 100 + "\n\n")

    f.write("一、模型设定\n")
    f.write("-" * 100 + "\n")
    f.write(f"被解释变量: {Y_var}\n")
    f.write(f"核心解释变量: {DID_var} (treat x post)\n")
    f.write(f"控制变量: {', '.join(control_vars)}\n")
    f.write(f"固定效应: 城市FE ({city_dummies.shape[1]} 个虚拟变量) + 年份FE ({year_dummies.shape[1]} 个虚拟变量)\n")
    f.write(f"标准误: 聚类到城市层面 ({G} 个城市)\n")
    f.write(f"样本数: {len(model_data)}\n\n")

    f.write("二、主要变量回归结果\n")
    f.write("-" * 100 + "\n")
    f.write(f"{'变量':<25} {'系数':>12} {'标准误':>12} {'t值':>10} {'p值':>10}\n")
    f.write("-" * 75 + "\n")
    for i in range(min(5, len(var_names))):
        stars = ''
        if p_values[i] < 0.01:
            stars = '***'
        elif p_values[i] < 0.05:
            stars = '**'
        elif p_values[i] < 0.1:
            stars = '*'
        f.write(f"{var_names[i]:<25} {beta[i]:>12.4f} {se[i]:>12.4f} {t_stats[i]:>10.4f} {p_values[i]:>10.4f} {stars}\n")

    f.write(f"\n... (省略 {len(var_names) - 5} 个固定效应虚拟变量)\n\n")

    f.write("三、DID系数\n")
    f.write("-" * 100 + "\n")
    f.write(f"DID系数: {did_coef:.4f}\n")
    f.write(f"标准误: {did_se:.4f}\n")
    f.write(f"t值: {did_t:.4f}\n")
    f.write(f"p值: {did_pval:.4f}\n")
    f.write(f"显著性: {sig}\n\n")

    f.write("四、模型拟合优度\n")
    f.write("-" * 100 + "\n")
    f.write(f"R-squared: {r_squared:.4f}\n")
    f.write(f"Adj. R-squared: {r_squared_adj:.4f}\n")
    f.write(f"F-statistic: {f_stat:.4f}\n")
    f.write(f"Prob(F-statistic): {f_pval:.4f}\n")
    f.write(f"城市数: {G}\n")
    f.write(f"年份数: {model_data['year'].nunique()}\n")

print("\n[OK] 结果已保存: did_twfe_numpy_results.txt")

# 可视化
fig, ax = plt.subplots(figsize=(10, 6))
groups = ['对照组\n政策前', '对照组\n政策后', '处理组\n政策前', '处理组\n政策后']
means = [control_pre, control_post, treat_pre, treat_post]
colors = ['#3498db', '#3498db', '#e74c3c', '#e74c3c']

x_pos = np.arange(4)
bars = ax.bar(x_pos, means, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(groups, fontsize=10)
ax.set_ylabel('碳排放（吨）', fontsize=12)
ax.set_title('DID分析：四组均值对比', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('did_twfe_numpy_results.png', dpi=300, bbox_inches='tight')
print("[OK] 图表已保存: did_twfe_numpy_results.png")

print("\n" + "=" * 100)
print("DID回归分析完成!")
print("=" * 100)
print(f"\nDID效应: {did_coef:.2f} 吨 {sig}")
print(f"模型: TWFE ({G} 个城市FE + {model_data['year'].nunique()} 个年份FE)")
print("\n输出文件:")
print("  1. did_twfe_numpy_results.txt")
print("  2. did_twfe_numpy_results.png")
