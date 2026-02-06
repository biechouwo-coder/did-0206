"""
DID回归分析 - 双向固定效应模型（TWFE）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import statsmodels.formula.api as smf

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

print("=" * 100)
print("DID回归分析 - 双向固定效应模型")
print("=" * 100)

# 步骤1: 读取数据
print("\n[步骤1] 读取数据")
print("-" * 100)

try:
    df = pd.read_excel('panel_data_2007_2023_validated.xlsx')
    print("读取数据: panel_data_2007_2023_validated.xlsx")
except:
    df = pd.read_excel('panel_data_2007_2023_corrected.xlsx')
    print("读取数据: panel_data_2007_2023_corrected.xlsx")

print(f"数据形状: {df.shape}")

# 步骤2: 变量设置
print("\n[步骤2] 变量设置")
print("-" * 100)

Y_var = '碳排放量_吨'
DID_var = 'did'
control_vars = ['ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重']

print(f"Y: {Y_var}")
print(f"DID: {DID_var}")
print(f"Controls: {', '.join(control_vars)}")

# 步骤3: 描述性统计
print("\n[步骤3] 描述性统计")
print("-" * 100)

stats = df.groupby(['treat', 'post'])[Y_var].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
print(stats)

# 四重差分
control_pre = df[(df['treat'] == 0) & (df['post'] == 0)][Y_var].mean()
control_post = df[(df['treat'] == 0) & (df['post'] == 1)][Y_var].mean()
treat_pre = df[(df['treat'] == 1) & (df['post'] == 0)][Y_var].mean()
treat_post = df[(df['treat'] == 1) & (df['post'] == 1)][Y_var].mean()

print(f"\n四重差分:")
print(f"对照组: {control_pre:.2f} -> {control_post:.2f} (差异: {control_post - control_pre:.2f})")
print(f"处理组: {treat_pre:.2f} -> {treat_post:.2f} (差异: {treat_post - treat_pre:.2f})")
print(f"DID效应: {(treat_post - treat_pre) - (control_post - control_pre):.2f}")

# 步骤4: 回归
print("\n[步骤4] TWFE回归")
print("-" * 100)

model_data = df[[Y_var, DID_var] + control_vars + ['city_name', 'year']].dropna()
print(f"样本数: {len(model_data)}")

# 构建公式
formula = Y_var + " ~ " + DID_var + " + " + " + " + control_vars[0] + " + " + control_vars[1] + " + " + control_vars[2] + " + " + control_vars[3] + " + C(city_name) + " + " + "C(year)"

print(f"\n回归模型:")
print(f"  Y_it = alpha + beta*DID_it + gamma*Controls_it + FE_city_i + FE_year_t + epsilon_it")
print(f"  FE: {model_data['city_name'].nunique()} 个城市 + {model_data['year'].nunique()} 个年份")
print(f"  标准误: 聚类到城市")

# 拟合
print("\n正在拟合模型...")
model = smf.ols(formula, data=model_data)
results = model.fit(cov_type='cluster', cov_kwds={'groups': model_data['city_name']})

# 输出
print("\n" + "=" * 100)
print("回归结果")
print("=" * 100)
print(results.summary())

# 解读DID系数
did_coef = results.params[DID_var]
did_pval = results.pvalues[DID_var]
did_se = results.bse[DID_var]
did_t = results.tvalues[DID_var]

print("\n" + "=" * 100)
print("结果解读")
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

print(f"\nR2: {results.rsquared:.4f}")
print(f"Adj.R2: {results.rsquared_adj:.4f}")
print(f"F: {results.fvalue:.2f}")

# 保存结果
with open('did_twfe_results.txt', 'w', encoding='utf-8') as f:
    f.write("DID回归结果 - TWFE模型\n")
    f.write("=" * 80 + "\n\n")
    f.write(results.summary().as_text())
    f.write("\n\n" + "=" * 80 + "\n")
    f.write(f"DID系数: {did_coef:.4f} ({sig})\n")
    f.write(f"样本数: {len(model_data)}\n")
    f.write(f"城市FE: {model_data['city_name'].nunique()}\n")
    f.write(f"年份FE: {model_data['year'].nunique()}\n")

print("\n[OK] 结果已保存: did_twfe_results.txt")

# 可视化
fig, ax = plt.subplots(figsize=(10, 6))
groups = ['对照组\n政策前', '对照组\n政策后', '处理组\n政策前', '处理组\n政策后']
means = [control_pre, control_post, treat_pre, treat_post]
colors = ['#3498db', '#3498db', '#e74c3c', '#e74c3c']

x_pos = np.arange(4)
bars = ax.bar(x_pos, means, color=colors, alpha=0.7, edgecolor='black')

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:,.0f}', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x_pos)
ax.set_xticklabels(groups, fontsize=10)
ax.set_ylabel('碳排放（吨）', fontsize=12)
ax.set_title('DID分析：四组均值对比', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('did_twfe_results.png', dpi=300, bbox_inches='tight')
print("[OK] 图表已保存: did_twfe_results.png")

print("\n" + "=" * 100)
print("分析完成!")
print("=" * 100)
print(f"DID效应: {did_coef:.2f} 吨 {sig}")
