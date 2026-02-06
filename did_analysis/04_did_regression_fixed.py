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
print("DID回归分析 - 双向固定效应模型（TWFE）")
print("=" * 100)

# 读取数据
print("\n【步骤1: 读取数据】")
print("-" * 100)

data_file = 'panel_data_2007_2023_validated.xlsx'
try:
    df = pd.read_excel(data_file)
    print(f"成功读取数据: {data_file}")
except FileNotFoundError:
    data_file = 'panel_data_2007_2023_corrected.xlsx'
    df = pd.read_excel(data_file)
    print(f"读取数据: {data_file}")

print(f"数据形状: {df.shape}")
print(f"年份范围: {df['year'].min()} - {df['year'].max()}")
print(f"城市数: {df['city_name'].nunique()}")

# 变量设置
print("\n【步骤2: 变量设置】")
print("-" * 100)

Y_var = '碳排放量_吨'
DID_var = 'did'
control_vars = ['ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重']

print(f"被解释变量: {Y_var}")
print(f"核心解释变量: {DID_var}")
print(f"控制变量: {', '.join(control_vars)}")
print(f"固定效应: city_name ({df['city_name'].nunique()} 个) + year ({df['year'].nunique()} 个)")
print(f"标准误聚类: 城市层面")

# 描述性统计
print("\n【步骤3: 描述性统计】")
print("-" * 100)

desc_stats = df.groupby(['treat', 'post'])[Y_var].agg([
    ('观测数', 'count'),
    ('均值', 'mean'),
    ('标准差', 'std'),
    ('最小值', 'min'),
    ('最大值', 'max')
]).round(2)

print("\n按treat和post分组的碳排放统计:")
print(desc_stats)

# 四重差分表
print("\n四重差分表:")
control_pre = df[(df['treat'] == 0) & (df['post'] == 0)][Y_var].mean()
control_post = df[(df['treat'] == 0) & (df['post'] == 1)][Y_var].mean()
treat_pre = df[(df['treat'] == 1) & (df['post'] == 0)][Y_var].mean()
treat_post = df[(df['treat'] == 1) & (df['post'] == 1)][Y_var].mean()

print(f"{'组别':<15} {'政策前':<15} {'政策后':<15} {'差异':<15}")
print("-" * 60)
print(f"{'对照组':<15} {control_pre:<15.2f} {control_post:<15.2f} {control_post - control_pre:<15.2f}")
print(f"{'处理组':<15} {treat_pre:<15.2f} {treat_post:<15.2f} {treat_post - treat_pre:<15.2f}")
print("-" * 60)
did_effect = (treat_post - treat_pre) - (control_post - control_pre)
print(f"{'DID效应':<15} {treat_post - treat_pre:<15.2f} {control_post - control_pre:<15.2f} {did_effect:<15.2f}")

# 回归分析
print("\n【步骤4: 模型回归】")
print("-" * 100)

# 准备数据
model_data = df[[Y_var, DID_var] + control_vars + ['city_name', 'year']].dropna()
print(f"删除缺失值后样本数: {len(model_data)}")

# 构建公式
controls_str = ' + '.join(control_vars)
formula = Y_var + " ~ " + DID_var + " + " + controls_str + " + " + "C(city_name) + " + " + "C(year)"

print(f"\n回归方程: {Y_var}_it = alpha + beta*DID_it + gamma*Controls_it + lambda_i + tau_t + epsilon_it")
print(f"  Controls: {', '.join(control_vars)}")
print(f"  FE: {model_data['city_name'].nunique()} 个城市 + {model_data['year'].nunique()} 个年份")
print(f"  标准误聚类: 城市")

# 拟合模型
print("\n正在拟合TWFE模型...")
model = smf.ols(formula, data=model_data)
results = model.fit(cov_type='cluster', cov_kwds={'groups': model_data['city_name']})

# 输出结果
print("\n" + "=" * 100)
print("回归结果")
print("=" * 100)
print(results.summary())

# 解读结果
print("\n【步骤5: 结果解读】")
print("-" * 100)

did_coef = results.params[DID_var]
did_se = results.bse[DID_var]
did_t = results.tvalues[DID_var]
did_pval = results.pvalues[DID_var]
did_ci_lower = results.conf_int().loc[DID_var, 0]
did_ci_upper = results.conf_int().loc[DID_var, 1]

print(f"\nDID系数（政策效应）:")
print(f"  系数: {did_coef:.4f}")
print(f"  标准误: {did_se:.4f}")
print(f"  t值: {did_t:.4f}")
print(f"  p值: {did_pval:.4f}")
print(f"  95%置信区间: [{did_ci_lower:.4f}, {did_ci_upper:.4f}]")

# 显著性
if did_pval < 0.01:
    sig_level = "*** (p<0.01)"
elif did_pval < 0.05:
    sig_level = "** (p<0.05)"
elif did_pval < 0.1:
    sig_level = "* (p<0.1)"
else:
    sig_level = "不显著"

print(f"  显著性: {sig_level}")

print(f"\n经济意义:")
if did_coef < 0:
    print(f"  低碳城市试点政策显著降低碳排放")
    print(f"  平均降低: {abs(did_coef):.2f} 吨")
    print(f"  相对降幅: {abs(did_coef) / treat_pre * 100:.2f}%")
else:
    print(f"  政策效应为正")

# 控制变量
print("\n【步骤6: 控制变量效应】")
print("-" * 100)
print(f"{'变量':<25} {'系数':<12} {'标准误':<12} {'t值':<10} {'显著性':<10}")
print("-" * 75)

for var in control_vars:
    if var in results.params:
        coef = results.params[var]
        se = results.bse[var]
        tval = results.tvalues[var]
        pval = results.pvalues[var]

        if pval < 0.01:
            sig = '***'
        elif pval < 0.05:
            sig = '**'
        elif pval < 0.1:
            sig = '*'
        else:
            sig = ''

        print(f"{var:<25} {coef:>10.4f}  {se:>10.4f}  {tval:>8.4f}  {sig:<10}")

# 模型拟合
print("\n【步骤7: 模型拟合优度】")
print("-" * 100)
print(f"R-squared: {results.rsquared:.4f}")
print(f"Adj. R-squared: {results.rsquared_adj:.4f}")
print(f"F-statistic: {results.fvalue:.2f}")
print(f"F-test p-value: {results.f_pvalue:.4f}")
print(f"AIC: {results.aic:.2f}")
print(f"BIC: {results.bic:.2f}")

# 可视化
print("\n【步骤8: 可视化】")
print("-" * 100)

# 四组均值图
fig, ax = plt.subplots(figsize=(10, 6))

groups = ['对照组-政策前', '对照组-政策后', '处理组-政策前', '处理组-政策后']
means = [control_pre, control_post, treat_pre, treat_post]
colors = ['#3498db', '#3498db', '#e74c3c', '#e74c3c']
hatches = ['' , 'xxx', '', 'xxx']

x_pos = np.arange(4)
bars = ax.bar(x_pos, means, color=colors, hatch=hatches, alpha=0.7, edgecolor='black', linewidth=1.5)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.annotate('', xy=(3, treat_post), xytext=(1, treat_post),
            arrowprops=dict(arrowstyle='->', color='red', lw=2), fontsize=12)
ax.text(2, treat_post + (treat_post - control_post)/2,
        f'DID效应\n{did_coef:,.0f} 吨\n({sig_level})',
        ha='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

ax.set_xlabel('组别', fontsize=12)
ax.set_ylabel('碳排放（吨）', fontsize=12)
ax.set_title('DID分析：四组均值对比', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(groups, rotation=15, ha='right')
ax.grid(True, alpha=0.3, axis='y')
ax.axvline(x=1.5, color='black', linestyle='--', linewidth=1, alpha=0.3)

plt.tight_layout()
plt.savefig('did_four_groups_comparison.png', dpi=300, bbox_inches='tight')
print("[OK] 四组均值对比图已保存")

# 趋势图
fig, ax = plt.subplots(figsize=(12, 6))

yearly_means = df.groupby(['year', 'treat'])[Y_var].mean().reset_index()
control_trend = yearly_means[yearly_means['treat'] == 0].sort_values('year')
treat_trend = yearly_means[yearly_means['treat'] == 1].sort_values('year')

ax.plot(control_trend['year'], control_trend[Y_var],
        marker='s', linewidth=2.5, label='对照组', color='#3498db')
ax.plot(treat_trend['year'], treat_trend[Y_var],
        marker='o', linewidth=2.5, label='处理组', color='#e74c3c')

ax.axvline(x=2009, color='green', linestyle='--', linewidth=2, label='政策实施年份')

y_pos = treat_trend[Y_var].mean()
ax.text(2016, y_pos, f'DID效应\n{did_coef:,.0f} 吨\n({sig_level})',
        fontsize=11, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

ax.set_xlabel('年份', fontsize=12)
ax.set_ylabel('碳排放（吨）', fontsize=12)
ax.set_title('处理组与对照组碳排放趋势（2007-2023）', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('did_trend_analysis.png', dpi=300, bbox_inches='tight')
print("[OK] 趋势分析图已保存")

# 保存结果
print("\n【步骤9: 保存结果】")
print("-" * 100)

results_file = 'did_regression_results.txt'
with open(results_file, 'w', encoding='utf-8') as f:
    f.write("=" * 100 + "\n")
    f.write("DID回归分析结果 - 双向固定效应模型\n")
    f.write("=" * 100 + "\n\n")

    f.write("一、模型设定\n")
    f.write("-" * 100 + "\n")
    f.write(f"被解释变量: {Y_var}\n")
    f.write(f"核心解释变量: {DID_var} (treat x post)\n")
    f.write(f"控制变量: {', '.join(control_vars)}\n")
    f.write(f"固定效应: 城市FE + 年份FE\n")
    f.write(f"标准误: 聚类到城市\n")
    f.write(f"样本数: {len(model_data)}\n\n")

    f.write("二、回归结果\n")
    f.write("-" * 100 + "\n")
    f.write(results.summary().as_text())

    f.write("\n\n三、DID系数\n")
    f.write("-" * 100 + "\n")
    f.write(f"系数: {did_coef:.4f}\n")
    f.write(f"标准误: {did_se:.4f}\n")
    f.write(f"t值: {did_t:.4f}\n")
    f.write(f"p值: {did_pval:.4f}\n")
    f.write(f"95%CI: [{did_ci_lower:.4f}, {did_ci_upper:.4f}]\n")
    f.write(f"显著性: {sig_level}\n\n")

    f.write("四、模型拟合\n")
    f.write("-" * 100 + "\n")
    f.write(f"R²: {results.rsquared:.4f}\n")
    f.write(f"调整R²: {results.rsquared_adj:.4f}\n")
    f.write(f"F统计量: {results.fvalue:.2f}\n")

print(f"[OK] 回归结果已保存: {results_file}")

print("\n" + "=" * 100)
print("DID回归分析完成!")
print("=" * 100)
print("\n主要发现:")
print(f"  1. DID系数: {did_coef:.2f} 吨 {sig_level}")
print(f"  2. 模型: TWFE ({model_data['city_name'].nunique()} 个城市FE + {model_data['year'].nunique()} 个年份FE)")
print(f"  3. 标准误: 聚类到城市层面")
print("\n输出文件:")
print("  1. did_regression_results.txt")
print("  2. did_four_groups_comparison.png")
print("  3. did_trend_analysis.png")
