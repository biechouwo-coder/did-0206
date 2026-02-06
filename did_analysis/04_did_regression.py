"""
DID回归分析 - 双向固定效应模型（TWFE）

理论模型:
碳排放强度_it = α + β·DID_it + γ·Controls_it + λ_i + τ_t + ε_it

其中:
- DID_it = treat_i × post_t (政策交互项)
- Controls: ln_real_gdp, ln_人口密度, ln_金融发展水平, 第二产业占GDP比重
- λ_i: 城市固定效应
- τ_t: 年份固定效应
- 标准误聚类到城市层面
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich import cov_cluster

# 设置中文显示
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

print("=" * 100)
print("DID回归分析 - 双向固定效应模型（TWFE）")
print("=" * 100)

# ============ 步骤1: 读取数据 ============
print("\n【步骤1: 读取数据】")
print("-" * 100)

# 读取面板数据
data_file = 'panel_data_2007_2023_validated.xlsx'
try:
    df = pd.read_excel(data_file)
    print(f"成功读取数据: {data_file}")
    print(f"数据形状: {df.shape}")
except FileNotFoundError:
    # 如果validated版本不存在，尝试corrected版本
    data_file = 'panel_data_2007_2023_corrected.xlsx'
    df = pd.read_excel(data_file)
    print(f"读取数据: {data_file}")
    print(f"数据形状: {df.shape}")

print(f"\n年份范围: {df['year'].min()} - {df['year'].max()}")
print(f"城市数: {df['city_name'].nunique()}")
print(f"总观测数: {len(df)}")

# ============ 步骤2: 变量设置 ============
print("\n【步骤2: 变量设置】")
print("-" * 100)

# 被解释变量
Y_var = '碳排放量_吨'
print(f"被解释变量 (Y): {Y_var}")

# 核心解释变量
DID_var = 'did'
print(f"核心解释变量 (DID): {DID_var} = treat × post")

# 控制变量
control_vars = ['ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重']
print(f"控制变量: {', '.join(control_vars)}")

# 检查变量是否存在
print("\n检查变量:")
for var in [Y_var, DID_var] + control_vars:
    if var in df.columns:
        missing = df[var].isnull().sum()
        print(f"  [OK] {var}: 存在，缺失值 {missing} 个")
    else:
        print(f"  [错误] {var}: 不存在")

# 检查固定效应变量
print(f"\n固定效应:")
print(f"  城市固定效应: city_name ({df['city_name'].nunique()} 个城市)")
print(f"  年份固定效应: year ({df['year'].nunique()} 个年份)")

# ============ 步骤3: 描述性统计 ============
print("\n【步骤3: 描述性统计】")
print("-" * 100)

# 按treat和post分组统计
desc_stats = df.groupby(['treat', 'post'])[Y_var].agg([
    ('观测数', 'count'),
    ('均值', 'mean'),
    ('标准差', 'std'),
    ('最小值', 'min'),
    ('最大值', 'max')
]).round(2)

print("\n按treat和post分组的碳排放统计:")
print(desc_stats)

# 计算四重差分表
print("\n四重差分表（Four-Way Table）:")
print(f"{'组别':<15} {'政策前均值':<15} {'政策后均值':<15} {'差异':<15}")
print("-" * 60)

control_pre = df[(df['treat'] == 0) & (df['post'] == 0)][Y_var].mean()
control_post = df[(df['treat'] == 0) & (df['post'] == 1)][Y_var].mean()
treat_pre = df[(df['treat'] == 1) & (df['post'] == 0)][Y_var].mean()
treat_post = df[(df['treat'] == 1) & (df['post'] == 1)][Y_var].mean()

print(f"{'对照组':<15} {control_pre:<15.2f} {control_post:<15.2f} {control_post - control_pre:<15.2f}")
print(f"{'处理组':<15} {treat_pre:<15.2f} {treat_post:<15.2f} {treat_post - treat_pre:<15.2f}")
print("-" * 60)
print(f"{'DID效应':<15} {treat_post - treat_pre:<15.2f} {control_post - control_pre:<15.2f} {(treat_post - treat_pre) - (control_post - control_pre):<15.2f}")

# ============ 步骤4: 模型回归 ============
print("\n【步骤4: 模型回归】")
print("-" * 100)

# 准备回归数据
model_data = df[[Y_var, DID_var] + control_vars + ['city_name', 'year']].copy()
model_data = model_data.dropna()

print(f"\n删除缺失值后样本数: {len(model_data)}")
print(f"完整样本比例: {len(model_data) / len(df) * 100:.2f}%")

# 构建回归公式
# C()表示分类变量（固定效应）
formula = f"{Y_var} ~ {DID_var} + {' + ' + '.join(control_vars) + '} + C(city_name) + C(year)'

print(f"\n回归方程:")
print(f"{Y_var}_it = α + β·{DID_var}_it + γ·Controls_it + λ_i + τ_t + ε_it")
print(f"其中:")
print(f"  - Controls: {', '.join(control_vars)}")
print(f"  - λ_i: 城市固定效应 ({model_data['city_name'].nunique()} 个)")
print(f"  - τ_t: 年份固定效应 ({model_data['year'].nunique()} 个)")
print(f"  - 标准误聚类到城市层面")

# 拟合模型
print(f"\n正在拟合TWFE模型...")
model = smf.ols(formula, data=model_data)
results = model.fit(cov_type='cluster', cov_kwds={'groups': model_data['city_name']})

# 输出回归结果
print("\n" + "=" * 100)
print("回归结果")
print("=" * 100)
print(results.summary())

# ============ 步骤5: 结果解读 ============
print("\n【步骤5: 结果解读】")
print("-" * 100)

# 提取关键系数
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

# 判断显著性
if did_pval < 0.01:
    sig_level = "*** (p<0.01)"
elif did_pval < 0.05:
    sig_level = "** (p<0.05)"
elif did_pval < 0.1:
    sig_level = "* (p<0.1)"
else:
    sig_level = "不显著"

print(f"  显著性: {sig_level}")

# 经济意义解释
print(f"\n经济意义解释:")
if did_coef < 0:
    print(f"  低碳城市试点政策显著降低了碳排放")
    print(f"  平均降低: {abs(did_coef):.2f} 吨")
    print(f"  相对处理组均值: {abs(did_coef) / treat_pre * 100:.2f}%")
else:
    print(f"  低碳城市试点政策增加了碳排放")
    print(f"  平均增加: {did_coef:.2f} 吨")

# ============ 步骤6: 控制变量效应 ============
print("\n【步骤6: 控制变量效应】")
print("-" * 100)

print(f"\n控制变量回归结果:")
print(f"{'变量':<25} {'系数':<12} {'标准误':<12} {'t值':<10} {'显著性':<10}")
print("-" * 80)

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

# ============ 步骤7: 模型拟合优度 ============
print("\n【步骤7: 模型拟合优度】")
print("-" * 100)

print(f"R-squared: {results.rsquared:.4f}")
print(f"Adj. R-squared: {results.rsquared_adj:.4f}")
print(f"F统计量: {results.fvalue:.2f}")
print(f"F检验p值: {results.f_pvalue:.4f}")
print(f"AIC: {results.aic:.2f}")
print(f"BIC: {results.bic:.2f}")

# ============ 步骤8: 固定效应检验 ============
print("\n【步骤8: 固定效应的重要性】")
print("-" * 100)

# 城市固定效应数量
n_cities = model_data['city_name'].nunique()
n_years = model_data['year'].nunique()

print(f"城市固定效应: {n_cities} 个（控制城市不随时间变化的特征）")
print(f"年份固定效应: {n_years} 个（控制宏观冲击和时间趋势）")
print(f"总计: {n_cities + n_years} 个固定效应")

print("\n固定效应的作用:")
print("  - 城市FE: 控制地理位置、资源禀赋、文化传统等")
print("  - 年份FE: 控制国家政策、经济周期、技术进步等")

# ============ 步骤9: 保存结果 ============
print("\n【步骤9: 保存回归结果】")
print("-" * 100)

# 保存回归结果表格
results_file = 'did_regression_results.txt'
with open(results_file, 'w', encoding='utf-8') as f:
    f.write("=" * 100 + "\n")
    f.write("DID回归分析结果 - 双向固定效应模型（TWFE）\n")
    f.write("=" * 100 + "\n\n")

    f.write("一、模型设定\n")
    f.write("-" * 100 + "\n")
    f.write(f"被解释变量: {Y_var}\n")
    f.write(f"核心解释变量: {DID_var} (treat × post)\n")
    f.write(f"控制变量: {', '.join(control_vars)}\n")
    f.write(f"固定效应: 城市固定效应 + 年份固定效应\n")
    f.write(f"标准误聚类: 城市层面\n")
    f.write(f"样本数: {len(model_data)}\n\n")

    f.write("二、回归结果\n")
    f.write("-" * 100 + "\n")
    f.write(results.summary().as_text())

    f.write("\n\n三、DID系数解读\n")
    f.write("-" * 100 + "\n")
    f.write(f"DID系数: {did_coef:.4f}\n")
    f.write(f"标准误: {did_se:.4f}\n")
    f.write(f"t值: {did_t:.4f}\n")
    f.write(f"p值: {did_pval:.4f}\n")
    f.write(f"95%置信区间: [{did_ci_lower:.4f}, {did_ci_upper:.4f}]\n")
    f.write(f"显著性: {sig_level}\n\n")

    f.write("四、经济意义\n")
    f.write("-" * 100 + "\n")
    if did_coef < 0:
        f.write(f"低碳城市试点政策显著降低了碳排放\n")
        f.write(f"平均降低: {abs(did_coef):.2f} 吨\n")
        f.write(f"相对处理组均值: {abs(did_coef) / treat_pre * 100:.2f}%\n")
    else:
        f.write(f"低碳城市试点政策增加了碳排放\n")
        f.write(f"平均增加: {did_coef:.2f} 吨\n")

    f.write("\n五、模型拟合\n")
    f.write("-" * 100 + "\n")
    f.write(f"R-squared: {results.rsquared:.4f}\n")
    f.write(f"Adj. R-squared: {results.rsquared_adj:.4f}\n")
    f.write(f"F统计量: {results.fvalue:.2f}\n")
    f.write(f"F检验p值: {results.f_pvalue:.4f}\n")

print(f"[OK] 回归结果已保存: {results_file}")

# ============ 步骤10: 可视化 ============
print("\n【步骤10: 可视化DID效应】")
print("-" * 100)

# 创建四组均值条形图
fig, ax = plt.subplots(figsize=(10, 6))

groups = ['对照组-政策前', '对照组-政策后', '处理组-政策前', '处理组-政策后']
means = [control_pre, control_post, treat_pre, treat_post]
colors = ['#3498db', '#3498db', '#e74c3c', '#e74c3c']
hatches = ['' , 'xxx', '', 'xxx']

x_pos = np.arange(4)
bars = ax.bar(x_pos, means, color=colors, hatch=hatches, alpha=0.7, edgecolor='black', linewidth=1.5)

# 添加数值标签
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:,.0f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# 标注政策效应
ax.annotate('', xy=(3, treat_post), xytext=(1, treat_post),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=12)
ax.text(2, treat_post + (treat_post - control_post)/2,
        f'DID效应\n{did_coef:,.0f} 吨',
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
print("[OK] 四组均值对比图已保存: did_four_groups_comparison.png")

# 创建时间趋势图（带政策效应标注）
fig, ax = plt.subplots(figsize=(12, 6))

# 计算每年的均值
yearly_means = df.groupby(['year', 'treat'])[Y_var].mean().reset_index()
control_trend = yearly_means[yearly_means['treat'] == 0].sort_values('year')
treat_trend = yearly_means[yearly_means['treat'] == 1].sort_values('year')

ax.plot(control_trend['year'], control_trend[Y_var],
        marker='s', linewidth=2.5, label='对照组', color='#3498db')
ax.plot(treat_trend['year'], treat_trend[Y_var],
        marker='o', linewidth=2.5, label='处理组', color='#e74c3c')

# 标注政策时点
ax.axvline(x=2009, color='green', linestyle='--', linewidth=2, label='政策实施年份')

# 标注政策效应
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
print("[OK] 趋势分析图已保存: did_trend_analysis.png")

print("\n" + "=" * 100)
print("DID回归分析完成!")
print("=" * 100)
print("\n主要发现:")
print(f"  1. DID系数: {did_coef:.2f} 吨 {sig_level}")
print(f"  2. 政策效应: {'显著降低碳排放' if did_coef < 0 and sig_level != '不显著' else '不显著' if sig_level == '不显著' else '增加碳排放'}")
print(f"  3. 模型包含: {n_cities} 个城市FE + {n_years} 个年份FE + {len(control_vars)} 个控制变量")
print(f"  4. 标准误: 聚类到城市层面")
print("\n输出文件:")
print("  1. did_regression_results.txt - 回归结果")
print("  2. did_four_groups_comparison.png - 四组均值对比图")
print("  3. did_trend_analysis.png - 趋势分析图")
