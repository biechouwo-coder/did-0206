"""
DID回归分析 - TWFE模型
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

# 读取数据
try:
    df = pd.read_excel('panel_data_2007_2023_validated.xlsx')
except:
    df = pd.read_excel('panel_data_2007_2023_corrected.xlsx')

print(f"\n数据: {df.shape}")

# 变量
Y_var = '碳排放量_吨'
control_vars = ['ln_real_gmp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重']

# 检查列名
print(f"\n可用变量: {list(df.columns)}")

# 修正列名（如果有需要）
if 'ln_real_gdp' not in df.columns:
    control_vars[0] = 'ln_real_gdp'

print(f"\nY: {Y_var}")
print(f"Controls: {control_vars}")

# 描述性统计
print("\n描述性统计:")
stats = df.groupby(['treat', 'post'])[Y_var].agg(['count', 'mean']).round(2)
print(stats)

# 四重差分
control_pre = df[(df['treat'] == 0) & (df['post'] == 0)][Y_var].mean()
control_post = df[(df['treat'] == 0) & (df['post'] == 1)][Y_var].mean()
treat_pre = df[(df['treat'] == 1) & (df['post'] == 0)][Y_var].mean()
treat_post = df[(df['treat'] == 1) & (df['post'] == 1)][Y_var].mean()

print(f"\n四重差分:")
print(f"对照组: {control_pre:.2f} -> {control_post:.2f}")
print(f"处理组: {treat_pre:.2f} -> {treat_post:.2f}")
print(f"DID效应: {(treat_post - treat_pre) - (control_post - control_pre):.2f}")

# 准备数据
model_data = df[[Y_var, 'did'] + control_vars + ['city_name', 'year']].dropna()
print(f"\n样本数: {len(model_data)}")

# 修正控制变量名（ln_real_gmp -> ln_real_gdp）
if 'ln_real_gmp' in model_data.columns and 'ln_real_gdp' not in model_data.columns:
    model_data.rename(columns={'ln_real_gmp': 'ln_real_gdp'}, inplace=True)
    control_vars[0] = 'ln_real_gdp'

# 构建公式 - 直接写出
formula = "碳排放量_吨 ~ did + ln_real_gdp + ln_人口密度 + ln_金融发展水平 + 第二产业占GDP比重 + C(city_name) + C(year)"

print(f"\n回归公式:")
print(formula)

# 拟合TWFE
print("\n拟合TWFE模型...")
model = smf.ols(formula, data=model_data)
results = model.fit(cov_type='cluster', cov_kwds={'groups': model_data['city_name']})

# 输出结果
print("\n" + "=" * 100)
print("回归结果")
print("=" * 100)
print(results.summary())

# 提取DID系数
did_coef = results.params['did']
did_se = results.bse['did']
did_pval = results.pvalues['did']
did_t = results.tvalues['did']

print("\n" + "=" * 100)
print("DID效应")
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

# 模型拟合
print(f"\nR-squared: {results.rsquared:.4f}")
print(f"Adj. R-squared: {results.rsquared_adj:.4f}")
print(f"F-statistic: {results.fvalue:.2f}")
print(f"Prob (F-statistic): {results.f_pvalue:.4f}")

# 保存结果
with open('did_twfe_results.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 100 + "\n")
    f.write("DID回归分析结果 - 双向固定效应模型（TWFE）\n")
    f.write("=" * 100 + "\n\n")
    
    f.write("一、模型设定\n")
    f.write("-" * 100 + "\n")
    f.write(f"被解释变量: 碳排放量_吨\n")
    f.write(f"核心解释变量: did = treat x post\n")
    f.write(f"控制变量: ln_real_gdp, ln_人口密度, ln_金融发展水平, 第二产业占GDP比重\n")
    f.write(f"固定效应: C(city_name) + C(year)\n")
    f.write(f"标准误: 聚类到城市层面\n")
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
    f.write(f"显著性: {sig}\n\n")
    
    f.write("四、模型拟合优度\n")
    f.write("-" * 100 + "\n")
    f.write(f"R-squared: {results.rsquared:.4f}\n")
    f.write(f"Adj. R-squared: {results.rsquared_adj:.4f}\n")
    f.write(f"F-statistic: {results.fvalue:.4f}\n")
    f.write(f"Prob(F-statistic): {results.f_pvalue:.4f}\n")

print("\n[OK] 结果已保存: did_twfe_results.txt")

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

# 标注DID效应
ax.annotate('', xy=(3, treat_post), xytext=(1, treat_post),
            arrowprops=dict(arrowstyle='->', color='red', lw=2), fontsize=12)
ax.text(2, treat_post + (treat_post - control_post)/2,
        f'DID效应\n{did_coef:,.0f} 吨\n({sig})',
        ha='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

ax.set_xticks(x_pos)
ax.set_xticklabels(groups, fontsize=10)
ax.set_ylabel('碳排放（吨）', fontsize=12)
ax.set_title('DID分析：四组均值对比', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('did_twfe_results.png', dpi=300, bbox_inches='tight')
print("[OK] 图表已保存: did_twfe_results.png")

print("\n" + "=" * 100)
print("DID回归分析完成!")
print("=" * 100)
print(f"\nDID效应: {did_coef:.2f} 吨 {sig}")
print(f"模型: TWFE ({model_data['city_name'].nunique()} 个城市FE + {model_data['year'].nunique()} 个年份FE)")
print(f"标准误: 聚类到城市")
print("\n输出文件:")
print("  1. did_twfe_results.txt")
print("  2. did_twfe_results.png")
