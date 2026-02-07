"""
基于PSM结果的DID分析 - 因变量：ln_碳排放量_吨
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

print("=" * 100)
print("基于PSM的DID分析 - 因变量：ln_碳排放量_吨")
print("=" * 100)

# =============================================================================
# 第一步：从PSM匹配结果中提取处理组城市
# =============================================================================
print("\n[第一步] 从PSM匹配结果中提取处理组城市")
print("-" * 100)

# 读取PSM匹配结果
psm_file = '../psm_analysis_2009/matched_data_2009.xlsx'
print(f"读取PSM匹配结果: {psm_file}")
df_psm = pd.read_excel(psm_file)

# 提取处理组和对照组城市
psm_treat_cities = df_psm[df_psm['treat'] == 1]['city_name'].tolist()
psm_control_cities = df_psm[df_psm['treat'] == 0]['city_name'].tolist()
all_matched_cities = psm_treat_cities + psm_control_cities

print(f"处理组城市数: {len(psm_treat_cities)}")
print(f"对照组城市数: {len(psm_control_cities)}")
print(f"总匹配城市数: {len(all_matched_cities)}")

# =============================================================================
# 第二步：构建面板数据集（2007-2023）
# =============================================================================
print("\n[第二步] 构建面板数据集（2007-2023）")
print("-" * 100)

# 读取总数据集
data_file = '../总数据集_已合并_含碳排放_new.xlsx'
print(f"读取总数据集: {data_file}")
df_total = pd.read_excel(data_file)

# 筛选匹配的城市
df_panel = df_total[df_total['city_name'].isin(all_matched_cities)].copy()
print(f"面板数据原始观测数: {len(df_panel)}")

# 筛选年份范围（2007-2023）
df_panel = df_panel[(df_panel['year'] >= 2007) & (df_panel['year'] <= 2023)]
print(f"筛选2007-2023年后观测数: {len(df_panel)}")

# =============================================================================
# 第三步：生成DID变量
# =============================================================================
print("\n[第三步] 生成DID变量")
print("-" * 100)

# 3.1 生成treat变量
df_panel['treat'] = df_panel['city_name'].isin(psm_treat_cities).astype(int)
print(f"处理组观测: {(df_panel['treat'] == 1).sum()}")
print(f"对照组观测: {(df_panel['treat'] == 0).sum()}")

# 3.2 生成post变量（政策实施时点：2009年）
policy_year = 2009
df_panel['post'] = (df_panel['year'] >= policy_year).astype(int)
print(f"政策前观测: {(df_panel['post'] == 0).sum()}")
print(f"政策后观测: {(df_panel['post'] == 1).sum()}")

# 3.3 生成交互项did
df_panel['did'] = df_panel['treat'] * df_panel['post']
print(f"did=1的观测数: {(df_panel['did'] == 1).sum()}")

# =============================================================================
# 第四步：数据验证
# =============================================================================
print("\n[第四步] 数据验证")
print("-" * 100)

Y_var = 'ln_碳排放量_吨'
print(f"因变量: {Y_var}")

# 检查因变量是否存在
if Y_var not in df_panel.columns:
    print(f"[错误] 数据集中不存在 {Y_var} 列")
    exit(1)

# 统计因变量
print(f"\n因变量描述性统计:")
print(df_panel[Y_var].describe())

# 检查缺失值
missing_y = df_panel[Y_var].isna().sum()
print(f"因变量缺失值: {missing_y}")

# 完整观测的城市
city_year_counts = df_panel.groupby('city_name')['year'].count()
complete_cities = city_year_counts[city_year_counts == 17].index.tolist()
print(f"有完整17年数据的城市数: {len(complete_cities)}")

# =============================================================================
# 第五步：平行趋势检验
# =============================================================================
print("\n[第五步] 平行趋势检验")
print("-" * 100)

# 筛选政策前数据（2007-2008）
df_pre = df_panel[df_panel['post'] == 0].copy()

# 计算处理组和对照组的平均值
pre_treat = df_pre[df_pre['treat'] == 1].groupby('year')[Y_var].mean()
pre_control = df_pre[df_pre['treat'] == 0].groupby('year')[Y_var].mean()

print(f"处理组政策前均值: {pre_treat.mean():.4f}")
print(f"对照组政策前均值: {pre_control.mean():.4f}")

# 线性趋势拟合
years_treat = pre_treat.index.values
values_treat = pre_treat.values
slope_treat, intercept_treat, r_value_treat, p_value_treat, std_err_treat = \
    stats.linregress(years_treat, values_treat)

years_control = pre_control.index.values
values_control = pre_control.values
slope_control, intercept_control, r_value_control, p_value_control, std_err_control = \
    stats.linregress(years_control, values_control)

print(f"\n处理组斜率: {slope_treat:.6f}")
print(f"对照组斜率: {slope_control:.6f}")
print(f"斜率差异: {slope_treat - slope_control:.6f}")

# 斜率差异的t检验
n1 = len(values_treat)
n2 = len(values_control)
se_diff = np.sqrt(std_err_treat**2 + std_err_control**2)
t_stat_slope = (slope_treat - slope_control) / se_diff

print(f"斜率差异t统计量: {t_stat_slope:.3f}")

# 临界值（自由度=2，alpha=0.05，双尾）
critical_value = 4.303  # df=2, alpha=0.05, two-tailed
print(f"临界值: {critical_value:.3f}")

if abs(t_stat_slope) < critical_value:
    print(f"[结论] 平行趋势假设成立 (t={t_stat_slope:.3f} < {critical_value:.3f})")
else:
    print(f"[警告] 平行趋势假设不成立 (t={t_stat_slope:.3f} >= {critical_value:.3f})")

# 绘制平行趋势图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 左图：全期趋势
years_all = sorted(df_panel['year'].unique())
treat_all = df_panel[df_panel['treat'] == 1].groupby('year')[Y_var].mean()
control_all = df_panel[df_panel['treat'] == 0].groupby('year')[Y_var].mean()

ax1.plot(years_all, treat_all, 'r-o', label='处理组', linewidth=2, markersize=8)
ax1.plot(years_all, control_all, 'b-s', label='对照组', linewidth=2, markersize=8)
ax1.axvline(x=policy_year, color='green', linestyle='--', linewidth=2, label=f'政策实施 ({policy_year}年)')
ax1.set_xlabel('年份', fontsize=12)
ax1.set_ylabel('ln(碳排放量)', fontsize=12)
ax1.set_title('平行趋势检验：全期趋势 (2007-2023)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# 右图：政策前放大图
ax2.plot(years_treat, values_treat, 'r-o', label='处理组', linewidth=2, markersize=8)
ax2.plot(years_control, values_control, 'b-s', label='对照组', linewidth=2, markersize=8)
# 添加趋势线
ax2.plot(years_treat, intercept_treat + slope_treat * years_treat, 'r--', alpha=0.5, linewidth=1.5)
ax2.plot(years_control, intercept_control + slope_control * years_control, 'b--', alpha=0.5, linewidth=1.5)
ax2.set_xlabel('年份', fontsize=12)
ax2.set_ylabel('ln(碳排放量)', fontsize=12)
ax2.set_title(f'政策前趋势放大 (t={t_stat_slope:.3f})', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('parallel_trend_ln_carbon.png', dpi=300, bbox_inches='tight')
print(f"\n[OK] 平行趋势图已保存: parallel_trend_ln_carbon.png")

# =============================================================================
# 第六步：四重差分分析
# =============================================================================
print("\n[第六步] 四重差分分析")
print("-" * 100)

control_pre = df_panel[(df_panel['treat'] == 0) & (df_panel['post'] == 0)][Y_var].mean()
control_post = df_panel[(df_panel['treat'] == 0) & (df_panel['post'] == 1)][Y_var].mean()
treat_pre = df_panel[(df_panel['treat'] == 1) & (df_panel['post'] == 0)][Y_var].mean()
treat_post = df_panel[(df_panel['treat'] == 1) & (df_panel['post'] == 1)][Y_var].mean()

print(f"对照组政策前均值: {control_pre:.4f}")
print(f"对照组政策后均值: {control_post:.4f}")
print(f"处理组政策前均值: {treat_pre:.4f}")
print(f"处理组政策后均值: {treat_post:.4f}")

diff_control = control_post - control_pre
diff_treat = treat_post - treat_pre
did_effect = diff_treat - diff_control

print(f"\n对照组变化: {diff_control:.4f}")
print(f"处理组变化: {diff_treat:.4f}")
print(f"DID效应: {did_effect:.4f}")

# =============================================================================
# 第七步：TWFE回归
# =============================================================================
print("\n[第七步] TWFE回归分析")
print("-" * 100)

# 准备回归数据
control_vars = ['ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重']
model_data = df_panel[[Y_var, 'did'] + control_vars + ['city_name', 'year']].dropna()
print(f"回归样本数: {len(model_data)}")

# 创建虚拟变量
city_dummies = pd.get_dummies(model_data['city_name'], prefix='C', drop_first=True)
year_dummies = pd.get_dummies(model_data['year'], prefix='C', drop_first=True)

# 构建X矩阵
X_vars = ['did'] + control_vars
X = model_data[X_vars].copy()
X.insert(0, 'const', 1.0)
X = pd.concat([X, city_dummies, year_dummies], axis=1)

# y向量
y = model_data[Y_var].values

# 转换为numpy数组
X = X.values.astype(float)
y = y.astype(float)

# OLS估计
print("正在估计TWFE模型...")
X_T_X = np.dot(X.T, X)
X_T_X_inv = np.linalg.inv(X_T_X)
X_T_y = np.dot(X.T, y)
beta = np.dot(X_T_X_inv, X_T_y)

# 预测值和残差
y_pred = np.dot(X, beta)
residuals = y - y_pred

# 计算聚类标准误
print("计算聚类稳健标准误...")
n = len(y)
k = X.shape[1]
G = model_data['city_name'].nunique()

city_ids = model_data['city_name'].astype('category').cat.codes.values

# 聚类方差-协方差矩阵
meat = np.zeros((k, k))
for g in range(G):
    mask = city_ids == g
    X_g = X[mask]
    e_g = residuals[mask]
    outer = np.dot(X_g.T, e_g.reshape(-1, 1))
    meat += np.dot(outer, outer.T)

scale = n / (n - k) * G / (G - 1)
meat = scale * meat

bread = X_T_X_inv
vcov_cluster = np.dot(bread, np.dot(meat, bread))

# 标准误、t值、p值
se = np.sqrt(np.diag(vcov_cluster))
t_stats = beta / se
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

var_names = ['const'] + X_vars + [f'C(city)' for _ in range(city_dummies.shape[1])] + [f'C(year)' for _ in range(year_dummies.shape[1])]

print(f"\n{'变量':<25} {'系数':>12} {'标准误':>12} {'t值':>10} {'p值':>10}")
print("-" * 75)
for i in range(min(5, len(var_names))):
    stars = ''
    if p_values[i] < 0.01:
        stars = '***'
    elif p_values[i] < 0.05:
        stars = '**'
    elif p_values[i] < 0.1:
        stars = '*'

    print(f"{var_names[i]:<25} {beta[i]:>12.4f} {se[i]:>12.4f} {t_stats[i]:>10.4f} {p_values[i]:>10.4f} {stars}")

print(f"\n... (省略 {len(var_names) - 5} 个固定效应虚拟变量)")

# DID系数
did_idx = 1
did_coef = beta[did_idx]
did_se = se[did_idx]
did_t = t_stats[did_idx]
did_pval = p_values[did_idx]

print("\n" + "=" * 100)
print("DID效应解读")
print("=" * 100)
print(f"\nDID系数: {did_coef:.6f}")
print(f"标准误: {did_se:.6f}")
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

# 解释：对数形式的DID系数
print(f"\n[解释] 对数形式的DID系数解读:")
print(f"  DID系数 = {did_coef:.6f}")
print(f"  即：政策使ln(碳排放量)变化 {did_coef:.6f} 个单位")
print(f"  百分比变化 = {(np.exp(did_coef) - 1) * 100:.2f}%")
print(f"  即：政策使碳排放量变化 {(np.exp(did_coef) - 1) * 100:.2f}%")

print(f"\n模型拟合优度:")
print(f"R-squared: {r_squared:.4f}")
print(f"Adj. R-squared: {r_squared_adj:.4f}")
print(f"F-statistic: {f_stat:.2f}")
print(f"Prob(F-statistic): {f_pval:.4f}")

# 保存结果
with open('did_twfe_results_ln_carbon.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 100 + "\n")
    f.write("DID回归分析结果 - 因变量：ln_碳排放量_吨\n")
    f.write("=" * 100 + "\n\n")

    f.write("一、模型设定\n")
    f.write("-" * 100 + "\n")
    f.write(f"被解释变量: {Y_var}\n")
    f.write(f"核心解释变量: did (treat x post)\n")
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
    f.write(f"DID系数: {did_coef:.6f}\n")
    f.write(f"标准误: {did_se:.6f}\n")
    f.write(f"t值: {did_t:.4f}\n")
    f.write(f"p值: {did_pval:.4f}\n")
    f.write(f"显著性: {sig}\n\n")

    f.write("四、系数解读\n")
    f.write("-" * 100 + "\n")
    f.write(f"对数形式DID系数: {did_coef:.6f}\n")
    f.write(f"百分比变化: {(np.exp(did_coef) - 1) * 100:.2f}%\n")
    f.write(f"解释: 政策使碳排放量变化 {(np.exp(did_coef) - 1) * 100:.2f}%\n\n")

    f.write("五、模型拟合优度\n")
    f.write("-" * 100 + "\n")
    f.write(f"R-squared: {r_squared:.4f}\n")
    f.write(f"Adj. R-squared: {r_squared_adj:.4f}\n")
    f.write(f"F-statistic: {f_stat:.4f}\n")
    f.write(f"Prob(F-statistic): {f_pval:.4f}\n")
    f.write(f"城市数: {G}\n")
    f.write(f"年份数: {model_data['year'].nunique()}\n")

print("\n[OK] 结果已保存: did_twfe_results_ln_carbon.txt")

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
            f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(groups, fontsize=10)
ax.set_ylabel('ln(碳排放量)', fontsize=12)
ax.set_title('DID分析：四组均值对比 (因变量：ln_碳排放量_吨)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('did_twfe_results_ln_carbon.png', dpi=300, bbox_inches='tight')
print("[OK] 图表已保存: did_twfe_results_ln_carbon.png")

# 保存面板数据
output_columns = [Y_var, 'treat', 'post', 'did'] + control_vars + ['city_name', 'year']
df_panel_output = df_panel[output_columns].copy()
df_panel_output.to_excel('panel_data_ln_carbon.xlsx', index=False)
print(f"[OK] 面板数据已保存: panel_data_ln_carbon.xlsx")

print("\n" + "=" * 100)
print("DID分析完成!")
print("=" * 100)
print(f"\n因变量: {Y_var}")
print(f"DID效应: {did_coef:.6f} ({(np.exp(did_coef) - 1) * 100:.2f}%) {sig}")
print(f"模型: TWFE ({G} 个城市FE + {model_data['year'].nunique()} 个年份FE)")
print(f"标准误: 聚类到城市")
print("\n输出文件:")
print("  1. panel_data_ln_carbon.xlsx - 面板数据")
print("  2. parallel_trend_ln_carbon.png - 平行趋势图")
print("  3. did_twfe_results_ln_carbon.txt - 回归结果")
print("  4. did_twfe_results_ln_carbon.png - 四组对比图")
