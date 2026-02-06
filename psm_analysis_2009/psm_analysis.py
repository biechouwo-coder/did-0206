"""
基期PSM分析 (2009年)
匹配变量: ln_real_gdp, ln_人口密度, ln_金融发展水平, 第二产业占GDP比重
卡尺: 0.05
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("警告: seaborn未安装，将使用matplotlib绘图")

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 100)
print("基期倾向得分匹配 (PSM) 分析 - 2009年")
print("=" * 100)

# ============ 步骤1: 读取数据 ============
print("\n【步骤1: 读取数据】")
print("-" * 100)

file_path = '../总数据集_已合并_含碳排放_new.xlsx'
df = pd.read_excel(file_path)

print(f"原始数据集形状: {df.shape}")
print(f"年份范围: {df['year'].min()} - {df['year'].max()}")

# 筛选2009年数据作为基期
df_base = df[df['year'] == 2009].copy()
print(f"\n2009年基期数据: {len(df_base)} 个观测")

# 查找政策变量
print("\n查找政策变量...")
policy_vars = [col for col in df.columns if any(keyword in col.lower() for keyword in ['treat', 'policy', 'treatment'])]

if not policy_vars:
    print("错误: 未找到政策变量!")
    print("请检查数据集中是否有treat、policy等变量")
    exit(1)

# 使用第一个找到的政策变量
treat_var = policy_vars[0]
print(f"使用政策变量: {treat_var}")

# 查看处理组分布
print(f"\n处理组分布:")
print(df_base[treat_var].value_counts())
print(f"处理组比例: {df_base[treat_var].mean():.2%}")

# ============ 步骤2: 准备匹配变量 ============
print("\n【步骤2: 准备匹配变量】")
print("-" * 100)

match_vars = ['ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重']

# 检查变量是否存在
missing_vars = [var for var in match_vars if var not in df_base.columns]
if missing_vars:
    print(f"错误: 以下匹配变量不存在: {missing_vars}")
    exit(1)

print("匹配变量:")
for var in match_vars:
    missing_count = df_base[var].isnull().sum()
    print(f"  - {var}: 缺失值 {missing_count} 个")

# 删除有缺失值的观测（包含city_name）
df_match = df_base[[treat_var] + match_vars + ['city_name']].dropna()
print(f"\n删除缺失值后样本数: {len(df_match)}")

# ============ 步骤3: 计算倾向得分 ============
print("\n【步骤3: 计算倾向得分】")
print("-" * 100)

X = df_match[match_vars]
y = df_match[treat_var]

# 使用Logistic回归计算倾向得分
logit_model = LogisticRegression(solver='lbfgs', max_iter=1000)
logit_model.fit(X, y)

# 计算倾向得分
df_match['propensity_score'] = logit_model.predict_proba(X)[:, 1]

print("倾向得分统计:")
print(df_match.groupby(treat_var)['propensity_score'].describe())

# 分离处理组和对照组（在添加倾向得分后）
df_treat = df_match[df_match[treat_var] == 1].copy()
df_control = df_match[df_match[treat_var] == 0].copy()

print(f"\n处理组样本数: {len(df_treat)}")
print(f"对照组样本数: {len(df_control)}")

# ============ 步骤4: 执行匹配 (卡尺=0.05) ============
print("\n【步骤4: 执行匹配 - 卡尺=0.05】")
print("-" * 100)

caliper = 0.05
print(f"卡尺: {caliper}")

# ============ 步骤3: 计算倾向得分 ============
print("\n【步骤3: 计算倾向得分】")
print("-" * 100)

X = df_match[match_vars]
y = df_match[treat_var]

# 使用Logistic回归计算倾向得分
logit_model = LogisticRegression(solver='lbfgs', max_iter=1000)
logit_model.fit(X, y)

# 计算倾向得分
df_match['propensity_score'] = logit_model.predict_proba(X)[:, 1]

print("倾向得分统计:")
print(df_match.groupby(treat_var)['propensity_score'].describe())

# ============ 步骤4: 执行匹配 (卡尺=0.05) ============
print("\n【步骤4: 执行匹配 - 卡尺=0.05】")
print("-" * 100)

caliper = 0.05
print(f"卡尺: {caliper}")

# 匹配函数
def match_with_caliper(treated_df, control_df, caliper, score_col='propensity_score'):
    """
    使用卡尺进行1:1近邻匹配

    参数:
    - treated_df: 处理组数据框
    - control_df: 对照组数据框
    - caliper: 卡尺宽度
    - score_col: 倾向得分列名

    返回:
    - matched_treated: 匹配成功的处理组
    - matched_control: 匹配成功的对照组
    """
    matched_treated = []
    matched_control = []
    unmatched_treated = []

    control_indices = control_df.index.tolist()
    control_scores = control_df[score_col].values
    is_matched = [False] * len(control_df)

    for idx_t, row_t in treated_df.iterrows():
        score_t = row_t[score_col]

        # 计算与所有未匹配对照组的距离
        distances = []
        for i, idx_c in enumerate(control_indices):
            if not is_matched[i]:
                score_c = control_scores[i]
                distance = abs(score_t - score_c)
                distances.append((distance, i, idx_c))

        if not distances:
            unmatched_treated.append(idx_t)
            continue

        # 找到最近的对照组
        distances.sort(key=lambda x: x[0])
        best_distance, best_i, best_idx_c = distances[0]

        # 检查是否在卡尺范围内
        if best_distance <= caliper:
            matched_treated.append(idx_t)
            matched_control.append(best_idx_c)
            is_matched[best_i] = True
        else:
            unmatched_treated.append(idx_t)

    return (
        treated_df.loc[matched_treated],
        control_df.loc[matched_control],
        len(matched_treated)
    )

# 执行匹配
matched_treat, matched_control, n_matched = match_with_caliper(
    df_treat, df_control, caliper
)

print(f"\n匹配结果:")
print(f"  处理组总数: {len(df_treat)}")
print(f"  对照组总数: {len(df_control)}")
print(f"  成功匹配: {n_matched} 对")
print(f"  未匹配处理组: {len(df_treat) - n_matched} 个")
print(f"  匹配成功率: {n_matched / len(df_treat) * 100:.2f}%")

# 合并匹配后的数据
df_matched = pd.concat([
    matched_treat.assign(matched='treated'),
    matched_control.assign(matched='control')
]).reset_index(drop=True)

# ============ 步骤5: 平衡性检验 ============
print("\n【步骤5: 平衡性检验】")
print("-" * 100)

def calculate_std_bias(data, treat_var, covariate):
    """
    计算标准化偏差 (Standardized Bias)

    Bias = (X_treat_mean - X_control_mean) / sqrt((SD_treat^2 + SD_control^2) / 2)
    """
    treat_mean = data[data[treat_var] == 1][covariate].mean()
    control_mean = data[data[treat_var] == 0][covariate].mean()
    treat_std = data[data[treat_var] == 1][covariate].std()
    control_std = data[data[treat_var] == 0][covariate].std()

    pooled_std = np.sqrt((treat_std**2 + control_std**2) / 2)
    bias = abs((treat_mean - control_mean) / pooled_std) * 100

    return bias, treat_mean, control_mean

print("\n匹配前后的标准化偏差:")
print(f"{'变量':<25} {'匹配前偏差(%)':<15} {'匹配后偏差(%)':<15} {'改善':<10}")
print("-" * 70)

balance_results = []
for var in match_vars:
    # 匹配前的偏差
    bias_before, _, _ = calculate_std_bias(df_match, treat_var, var)

    # 匹配后的偏差
    bias_after, treat_mean, control_mean = calculate_std_bias(df_matched, treat_var, var)

    improvement = bias_before - bias_after
    balance_results.append({
        'variable': var,
        'bias_before': bias_before,
        'bias_after': bias_after,
        'improvement': improvement,
        'treat_mean': treat_mean,
        'control_mean': control_mean
    })

    status = "[OK]" if bias_after < 10 else "[WARN]"
    print(f"{var:<25} {bias_before:<15.2f} {bias_after:<15.2f} {improvement:<10.2f} {status}")

print("\n判断标准: 匹配后偏差 < 10% 为平衡性良好")

# 计算平均偏差
avg_bias_before = np.mean([r['bias_before'] for r in balance_results])
avg_bias_after = np.mean([r['bias_after'] for r in balance_results])
print(f"\n平均标准化偏差:")
print(f"  匹配前: {avg_bias_before:.2f}%")
print(f"  匹配后: {avg_bias_after:.2f}%")
print(f"  改善: {avg_bias_before - avg_bias_after:.2f}%")

# ============ 步骤6: 倾向得分分布可视化 ============
print("\n【步骤6: 生成可视化图表】")
print("-" * 100)

# 图1: 匹配前后倾向得分分布
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 匹配前
ax1 = axes[0]
for treat_status in [0, 1]:
    data = df_match[df_match[treat_var] == treat_status]['propensity_score']
    label = '处理组' if treat_status == 1 else '对照组'
    ax1.hist(data, bins=30, alpha=0.6, label=label, density=True)
ax1.set_xlabel('倾向得分')
ax1.set_ylabel('密度')
ax1.set_title('匹配前倾向得分分布')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 匹配后
ax2 = axes[1]
for treat_status in [0, 1]:
    data = df_matched[df_matched[treat_var] == treat_status]['propensity_score']
    label = '处理组' if treat_status == 1 else '对照组'
    ax2.hist(data, bins=30, alpha=0.6, label=label, density=True)
ax2.set_xlabel('倾向得分')
ax2.set_ylabel('密度')
ax2.set_title('匹配后倾向得分分布')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('psm_score_distribution.png', dpi=300, bbox_inches='tight')
print("[OK] 已保存: psm_score_distribution.png")

# 图2: 各变量匹配前后对比
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, var in enumerate(match_vars):
    ax = axes[i]

    # 绘制匹配前的均值
    means_before = [
        df_match[df_match[treat_var] == 0][var].mean(),
        df_match[df_match[treat_var] == 1][var].mean()
    ]

    # 绘制匹配后的均值
    means_after = [
        df_matched[df_matched[treat_var] == 0][var].mean(),
        df_matched[df_matched[treat_var] == 1][var].mean()
    ]

    x = np.arange(2)
    width = 0.35

    ax.bar(x - width/2, means_before, width, label='匹配前', alpha=0.7)
    ax.bar(x + width/2, means_after, width, label='匹配后', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(['对照组', '处理组'])
    ax.set_ylabel('均值')
    ax.set_title(var)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('psm_balance_check.png', dpi=300, bbox_inches='tight')
print("[OK] 已保存: psm_balance_check.png")

# 图3: 标准化偏差对比
fig, ax = plt.subplots(figsize=(10, 6))

vars_short = [v.replace('ln_', '').replace('第二产业占GDP比重', '二产占比') for v in match_vars]

x = np.arange(len(match_vars))
width = 0.35

biases_before = [r['bias_before'] for r in balance_results]
biases_after = [r['bias_after'] for r in balance_results]

ax.bar(x - width/2, biases_before, width, label='匹配前', alpha=0.7, color='coral')
ax.bar(x + width/2, biases_after, width, label='匹配后', alpha=0.7, color='lightblue')

ax.axhline(y=10, color='red', linestyle='--', linewidth=1, label='10%阈值')
ax.set_xticks(x)
ax.set_xticklabels(vars_short, rotation=15, ha='right')
ax.set_ylabel('标准化偏差 (%)')
ax.set_title('匹配前后标准化偏差对比')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('psm_std_bias.png', dpi=300, bbox_inches='tight')
print("[OK] 已保存: psm_std_bias.png")

# ============ 步骤7: 保存匹配结果 ============
print("\n【步骤7: 保存匹配结果】")
print("-" * 100)

# 保存匹配后的数据
df_matched_final = df_matched[[treat_var] + match_vars + ['propensity_score', 'city_name']].copy()
df_matched_final['year'] = 2009

# 重新排列列顺序
cols = ['city_name', 'year', treat_var] + match_vars + ['propensity_score']
df_matched_final = df_matched_final[cols]

output_file = 'matched_data_2009.xlsx'
df_matched_final.to_excel(output_file, index=False)
print(f"[OK] 匹配数据已保存: {output_file}")
print(f"  样本数: {len(df_matched_final)} ({len(matched_treat)} 对)")

# 保存匹配ID对（用于后续DID分析）
matched_pairs = pd.DataFrame({
    'treated_city': matched_treat['city_name'].values if 'city_name' in matched_treat.columns else matched_treat.index,
    'control_city': matched_control['city_name'].values if 'city_name' in matched_control.columns else matched_control.index,
    'treated_ps': matched_treat['propensity_score'].values,
    'control_ps': matched_control['propensity_score'].values,
    'ps_distance': abs(matched_treat['propensity_score'].values - matched_control['propensity_score'].values)
})

pairs_file = 'matched_pairs_2009.xlsx'
matched_pairs.to_excel(pairs_file, index=False)
print(f"[OK] 匹配对已保存: {pairs_file}")

# 保存统计报告
report_file = 'psm_report_2009.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("=" * 100 + "\n")
    f.write("基期倾向得分匹配 (PSM) 分析报告\n")
    f.write("=" * 100 + "\n\n")

    f.write("一、分析设置\n")
    f.write("-" * 100 + "\n")
    f.write(f"基期年份: 2009\n")
    f.write(f"匹配变量: {', '.join(match_vars)}\n")
    f.write(f"卡尺: {caliper}\n")
    f.write(f"政策变量: {treat_var}\n\n")

    f.write("二、样本信息\n")
    f.write("-" * 100 + "\n")
    f.write(f"原始样本数: {len(df_base)}\n")
    f.write(f"有效样本数: {len(df_match)} (删除缺失值后)\n")
    f.write(f"处理组: {len(df_treat)} 个\n")
    f.write(f"对照组: {len(df_control)} 个\n\n")

    f.write("三、匹配结果\n")
    f.write("-" * 100 + "\n")
    f.write(f"成功匹配: {n_matched} 对\n")
    f.write(f"未匹配处理组: {len(df_treat) - n_matched} 个\n")
    f.write(f"匹配成功率: {n_matched / len(df_treat) * 100:.2f}%\n\n")

    f.write("四、平衡性检验\n")
    f.write("-" * 100 + "\n")
    f.write(f"{'变量':<25} {'匹配前偏差(%)':<15} {'匹配后偏差(%)':<15} {'改善':<15}\n")
    f.write("-" * 80 + "\n")
    for r in balance_results:
        status = "[OK] 平衡良好" if r['bias_after'] < 10 else "[WARN] 需关注"
        f.write(f"{r['variable']:<25} {r['bias_before']:<15.2f} {r['bias_after']:<15.2f} {r['improvement']:<15.2f}\n")
        f.write(f"{'':<25} 处理组均值: {r['treat_mean']:.4f}, 对照组均值: {r['control_mean']:.4f} [{status}]\n\n")

    f.write(f"平均标准化偏差:\n")
    f.write(f"  匹配前: {avg_bias_before:.2f}%\n")
    f.write(f"  匹配后: {avg_bias_after:.2f}%\n")
    f.write(f"  改善: {avg_bias_before - avg_bias_after:.2f}%\n\n")

    f.write("五、结论\n")
    f.write("-" * 100 + "\n")
    if avg_bias_after < 10:
        f.write("[OK] 平衡性检验通过: 平均标准化偏差小于10%\n")
        f.write("[OK] 匹配质量良好，可用于后续DID分析\n")
    else:
        f.write("[WARN] 平衡性检验未完全通过: 部分变量偏差较大\n")
        f.write("[WARN] 建议: 调整匹配变量或卡尺参数重新匹配\n")

    f.write("\n" + "=" * 100 + "\n")

print(f"[OK] 分析报告已保存: {report_file}")

print("\n" + "=" * 100)
print("PSM分析完成!")
print("=" * 100)
print("\n输出文件:")
print(f"  1. matched_data_2009.xlsx - 匹配后的数据")
print(f"  2. matched_pairs_2009.xlsx - 匹配对信息")
print(f"  3. psm_report_2009.txt - 分析报告")
print(f"  4. psm_score_distribution.png - 倾向得分分布图")
print(f"  5. psm_balance_check.png - 平衡性检验图")
print(f"  6. psm_std_bias.png - 标准化偏差对比图")
