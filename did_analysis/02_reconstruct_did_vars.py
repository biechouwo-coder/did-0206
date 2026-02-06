"""
基于PSM结果显式构造DID变量的脚本（修正版）
核心改进：从PSM匹配结果中提取处理组城市，在面板数据中重新生成treat、post、did变量
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats

# 设置中文显示
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

print("=" * 100)
print("基于PSM结果显式构造DID变量")
print("=" * 100)

# ============ 步骤1: 读取PSM匹配结果，提取处理组城市 ============
print("\n【步骤1: 从PSM匹配结果中提取处理组城市】")
print("-" * 100)

# 读取PSM匹配结果
psm_file = '../psm_analysis_2009/matched_data_2009.xlsx'
df_psm = pd.read_excel(psm_file)

# 提取处理组城市列表（关键步骤！）
psm_treat_cities = df_psm[df_psm['treat'] == 1]['city_name'].unique().tolist()
psm_control_cities = df_psm[df_psm['treat'] == 0]['city_name'].unique().tolist()

print(f"PSM匹配处理组城市数: {len(psm_treat_cities)}")
print(f"PSM匹配对照组城市数: {len(psm_control_cities)}")
print(f"总匹配城市数: {len(psm_treat_cities) + len(psm_control_cities)}")

print(f"\n处理组城市示例（前10个）:")
for i, city in enumerate(psm_treat_cities[:10], 1):
    print(f"  {i:2d}. {city}")

# ============ 步骤2: 读取原始数据并筛选匹配城市 ============
print("\n【步骤2: 读取原始数据并筛选匹配城市】")
print("-" * 100)

# 读取原始数据
original_file = '../总数据集_已合并_含碳排放_new.xlsx'
df_original = pd.read_excel(original_file)

print(f"原始数据集: {df_original.shape}")
print(f"年份范围: {df_original['year'].min()} - {df_original['year'].max()}")

# 筛选匹配城市
all_matched_cities = psm_treat_cities + psm_control_cities
df_panel = df_original[df_original['city_name'].isin(all_matched_cities)].copy()

print(f"\n面板数据（筛选后）: {df_panel.shape}")
print(f"城市数: {df_panel['city_name'].nunique()}")
print(f"观测总数: {len(df_panel)}")

# ============ 步骤3: 显式生成DID核心变量 ============
print("\n【步骤3: 基于PSM结果显式生成DID变量】")
print("-" * 100)

print("\n[关键步骤] 重新生成treat变量...")

# 3.1 生成treat变量（基于PSM匹配结果）
df_panel['treat_new'] = df_panel['city_name'].isin(psm_treat_cities).astype(int)

print(f"\ntreat变量分布:")
print(df_panel['treat_new'].value_counts())
print(f"  处理组观测数: {(df_panel['treat_new'] == 1).sum()}")
print(f"  对照组观测数: {(df_panel['treat_new'] == 0).sum()}")

# 3.2 生成post变量（2009年及以后=1）
policy_year = 2009
df_panel['post_new'] = (df_panel['year'] >= policy_year).astype(int)

print(f"\npost变量分布（政策实施年份={policy_year}）:")
print(df_panel['post_new'].value_counts())
print(f"  政策前观测数: {(df_panel['post_new'] == 0).sum()}")
print(f"  政策后观测数: {(df_panel['post_new'] == 1).sum()}")

# 3.3 生成did交互项
df_panel['did_new'] = df_panel['treat_new'] * df_panel['post_new']

print(f"\ndid交互项分布:")
did_cross = pd.crosstab([df_panel['treat_new'], df_panel['post_new']],
                        df_panel['did_new'],
                        margins=True)
print(did_cross)

# 3.4 对比原始treat变量（如果存在）
if 'treat' in df_panel.columns:
    print("\n[对比检查] 新生成的treat变量与原始treat变量:")
    consistency = (df_panel['treat_new'] == df_panel['treat']).all()
    print(f"是否完全一致: {consistency}")

    if not consistency:
        diff_count = (df_panel['treat_new'] != df_panel['treat']).sum()
        print(f"不一致的观测数: {diff_count}")

        # 找出差异
        diff_df = df_panel[df_panel['treat_new'] != df_panel['treat']][['city_name', 'year', 'treat', 'treat_new']].drop_duplicates()
        if len(diff_df) > 0:
            print(f"\n差异示例（前10条）:")
            print(diff_df.head(10))
    else:
        print("[OK] 新变量与原始变量完全一致")
else:
    print("\n[信息] 原始数据中没有treat变量，使用新生成的treat_new")

# ============ 步骤4: 使用新生成的变量 ============
print("\n【步骤4: 使用新生成的DID变量】")
print("-" * 100)

# 将新变量设为主变量（覆盖原有变量）
df_panel['treat'] = df_panel['treat_new']
df_panel['post'] = df_panel['post_new']
df_panel['did'] = df_panel['did_new']

# 可以选择删除临时变量，或者保留以备查
# df_panel = df_panel.drop(columns=['treat_new', 'post_new', 'did_new'])

print("已完成变量更新，现在treat/post/did都是基于PSM匹配结果生成的")

# ============ 步骤5: 验证数据结构 ============
print("\n【步骤5: 验证面板数据结构】")
print("-" * 100)

# 检查每个城市的观测数
city_obs = df_panel.groupby('city_name').size()
print(f"\n每个城市观测数:")
print(f"  平均: {city_obs.mean():.2f}")
print(f"  最小: {city_obs.min()}")
print(f"  最大: {city_obs.max()}")

complete_cities = city_obs[city_obs == 17].index.tolist()
print(f"\n完整观测城市（17年）: {len(complete_cities)}/{len(city_obs)}")

# 检查平衡性
for treat_status in [0, 1]:
    cities = psm_control_cities if treat_status == 0 else psm_treat_cities
    obs_counts = df_panel[df_panel['city_name'].isin(cities)].groupby('city_name').size()
    group_name = "对照组" if treat_status == 0 else "处理组"
    print(f"{group_name}: 所有城市都有 {obs_counts.min()}-{obs_counts.max()} 个观测")

# ============ 步骤6: 描述性统计（基于新变量）============
print("\n【步骤6: 描述性统计（基于新生成变量）】")
print("-" * 100)

# 确定碳排放变量
carbon_vars = [col for col in df_panel.columns if '碳排放' in col or 'carbon' in col.lower()]
if carbon_vars:
    carbon_var = carbon_vars[0]
    print(f"使用碳排放变量: {carbon_var}\n")

    print("按treat和post分组的碳排放统计:")
    carbon_stats = df_panel.groupby(['treat', 'post'])[carbon_var].agg([
        ('观测数', 'count'),
        ('均值', 'mean'),
        ('标准差', 'std'),
        ('最小值', 'min'),
        ('最大值', 'max')
    ]).round(4)
    print(carbon_stats)
else:
    print("[警告] 未找到碳排放变量")
    carbon_var = None

# ============ 步骤7: 平行趋势检验 ============
print("\n【步骤7: 平行趋势检验】")
print("-" * 100)

if carbon_var:
    # 计算每年处理组和对照组的均值
    yearly_stats = df_panel.groupby(['year', 'treat'])[carbon_var].mean().reset_index()
    yearly_stats_treat = yearly_stats[yearly_stats['treat'] == 1].sort_values('year')
    yearly_stats_control = yearly_stats[yearly_stats['treat'] == 0].sort_values('year')

    # 绘制趋势图
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(yearly_stats_treat['year'],
            yearly_stats_treat[carbon_var],
            marker='o',
            linewidth=2.5,
            label='处理组（基于PSM匹配）',
            color='#e74c3c')

    ax.plot(yearly_stats_control['year'],
            yearly_stats_control[carbon_var],
            marker='s',
            linewidth=2.5,
            label='对照组（基于PSM匹配）',
            color='#3498db')

    ax.axvline(x=policy_year, color='green', linestyle='--', linewidth=2,
               label=f'政策实施年份 ({policy_year})')

    ax.set_xlabel('年份', fontsize=12)
    ax.set_ylabel('碳排放', fontsize=12)
    ax.set_title('处理组与对照组碳排放趋势（基于PSM匹配结果）', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=':')

    all_years = sorted(df_panel['year'].unique())
    ax.set_xticks(all_years)
    ax.set_xticklabels(all_years, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('parallel_trend_plot_psm_based.png', dpi=300, bbox_inches='tight')
    print("[OK] 趋势图已保存: parallel_trend_plot_psm_based.png")

    # 绘制政策前放大图（带斜率）
    fig, ax = plt.subplots(figsize=(10, 6))

    pre_policy = yearly_stats[yearly_stats['year'] < policy_year]
    pre_policy_treat = pre_policy[pre_policy['treat'] == 1]
    pre_policy_control = pre_policy[pre_policy['treat'] == 0]

    ax.plot(pre_policy_treat['year'],
            pre_policy_treat[carbon_var],
            marker='o',
            linewidth=2.5,
            label='处理组',
            color='#e74c3c')

    ax.plot(pre_policy_control['year'],
            pre_policy_control[carbon_var],
            marker='s',
            linewidth=2.5,
            label='对照组',
            color='#3498db')

    ax.set_xlabel('年份', fontsize=12)
    ax.set_ylabel('碳排放', fontsize=12)
    ax.set_title('政策前趋势（2007-2008）- 平行趋势检验', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=':')

    # 计算斜率
    slope_treat, _, r_treat, p_treat, _ = stats.linregress(
        pre_policy_treat['year'], pre_policy_treat[carbon_var])
    slope_control, _, r_control, p_control, _ = stats.linregress(
        pre_policy_control['year'], pre_policy_control[carbon_var])

    # 斜率差异检验
    n1 = len(pre_policy_treat)
    n2 = len(pre_policy_control)
    se_diff = np.sqrt(((n1-1)*pre_policy_treat[carbon_var].std()**2 +
                       (n2-1)*pre_policy_control[carbon_var].std()**2) /
                      (n1+n2-2) * (1/n1 + 1/n2))

    t_stat = (slope_treat - slope_control) / se_diff if se_diff > 0 else 0

    textstr = f'处理组斜率: {slope_treat:.2e} (R²={r_treat**2:.3f})\n'
    textstr += f'对照组斜率: {slope_control:.2e} (R²={r_control**2:.3f})\n'
    textstr += f'斜率差异: {abs(slope_treat - slope_control):.2e}\n'
    textstr += f'斜率差异t值: {t_stat:.3f}'

    ax.text(0.5, 0.95, textstr,
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('parallel_trend_pre_policy_psm_based.png', dpi=300, bbox_inches='tight')
    print("[OK] 政策前趋势图已保存: parallel_trend_pre_policy_psm_based.png")

    print(f"\n平行趋势检验结果:")
    print(f"  处理组斜率: {slope_treat:.6f}")
    print(f"  对照组斜率: {slope_control:.6f}")
    print(f"  斜率差异: {abs(slope_treat - slope_control):.6f}")
    print(f"  斜率差异t统计量: {t_stat:.3f}")

    # 判断平行趋势
    alpha = 0.05
    t_critical = 4.303  # 小样本t临界值（df=1, alpha=0.05, 双尾）
    if abs(t_stat) < t_critical:
        print(f"\n[结论] 在{alpha*100}%显著性水平下，无法拒绝斜率相等的假设")
        print("       平行趋势假设成立 [OK]")
    else:
        print(f"\n[警告] 在{alpha*100}%显著性水平下，拒绝斜率相等的假设")
        print("        平行趋势假设可能不成立 [WARN]")

# ============ 步骤8: 保存修正后的面板数据 ============
print("\n【步骤8: 保存修正后的面板数据】")
print("-" * 100)

output_file = 'panel_data_2007_2023_corrected.xlsx'
df_panel.to_excel(output_file, index=False)
print(f"[OK] 修正后的面板数据已保存: {output_file}")

# ============ 步骤9: 生成验证报告 ============
print("\n【步骤9: 生成验证报告】")
print("-" * 100)

report_file = 'did_variable_verification_report.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("=" * 100 + "\n")
    f.write("DID变量生成验证报告\n")
    f.write("=" * 100 + "\n\n")

    f.write("一、数据来源\n")
    f.write("-" * 100 + "\n")
    f.write(f"PSM匹配结果: {psm_file}\n")
    f.write(f"原始数据: {original_file}\n\n")

    f.write("二、处理组城市提取\n")
    f.write("-" * 100 + "\n")
    f.write(f"从PSM匹配结果中提取处理组城市: {len(psm_treat_cities)} 个\n")
    f.write(f"从PSM匹配结果中提取对照组城市: {len(psm_control_cities)} 个\n")
    f.write(f"总匹配城市: {len(all_matched_cities)} 个\n\n")

    f.write("三、DID变量生成\n")
    f.write("-" * 100 + "\n")
    f.write("1. treat变量:\n")
    f.write("   - 基于PSM匹配结果中的处理组城市列表生成\n")
    f.write(f"   - 处理组观测: {(df_panel['treat'] == 1).sum()} 个\n")
    f.write(f"   - 对照组观测: {(df_panel['treat'] == 0).sum()} 个\n\n")

    f.write("2. post变量:\n")
    f.write(f"   - 政策实施年份: {policy_year}\n")
    f.write(f"   - 政策前观测: {(df_panel['post'] == 0).sum()} 个\n")
    f.write(f"   - 政策后观测: {(df_panel['post'] == 1).sum()} 个\n\n")

    f.write("3. did交互项:\n")
    f.write("   - did = treat × post\n")
    f.write(f"   - did=1的观测数: {(df_panel['did'] == 1).sum()} 个\n\n")

    f.write("四、数据验证\n")
    f.write("-" * 100 + "\n")
    f.write(f"面板数据观测总数: {len(df_panel)} 条\n")
    f.write(f"城市数: {df_panel['city_name'].nunique()} 个\n")
    f.write(f"年份范围: {df_panel['year'].min()} - {df_panel['year'].max()}\n")
    f.write(f"完整观测城市: {len(complete_cities)} 个\n\n")

    if carbon_var:
        f.write("五、平行趋势检验\n")
        f.write("-" * 100 + "\n")
        f.write(f"处理组斜率: {slope_treat:.6f}\n")
        f.write(f"对照组斜率: {slope_control:.6f}\n")
        f.write(f"斜率差异: {abs(slope_treat - slope_control):.6f}\n")
        f.write(f"斜率差异t统计量: {t_stat:.3f}\n\n")

        if abs(t_stat) < t_critical:
            f.write("[结论] 平行趋势假设成立 [OK]\n")
        else:
            f.write("[警告] 平行趋势假设可能不成立 [WARN]\n")

    f.write("\n六、输出文件\n")
    f.write("-" * 100 + "\n")
    f.write("1. panel_data_2007_2023_corrected.xlsx - 修正后的面板数据\n")
    f.write("2. parallel_trend_plot_psm_based.png - 全期趋势图\n")
    f.write("3. parallel_trend_pre_policy_psm_based.png - 政策前趋势图\n")
    f.write("4. did_variable_verification_report.txt - 本报告\n")
    f.write("\n" + "=" * 100 + "\n")

print(f"[OK] 验证报告已保存: {report_file}")

print("\n" + "=" * 100)
print("DID变量显式构造完成!")
print("=" * 100)
print("\n关键改进:")
print("  [OK] treat变量：从PSM匹配结果中提取处理组城市后生成")
print("  [OK] post变量：基于政策实施年份（2009年）生成")
print("  [OK] did变量：treat × post 交互项")
print("  [OK] 完整验证：对比原始变量（如存在），确保数据一致性")
print("\n输出文件:")
print("  1. panel_data_2007_2023_corrected.xlsx")
print("  2. parallel_trend_plot_psm_based.png")
print("  3. parallel_trend_pre_policy_psm_based.png")
print("  4. did_variable_verification_report.txt")
