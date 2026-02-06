"""
基于PSM结果的DID分析（简化版）
步骤：
1. 读取PSM匹配结果，提取匹配城市
2. 构造2007-2023年面板数据集
3. 设定DID核心变量（treat, post, did）
4. 执行平行趋势检验（趋势图）
5. 生成分析报告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文显示
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

print("=" * 100)
print("基于PSM结果的DID分析")
print("=" * 100)

# ============ 步骤1: 读取PSM匹配结果 ============
print("\n【步骤1: 读取PSM匹配结果】")
print("-" * 100)

# 读取匹配后的数据
matched_data_file = '../psm_analysis_2009/matched_data_2009.xlsx'
df_matched = pd.read_excel(matched_data_file)

print(f"匹配数据样本数: {len(df_matched)}")
print(f"处理组: {(df_matched['treat'] == 1).sum()} 个")
print(f"对照组: {(df_matched['treat'] == 0).sum()} 个")

# 提取匹配城市列表
matched_cities = df_matched['city_name'].unique().tolist()
print(f"\n匹配城市总数: {len(matched_cities)} 个")
print(f"\n处理组城市示例: {df_matched[df_matched['treat'] == 1]['city_name'].head(5).tolist()}")
print(f"对照组城市示例: {df_matched[df_matched['treat'] == 0]['city_name'].head(5).tolist()}")

# ============ 步骤2: 构造面板数据集 ============
print("\n【步骤2: 构造2007-2023年面板数据集】")
print("-" * 100)

# 读取原始数据
original_data_file = '../总数据集_已合并_含碳排放_new.xlsx'
df_original = pd.read_excel(original_data_file)

print(f"原始数据集: {df_original.shape}")
print(f"年份范围: {df_original['year'].min()} - {df_original['year'].max()}")

# 筛选匹配城市的数据
df_panel = df_original[df_original['city_name'].isin(matched_cities)].copy()

print(f"\n面板数据集（匹配城市）: {df_panel.shape}")
print(f"城市数: {df_panel['city_name'].nunique()}")
print(f"年份数: {df_panel['year'].nunique()}")
print(f"观测总数: {len(df_panel)}")

# 检查每个城市的观测数
city_obs = df_panel.groupby('city_name').size()
print(f"\n每个城市平均观测数: {city_obs.mean():.2f}")
print(f"观测数范围: {city_obs.min()} - {city_obs.max()}")

# 确保所有城市都有完整的年份数据
expected_years = 17  # 2007-2023
complete_cities = city_obs[city_obs == expected_years].index.tolist()
incomplete_cities = city_obs[city_obs != expected_years].index.tolist()

print(f"\n完整观测城市（{expected_years}年）: {len(complete_cities)} 个")
if incomplete_cities:
    print(f"不完整观测城市: {len(incomplete_cities)} 个")

# ============ 步骤3: 检查并设定DID核心变量 ============
print("\n【步骤3: 检查并设定DID核心变量】")
print("-" * 100)

# 检查变量是否存在
print("\n检查DID核心变量:")
for var in ['treat', 'post', 'did']:
    if var in df_panel.columns:
        print(f"  [OK] {var}: 存在")
        if var == 'treat':
            print(f"        处理组: {(df_panel[var] == 1).sum()} 个观测")
            print(f"        对照组: {(df_panel[var] == 0).sum()} 个观测")
        elif var == 'post':
            print(f"        政策前: {(df_panel[var] == 0).sum()} 个观测")
            print(f"        政策后: {(df_panel[var] == 1).sum()} 个观测")
    else:
        print(f"  [X] {var}: 不存在")

# 检查碳排放变量
print("\n检查碳排放相关变量:")
carbon_vars = [col for col in df_panel.columns if '碳排放' in col or 'carbon' in col.lower()]
if carbon_vars:
    print(f"  找到碳排放变量: {carbon_vars}")
    carbon_var = carbon_vars[0]  # 使用第一个碳排放变量
else:
    print("  未找到明确的碳排放变量")
    print("  可用变量:")
    for col in df_panel.columns:
        print(f"    - {col}")
    carbon_var = None

# 显示DID变量交叉表
print("\nDID变量交叉表:")
did_cross = pd.crosstab([df_panel['treat'], df_panel['post']],
                        df_panel['did'],
                        margins=True)
print(did_cross)

# ============ 步骤4: 描述性统计 ============
print("\n【步骤4: 描述性统计】")
print("-" * 100)

# 按处理组和政策前后分组统计
if carbon_var:
    print("\n按treat和post分组的碳排放统计:")
    carbon_stats = df_panel.groupby(['treat', 'post'])[carbon_var].agg([
        ('观测数', 'count'),
        ('均值', 'mean'),
        ('标准差', 'std'),
        ('最小值', 'min'),
        ('最大值', 'max')
    ]).round(4)
    print(carbon_stats)

# ============ 步骤5: 平行趋势检验 - 趋势图 ============
print("\n【步骤5: 平行趋势检验 - 绘制趋势图】")
print("-" * 100)

if carbon_var:
    # 计算每年处理组和对照组的均值
    yearly_stats = df_panel.groupby(['year', 'treat'])[carbon_var].mean().reset_index()
    yearly_stats_treat = yearly_stats[yearly_stats['treat'] == 1].sort_values('year')
    yearly_stats_control = yearly_stats[yearly_stats['treat'] == 0].sort_values('year')

    # 绘制趋势图
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制处理组和对照组的趋势
    ax.plot(yearly_stats_treat['year'],
            yearly_stats_treat[carbon_var],
            marker='o',
            linewidth=2.5,
            label='处理组',
            color='#e74c3c')

    ax.plot(yearly_stats_control['year'],
            yearly_stats_control[carbon_var],
            marker='s',
            linewidth=2.5,
            label='对照组',
            color='#3498db')

    # 添加政策实施年份的垂直线
    if 2009 in df_panel['year'].values:
        ax.axvline(x=2009, color='green', linestyle='--', linewidth=2,
                   label='政策实施年份 (2009)')

    ax.set_xlabel('年份', fontsize=12)
    ax.set_ylabel('碳排放', fontsize=12)
    ax.set_title('处理组与对照组碳排放趋势（2007-2023）', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=':')

    # 设置x轴刻度
    all_years = sorted(df_panel['year'].unique())
    ax.set_xticks(all_years)
    ax.set_xticklabels(all_years, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('parallel_trend_plot.png', dpi=300, bbox_inches='tight')
    print("[OK] 趋势图已保存: parallel_trend_plot.png")

    # 绘制政策前放大图
    fig, ax = plt.subplots(figsize=(10, 6))

    pre_policy = yearly_stats[yearly_stats['year'] < 2009]
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

    # 添加两条线的斜率
    from scipy import stats
    slope_treat, _, _, _, _ = stats.linregress(pre_policy_treat['year'], pre_policy_treat[carbon_var])
    slope_control, _, _, _, _ = stats.linregress(pre_policy_control['year'], pre_policy_control[carbon_var])

    ax.text(0.5, 0.95, f'处理组斜率: {slope_treat:.2e}\n对照组斜率: {slope_control:.2e}\n斜率差异: {abs(slope_treat - slope_control):.2e}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('parallel_trend_pre_policy.png', dpi=300, bbox_inches='tight')
    print("[OK] 政策前趋势图已保存: parallel_trend_pre_policy.png")
else:
    print("[跳过] 未找到碳排放变量，无法绘制趋势图")

# ============ 步骤6: 保存面板数据 ============
print("\n【步骤6: 保存面板数据】")
print("-" * 100)

# 保存面板数据
panel_data_file = 'panel_data_2007_2023.xlsx'
df_panel.to_excel(panel_data_file, index=False)
print(f"[OK] 面板数据已保存: {panel_data_file}")
print(f"  观测数: {len(df_panel)}")
print(f"  城市数: {df_panel['city_name'].nunique()}")
print(f"  年份范围: {df_panel['year'].min()} - {df_panel['year'].max()}")

# ============ 步骤7: 生成分析报告 ============
print("\n【步骤7: 生成分析报告】")
print("-" * 100)

report_file = 'did_analysis_report.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("=" * 100 + "\n")
    f.write("基于PSM结果的DID分析报告\n")
    f.write("=" * 100 + "\n\n")

    f.write("一、数据构造\n")
    f.write("-" * 100 + "\n")
    f.write(f"PSM匹配城市数: {len(matched_cities)} 个\n")
    f.write(f"面板数据观测数: {len(df_panel)} 条\n")
    f.write(f"城市数: {df_panel['city_name'].nunique()} 个\n")
    f.write(f"年份范围: {df_panel['year'].min()} - {df_panel['year'].max()}\n")
    f.write(f"处理组观测: {(df_panel['treat'] == 1).sum()} 个\n")
    f.write(f"对照组观测: {(df_panel['treat'] == 0).sum()} 个\n\n")

    f.write("二、DID模型设定\n")
    f.write("-" * 100 + "\n")
    if carbon_var:
        f.write(f"因变量: {carbon_var}\n")
    else:
        f.write("因变量: 待确定（未找到明确的碳排放变量）\n")
    f.write("核心自变量: did (treat × post)\n")
    f.write("treat: 处理组虚拟变量（1=试点城市，0=非试点城市）\n")
    f.write("post: 时间虚拟变量（1=政策实施后，0=政策实施前）\n")
    f.write("政策实施年份: 2009年\n\n")

    if carbon_var:
        f.write("三、描述性统计\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'分组':<20} {'观测数':<10} {'均值':<12} {'标准差':<12}\n")
        f.write("-" * 60 + "\n")
        for (treat_val, post_val), group in df_panel.groupby(['treat', 'post']):
            group_name = f"{'处理组' if treat_val == 1 else '对照组'}-{'政策后' if post_val == 1 else '政策前'}"
            f.write(f"{group_name:<20} {len(group):<10} "
                   f"{group[carbon_var].mean():<12.4f} "
                   f"{group[carbon_var].std():<12.4f}\n")
        f.write("\n")

    f.write("四、平行趋势检验\n")
    f.write("-" * 100 + "\n")
    f.write("1. 趋势图: parallel_trend_plot.png\n")
    f.write("   处理组和对照组在政策实施前（2007-2008）的碳排放趋势\n")
    f.write("   查看政策前放大图: parallel_trend_pre_policy.png\n\n")
    f.write("2. 判断标准:\n")
    f.write("   - 2009年之前两组趋势应基本平行（斜率相近）\n")
    f.write("   - 如果斜率差异较大，说明平行趋势假设可能不成立\n\n")

    f.write("五、输出文件\n")
    f.write("-" * 100 + "\n")
    f.write("1. panel_data_2007_2023.xlsx - 面板数据集\n")
    f.write("2. parallel_trend_plot.png - 平行趋势图（全期）\n")
    f.write("3. parallel_trend_pre_policy.png - 政策前趋势图（放大）\n")
    f.write("4. did_analysis_report.txt - 本报告\n")
    f.write("\n" + "=" * 100 + "\n")

print(f"[OK] 分析报告已保存: {report_file}")

print("\n" + "=" * 100)
print("DID分析完成!")
print("=" * 100)
print("\n输出文件:")
print("  1. panel_data_2007_2023.xlsx - 面板数据集")
print("  2. parallel_trend_plot.png - 平行趋势检验图（全期）")
print("  3. parallel_trend_pre_policy.png - 政策前趋势图（放大）")
print("  4. did_analysis_report.txt - 分析报告")
