"""
基于PSM结果显式构造DID变量的脚本（增强版）
改进：添加列名检测、错误处理和验证机制
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
import sys

# 设置中文显示
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

print("=" * 100)
print("基于PSM结果显式构造DID变量（增强版）")
print("=" * 100)

# ============ 辅助函数：列名检测 ============
def detect_column_name(df, possible_names, column_description):
    """
    检测DataFrame中是否存在指定的列名（支持模糊匹配）

    参数:
    - df: DataFrame
    - possible_names: 可能的列名列表
    - column_description: 列的描述（用于错误提示）

    返回:
    - 找到的列名，如果找不到则返回None
    """
    # 首先尝试精确匹配
    for name in possible_names:
        if name in df.columns:
            return name

    # 如果精确匹配失败，尝试模糊匹配
    for name in possible_names:
        for col in df.columns:
            if name.lower() in col.lower() or col.lower() in name.lower():
                print(f"  [提示] 使用模糊匹配: '{col}' 匹配到 '{name}'")
                return col

    # 如果还是找不到，返回None
    return None

def validate_required_columns(df, required_columns, file_description):
    """
    验证DataFrame中是否包含所有必需的列

    参数:
    - df: DataFrame
    - required_columns: 必需列的字典 {列描述: 可能的列名列表}
    - file_description: 文件描述（用于错误提示）

    返回:
    - 找到的列名映射字典
    - 如果缺少关键列，则退出程序
    """
    print(f"\n正在检测 {file_description} 的列名...")
    found_columns = {}
    missing_columns = []

    for desc, possible_names in required_columns.items():
        col_name = detect_column_name(df, possible_names, desc)
        if col_name:
            found_columns[desc] = col_name
            print(f"  [OK] {desc}: 找到列名 '{col_name}'")
        else:
            missing_columns.append(desc)
            print(f"  [错误] {desc}: 未找到列名（尝试了: {possible_names}）")

    if missing_columns:
        print(f"\n严重错误: {file_description} 缺少以下关键列:")
        for col in missing_columns:
            print(f"  - {col}")
        print(f"\n请检查Excel文件中的列名是否正确")
        print(f"当前文件的列名: {list(df.columns)}")
        sys.exit(1)

    return found_columns

# ============ 步骤0: 配置参数 ============
print("\n【步骤0: 配置参数】")
print("-" * 100)

# 定义政策实施年份（可修改）
POLICY_YEAR = 2009

# 定义可能的列名（用于模糊匹配）
PSM_REQUIRED_COLUMNS = {
    'treat': ['treat', 'treat_var', 'treatment', 'group'],
    'city_name': ['city_name', 'city', '城市名称', 'cityname']
}

ORIGINAL_REQUIRED_COLUMNS = {
    'year': ['year', '时间', '年份'],
    'city_name': ['city_name', 'city', '城市名称', 'cityname'],
    'carbon': ['碳排放量_吨', '碳排放', 'carbon_emissions', 'carbon', 'emissions']
}

print(f"政策实施年份: {POLICY_YEAR}")
print(f"将检测以下关键列:")
for desc, names in PSM_REQUIRED_COLUMNS.items():
    print(f"  - {desc}: {names}")
for desc, names in ORIGINAL_REQUIRED_COLUMNS.items():
    print(f"  - {desc}: {names}")

# ============ 步骤1: 读取PSM匹配结果 ============
print("\n【步骤1: 从PSM匹配结果中提取处理组城市】")
print("-" * 100)

# 读取PSM匹配结果
psm_file = '../psm_analysis_2009/matched_data_2009.xlsx'
try:
    df_psm = pd.read_excel(psm_file)
    print(f"成功读取PSM文件: {psm_file}")
    print(f"文件形状: {df_psm.shape}")
except FileNotFoundError:
    print(f"\n严重错误: 找不到PSM文件 '{psm_file}'")
    print(f"请确保PSM分析已完成，且文件路径正确")
    sys.exit(1)
except Exception as e:
    print(f"\n严重错误: 读取PSM文件失败")
    print(f"错误信息: {e}")
    sys.exit(1)

# 检测PSM文件的列名
psm_columns = validate_required_columns(df_psm, PSM_REQUIRED_COLUMNS, "PSM匹配结果文件")

# 使用检测到的列名
treat_col_psm = psm_columns['treat']
city_col_psm = psm_columns['city_name']

# 提取处理组城市列表
psm_treat_cities = df_psm[df_psm[treat_col_psm] == 1][city_col_psm].unique().tolist()
psm_control_cities = df_psm[df_psm[treat_col_psm] == 0][city_col_psm].unique().tolist()

print(f"\nPSM匹配处理组城市数: {len(psm_treat_cities)}")
print(f"PSM匹配对照组城市数: {len(psm_control_cities)}")
print(f"总匹配城市数: {len(psm_treat_cities) + len(psm_control_cities)}")

print(f"\n处理组城市示例（前10个）:")
for i, city in enumerate(psm_treat_cities[:10], 1):
    print(f"  {i:2d}. {city}")

# ============ 步骤2: 读取原始数据 ============
print("\n【步骤2: 读取原始数据并筛选匹配城市】")
print("-" * 100)

# 读取原始数据
original_file = '../总数据集_已合并_含碳排放_new.xlsx'
try:
    df_original = pd.read_excel(original_file)
    print(f"成功读取原始数据文件: {original_file}")
    print(f"文件形状: {df_original.shape}")
except FileNotFoundError:
    print(f"\n严重错误: 找不到原始数据文件 '{original_file}'")
    print(f"请确保文件路径正确")
    sys.exit(1)
except Exception as e:
    print(f"\n严重错误: 读取原始数据文件失败")
    print(f"错误信息: {e}")
    sys.exit(1)

# 检测原始数据的列名
original_columns = validate_required_columns(df_original, ORIGINAL_REQUIRED_COLUMNS, "原始数据文件")

# 使用检测到的列名
year_col = original_columns['year']
city_col_original = original_columns['city_name']
carbon_col = original_columns.get('carbon')  # 碳排放列（可选）

print(f"\n原始数据集信息:")
print(f"  年份范围: {df_original[year_col].min()} - {df_original[year_col].max()}")
print(f"  城市数: {df_original[city_col_original].nunique()}")

# 筛选匹配城市
all_matched_cities = psm_treat_cities + psm_control_cities
df_panel = df_original[df_original[city_col_original].isin(all_matched_cities)].copy()

print(f"\n面板数据（筛选后）:")
print(f"  观测总数: {len(df_panel)}")
print(f"  城市数: {df_panel[city_col_original].nunique()}")

# 验证是否所有匹配城市都在原始数据中
matched_cities_in_data = df_panel[city_col_original].unique().tolist()
missing_cities = set(all_matched_cities) - set(matched_cities_in_data)
if missing_cities:
    print(f"\n警告: 以下PSM匹配城市在原始数据中未找到:")
    for city in missing_cities:
        print(f"  - {city}")

# ============ 步骤3: 显式生成DID核心变量 ============
print("\n【步骤3: 基于PSM结果显式生成DID变量】")
print("-" * 100)

print(f"\n[关键步骤] 重新生成treat变量...")
print(f"  政策实施年份: {POLICY_YEAR}")

# 3.1 生成treat变量（基于PSM匹配结果）
df_panel['treat_new'] = df_panel[city_col_original].isin(psm_treat_cities).astype(int)

print(f"\ntreat变量分布:")
print(df_panel['treat_new'].value_counts())
print(f"  处理组观测数: {(df_panel['treat_new'] == 1).sum()}")
print(f"  对照组观测数: {(df_panel['treat_new'] == 0).sum()}")

# 3.2 生成post变量
df_panel['post_new'] = (df_panel[year_col] >= POLICY_YEAR).astype(int)

print(f"\npost变量分布（政策实施年份={POLICY_YEAR}）:")
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

# ============ 步骤4: 数据质量验证 ============
print("\n【步骤4: 数据质量验证】")
print("-" * 100)

# 验证平衡面板
city_obs = df_panel.groupby(city_col_original).size()
print(f"\n每个城市观测数:")
print(f"  平均: {city_obs.mean():.2f}")
print(f"  最小: {city_obs.min()}")
print(f"  最大: {city_obs.max()}")

expected_years = len(df_panel[year_col].unique())
complete_cities = city_obs[city_obs == expected_years].index.tolist()

print(f"\n完整观测城市（{expected_years}年）: {len(complete_cities)}/{len(city_obs)}")

if len(complete_cities) < len(city_obs):
    incomplete_cities = city_obs[city_obs != expected_years].index.tolist()
    print(f"警告: 以下城市观测数不完整:")
    for city in incomplete_cities:
        print(f"  - {city}: {city_obs[city]} 年")

# 统一列名（用于后续分析）
df_panel['treat'] = df_panel['treat_new']
df_panel['post'] = df_panel['post_new']
df_panel['did'] = df_panel['did_new']
df_panel['city_name'] = df_panel[city_col_original]
df_panel['year'] = df_panel[year_col]

print("\n列名已统一为: treat, post, did, city_name, year")

# ============ 步骤5: 平行趋势检验 ============
print("\n【步骤5: 平行趋势检验】")
print("-" * 100)

if carbon_col:
    print(f"使用碳排放变量: {carbon_col}\n")

    # 计算每年处理组和对照组的均值
    yearly_stats = df_panel.groupby(['year', 'treat'])[carbon_col].mean().reset_index()
    yearly_stats_treat = yearly_stats[yearly_stats['treat'] == 1].sort_values('year')
    yearly_stats_control = yearly_stats[yearly_stats['treat'] == 0].sort_values('year')

    # 绘制趋势图
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(yearly_stats_treat['year'],
            yearly_stats_treat[carbon_col],
            marker='o',
            linewidth=2.5,
            label='处理组（基于PSM匹配）',
            color='#e74c3c')

    ax.plot(yearly_stats_control['year'],
            yearly_stats_control[carbon_col],
            marker='s',
            linewidth=2.5,
            label='对照组（基于PSM匹配）',
            color='#3498db')

    ax.axvline(x=POLICY_YEAR, color='green', linestyle='--', linewidth=2,
               label=f'政策实施年份 ({POLICY_YEAR})')

    ax.set_xlabel('年份', fontsize=12)
    ax.set_ylabel('碳排放', fontsize=12)
    ax.set_title('处理组与对照组碳排放趋势（基于PSM匹配结果）', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=':')

    all_years = sorted(df_panel['year'].unique())
    ax.set_xticks(all_years)
    ax.set_xticklabels(all_years, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('parallel_trend_plot_validated.png', dpi=300, bbox_inches='tight')
    print("[OK] 趋势图已保存: parallel_trend_plot_validated.png")

    # 绘制政策前放大图（带斜率检验）
    fig, ax = plt.subplots(figsize=(10, 6))

    pre_policy = yearly_stats[yearly_stats['year'] < POLICY_YEAR]
    pre_policy_treat = pre_policy[pre_policy['treat'] == 1]
    pre_policy_control = pre_policy[pre_policy['treat'] == 0]

    ax.plot(pre_policy_treat['year'],
            pre_policy_treat[carbon_col],
            marker='o',
            linewidth=2.5,
            label='处理组',
            color='#e74c3c')

    ax.plot(pre_policy_control['year'],
            pre_policy_control[carbon_col],
            marker='s',
            linewidth=2.5,
            label='对照组',
            color='#3498db')

    ax.set_xlabel('年份', fontsize=12)
    ax.set_ylabel('碳排放', fontsize=12)
    ax.set_title(f'政策前趋势（{POLICY_YEAR-2}-{POLICY_YEAR-1}）- 平行趋势检验', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=':')

    # 计算斜率
    slope_treat, _, r_treat, p_treat, _ = stats.linregress(
        pre_policy_treat['year'], pre_policy_treat[carbon_col])
    slope_control, _, r_control, p_control, _ = stats.linregress(
        pre_policy_control['year'], pre_policy_control[carbon_col])

    # 斜率差异检验
    n1 = len(pre_policy_treat)
    n2 = len(pre_policy_control)
    se_diff = np.sqrt(((n1-1)*pre_policy_treat[carbon_col].std()**2 +
                       (n2-1)*pre_policy_control[carbon_col].std()**2) /
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
    plt.savefig('parallel_trend_pre_policy_validated.png', dpi=300, bbox_inches='tight')
    print("[OK] 政策前趋势图已保存: parallel_trend_pre_policy_validated.png")

    print(f"\n平行趋势检验结果:")
    print(f"  处理组斜率: {slope_treat:.6f}")
    print(f"  对照组斜率: {slope_control:.6f}")
    print(f"  斜率差异: {abs(slope_treat - slope_control):.6f}")
    print(f"  斜率差异t统计量: {t_stat:.3f}")

    # 判断平行趋势
    alpha = 0.05
    t_critical = 4.303
    if abs(t_stat) < t_critical:
        print(f"\n[结论] 在{alpha*100}%显著性水平下，无法拒绝斜率相等的假设")
        print("       平行趋势假设成立 [OK]")
    else:
        print(f"\n[警告] 在{alpha*100}%显著性水平下，拒绝斜率相等的假设")
        print("        平行趋势假设可能不成立 [WARN]")
else:
    print("[跳过] 未找到碳排放变量，无法绘制趋势图")

# ============ 步骤6: 保存数据 ============
print("\n【步骤6: 保存面板数据】")
print("-" * 100)

output_file = 'panel_data_2007_2023_validated.xlsx'
df_panel.to_excel(output_file, index=False)
print(f"[OK] 面板数据已保存: {output_file}")

# ============ 步骤7: 生成验证报告 ============
print("\n【步骤7: 生成验证报告】")
print("-" * 100)

report_file = 'did_validation_report.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("=" * 100 + "\n")
    f.write("DID变量生成验证报告（增强版）\n")
    f.write("=" * 100 + "\n\n")

    f.write("一、配置参数\n")
    f.write("-" * 100 + "\n")
    f.write(f"政策实施年份: {POLICY_YEAR}\n\n")

    f.write("二、列名检测\n")
    f.write("-" * 100 + "\n")
    f.write("PSM文件列名:\n")
    for desc, col in psm_columns.items():
        f.write(f"  {desc}: '{col}'\n")
    f.write("\n原始数据文件列名:\n")
    for desc, col in original_columns.items():
        f.write(f"  {desc}: '{col}'\n")
    f.write("\n")

    f.write("三、数据来源\n")
    f.write("-" * 100 + "\n")
    f.write(f"PSM匹配结果: {psm_file}\n")
    f.write(f"原始数据: {original_file}\n\n")

    f.write("四、处理组城市提取\n")
    f.write("-" * 100 + "\n")
    f.write(f"从PSM匹配结果中提取处理组城市: {len(psm_treat_cities)} 个\n")
    f.write(f"从PSM匹配结果中提取对照组城市: {len(psm_control_cities)} 个\n")
    f.write(f"总匹配城市: {len(all_matched_cities)} 个\n\n")

    f.write("五、DID变量生成\n")
    f.write("-" * 100 + "\n")
    f.write(f"1. treat变量: 基于PSM匹配结果中的处理组城市列表生成\n")
    f.write(f"   处理组观测: {(df_panel['treat'] == 1).sum()} 个\n")
    f.write(f"   对照组观测: {(df_panel['treat'] == 0).sum()} 个\n\n")
    f.write(f"2. post变量: 政策实施年份={POLICY_YEAR}\n")
    f.write(f"   政策前观测: {(df_panel['post'] == 0).sum()} 个\n")
    f.write(f"   政策后观测: {(df_panel['post'] == 1).sum()} 个\n\n")
    f.write(f"3. did交互项: did = treat × post\n")
    f.write(f"   did=1的观测数: {(df_panel['did'] == 1).sum()} 个\n\n")

    f.write("六、数据验证\n")
    f.write("-" * 100 + "\n")
    f.write(f"面板数据观测总数: {len(df_panel)} 条\n")
    f.write(f"城市数: {df_panel['city_name'].nunique()} 个\n")
    f.write(f"年份范围: {df_panel['year'].min()} - {df_panel['year'].max()}\n")
    f.write(f"完整观测城市: {len(complete_cities)} 个\n\n")

    if carbon_col:
        f.write("七、平行趋势检验\n")
        f.write("-" * 100 + "\n")
        f.write(f"处理组斜率: {slope_treat:.6f}\n")
        f.write(f"对照组斜率: {slope_control:.6f}\n")
        f.write(f"斜率差异: {abs(slope_treat - slope_control):.6f}\n")
        f.write(f"斜率差异t统计量: {t_stat:.3f}\n\n")

        if abs(t_stat) < t_critical:
            f.write("[结论] 平行趋势假设成立 [OK]\n")
        else:
            f.write("[警告] 平行趋势假设可能不成立 [WARN]\n")

    f.write("\n八、输出文件\n")
    f.write("-" * 100 + "\n")
    f.write("1. panel_data_2007_2023_validated.xlsx - 验证后的面板数据\n")
    f.write("2. parallel_trend_plot_validated.png - 全期趋势图\n")
    f.write("3. parallel_trend_pre_policy_validated.png - 政策前趋势图\n")
    f.write("4. did_validation_report.txt - 本报告\n")
    f.write("\n" + "=" * 100 + "\n")

print(f"[OK] 验证报告已保存: {report_file}")

print("\n" + "=" * 100)
print("DID变量显式构造完成（增强版）!")
print("=" * 100)
print("\n关键特性:")
print("  [OK] 自动检测列名（支持模糊匹配）")
print("  [OK] 完整的错误处理和验证")
print("  [OK] 清晰的错误提示信息")
print("  [OK] 基于PSM结果显式生成DID变量")
print(f"  [OK] 政策实施年份: {POLICY_YEAR}")
print("\n输出文件:")
print("  1. panel_data_2007_2023_validated.xlsx")
print("  2. parallel_trend_plot_validated.png")
print("  3. parallel_trend_pre_policy_validated.png")
print("  4. did_validation_report.txt")

print("\n" + "=" * 100)
print("下一步:")
print("=" * 100)
print("请立即打开以下文件，验证平行趋势假设:")
print("  - parallel_trend_pre_policy_validated.png")
print("\n检查要点:")
print("  1. 2009年之前，处理组和对照组的趋势是否基本平行")
print("  2. 斜率差异的t值是否小于临界值（4.303）")
print("  3. 如果平行趋势成立，可以进行后续DID回归分析")
print("=" * 100)
