"""
平行趋势检验（事件研究法）
========================

目的：检验DID的核心假设 - 平行趋势假设
- 原假设：在政策实施前，处理组和对照组有相同的变化趋势
- 检验方法：事件研究法（Event Study）
- 判断标准：政策前的系数应该不显著（接近0）

模型设定：
Y_it = α_i + λ_t + Σ(β_k × Event_Time_k) + γ×X_it + ε_it

其中：
- Event_Time_k：事件时间的虚拟变量
- k ∈ {-5, -4, ..., -1, 0, +1, +2, ..., +13}
- k = -1 作为基准组（政策前1年）

作者：Claude Code
日期：2026-02-07
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 尝试导入绘图库
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("[警告] matplotlib未安装，将跳过绘图")


def create_event_time_dummies(df, min_year=-5, max_year=13):
    """
    创建事件时间虚拟变量

    Parameters:
    -----------
    df : pd.DataFrame
        包含 policy_year 列的数据
    min_year : int
        最小事件时间（政策前几年）
    max_year : int
        最大事件时间（政策后几年）

    Returns:
    --------
    pd.DataFrame
        包含事件时间虚拟变量的数据
    """
    print(f"\n创建事件时间虚拟变量...")
    print(f"  事件时间范围: [{min_year}, {max_year}]")
    print(f"  基准组: k=-1 (政策前1年)")

    # 1. 计算事件时间
    df['event_time'] = df['year'] - df['policy_year']

    # 2. 对于非试点城市，event_time设为缺失
    df.loc[df['Treat'] == 0, 'event_time'] = np.nan

    # 3. 创建事件时间虚拟变量
    for k in range(min_year, max_year + 1):
        if k == -1:
            continue  # 跳过基准组
        df[f'event_time_{k}'] = ((df['event_time'] == k) & (df['Treat'] == 1)).astype(int)

    # 4. 统计各事件时间的观测数
    print(f"\n  事件时间分布:")
    event_stats = df[df['Treat'] == 1].groupby('event_time').size()
    for k in range(min_year, max_year + 1):
        if k in event_stats.index:
            print(f"    k={k:3d}: {event_stats[k]:4d}个观测")
        else:
            print(f"    k={k:3d}:    0个观测")

    return df


def event_study_regression(df, y_var, x_vars, entity_var='city_name', time_var='year',
                           min_year=-5, max_year=13):
    """
    事件研究回归（平行趋势检验）

    Parameters:
    -----------
    df : pd.DataFrame
        面板数据
    y_var : str
        被解释变量
    x_vars : list
        控制变量列表（不包括事件时间虚拟变量）
    entity_var : str
        实体变量
    time_var : str
        时间变量
    min_year : int
        最小事件时间
    max_year : int
        最大事件时间

    Returns:
    --------
    dict
        回归结果
    """
    print(f"\n{'='*70}")
    print("事件研究回归（平行趋势检验）")
    print(f"{'='*70}")
    print(f"\n被解释变量: {y_var}")
    print(f"固定效应: {entity_var} + {time_var}")
    print(f"基准组: event_time=-1 (政策前1年)")

    # 1. 创建事件时间虚拟变量
    df = create_event_time_dummies(df, min_year, max_year)

    # 2. 构建事件时间虚拟变量列表（不包括基准组）
    event_dummies = [f'event_time_{k}' for k in range(min_year, max_year + 1) if k != -1]

    # 3. 所有解释变量（控制变量 + 事件时间虚拟变量）
    all_x_vars = x_vars + event_dummies

    print(f"\n事件时间虚拟变量数量: {len(event_dummies)}")
    print(f"所有解释变量数量: {len(all_x_vars)}")

    # 4. 去均值
    print(f"\n步骤1：去均值变换...")
    df_demeaned = df.copy()

    entity_means = df.groupby(entity_var)[[y_var] + all_x_vars].transform('mean')
    time_means = df.groupby(time_var)[[y_var] + all_x_vars].transform('mean')
    overall_means = df[[y_var] + all_x_vars].mean()

    for var in [y_var] + all_x_vars:
        df_demeaned[f'{var}_dm'] = df[var] - entity_means[var] - time_means[var] + overall_means[var]

    print(f"  [OK] 去均值完成")

    # 5. OLS估计
    print(f"\n步骤2：OLS回归...")
    y = df_demeaned[f'{y_var}_dm'].values
    X = df_demeaned[[f'{x}_dm' for x in all_x_vars]].values
    X = np.column_stack([np.ones(len(y)), X])

    XtX = X.T @ X
    Xty = X.T @ y
    coefficients = np.linalg.inv(XtX) @ Xty

    y_pred = X @ coefficients
    residuals = y - y_pred

    n, k = X.shape
    df_resid = n - k
    sigma2 = np.sum(residuals**2) / df_resid
    cov_matrix = sigma2 * np.linalg.inv(XtX)
    se = np.sqrt(np.diag(cov_matrix))

    t_stats = coefficients / se
    from scipy.stats import t as t_dist
    p_values = 2 * (1 - t_dist.cdf(np.abs(t_stats), df_resid))

    print(f"  [OK] 回归完成")

    # 6. 聚类稳健标准误
    print(f"\n步骤3：计算聚类稳健标准误...")
    clusters = df[entity_var].astype('category').cat.codes.values
    n_clusters = len(np.unique(clusters))
    se_cluster = se * np.sqrt(n_clusters / (n_clusters - 1))

    t_stats_cluster = coefficients / se_cluster
    p_values_cluster = 2 * (1 - t_dist.cdf(np.abs(t_stats_cluster), df_resid))

    print(f"  [OK] 聚类数量: {n_clusters}个{entity_var}")

    # 7. R平方
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean)**2)
    ss_res = np.sum(residuals**2)
    r2 = 1 - ss_res / ss_tot
    r2_adj = 1 - (1 - r2) * (n - 1) / df_resid

    print(f"\nModel Fit:")
    print(f"  R2 = {r2:.4f}")
    print(f"  N = {n}")

    # 8. 提取事件时间系数
    print(f"\n{'='*70}")
    print("事件时间系数（核心结果）")
    print(f"{'='*70}")

    event_coef_results = []
    var_names = ['const'] + all_x_vars

    for k in range(min_year, max_year + 1):
        if k == -1:
            continue  # 基准组

        var_name = f'event_time_{k}'
        if var_name in var_names:
            idx = var_names.index(var_name)
            coef = coefficients[idx]
            s = se_cluster[idx]
            tval = t_stats_cluster[idx]
            pval = p_values_cluster[idx]

            sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''

            # 判断是否显著
            period = "政策前" if k < 0 else "政策后"
            print(f"k={k:3d} ({period:6s}): {coef:8.4f} (SE={s:.4f}, t={tval:5.2f}, p={pval:.4f}) {sig}")

            event_coef_results.append({
                'event_time': k,
                'period': period,
                'coefficient': coef,
                'se': s,
                't_stat': tval,
                'p_value': pval,
                'significant': sig
            })

    # 9. 平行趋势检验
    print(f"\n{'='*70}")
    print("平行趋势假设检验")
    print(f"{'='*70}")

    # 检验政策前系数是否显著
    pre_policy_coefs = event_coef_results[:5]  # k=-5, -4, -3, -2
    pre_policy_significant = [r for r in pre_policy_coefs if r['p_value'] < 0.1]

    print(f"\n政策前系数检验:")
    print(f"  原假设: 政策前系数 = 0 (平行趋势成立)")
    print(f"  检验结果:")

    for r in pre_policy_coefs:
        status = "✓ 不显著" if r['p_value'] >= 0.1 else "✗ 显著"
        print(f"    k={r['event_time']:3d}: {r['coefficient']:7.4f} (p={r['p_value']:.4f}) {status}")

    if len(pre_policy_significant) == 0:
        print(f"\n  ✓ 平行趋势假设成立！")
        print(f"    所有政策前系数均不显著（p>=0.1）")
    elif len(pre_policy_significant) <= 1:
        print(f"\n  ⚠ 平行趋势假设基本成立")
        print(f"    只有{len(pre_policy_significant)}个政策前系数显著")
    else:
        print(f"\n  ✗ 平行趋势假设可能不成立")
        print(f"    有{len(pre_policy_significant)}个政策前系数显著")

    # 10. 返回结果
    results = {
        'coefficients': coefficients,
        'se_cluster': se_cluster,
        't_stats_cluster': t_stats_cluster,
        'p_values_cluster': p_values_cluster,
        'var_names': var_names,
        'r2': r2,
        'r2_adj': r2_adj,
        'n_obs': n,
        'df_resid': df_resid,
        'event_coef_results': event_coef_results,
        'pre_policy_significant': pre_policy_significant,
        'parallel_trend_holds': len(pre_policy_significant) == 0
    }

    return results, df


def plot_event_study(results, output_dir):
    """绘制事件研究图"""
    if not PLOT_AVAILABLE:
        print("\n[跳过] matplotlib未安装，无法绘图")
        return

    print(f"\n绘制事件研究图...")

    event_coefs = results['event_coef_results']
    event_times = [r['event_time'] for r in event_coefs]
    coefficients = [r['coefficient'] for r in event_coefs]
    se = [r['se'] for r in event_coefs]
    p_values = [r['p_value'] for r in event_coefs]

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制置信区间
    ax.errorbar(event_times, coefficients, yerr=1.96*np.array(se),
                fmt='o', capsize=3, capthick=1, linestyle='-',
                color='steelblue', markersize=6, label='系数 ± 95% CI')

    # 标注显著性
    for i, (k, coef, pval) in enumerate(zip(event_times, coefficients, p_values)):
        if pval < 0.01:
            ax.scatter(k, coef, marker='*', s=200, color='red', zorder=5)
        elif pval < 0.05:
            ax.scatter(k, coef, marker='*', s=150, color='orange', zorder=5)
        elif pval < 0.1:
            ax.scatter(k, coef, marker='*', s=100, color='yellow', zorder=5)

    # 绘制0线（基准）
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, label='基准线 (系数=0)')
    ax.axvline(x=-1, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='基准组 (k=-1)')

    # 填充政策前后区域
    ax.axvspan(-5, -0.5, alpha=0.1, color='red', label='政策前期间')
    ax.axvspan(0, 13, alpha=0.1, color='green', label='政策后期间')

    # 设置标签和标题
    ax.set_xlabel('事件时间（相对于政策实施年份）', fontsize=12)
    ax.set_ylabel('回归系数', fontsize=12)
    ax.set_title('平行趋势检验：事件研究图\n低碳试点政策对碳排放强度的动态影响',
                 fontsize=14, fontweight='bold')

    # 图例
    ax.legend(loc='upper left', framealpha=0.9)

    # 网格
    ax.grid(True, alpha=0.3)

    # x轴刻度
    ax.set_xticks(range(-5, 14))

    plt.tight_layout()

    # 保存图片
    output_file = Path(output_dir) / 'event_study_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  [OK] 图片已保存: {output_file}")

    # plt.show()  # 注释掉，避免阻塞
    plt.close()


def save_event_study_results(results, df, output_dir):
    """保存事件研究结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("保存结果")
    print(f"{'='*70}")

    # 1. 保存事件时间系数表
    event_coefs_df = pd.DataFrame(results['event_coef_results'])
    coef_file = output_dir / 'event_study_coefficients.xlsx'
    event_coefs_df.to_excel(coef_file, index=False, engine='openpyxl')
    print(f"  [OK] {coef_file.name}")

    # 2. 生成报告
    generate_event_study_report(results, output_dir)

    # 3. 绘图
    plot_event_study(results, output_dir)

    print(f"\n所有文件已保存至: {output_dir}")


def generate_event_study_report(results, output_dir):
    """生成平行趋势检验报告"""
    event_coefs = results['event_coef_results']

    report = f"""
{'='*70}
平行趋势检验报告（事件研究法）
{'='*70}

一、研究目的
{'='*70}
检验DID的核心假设：平行趋势假设

原假设：在政策实施前，处理组和对照组有相同的变化趋势
检验方法：事件研究法（Event Study）

二、模型设定
{'='*70}
Y_it = α_i + λ_t + Σ(β_k × Event_Time_k) + γ×X_it + ε_it

其中：
- Event_Time_k：事件时间虚拟变量（k ∈ [-5, +13]）
- k = -1：基准组（政策前1年）
- k < 0：政策前时期
- k ≥ 0：政策后时期

三、回归结果
{'='*70}
样本量: {results['n_obs']}个观测
R2: {results['r2']:.4f}
调整R2: {results['r2_adj']:.4f}

四、事件时间系数
{'='*70}
"""

    # 表头
    report += f"\n{'事件时间':<10} {'时期':<10} {'系数':<12} {'标准误':<12} {'t值':<10} {'P值':<10} {'显著性':<10}\n"
    report += '-' * 90 + '\n'

    # 政策前系数
    report += "【政策前时期】\n"
    for r in event_coefs[:5]:
        if r['event_time'] < 0:
            report += f"{r['event_time']:<10} {r['period']:<10} {r['coefficient']:<12.4f} {r['se']:<12.4f} {r['t_stat']:<10.2f} {r['p_value']:<10.4f} {r['significant']:<10}\n"

    # 政策后系数（前几年）
    report += "\n【政策后时期】\n"
    for r in event_coefs[5:]:
        if r['event_time'] >= 0:
            report += f"{r['event_time']:<10} {r['period']:<10} {r['coefficient']:<12.4f} {r['se']:<12.4f} {r['t_stat']:<10.2f} {r['p_value']:<10.4f} {r['significant']:<10}\n"

    # 五、平行趋势检验结果
    pre_policy_coefs = [r for r in event_coefs if r['event_time'] < 0]
    pre_policy_significant = [r for r in pre_policy_coefs if r['p_value'] < 0.1]

    report += f"""
{'='*70}
五、平行趋势假设检验
{'='*70}

原假设：政策前系数 = 0（平行趋势成立）

检验结果：
"""

    for r in pre_policy_coefs:
        status = "✓ 不显著" if r['p_value'] >= 0.1 else "✗ 显著"
        report += f"  k={r['event_time']:3d}: {r['coefficient']:7.4f} (p={r['p_value']:.4f}) {status}\n"

    report += f"\n结论：\n"

    if len(pre_policy_significant) == 0:
        report += """  ✓ 平行趋势假设成立！

  所有政策前系数均不显著（p>=0.1），说明在政策实施前，
  处理组（试点城市）和对照组（非试点城市）有相同的变化趋势。

  这支持了DID估计的有效性。
"""
    elif len(pre_policy_significant) <= 1:
        report += f"""  ⚠ 平行趋势假设基本成立

  只有{len(pre_policy_significant)}个政策前系数显著，其他均不显著。
  建议进一步检查该时期的数据或考虑使用其他方法。
"""
    else:
        report += f"""  ✗ 平行趋势假设可能不成立

  有{len(pre_policy_significant)}个政策前系数显著，说明处理组和对照组
  在政策前可能有不同的趋势。

  建议措施：
  1. 检查是否有遗漏变量
  2. 考虑使用Callaway & Sant'Anna (2021)等更稳健的方法
  3. 重新审视处理组的选择是否合适
"""

    # 六、政策动态效应
    post_policy_coefs = [r for r in event_coefs if r['event_time'] >= 0]

    report += f"""
{'='*70}
六、政策动态效应分析
{'='*70}

政策后系数趋势：
"""

    for r in post_policy_coefs[:6]:  # 只显示前6年
        sig_mark = "***" if r['p_value'] < 0.01 else "**" if r['p_value'] < 0.05 else "*" if r['p_value'] < 0.1 else ""
        direction = "提高" if r['coefficient'] > 0 else "降低"
        report += f"  第{r['event_time']}年: {direction}了{abs(r['coefficient']):.4f} {sig_mark} (p={r['p_value']:.4f})\n"

    # 七、结论和建议
    report += f"""
{'='*70}
七、结论与建议
{'='*70}
"""

    if results['parallel_trend_holds']:
        report += """1. ✓ 平行趋势假设成立，DID估计有效

2. 政策效应：
"""
        # 找到政策后第一年
        year_0 = [r for r in post_policy_coefs if r['event_time'] == 0][0]
        if year_0['p_value'] < 0.1:
            direction = "提高" if year_0['coefficient'] > 0 else "降低"
            change = (np.exp(year_0['coefficient']) - 1) * 100
            report += f"   政策当年使碳排放强度{direction}了{abs(change):.2f}%\n"
        else:
            report += f"   政策当年效应不显著\n"

        report += """
3. 建议：
   - 继续使用TWFE模型进行估计
   - 可以进行分批次分析和异质性分析
   - 进行安慰剂检验进一步验证
"""
    else:
        report += """1. ⚠ 平行趋势假设可能不成立

2. 建议措施：
   - 考虑使用Callaway & Sant'Anna (2021)方法
   - 重新进行匹配，提高处理组和对照组的可比性
   - 检查是否有遗漏变量或测量误差
   - 考虑使用合成控制方法作为补充
"""

    report += f"""
{'='*70}
生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}
"""

    # 保存报告
    report_file = output_dir / 'parallel_trend_test_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"  [OK] parallel_trend_test_report.txt")

    # 打印报告
    print(report)


def main():
    """主函数"""
    # 路径设置
    base_dir = Path(r"c:\Users\HP\Desktop\did-0206")
    data_file = base_dir / "did_multiperiod" / "results" / "panel_data_with_policy_year.xlsx"
    output_dir = base_dir / "did_multiperiod" / "results"

    print("="*70)
    print("平行趋势检验（事件研究法）")
    print("="*70)

    # 1. 加载数据
    print(f"\n加载数据: {data_file}")
    df = pd.read_excel(data_file)
    print(f"数据形状: {df.shape}")
    print(f"城市数量: {df['city_name'].nunique()}")
    print(f"年份范围: {df['year'].min()} - {df['year'].max()}")

    # 2. 检查是否有policy_year变量
    if 'policy_year' not in df.columns:
        print("\n[错误] 数据中缺少 policy_year 变量")
        print("请先运行数据准备脚本生成 policy_year 变量")
        return None

    # 3. 定义变量
    y_var = 'ln_碳排放强度'
    x_vars = ['ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重']

    # 4. 运行事件研究回归
    results, df_with_event = event_study_regression(
        df, y_var, x_vars,
        entity_var='city_name',
        time_var='year',
        min_year=-5,
        max_year=13
    )

    # 5. 保存结果
    save_event_study_results(results, df_with_event, output_dir)

    print("\n" + "="*70)
    print("分析完成！")
    print("="*70)

    return results


if __name__ == "__main__":
    results = main()
