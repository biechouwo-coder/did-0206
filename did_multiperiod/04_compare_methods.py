"""
对比numpy实现和linearmodels实现的双重差分回归结果
==================================================

对比两种实现方式的:
1. 回归系数
2. 标准误
3. t统计量
4. P值
5. R平方

作者: Claude Code
日期: 2026-02-07
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def compare_results():
    """对比两种方法的结果"""
    print("="*70)
    print("对比numpy实现和linearmodels实现的DID回归结果")
    print("="*70)

    base_dir = Path(r"c:\Users\HP\Desktop\did-0206")
    data_file = base_dir / "did_multiperiod" / "data" / "panel_data_final.xlsx"

    # 加载数据
    print(f"\n加载数据: {data_file}")
    df = pd.read_excel(data_file)

    # 创建ln_碳排放强度
    if 'ln_碳排放强度' not in df.columns:
        df['ln_碳排放强度'] = df['ln_碳排放量_吨'] - df['ln_real_gdp']

    # 1%缩尾
    y_lower = df['ln_碳排放强度'].quantile(0.01)
    y_upper = df['ln_碳排放强度'].quantile(0.99)
    df['ln_碳排放强度'] = df['ln_碳排放强度'].clip(lower=y_lower, upper=y_upper)

    y_var = 'ln_碳排放强度'
    x_vars = ['DID', 'ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重']

    # =================================================================
    # 方法1: numpy实现(Within Transformation)
    # =================================================================
    print(f"\n{'='*70}")
    print("方法1: numpy实现(Within Transformation + OLS)")
    print(f"{'='*70}")

    # 去均值
    df_dm = df.copy()
    entity_means = df.groupby('city_name')[[y_var] + x_vars].transform('mean')
    time_means = df.groupby('year')[[y_var] + x_vars].transform('mean')
    overall_means = df[[y_var] + x_vars].mean()

    for var in [y_var] + x_vars:
        df_dm[f'{var}_dm'] = df[var] - entity_means[var] - time_means[var] + overall_means[var]

    # OLS估计
    y = df_dm[f'{y_var}_dm'].values
    X = df_dm[[f'{x}_dm' for x in x_vars]].values
    X = np.column_stack([np.ones(len(y)), X])

    XtX = X.T @ X
    Xty = X.T @ y
    coef_numpy = np.linalg.inv(XtX) @ Xty

    # 残差和标准误
    y_pred = X @ coef_numpy
    residuals = y - y_pred
    n, k = X.shape
    df_resid = n - k
    sigma2 = np.sum(residuals**2) / df_resid
    cov_matrix = sigma2 * np.linalg.inv(XtX)
    se_numpy = np.sqrt(np.diag(cov_matrix))

    # 聚类稳健标准误(简化版)
    n_clusters = df['city_name'].nunique()
    se_cluster_numpy = se_numpy * np.sqrt(n_clusters / (n_clusters - 1))

    # t统计量和p值
    from scipy.stats import t as t_dist
    t_stats_numpy = coef_numpy / se_cluster_numpy
    p_values_numpy = 2 * (1 - t_dist.cdf(np.abs(t_stats_numpy), df_resid))

    # R平方
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean)**2)
    ss_res = np.sum(residuals**2)
    r2_numpy = 1 - ss_res / ss_tot

    print(f"  [OK] numpy实现完成")
    print(f"  R2 = {r2_numpy:.4f}")
    print(f"  观测数 = {n}")

    # =================================================================
    # 方法2: linearmodels实现
    # =================================================================
    print(f"\n{'='*70}")
    print("方法2: linearmodels实现(PanelOLS)")
    print(f"{'='*70}")

    try:
        from linearmodels.panel import PanelOLS

        # 设置面板索引
        df_panel = df.set_index(['city_name', 'year'])

        # 构建公式
        formula = f"{y_var} ~ {' + '.join(x_vars)} + EntityEffects + TimeEffects"

        # 拟合模型
        model = PanelOLS.from_formula(formula, data=df_panel)
        result_lm = model.fit(cov_type='clustered', cluster_entity=True)

        # 提取结果
        coef_lm = result_lm.params.values
        se_lm = result_lm.std_errors.values
        t_stats_lm = result_lm.tstats.values
        p_values_lm = result_lm.pvalues.values
        r2_within_lm = result_lm.rsquared_within
        r2_between_lm = result_lm.rsquared_between
        r2_overall_lm = result_lm.rsquared_overall

        print(f"  [OK] linearmodels实现完成")
        print(f"  R2(Within) = {r2_within_lm:.4f}")
        print(f"  R2(Between) = {r2_between_lm:.4f}")
        print(f"  R2(Overall) = {r2_overall_lm:.4f}")

        LINEARMODELS_AVAILABLE = True
    except ImportError:
        print(f"  [X] linearmodels未安装,跳过对比")
        LINEARMODELS_AVAILABLE = False
        result_lm = None

    # =================================================================
    # 对比结果
    # =================================================================
    if LINEARMODELS_AVAILABLE:
        print(f"\n{'='*70}")
        print("结果对比")
        print(f"{'='*70}")

        # 变量名
        var_names = ['const'] + x_vars

        # 创建对比表
        comparison_data = []
        for i, var in enumerate(var_names):
            # numpy结果
            coef_np = coef_numpy[i]
            se_np = se_cluster_numpy[i]
            t_np = t_stats_numpy[i]
            p_np = p_values_numpy[i]

            # linearmodels结果
            coef_lm_val = coef_lm[i]
            se_lm_val = se_lm[i]
            t_lm_val = t_stats_lm[i]
            p_lm_val = p_values_lm[i]

            # 差异
            coef_diff = abs(coef_np - coef_lm_val)
            coef_diff_pct = (coef_diff / abs(coef_lm_val)) * 100 if coef_lm_val != 0 else 0

            se_diff = abs(se_np - se_lm_val)
            se_diff_pct = (se_diff / se_lm_val) * 100 if se_lm_val != 0 else 0

            comparison_data.append({
                '变量': var,
                '系数_np': coef_np,
                '系数_lm': coef_lm_val,
                '系数差异': coef_diff,
                '系数差异%': coef_diff_pct,
                'SE_np': se_np,
                'SE_lm': se_lm_val,
                'SE差异': se_diff,
                'SE差异%': se_diff_pct,
                't值_np': t_np,
                't值_lm': t_lm_val,
                'P值_np': p_np,
                'P值_lm': p_lm_val,
            })

        comp_df = pd.DataFrame(comparison_data)

        # 打印对比表
        print(f"\n{'='*90}")
        print("回归系数对比")
        print(f"{'='*90}")
        print(f"\n{'变量':<15} {'numpy':>12} {'linearmodels':>14} {'差异':>12} {'差异%':>10}")
        print('-' * 70)

        for _, row in comp_df.iterrows():
            print(f"{row['变量']:<15} {row['系数_np']:>12.4f} {row['系数_lm']:>14.4f} "
                  f"{row['系数差异']:>12.4f} {row['系数差异%']:>9.2f}%")

        print(f"\n{'='*90}")
        print("标准误对比")
        print(f"{'='*90}")
        print(f"\n{'变量':<15} {'numpy':>12} {'linearmodels':>14} {'差异':>12} {'差异%':>10}")
        print('-' * 70)

        for _, row in comp_df.iterrows():
            print(f"{row['变量']:<15} {row['SE_np']:>12.4f} {row['SE_lm']:>14.4f} "
                  f"{row['SE差异']:>12.4f} {row['SE差异%']:>9.2f}%")

        print(f"\n{'='*90}")
        print("t统计量和P值对比")
        print(f"{'='*90}")
        print(f"\n{'变量':<15} {'t值(np)':>10} {'t值(lm)':>10} {'P值(np)':>10} {'P值(lm)':>10}")
        print('-' * 70)

        for _, row in comp_df.iterrows():
            print(f"{row['变量']:<15} {row['t值_np']:>10.2f} {row['t值_lm']:>10.2f} "
                  f"{row['P值_np']:>10.4f} {row['P值_lm']:>10.4f}")

        # =================================================================
        # 核心结论
        # =================================================================
        print(f"\n{'='*70}")
        print("核心结论")
        print(f"{'='*70}")

        did_row = comp_df[comp_df['变量'] == 'DID'].iloc[0]

        print(f"\nDID系数对比:")
        print(f"  numpy实现:         {did_row['系数_np']:.4f} (SE={did_row['SE_np']:.4f}, t={did_row['t值_np']:.2f})")
        print(f"  linearmodels实现:  {did_row['系数_lm']:.4f} (SE={did_row['SE_lm']:.4f}, t={did_row['t值_lm']:.2f})")
        print(f"  差异:              {did_row['系数差异']:.4f} ({did_row['系数差异%']:.2f}%)")

        print(f"\n标准误对比:")
        print(f"  numpy实现(简化版):      {did_row['SE_np']:.4f}")
        print(f"  linearmodels(三明治):   {did_row['SE_lm']:.4f}")
        print(f"  差异:                   {did_row['SE差异']:.4f} ({did_row['SE差异%']:.2f}%)")

        # 判断一致性
        coef_consistent = did_row['系数差异_pct'] < 0.01  # 系数差异<0.01%
        se_consistent = did_row['SE差异_pct'] < 5  # 标准误差异<5%

        print(f"\n一致性判断:")
        print(f"  回归系数: {'✓ 完全一致' if coef_consistent else '⚠ 存在差异'}")
        print(f"  标准误:   {'✓ 基本一致' if se_consistent else '⚠ 存在差异'}")

        # =================================================================
        # 方法优缺点总结
        # =================================================================
        print(f"\n{'='*70}")
        print("方法优缺点总结")
        print(f"{'='*70}")

        print(f"\nnumpy实现:")
        print(f"  ✓ 优点:")
        print(f"    - 完全透明,易于理解原理")
        print(f"    - 无需额外依赖")
        print(f"    - 可以灵活修改每一步")
        print(f"  ✗ 缺点:")
        print(f"    - 聚类标准误使用简化算法,可能不准确")
        print(f"    - 缺少完整的统计检验")
        print(f"    - 需要手动编写所有功能")

        print(f"\nlinearmodels实现:")
        print(f"  ✓ 优点:")
        print(f"    - 专业的计量经济学库,结果可靠")
        print(f"    - 使用三明治估计量,标准误更准确")
        print(f"    - 自动提供固定效应F检验")
        print(f"    - 提供R2(Within/Between/Overall)")
        print(f"    - 符合学术发表标准")
        print(f"  ✗ 缺点:")
        print(f"    - 需要安装额外库")
        print(f"    - 相对不透明")

        # =================================================================
        # 建议
        # =================================================================
        print(f"\n{'='*70}")
        print("使用建议")
        print(f"{'='*70}")

        if coef_consistent and se_consistent:
            print(f"\n✓ 两种方法结果基本一致")
            print(f"\n建议:")
            print(f"  1. 主要使用linearmodels(更准确、更专业)")
            print(f"  2. 保留numpy实现作为验证和学习工具")
            print(f"  3. 在论文中使用linearmodels的结果(更可靠)")
        else:
            print(f"\n⚠ 两种方法结果存在差异")
            print(f"\n建议:")
            print(f"  1. 使用linearmodels(更准确的统计推断)")
            print(f"  2. 检查numpy实现的聚类标准误计算")
            print(f"  3. 在论文中报告linearmodels的结果")

        # =================================================================
        # 保存对比结果
        # =================================================================
        output_dir = base_dir / "did_multiperiod" / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存对比表
        comparison_file = output_dir / 'method_comparison.xlsx'
        comp_df.to_excel(comparison_file, index=False, engine='openpyxl')
        print(f"\n[OK] 对比结果已保存至: {comparison_file}")

        # 生成对比报告
        generate_comparison_report(comp_df, r2_numpy, r2_within_lm, output_dir)

        print(f"\n{'='*70}")
        print("对比完成!")
        print(f"{'='*70}")

    else:
        print(f"\n[提示] 请安装linearmodels库后再次运行:")
        print(f"  pip install linearmodels")


def generate_comparison_report(comp_df, r2_numpy, r2_lm, output_dir):
    """生成方法对比报告"""
    did_row = comp_df[comp_df['变量'] == 'DID'].iloc[0]

    report = f"""
{'='*70}
numpy vs linearmodels 方法对比报告
{'='*70}

一、对比说明
{'='*70}
本报告对比了两种多时点DID回归实现方式:
1. numpy实现: Within Transformation + OLS(手动实现)
2. linearmodels实现: PanelOLS(专业计量经济学库)

二、DID系数对比(核心结果)
{'='*70}
numpy实现:         {did_row['系数_np']:.4f} (SE={did_row['SE_np']:.4f}, t={did_row['t值_np']:.2f}, p={did_row['P值_np']:.4f})
linearmodels实现:  {did_row['系数_lm']:.4f} (SE={did_row['SE_lm']:.4f}, t={did_row['t值_lm']:.2f}, p={did_row['P值_lm']:.4f})

系数差异:  {did_row['系数差异']:.4f} ({did_row['系数差异%']:.2f}%)
标准误差异: {did_row['SE差异']:.4f} ({did_row['SE差异%']:.2f}%)

三、R平方对比
{'='*70}
numpy R2:             {r2_numpy:.4f}
linearmodels R2(Within):  {r2_lm:.4f}

四、一致性评价
{'='*70}
"""

    coef_consistent = did_row['系数差异_pct'] < 0.01
    se_consistent = did_row['SE差异_pct'] < 5

    if coef_consistent:
        report += "✓ 回归系数完全一致(差异<0.01%)\n\n"
    else:
        report += f"⚠ 回归系数存在差异({did_row['系数差异%']:.2f}%)\n\n"

    if se_consistent:
        report += "✓ 标准误基本一致(差异<5%)\n\n"
    else:
        report += f"⚠ 标准误存在差异({did_row['SE差异%']:.2f}%)\n"
        report += "  原因: numpy使用简化版聚类标准误,linearmodels使用三明治估计量\n\n"

    report += f"""五、方法优缺点
{'='*70}
numpy实现:
  ✓ 优点: 完全透明、易于理解原理、无需额外依赖
  ✗ 缺点: 聚类标准误简化、缺少完整检验、需手动编写

linearmodels实现:
  ✓ 优点: 专业可靠、标准误准确、完整检验、符合学术标准
  ✗ 缺点: 需要额外安装

六、使用建议
{'='*70}
"""

    if coef_consistent and se_consistent:
        report += """✓ 两种方法结果基本一致

建议:
  1. 主要使用linearmodels(更准确、更专业)
  2. 保留numpy实现作为验证和学习工具
  3. 在论文中使用linearmodels的结果
"""
    else:
        report += """⚠ 两种方法结果存在差异

建议:
  1. 使用linearmodels(更准确的统计推断)
  2. 检查numpy实现的聚类标准误计算
  3. 在论文中报告linearmodels的结果
"""

    report += f"""
{'='*70}
七、技术说明
{'='*70}
numpy实现的聚类标准误(简化版):
  SE_cluster = SE_ols * sqrt(n_clusters / (n_clusters - 1))

linearmodels的聚类标准误(三明治估计量):
  VCE = (X'X)^(-1) * X' * Ω * X * (X'X)^(-1)
  其中 Ω 是聚类调整的残差外积矩阵

三明治估计量是更准确的方法,被学术界广泛采用。

{'='*70}
生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}
"""

    # 保存报告
    report_file = output_dir / 'method_comparison_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"  [OK] method_comparison_report.txt")

    # 打印报告
    print(report)


if __name__ == "__main__":
    compare_results()
