"""
多时点双重差分(DID)回归分析(linearmodels实现)
==========================================

模型设定:
Y_it = α_i + λ_t + β·DID_it + γ·X_it + ε_it

使用linearmodels.panel.PanelOLS实现

优势:
1. 专业的计量经济学库
2. 准确的聚类稳健标准误(三明治估计量)
3. 自动处理固定效应
4. 丰富的诊断统计量

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

# 尝试导入linearmodels
try:
    from linearmodels.panel import PanelOLS
    LINEARMODELS_AVAILABLE = True
except ImportError:
    LINEARMODELS_AVAILABLE = False
    print("[警告] linearmodels库未安装!")
    print("请运行: pip install linearmodels")


def did_regression_linearmodels(df, y_var, x_vars, entity_var='city_name', time_var='year'):
    """
    使用linearmodels进行双向固定效应回归

    Parameters:
    -----------
    df : pd.DataFrame
        面板数据
    y_var : str
        被解释变量
    x_vars : list
        解释变量列表
    entity_var : str
        实体变量
    time_var : str
        时间变量

    Returns:
    --------
    dict
        回归结果
    """
    if not LINEARMODELS_AVAILABLE:
        raise ImportError("linearmodels库未安装,请运行: pip install linearmodels")

    print(f"\n{'='*70}")
    print("双向固定效应回归(TWFE) - linearmodels实现")
    print(f"{'='*70}")
    print(f"\n被解释变量: {y_var}")
    print(f"解释变量: {', '.join(x_vars)}")
    print(f"固定效应: {entity_var} + {time_var}")
    print(f"标准误: 聚类到{entity_var}层面")

    # 1. 设置面板索引
    print(f"\n步骤1: 设置面板索引...")
    df_panel = df.set_index([entity_var, time_var]).copy()
    print(f"  [OK] 面板索引已设置: {entity_var} + {time_var}")
    print(f"  数据形状: {df_panel.shape}")
    print(f"  实体数量: {df_panel.index.get_level_values(0).nunique()}")
    print(f"  时间数量: {df_panel.index.get_level_values(1).nunique()}")

    # 2. 构建公式
    formula = f"{y_var} ~ {' + '.join(x_vars)} + EntityEffects + TimeEffects"
    print(f"\n步骤2: 构建回归公式...")
    print(f"  {formula}")

    # 3. 拟合模型
    print(f"\n步骤3: 拟合TWFE模型...")
    model = PanelOLS.from_formula(
        formula,
        data=df_panel
    )

    # 使用聚类稳健标准误
    result = model.fit(
        cov_type='clustered',  # 聚类稳健标准误
        cluster_entity=True    # 在实体层面聚类
    )

    print(f"  [OK] 回归完成")

    # 4. 打印结果
    print(f"\n{'='*70}")
    print("回归结果(linearmodels)")
    print(f"{'='*70}")
    print(result)

    # 5. 提取结果
    print(f"\n{'='*70}")
    print("回归系数表(聚类稳健标准误)")
    print(f"{'='*70}")

    # 提取系数和标准误
    coef_df = pd.DataFrame({
        '系数': result.params,
        '标准误': result.std_errors,
        't值': result.tstats,
        'P值': result.pvalues,
        '显著性': ['***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
                  for p in result.pvalues]
    })

    print(coef_df.to_string())

    # 6. 模型拟合统计
    print(f"\n{'='*70}")
    print("模型拟合统计")
    print(f"{'='*70}")
    print(f"观测数: {result.nobs:.0f}")
    print(f"R2(Within): {result.rsquared_within:.4f}")
    print(f"R2(Between): {result.rsquared_between:.4f}")
    print(f"R2(Overall): {result.rsquared_overall:.4f}")
    print(f"实体数量: {result.entity_info['total']:.0f}")
    print(f"时间数量: {result.time_info['total']:.0f}")

    # 7. 固定效应检验
    print(f"\n{'='*70}")
    print("固定效应检验")
    print(f"{'='*70}")
    print(f"实体F检验: {result.f_statistic_entity[0]:.2f} (p={result.f_statistic_entity[1]:.4f})")
    print(f"时间F检验: {result.f_statistic_time[0]:.2f} (p={result.f_statistic_time[1]:.4f})")

    # 8. DID政策效应解释
    did_coef = result.params['DID']
    did_se = result.std_errors['DID']
    did_t = result.tstats['DID']
    did_p = result.pvalues['DID']

    print(f"\n{'='*70}")
    print("政策效应解释")
    print(f"{'='*70}")
    print(f"\nDID系数: {did_coef:.4f}")
    print(f"标准误: {did_se:.4f}")
    print(f"t值: {did_t:.2f}")
    print(f"P值: {did_p:.4f}")

    if did_p < 0.05:
        intensity_change = (1 - np.exp(did_coef)) * 100
        print(f"\n[OK] 政策效应显著 (p={did_p:.4f}{'***' if did_p<0.01 else '**'})")
        print(f"\n经济含义:")
        print(f"  低碳试点政策使碳排放强度{'降低' if did_coef<0 else '提高'}了{abs(intensity_change):.2f}%")
        print(f"  解释: 在控制了城市固定效应、年份固定效应和其他控制变量后,")
        print(f"        低碳试点政策使试点城市的碳排放强度比非试点城市")
        print(f"        {'显著降低' if did_coef<0 else '显著提高'}了{abs(intensity_change):.2f}%")
    else:
        print(f"\n[X] 政策效应不显著 (p={did_p:.4f})")
        print(f"  无法拒绝政策无效的原假设")

    # 9. 返回结果
    results = {
        'params': result.params,
        'std_errors': result.std_errors,
        'tstats': result.tstats,
        'pvalues': result.pvalues,
        'rsquared_within': result.rsquared_within,
        'rsquared_between': result.rsquared_between,
        'rsquared_overall': result.rsquared_overall,
        'nobs': result.nobs,
        'df_resid': result.df_resid,
        'entity_effects': result.f_statistic_entity,
        'time_effects': result.f_statistic_time,
        'did_coef': did_coef,
        'did_se': did_se,
        'did_p': did_p,
        'intensity_change': (1 - np.exp(did_coef)) * 100 if did_p < 0.05 else None,
        'result_object': result  # 保存完整结果对象
    }

    return results, coef_df


def save_results_linearmodels(results, coef_df, df, output_dir):
    """保存linearmodels的结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("保存结果(linearmodels)")
    print(f"{'='*70}")

    # 1. 保存系数表
    coef_file = output_dir / 'linearmodels_coefficients.xlsx'
    coef_df.to_excel(coef_file, engine='openpyxl')
    print(f"  [OK] {coef_file.name}")

    # 2. 保存详细结果(完整summary)
    summary_file = output_dir / 'linearmodels_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(str(results['result_object']))
    print(f"  [OK] {summary_file.name}")

    # 3. 保存数据
    data_file = output_dir / 'panel_data_with_intensity.xlsx'
    df.to_excel(data_file, index=False, engine='openpyxl')
    print(f"  [OK] {data_file.name}")

    # 4. 生成报告
    generate_report_linearmodels(results, output_dir)

    print(f"\n所有文件已保存至: {output_dir}")


def generate_report_linearmodels(results, output_dir):
    """生成linearmodels分析报告"""
    did_p = results['did_p']
    did_coef = results['did_coef']

    report = f"""
{'='*70}
多时点双重差分回归分析报告(linearmodels)
{'='*70}

一、模型设定
{'='*70}
模型类型: 双向固定效应模型(TWFE)
估计方法: linearmodels.panel.PanelOLS
库版本: linearmodels (专业计量经济学库)

被解释变量: ln_碳排放强度
核心解释变量: DID(政策变量)

控制变量:
  - ln_real_gdp(实际GDP对数)
  - ln_人口密度(人口密度对数)
  - ln_金融发展水平(金融发展水平对数)
  - 第二产业占GDP比重

固定效应:
  - 城市固定效应(EntityEffects)
  - 年份固定效应(TimeEffects)

标准误: 聚类稳健标准误(聚类到城市层面)
方法: 三明治估计量(Sandwich Estimator)

二、回归结果
{'='*70}
样本量: {results['nobs']:.0f}个观测
实体数量: {results['result_object'].entity_info['total']:.0f}个城市
时间数量: {results['result_object'].time_info['total']:.0f}年

R2(Within): {results['rsquared_within']:.4f}
R2(Between): {results['rsquared_between']:.4f}
R2(Overall): {results['rsquared_overall']:.4f}

三、固定效应检验
{'='*70}
实体固定效应F检验: {results['entity_effects'][0]:.2f} (p={results['entity_effects'][1]:.4f})
  {'✓ 拒绝原假设,城市固定效应显著' if results['entity_effects'][1]<0.05 else '✗ 无法拒绝原假设'}

时间固定效应F检验: {results['time_effects'][0]:.2f} (p={results['time_effects'][1]:.4f})
  {'✓ 拒绝原假设,年份固定效应显著' if results['time_effects'][1]<0.05 else '✗ 无法拒绝原假设'}

四、政策效应(核心结果)
{'='*70}
"""

    if did_p < 0.05:
        intensity_change = results['intensity_change']
        report += f"""DID系数: {did_coef:.4f} ({'***' if did_p<0.01 else '**'} p={did_p:.4f})
标准误: {results['did_se']:.4f}
t值: {did_coef/results['did_se']:.2f}

{'='*70}
经济含义:
{'='*70}
✓ 政策效应显著!

低碳试点政策使碳排放强度{'降低' if did_coef<0 else '提高'}了{abs(intensity_change):.2f}%。

解释:
  在控制了城市固定效应、年份固定效应和其他控制变量后,
  低碳试点政策使试点城市的碳排放强度比非试点城市
  {'显著降低' if did_coef<0 else '显著提高'}了{abs(intensity_change):.2f}%。

  这一结果在{did_p*100:.1f}%的显著性水平下成立。

  linearmodels使用专业的三明治估计量计算聚类稳健标准误,
  结果可靠且符合学术标准。
"""
    else:
        report += f"""DID系数: {did_coef:.4f} (p={did_p:.4f})
标准误: {results['did_se']:.4f}

✗ 政策效应不显著

无法拒绝政策无效的原假设。建议:
1. 检查模型设定
2. 尝试不同的控制变量组合
3. 进行分样本分析
"""

    report += f"""
{'='*70}
五、所有变量回归系数
{'='*70}
{'变量':<25} {'系数':<12} {'标准误':<12} {'t值':<10} {'P值':<10}
{'-'*80}
"""
    for var in results['params'].index:
        report += f"{var:<25} {results['params'][var]:<12.4f} {results['std_errors'][var]:<12.4f} {results['tstats'][var]:<10.2f} {results['pvalues'][var]:<10.4f}\n"

    report += f"""
{'='*70}
六、linearmodels优势说明
{'='*70}
✓ 使用专业的计量经济学库,结果更可靠
✓ 准确的聚类稳健标准误(三明治估计量)
✓ 完整的固定效应F检验
✓ 多维度的R2统计量(Within/Between/Overall)
✓ 符合学术发表标准

七、稳健性说明
{'='*70}
1. ✓ 使用PSM匹配样本,保证处理组与对照组可比性
2. ✓ 控制城市和年份双向固定效应
3. ✓ 使用聚类稳健标准误(聚类到城市层面)
4. ✓ linearmodels提供准确的统计推断

八、结论与建议
{'='*70}
"""

    if did_p < 0.05:
        report += """✓ 核心结果显著,支持政策有效的假设

建议:
1. ✓ 进行平行趋势检验(事件研究法)
2. ✓ 按批次分析(第一批、第二批、第三批)
3. ✓ 异质性分析(地区、城市规模等)
4. ✓ 安慰剂检验
5. ✓ 尝试Callaway & Sant'Anna(2021)等新方法
"""
    else:
        report += """政策效应不显著,需要进一步分析

建议:
1. 检查平行趋势假设是否成立
2. 尝试不同的模型设定
3. 检查是否存在滞后效应
"""

    report += f"""
{'='*70}
生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}
"""

    # 保存报告
    report_file = output_dir / 'did_regression_report_linearmodels.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"  [OK] did_regression_report_linearmodels.txt")

    # 打印报告
    print(report)


def main():
    """主函数"""
    if not LINEARMODELS_AVAILABLE:
        print("\n" + "="*70)
        print("错误: linearmodels库未安装")
        print("="*70)
        print("\n请先安装linearmodels库:")
        print("  pip install linearmodels")
        print("\n或者使用项目中的其他回归脚本:")
        print("  - 02_did_regression.py (numpy实现)")
        print("  - 02_did_regression_simple.py (简化实现)")
        return None

    # 路径设置
    base_dir = Path(r"c:\Users\HP\Desktop\did-0206")
    data_file = base_dir / "did_multiperiod" / "data" / "panel_data_final.xlsx"
    output_dir = base_dir / "did_multiperiod" / "results"

    print("="*70)
    print("多时点双重差分(DID)回归分析 - linearmodels版本")
    print("="*70)

    # 1. 加载数据
    print(f"\n加载数据: {data_file}")
    df = pd.read_excel(data_file)
    print(f"数据形状: {df.shape}")
    print(f"城市数量: {df['city_name'].nunique()}")
    print(f"年份范围: {df['year'].min()} - {df['year'].max()}")

    # 2. 创建ln_碳排放强度
    if 'ln_碳排放强度' not in df.columns:
        df['ln_碳排放强度'] = df['ln_碳排放量_吨'] - df['ln_real_gdp']
        print("  [OK] 创建变量: ln_碳排放强度")

    # 2.1 对因变量进行1%缩尾处理
    print(f"\n对ln_碳排放强度进行1%缩尾处理...")
    y_lower = df['ln_碳排放强度'].quantile(0.01)
    y_upper = df['ln_碳排放强度'].quantile(0.99)
    print(f"  缩尾前: 1%分位数={y_lower:.4f}, 99%分位数={y_upper:.4f}")

    df['ln_碳排放强度'] = df['ln_碳排放强度'].clip(lower=y_lower, upper=y_upper)

    n_winsorized = ((df['ln_碳排放强度'] == y_lower) | (df['ln_碳排放强度'] == y_upper)).sum()
    print(f"  [OK] 缩尾完成: {n_winsorized}个观测被调整")

    # 3. 定义回归变量
    y_var = 'ln_碳排放强度'
    x_vars = ['DID', 'ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重']

    # 4. 运行TWFE回归(linearmodels)
    results, coef_df = did_regression_linearmodels(df, y_var, x_vars, 'city_name', 'year')

    # 5. 保存结果
    save_results_linearmodels(results, coef_df, df, output_dir)

    print("\n" + "="*70)
    print("分析完成!")
    print("="*70)
    print("\n使用linearmodels库的优势:")
    print("  ✓ 专业的计量经济学库,结果更可靠")
    print("  ✓ 准确的聚类稳健标准误(三明治估计量)")
    print("  ✓ 完整的固定效应F检验")
    print("  ✓ 符合学术发表标准")

    return results


if __name__ == "__main__":
    results = main()
