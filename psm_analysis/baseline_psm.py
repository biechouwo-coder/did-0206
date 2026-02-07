"""
基期倾向得分匹配（PSM）分析
=====================================

基期设定：2009年
匹配变量：ln_real_gdp, ln_人口密度, ln_金融发展水平, 第二产业占GDP比重
匹配方法：最近邻匹配（1:1）
卡尺（caliper）：0.05
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 尝试导入sklearn，如果没有安装则提示用户安装
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("警告: scikit-learn未安装，将使用手动实现的方式")


def prepare_baseline_data(df, baseline_year=2009, city_col='city_name', year_col='year'):
    """
    准备基期数据

    Parameters:
    -----------
    df : pd.DataFrame
        完整数据集
    baseline_year : int
        基期年份，默认为2009年
    city_col : str
        城市列名
    year_col : str
        年份列名

    Returns:
    --------
    pd.DataFrame
        基期数据
    """
    # 提取基期数据
    baseline_df = df[df[year_col] == baseline_year].copy()

    # 删除缺失值的行
    baseline_df = baseline_df.dropna(subset=['Treat', 'ln_real_gdp',
                                              'ln_人口密度', 'ln_金融发展水平',
                                              '第二产业占GDP比重'])

    print(f"\n{'='*70}")
    print(f"基期数据统计（{baseline_year}年）")
    print(f"{'='*70}")
    print(f"处理组（试点城市）数量: {(baseline_df['Treat'] == 1).sum()}")
    print(f"对照组（非试点城市）数量: {(baseline_df['Treat'] == 0).sum()}")
    print(f"总样本数: {len(baseline_df)}")

    return baseline_df


def calculate_propensity_score(baseline_df, covariates):
    """
    计算倾向得分

    使用Logistic回归预测倾向得分

    Parameters:
    -----------
    baseline_df : pd.DataFrame
        基期数据
    covariates : list
        协变量列表

    Returns:
    --------
    pd.DataFrame
        包含倾向得分的数据
    """
    from sklearn.linear_model import LogisticRegression

    X = baseline_df[covariates].values
    y = baseline_df['Treat'].values

    # 使用Logistic回归计算倾向得分
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X, y)

    # 预测概率（倾向得分）
    propensity_scores = lr.predict_proba(X)[:, 1]

    baseline_df['propensity_score'] = propensity_scores

    print(f"\n{'='*70}")
    print("倾向得分统计")
    print(f"{'='*70}")
    print(f"处理组倾向得分均值: {baseline_df[baseline_df['Treat']==1]['propensity_score'].mean():.4f}")
    print(f"对照组倾向得分均值: {baseline_df[baseline_df['Treat']==0]['propensity_score'].mean():.4f}")
    print(f"倾向得分标准差: {baseline_df['propensity_score'].std():.4f}")

    return baseline_df


def perform_matching(baseline_df, caliper=0.05, ratio=1):
    """
    执行倾向得分匹配

    使用最近邻匹配方法

    Parameters:
    -----------
    baseline_df : pd.DataFrame
        包含倾向得分的基期数据
    caliper : float
        卡尺，默认0.05
    ratio : int
        匹配比例，默认1:1

    Returns:
    --------
    pd.DataFrame
        匹配后的数据
    dict
        匹配统计信息
    """
    # 分离处理组和对照组
    treated = baseline_df[baseline_df['Treat'] == 1].copy()
    control = baseline_df[baseline_df['Treat'] == 0].copy()

    print(f"\n{'='*70}")
    print("执行倾向得分匹配")
    print(f"{'='*70}")
    print(f"匹配方法: 最近邻匹配（1:{ratio}）")
    print(f"卡尺: {caliper}")

    matched_controls = []
    unmatched_treated = []

    # 对每个处理组单位寻找匹配的对照组
    for idx, treated_unit in treated.iterrows():
        treated_ps = treated_unit['propensity_score']

        # 计算与对照组的倾向得分距离
        control['ps_distance'] = abs(control['propensity_score'] - treated_ps)

        # 在卡尺范围内寻找最近的匹配
        eligible_controls = control[control['ps_distance'] <= caliper]

        if len(eligible_controls) > 0:
            # 选择距离最近的control
            best_match = eligible_controls.loc[eligible_controls['ps_distance'].idxmin()]
            matched_controls.append(best_match)
        else:
            # 没有找到匹配
            unmatched_treated.append(treated_unit)

    # 将匹配的对照组转换为DataFrame
    if matched_controls:
        matched_control_df = pd.DataFrame(matched_controls)
    else:
        matched_control_df = pd.DataFrame()

    # 统计信息
    n_treated = len(treated)
    n_matched = len(matched_controls)
    n_unmatched = len(unmatched_treated)
    match_rate = n_matched / n_treated * 100 if n_treated > 0 else 0

    stats = {
        'n_treated': n_treated,
        'n_matched': n_matched,
        'n_unmatched': n_unmatched,
        'match_rate': match_rate,
        'caliper': caliper
    }

    print(f"\n匹配结果:")
    print(f"  处理组数量: {n_treated}")
    print(f"  成功匹配数量: {n_matched}")
    print(f"  未匹配数量: {n_unmatched}")
    print(f"  匹配成功率: {match_rate:.2f}%")

    return matched_control_df, stats


def check_balance(baseline_df, matched_control_df, covariates):
    """
    检查匹配后的协变量平衡性

    Parameters:
    -----------
    baseline_df : pd.DataFrame
        完整的基期数据
    matched_control_df : pd.DataFrame
        匹配后的对照组
    covariates : list
        协变量列表

    Returns:
    --------
    pd.DataFrame
        平衡性检验结果
    """
    # 获取匹配后的处理组
    treated_matched = baseline_df[baseline_df['Treat'] == 1].copy()

    # 计算匹配前后的标准化差异
    balance_results = []

    for covar in covariates:
        # 匹配前
        treated_mean_before = baseline_df[baseline_df['Treat'] == 1][covar].mean()
        control_mean_before = baseline_df[baseline_df['Treat'] == 0][covar].mean()
        pooled_std_before = (
            (baseline_df[baseline_df['Treat'] == 1][covar].std()**2 +
             baseline_df[baseline_df['Treat'] == 0][covar].std()**2) / 2
        ) ** 0.5
        std_diff_before = abs(treated_mean_before - control_mean_before) / pooled_std_before * 100

        # 匹配后
        treated_mean_after = treated_matched[covar].mean()
        control_mean_after = matched_control_df[covar].mean()
        pooled_std_after = (
            (treated_matched[covar].std()**2 +
             matched_control_df[covar].std()**2) / 2
        ) ** 0.5
        std_diff_after = abs(treated_mean_after - control_mean_after) / pooled_std_after * 100

        balance_results.append({
            '变量': covar,
            '匹配前-处理组均值': treated_mean_before,
            '匹配前-对照组均值': control_mean_before,
            '匹配前-标准化差异(%)': std_diff_before,
            '匹配后-处理组均值': treated_mean_after,
            '匹配后-对照组均值': control_mean_after,
            '匹配后-标准化差异(%)': std_diff_after,
            '偏差减少(%)': (std_diff_before - std_diff_after) / std_diff_before * 100 if std_diff_before > 0 else 0
        })

    balance_df = pd.DataFrame(balance_results)

    print(f"\n{'='*70}")
    print("协变量平衡性检验")
    print(f"{'='*70}")
    print(balance_df.to_string(index=False))

    # 判断平衡性（通常认为标准化差异<10%为可接受）
    balanced = balance_df['匹配后-标准化差异(%)'] < 10
    print(f"\n平衡性判断（标准化差异<10%）:")
    print(f"  满足平衡性变量数: {balanced.sum()}/{len(balanced)}")

    return balance_df


def save_results(baseline_df, matched_control_df, stats, balance_df, output_dir):
    """
    保存匹配结果

    Parameters:
    -----------
    baseline_df : pd.DataFrame
        基期数据
    matched_control_df : pd.DataFrame
        匹配的对照组
    stats : dict
        匹配统计信息
    balance_df : pd.DataFrame
        平衡性检验结果
    output_dir : Path
        输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 保存匹配后的处理组
    treated_matched = baseline_df[baseline_df['Treat'] == 1].copy()
    treated_matched['matched'] = 'matched' if len(matched_control_df) > 0 else 'unmatched'

    # 2. 保存匹配后的完整数据
    matched_control_df['matched'] = 'matched'
    matched_data = pd.concat([treated_matched, matched_control_df], ignore_index=True)
    matched_data.to_excel(output_dir / 'matched_data.xlsx', index=False, engine='openpyxl')

    # 3. 保存平衡性检验结果
    balance_df.to_excel(output_dir / 'balance_test.xlsx', index=False, engine='openpyxl')

    # 4. 保存匹配统计信息
    stats_df = pd.DataFrame([stats])
    stats_df.to_excel(output_dir / 'matching_stats.xlsx', index=False, engine='openpyxl')

    # 5. 保存倾向得分
    propensity_df = baseline_df[['city_name', 'Treat', 'propensity_score']].copy()
    propensity_df.to_excel(output_dir / 'propensity_scores.xlsx', index=False, engine='openpyxl')

    print(f"\n{'='*70}")
    print("结果保存完成")
    print(f"{'='*70}")
    print(f"输出目录: {output_dir}")
    print(f"  - matched_data.xlsx: 匹配后的数据")
    print(f"  - balance_test.xlsx: 平衡性检验结果")
    print(f"  - matching_stats.xlsx: 匹配统计信息")
    print(f"  - propensity_scores.xlsx: 倾向得分")


def main():
    """主函数"""
    # 路径设置
    base_dir = Path(r"c:\Users\HP\Desktop\did-0206")
    input_file = base_dir / "总数据集_已合并_含碳排放_new.xlsx"
    output_dir = base_dir / "psm_analysis" / "results"

    print("="*70)
    print("基期倾向得分匹配（PSM）分析")
    print("="*70)

    # 1. 读取数据
    print(f"\n正在读取数据: {input_file}")
    df = pd.read_excel(input_file)
    print(f"数据形状: {df.shape}")

    # 2. 准备基期数据（2009年）
    baseline_df = prepare_baseline_data(df, baseline_year=2009)

    # 3. 定义协变量
    covariates = ['ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重']

    print(f"\n匹配变量:")
    for covar in covariates:
        print(f"  - {covar}")

    # 4. 计算倾向得分
    baseline_df = calculate_propensity_score(baseline_df, covariates)

    # 5. 执行匹配
    matched_control_df, stats = perform_matching(baseline_df, caliper=0.05, ratio=1)

    # 6. 检查平衡性
    if len(matched_control_df) > 0:
        balance_df = check_balance(baseline_df, matched_control_df, covariates)

        # 7. 保存结果
        save_results(baseline_df, matched_control_df, stats, balance_df, output_dir)
    else:
        print("\n警告: 没有成功匹配的样本，请检查卡尺设置或数据分布")

    print("\n" + "="*70)
    print("分析完成！")
    print("="*70)

    return baseline_df, matched_control_df, stats


if __name__ == "__main__":
    baseline_df, matched_control_df, stats = main()
