"""
多时点DID分析 - 数据准备阶段
=====================================

目标：从PSM匹配结果中筛选出最终的面板数据

步骤：
1. 从matched_data.xlsx中提取所有城市名称（处理组+对照组）
2. 从总数据集中筛选这些城市2007-2023年的所有数据
3. 生成最终的面板数据集df_final

作者：Claude Code
日期：2026-02-07
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def extract_city_list_from_psm(matched_data_path):
    """
    从PSM匹配结果中提取城市名单

    Parameters:
    -----------
    matched_data_path : str or Path
        matched_data.xlsx文件路径

    Returns:
    --------
    list
        城市名称列表
    """
    print(f"\n{'='*70}")
    print("步骤1：从PSM匹配结果中提取城市名单")
    print(f"{'='*70}")

    # 读取匹配后的数据
    matched_df = pd.read_excel(matched_data_path)

    print(f"匹配数据形状: {matched_df.shape}")
    print(f"处理组数量: {(matched_df['Treat'] == 1).sum()}")
    print(f"对照组数量: {(matched_df['Treat'] == 0).sum()}")

    # 提取城市名称（去重）
    city_list = matched_df['city_name'].unique().tolist()

    print(f"\n入围城市总数: {len(city_list)}个")
    print(f"城市名单前10个: {city_list[:10]}")

    return city_list, matched_df


def filter_panel_data(total_data_path, city_list, year_start=2007, year_end=2023):
    """
    筛选面板数据

    Parameters:
    -----------
    total_data_path : str or Path
        总数据集文件路径
    city_list : list
        入围城市名单
    year_start : int
        起始年份，默认2007
    year_end : int
        结束年份，默认2023

    Returns:
    --------
    pd.DataFrame
        筛选后的面板数据
    """
    print(f"\n{'='*70}")
    print(f"步骤2：筛选面板数据（{year_start}-{year_end}年）")
    print(f"{'='*70}")

    # 读取总数据集
    total_df = pd.read_excel(total_data_path)

    print(f"\n原始数据集形状: {total_df.shape}")
    print(f"原始城市数量: {total_df['city_name'].nunique()}")
    print(f"原始年份范围: {total_df['year'].min()} - {total_df['year'].max()}")

    # 筛选城市
    df_filtered = total_df[total_df['city_name'].isin(city_list)].copy()

    print(f"\n筛选后数据形状: {df_filtered.shape}")
    print(f"筛选后城市数量: {df_filtered['city_name'].nunique()}")

    # 筛选年份
    df_filtered = df_filtered[
        (df_filtered['year'] >= year_start) &
        (df_filtered['year'] <= year_end)
    ].copy()

    print(f"最终数据形状: {df_filtered.shape}")
    print(f"最终年份范围: {df_filtered['year'].min()} - {df_filtered['year'].max()}")

    # 检查数据完整性
    print(f"\n数据完整性检查:")
    for city in city_list[:5]:  # 检查前5个城市
        city_data = df_filtered[df_filtered['city_name'] == city]
        years = city_data['year'].sort_values().tolist()
        print(f"  {city}: {len(years)}个观测, 年份范围 {min(years)}-{max(years)}")

    print(f"  ...")

    return df_filtered


def validate_final_dataset(df_final, city_list):
    """
    验证最终数据集

    Parameters:
    -----------
    df_final : pd.DataFrame
        最终数据集
    city_list : list
        入围城市名单
    """
    print(f"\n{'='*70}")
    print("步骤3：验证最终数据集")
    print(f"{'='*70}")

    # 基本统计
    print(f"\n数据集形状: {df_final.shape}")
    print(f"城市数量: {df_final['city_name'].nunique()}")
    print(f"年份范围: {df_final['year'].min()} - {df_final['year'].max()}")
    print(f"时间跨度: {df_final['year'].max() - df_final['year'].min() + 1}年")

    # 处理组与对照组
    print(f"\n处理组（试点城市）:")
    treated_cities = df_final[df_final['Treat'] == 1]['city_name'].unique()
    print(f"  数量: {len(treated_cities)}个")
    print(f"  前10个: {treated_cities[:10].tolist()}")

    print(f"\n对照组（匹配的非试点城市）:")
    control_cities = df_final[df_final['Treat'] == 0]['city_name'].unique()
    print(f"  数量: {len(control_cities)}个")
    print(f"  前10个: {control_cities[:10].tolist()}")

    # 各年份数据分布
    print(f"\n各年份数据分布:")
    year_counts = df_final.groupby('year').size()
    for year, count in year_counts.items():
        treated_count = df_final[(df_final['year'] == year) & (df_final['Treat'] == 1)].shape[0]
        control_count = df_final[(df_final['year'] == year) & (df_final['Treat'] == 0)].shape[0]
        print(f"  {year}年: 总计{count:3d}个观测 (处理组{treated_count:3d}, 对照组{control_count:3d})")

    # DID变量统计
    print(f"\nDID变量统计:")
    print(f"  Treat=1观测数: {(df_final['Treat'] == 1).sum()}")
    print(f"  Treat=0观测数: {(df_final['Treat'] == 0).sum()}")
    print(f"  Post=1观测数: {(df_final['Post'] == 1).sum()}")
    print(f"  DID=1观测数: {(df_final['DID'] == 1).sum()}")

    # 检查缺失值
    print(f"\n关键变量缺失值检查:")
    key_vars = ['Treat', 'Post', 'DID', 'ln_碳排放量_吨', 'ln_real_gdp', 'ln_人口密度',
                'ln_金融发展水平', '第二产业占GDP比重']
    for var in key_vars:
        if var in df_final.columns:
            missing_count = df_final[var].isna().sum()
            missing_pct = missing_count / len(df_final) * 100
            print(f"  {var}: {missing_count}个缺失 ({missing_pct:.2f}%)")
        else:
            print(f"  {var}: [变量不存在]")

    return df_final


def save_results(df_final, matched_baseline_df, output_dir):
    """
    保存结果

    Parameters:
    -----------
    df_final : pd.DataFrame
        最终面板数据
    matched_baseline_df : pd.DataFrame
        基期匹配数据
    output_dir : Path
        输出目录
    """
    print(f"\n{'='*70}")
    print("步骤4：保存结果")
    print(f"{'='*70}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 保存最终面板数据
    output_file = output_dir / 'panel_data_final.xlsx'
    df_final.to_excel(output_file, index=False, engine='openpyxl')
    print(f"  [OK] panel_data_final.xlsx ({len(df_final)}个观测)")

    # 2. 保存基期匹配数据（供参考）
    baseline_file = output_dir / 'baseline_2009_matched.xlsx'
    matched_baseline_df.to_excel(baseline_file, index=False, engine='openpyxl')
    print(f"  [OK] baseline_2009_matched.xlsx (基期匹配数据)")

    # 3. 保存城市名单
    city_list = df_final['city_name'].unique().tolist()
    city_df = pd.DataFrame({
        'city_name': city_list,
        'is_treated': [1 if c in df_final[df_final['Treat'] == 1]['city_name'].values else 0
                       for c in city_list]
    })
    city_df.to_excel(output_dir / 'city_list.xlsx', index=False, engine='openpyxl')
    print(f"  [OK] city_list.xlsx ({len(city_list)}个城市)")

    # 4. 生成数据摘要报告
    report = f"""
{'='*70}
多时点DID分析 - 数据准备完成
{'='*70}

一、数据集基本信息
{'='*70}
数据集名称: panel_data_final.xlsx
观测数量: {len(df_final)}
城市数量: {df_final['city_name'].nunique()}
年份范围: {df_final['year'].min()} - {df_final['year'].max()}
时间跨度: {df_final['year'].max() - df_final['year'].min() + 1}年

处理组城市: {len(df_final[df_final['Treat'] == 1]['city_name'].unique())}个
对照组城市: {len(df_final[df_final['Treat'] == 0]['city_name'].unique())}个

二、数据来源
{'='*70}
1. 城市筛选: PSM基期匹配结果（2009年，卡尺=0.05）
2. 原始数据: 总数据集_已合并_含碳排放_new.xlsx
3. 筛选范围: 入围城市的2007-2023年全数据

三、DID变量说明
{'='*70}
- Treat: 分组变量（1=试点城市，0=非试点城市）
- Post: 时间变量（1≥政策实施年份，0=其他）
- DID: 政策变量 = Treat × Post

四、批次信息
{'='*70}
第一批（2010年）: {df_final[(df_final['Treat']==1) & (df_final['year']>=2010)]['city_name'].nunique()}个试点城市
第二批（2012年）: {df_final[(df_final['Treat']==1) & (df_final['year']>=2012)]['city_name'].nunique()}个试点城市
第三批（2017年）: {df_final[(df_final['Treat']==1) & (df_final['year']>=2017)]['city_name'].nunique()}个试点城市

五、下一步
{'='*70}
1. 使用此数据进行多时点DID回归分析
2. 检验平行趋势假设
3. 进行异质性分析和稳健性检验

{'='*70}
生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}
"""

    report_file = output_dir / 'data_preparation_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n  [OK] data_preparation_report.txt")
    print(f"\n所有文件已保存至: {output_dir}")

    return report


def main():
    """主函数"""
    # 路径设置
    base_dir = Path(r"c:\Users\HP\Desktop\did-0206")
    matched_data_path = base_dir / "psm_baseline_2009" / "results" / "matched_data.xlsx"
    total_data_path = base_dir / "总数据集_已合并_含碳排放_new.xlsx"
    output_dir = base_dir / "did_multiperiod" / "data"

    print("="*70)
    print("多时点DID分析 - 数据准备阶段")
    print("="*70)

    # 步骤1：提取城市名单
    city_list, matched_baseline_df = extract_city_list_from_psm(matched_data_path)

    # 步骤2：筛选面板数据
    df_final = filter_panel_data(
        total_data_path=total_data_path,
        city_list=city_list,
        year_start=2007,
        year_end=2023
    )

    # 步骤3：验证最终数据集
    validate_final_dataset(df_final, city_list)

    # 步骤4：保存结果
    report = save_results(df_final, matched_baseline_df, output_dir)

    # 打印报告
    print(report)

    print("\n" + "="*70)
    print("数据准备完成！")
    print("="*70)

    return df_final, city_list


if __name__ == "__main__":
    df_final, city_list = main()
