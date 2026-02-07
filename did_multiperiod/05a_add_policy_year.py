"""
添加 policy_year 变量到面板数据
==============================

policy_year 定义：
- 对于试点城市（Treat=1）：政策首次实施的年份
- 对于非试点城市（Treat=0）：设为NaN（不使用）

作者：Claude Code
日期：2026-02-07
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path


def main():
    """主函数"""
    print("="*70)
    print("添加 policy_year 变量")
    print("="*70)

    # 路径设置
    base_dir = Path(r"c:\Users\HP\Desktop\did-0206")
    data_file = base_dir / "did_multiperiod" / "results" / "panel_data_with_intensity.xlsx"
    output_file = base_dir / "did_multiperiod" / "results" / "panel_data_with_policy_year.xlsx"

    # 1. 加载数据
    print(f"\n加载数据: {data_file}")
    df = pd.read_excel(data_file)
    print(f"数据形状: {df.shape}")

    # 2. 计算 policy_year
    print(f"\n计算 policy_year...")
    print(f"  方法: 对于每个试点城市(Treat=1)，找到第一个Post=1的年份")

    # 初始化policy_year列
    df['policy_year'] = np.nan

    # 对每个试点城市，找到第一个Post=1的年份
    treat_cities = df[df['Treat'] == 1]['city_name'].unique()
    for city in treat_cities:
        city_data = df[df['city_name'] == city]
        post_1_data = city_data[city_data['Post'] == 1]
        if len(post_1_data) > 0:
            policy_year = post_1_data['year'].min()
            df.loc[df['city_name'] == city, 'policy_year'] = policy_year

    # 3. 验证
    print(f"\npolicy_year 分布:")
    policy_year_counts = df[df['Treat']==1].groupby('policy_year').size()
    for year, count in policy_year_counts.items():
        if not pd.isna(year):
            print(f"  {int(year)}年: {count}个城市")

    # 4. 统计
    print(f"\n统计信息:")
    print(f"  试点城市总数: {df[df['Treat']==1]['city_name'].nunique()}")
    print(f"  有policy_year的城市: {df[df['Treat']==1]['policy_year'].notna().sum() / df[df['Treat']==1]['year'].nunique():.0f}")

    # 5. 保存数据
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"\n[OK] 数据已保存至: {output_file}")

    print("\n" + "="*70)
    print("完成！")
    print("="*70)

    return df


if __name__ == "__main__":
    df = main()
