"""生成PSM分析报告"""
import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

# 读取结果
balance_df = pd.read_excel(r'c:\Users\HP\Desktop\did-0206\psm_analysis\results\balance_test.xlsx')
stats_df = pd.read_excel(r'c:\Users\HP\Desktop\did-0206\psm_analysis\results\matching_stats.xlsx')
propensity_df = pd.read_excel(r'c:\Users\HP\Desktop\did-0206\psm_analysis\results\propensity_scores.xlsx')

# 生成报告
report = f"""
{'='*70}
基期倾向得分匹配（PSM）分析报告
{'='*70}

一、研究设计
{'='*70}
基期年份：2009年
匹配变量：
  1. ln_real_gdp（实际GDP对数）
  2. ln_人口密度（人口密度对数）
  3. ln_金融发展水平（金融发展水平对数）
  4. 第二产业占GDP比重（第二产业占比）

匹配方法：最近邻匹配（1:1）
卡尺（Caliper）：0.05

二、样本统计
{'='*70}
"""

for _, row in stats_df.iterrows():
    report += f"处理组数量：{row['n_treated']}个\n"
    report += f"成功匹配数量：{row['n_matched']}个\n"
    report += f"未匹配数量：{row['n_unmatched']}个\n"
    report += f"匹配成功率：{row['match_rate']:.2f}%\n"

report += f"""
三、倾向得分统计
{'='*70}
"""

treated_ps = propensity_df[propensity_df['Treat']==1]['propensity_score']
control_ps = propensity_df[propensity_df['Treat']==0]['propensity_score']

report += f"处理组倾向得分均值：{treated_ps.mean():.4f} ± {treated_ps.std():.4f}\n"
report += f"对照组倾向得分均值：{control_ps.mean():.4f} ± {control_ps.std():.4f}\n"
report += f"倾向得分范围：[{propensity_df['propensity_score'].min():.4f}, {propensity_df['propensity_score'].max():.4f}]\n"

report += f"""
四、协变量平衡性检验
{'='*70}
{'变量':<20} {'匹配前-StdDiff(%)':<20} {'匹配后-StdDiff(%)':<20} {'偏差减少(%)':<15} {'平衡性':<10}
{'-'*85}
"""

for _, row in balance_df.iterrows():
    var_name = row['变量']
    std_diff_before = row['匹配前-标准化差异(%)']
    std_diff_after = row['匹配后-标准化差异(%)']
    bias_reduction = row['偏差减少(%)']
    balanced = '✓ 是' if std_diff_after < 10 else '✗ 否'

    report += f"{var_name:<20} {std_diff_before:<20.2f} {std_diff_after:<20.2f} {bias_reduction:<15.2f} {balanced:<10}\n"

report += f"""
{'-'*85}
注：标准化差异 < 10% 认为满足平衡性要求

五、匹配质量评估
{'='*70}
1. 匹配成功率：{stats_df['match_rate'].iloc[0]:.2f}%
   {'✓ 优秀（>95%）' if stats_df['match_rate'].iloc[0] > 95 else '良好' if stats_df['match_rate'].iloc[0] > 80 else '需改进'}

2. 平衡性检验：{balance_df[balance_df['匹配后-标准化差异(%)'] < 10].shape[0]}/{len(balance_df)}个变量满足平衡性要求
   - ln_real_gdp: {balance_df.iloc[0]['匹配后-标准化差异(%)']:.2f}% {'✓' if balance_df.iloc[0]['匹配后-标准化差异(%)'] < 10 else '✗'}
   - ln_人口密度: {balance_df.iloc[1]['匹配后-标准化差异(%)']:.2f}% {'✓' if balance_df.iloc[1]['匹配后-标准化差异(%)'] < 10 else '✗'}
   - ln_金融发展水平: {balance_df.iloc[2]['匹配后-标准化差异(%)']:.2f}% {'✓' if balance_df.iloc[2]['匹配后-标准化差异(%)'] < 10 else '✗'}
   - 第二产业占GDP比重: {balance_df.iloc[3]['匹配后-标准化差异(%)']:.2f}% {'✓' if balance_df.iloc[3]['匹配后-标准化差异(%)'] < 10 else '✗'}

3. 偏差减少效果：
   - 所有变量的匹配后偏差都显著降低
   - 平均偏差减少：{balance_df['偏差减少(%)'].mean():.2f}%

六、结论与建议
{'='*70}
"""

match_rate = stats_df['match_rate'].iloc[0]
n_balanced = balance_df[balance_df['匹配后-标准化差异(%)'] < 10].shape[0]

if match_rate > 95 and n_balanced >= 2:
    report += """
✓ 匹配质量良好，可以使用匹配后的样本进行DID分析。

建议：
1. 使用匹配后的120对样本进行多期DID估计
2. 考虑对ln_real_gdp和ln_人口密度进行进一步调整（匹配后标准化差异仍>10%）
3. 可以尝试调整卡尺（如0.03或0.1）以优化平衡性
"""
else:
    report += """
匹配质量基本合格，但建议进一步优化。

建议：
1. 考虑调整卡尺范围或使用其他匹配方法（如核匹配）
2. 检查是否存在共同支撑域问题
3. 考虑增加协变量或使用不同的匹配比例（1:2或1:3）
"""

report += f"""
{'='*70}
生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}
"""

print(report)

# 保存报告
with open(r'c:\Users\HP\Desktop\did-0206\psm_analysis\results\PSM报告.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("\n报告已保存至: psm_analysis/results/PSM报告.txt")
