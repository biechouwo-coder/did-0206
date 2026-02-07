"""
增强版基期PSM分析（2009年基期）
=====================================

功能：
1. 倾向得分计算（Logistic回归）
2. 1:1最近邻匹配（卡尺=0.05）
3. 平衡性检验
4. 可视化分析
5. 匹配质量评估

作者：Claude Code
日期：2026-02-07
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 导入sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve


class PSMAnalyzer:
    """倾向得分匹配分析器"""

    def __init__(self, data_path, baseline_year=2009, caliper=0.05):
        """
        初始化PSM分析器

        Parameters:
        -----------
        data_path : str or Path
            数据文件路径
        baseline_year : int
            基期年份，默认2009年
        caliper : float
            匹配卡尺，默认0.05
        """
        self.data_path = Path(data_path)
        self.baseline_year = baseline_year
        self.caliper = caliper
        self.covariates = ['ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重']

        # 存储结果
        self.df = None
        self.baseline_df = None
        self.matched_df = None
        self.matching_stats = {}
        self.balance_df = None

    def load_data(self):
        """加载数据"""
        print(f"\n{'='*70}")
        print("步骤1：加载数据")
        print(f"{'='*70}")
        print(f"数据文件: {self.data_path}")

        self.df = pd.read_excel(self.data_path)
        print(f"数据形状: {self.df.shape}")
        print(f"年份范围: {self.df['year'].min()} - {self.df['year'].max()}")
        print(f"城市数量: {self.df['city_name'].nunique()}")

        return self.df

    def prepare_baseline_data(self):
        """准备基期数据"""
        print(f"\n{'='*70}")
        print(f"步骤2：准备基期数据（{self.baseline_year}年）")
        print(f"{'='*70}")

        # 提取基期数据
        baseline = self.df[self.df['year'] == self.baseline_year].copy()

        # 删除缺失值
        baseline = baseline.dropna(subset=self.covariates + ['Treat'])

        self.baseline_df = baseline

        # 统计
        n_treated = (baseline['Treat'] == 1).sum()
        n_control = (baseline['Treat'] == 0).sum()

        print(f"处理组（试点城市）: {n_treated}个")
        print(f"对照组（非试点城市）: {n_control}个")
        print(f"总样本数: {len(baseline)}")

        return baseline

    def calculate_propensity_scores(self):
        """计算倾向得分"""
        print(f"\n{'='*70}")
        print("步骤3：计算倾向得分（Logistic回归）")
        print(f"{'='*70}")

        X = self.baseline_df[self.covariates].values
        y = self.baseline_df['Treat'].values

        # 标准化协变量
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Logistic回归
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_scaled, y)

        # 预测概率
        propensity_scores = lr.predict_proba(X_scaled)[:, 1]

        self.baseline_df['propensity_score'] = propensity_scores

        # 统计
        treated_ps = self.baseline_df[self.baseline_df['Treat'] == 1]['propensity_score']
        control_ps = self.baseline_df[self.baseline_df['Treat'] == 0]['propensity_score']

        print(f"处理组倾向得分: {treated_ps.mean():.4f} ± {treated_ps.std():.4f}")
        print(f"对照组倾向得分: {control_ps.mean():.4f} ± {control_ps.std():.4f}")
        print(f"倾向得分范围: [{propensity_scores.min():.4f}, {propensity_scores.max():.4f}]")

        # 显示模型系数
        print(f"\nLogistic回归系数:")
        for covar, coef in zip(self.covariates, lr.coef_[0]):
            print(f"  {covar}: {coef:.4f}")

        return propensity_scores

    def perform_matching(self):
        """执行倾向得分匹配"""
        print(f"\n{'='*70}")
        print("步骤4：执行倾向得分匹配")
        print(f"{'='*70}")
        print(f"匹配方法: 最近邻匹配（1:1）")
        print(f"卡尺: {self.caliper}")

        # 分离处理组和对照组
        treated = self.baseline_df[self.baseline_df['Treat'] == 1].copy()
        control = self.baseline_df[self.baseline_df['Treat'] == 0].copy()

        matched_controls = []
        unmatched_treated = []
        used_control_indices = []

        # 对每个处理组单位寻找匹配
        for idx_t, treated_unit in treated.iterrows():
            treated_ps = treated_unit['propensity_score']

            # 计算距离
            control['ps_distance'] = abs(control['propensity_score'] - treated_ps)

            # 排除已使用的对照组
            eligible_controls = control[~control.index.isin(used_control_indices)]
            eligible_controls = eligible_controls[eligible_controls['ps_distance'] <= self.caliper]

            if len(eligible_controls) > 0:
                # 选择最近的
                best_match = eligible_controls.loc[eligible_controls['ps_distance'].idxmin()]
                matched_controls.append(best_match)
                used_control_indices.append(best_match.name)
            else:
                unmatched_treated.append(treated_unit)

        # 构建匹配后的数据
        matched_treated = treated[~treated.index.isin([u.name for u in unmatched_treated])].copy()

        if len(matched_controls) > 0:
            matched_control_df = pd.DataFrame(matched_controls)
            matched_control_df['matched'] = 1
            matched_treated['matched'] = 1
            self.matched_df = pd.concat([matched_treated, matched_control_df], ignore_index=True)
        else:
            self.matched_df = pd.DataFrame()

        # 统计信息
        self.matching_stats = {
            'n_treated': len(treated),
            'n_matched': len(matched_controls),
            'n_unmatched': len(unmatched_treated),
            'match_rate': len(matched_controls) / len(treated) * 100 if len(treated) > 0 else 0,
            'caliper': self.caliper,
            'baseline_year': self.baseline_year
        }

        print(f"\n匹配结果:")
        print(f"  处理组数量: {self.matching_stats['n_treated']}")
        print(f"  成功匹配: {self.matching_stats['n_matched']}")
        print(f"  未匹配: {self.matching_stats['n_unmatched']}")
        print(f"  匹配成功率: {self.matching_stats['match_rate']:.2f}%")

        return self.matched_df

    def check_balance(self):
        """检查协变量平衡性"""
        print(f"\n{'='*70}")
        print("步骤5：协变量平衡性检验")
        print(f"{'='*70}")

        if self.matched_df is None or len(self.matched_df) == 0:
            print("错误: 没有匹配成功的样本")
            return None

        # 获取匹配后的处理组和对照组
        treated_matched = self.matched_df[self.matched_df['Treat'] == 1]
        control_matched = self.matched_df[self.matched_df['Treat'] == 0]

        # 匹配前
        treated_before = self.baseline_df[self.baseline_df['Treat'] == 1]
        control_before = self.baseline_df[self.baseline_df['Treat'] == 0]

        balance_results = []

        for covar in self.covariates:
            # 匹配前
            t_mean_before = treated_before[covar].mean()
            c_mean_before = control_before[covar].mean()
            pooled_std_before = np.sqrt((treated_before[covar].std()**2 + control_before[covar].std()**2) / 2)
            std_diff_before = abs(t_mean_before - c_mean_before) / pooled_std_before * 100

            # 匹配后
            t_mean_after = treated_matched[covar].mean()
            c_mean_after = control_matched[covar].mean()
            pooled_std_after = np.sqrt((treated_matched[covar].std()**2 + control_matched[covar].std()**2) / 2)
            std_diff_after = abs(t_mean_after - c_mean_after) / pooled_std_after * 100

            # 偏差减少
            bias_reduction = (std_diff_before - std_diff_after) / std_diff_before * 100 if std_diff_before > 0 else 0

            balance_results.append({
                '变量': covar,
                '匹配前-处理组': t_mean_before,
                '匹配前-对照组': c_mean_before,
                '匹配前-StdDiff(%)': std_diff_before,
                '匹配后-处理组': t_mean_after,
                '匹配后-对照组': c_mean_after,
                '匹配后-StdDiff(%)': std_diff_after,
                '偏差减少(%)': bias_reduction,
                '平衡性': 'Y' if std_diff_after < 10 else 'N'
            })

        self.balance_df = pd.DataFrame(balance_results)

        # 打印结果
        print(f"\n{'变量':<25} {'匹配前StdDiff(%)':<15} {'匹配后StdDiff(%)':<15} {'偏差减少(%)':<15} {'平衡性':<10}")
        print('-' * 80)
        for _, row in self.balance_df.iterrows():
            print(f"{row['变量']:<25} {row['匹配前-StdDiff(%)']:<15.2f} {row['匹配后-StdDiff(%)']:<15.2f} {row['偏差减少(%)']:<15.2f} {row['平衡性']:<10}")

        n_balanced = (self.balance_df['平衡性'] == '[OK]').sum()
        print(f"\n满足平衡性(<10%)的变量: {n_balanced}/{len(self.balance_df)}")

        return self.balance_df

    def visualize(self, output_dir):
        """生成可视化图表"""
        print(f"\n{'='*70}")
        print("步骤6：生成可视化图表")
        print(f"{'='*70}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 倾向得分分布图
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 匹配前
        treated_ps = self.baseline_df[self.baseline_df['Treat'] == 1]['propensity_score']
        control_ps = self.baseline_df[self.baseline_df['Treat'] == 0]['propensity_score']

        axes[0].hist(treated_ps, bins=20, alpha=0.6, label='处理组', color='red', density=True)
        axes[0].hist(control_ps, bins=20, alpha=0.6, label='对照组', color='blue', density=True)
        axes[0].set_xlabel('倾向得分')
        axes[0].set_ylabel('密度')
        axes[0].set_title('匹配前倾向得分分布')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 匹配后
        if self.matched_df is not None and len(self.matched_df) > 0:
            treated_matched_ps = self.matched_df[self.matched_df['Treat'] == 1]['propensity_score']
            control_matched_ps = self.matched_df[self.matched_df['Treat'] == 0]['propensity_score']

            axes[1].hist(treated_matched_ps, bins=20, alpha=0.6, label='处理组', color='red', density=True)
            axes[1].hist(control_matched_ps, bins=20, alpha=0.6, label='对照组', color='blue', density=True)
            axes[1].set_xlabel('倾向得分')
            axes[1].set_ylabel('密度')
            axes[1].set_title('匹配后倾向得分分布')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'ps_distribution.png', dpi=300, bbox_inches='tight')
        print(f"  [OK] 保存: ps_distribution.png")
        plt.close()

        # 2. 标准化差异对比图
        if self.balance_df is not None:
            fig, ax = plt.subplots(figsize=(10, 6))

            x = np.arange(len(self.balance_df))
            width = 0.35

            before_diff = self.balance_df['匹配前-StdDiff(%)'].values
            after_diff = self.balance_df['匹配后-StdDiff(%)'].values

            bars1 = ax.bar(x - width/2, before_diff, width, label='匹配前', color='coral', alpha=0.8)
            bars2 = ax.bar(x + width/2, after_diff, width, label='匹配后', color='lightblue', alpha=0.8)

            ax.axhline(y=10, color='red', linestyle='--', linewidth=1.5, label='平衡性阈值(10%)')

            ax.set_xlabel('协变量', fontsize=12)
            ax.set_ylabel('标准化差异 (%)', fontsize=12)
            ax.set_title('匹配前后标准化差异对比', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([c.replace('ln_', '').replace('_', '\n') for c in self.balance_df['变量']], rotation=0, fontsize=9)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plt.savefig(output_dir / 'std_bias_comparison.png', dpi=300, bbox_inches='tight')
            print(f"  [OK] 保存: std_bias_comparison.png")
            plt.close()

        # 3. 偏差减少图
        if self.balance_df is not None:
            fig, ax = plt.subplots(figsize=(10, 6))

            bias_reduction = self.balance_df['偏差减少(%)'].values
            colors = ['green' if br > 50 else 'orange' if br > 30 else 'red' for br in bias_reduction]

            bars = ax.barh(range(len(self.balance_df)), bias_reduction, color=colors, alpha=0.7)
            ax.set_yticks(range(len(self.balance_df)))
            ax.set_yticklabels([c.replace('ln_', '') for c in self.balance_df['变量']])
            ax.set_xlabel('偏差减少 (%)', fontsize=12)
            ax.set_title('协变量偏差减少效果', fontsize=14, fontweight='bold')
            ax.axvline(x=50, color='red', linestyle='--', linewidth=1.5, label='50%阈值')
            ax.grid(True, alpha=0.3, axis='x')
            ax.legend()

            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars, bias_reduction)):
                ax.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=10)

            plt.tight_layout()
            plt.savefig(output_dir / 'bias_reduction.png', dpi=300, bbox_inches='tight')
            print(f"  [OK] 保存: bias_reduction.png")
            plt.close()

        print(f"\n所有图表已保存至: {output_dir}")

    def save_results(self, output_dir):
        """保存结果"""
        print(f"\n{'='*70}")
        print("步骤7：保存结果")
        print(f"{'='*70}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 匹配后的数据
        if self.matched_df is not None:
            self.matched_df.to_excel(output_dir / 'matched_data.xlsx', index=False, engine='openpyxl')
            print(f"  [OK] matched_data.xlsx ({len(self.matched_df)}个样本)")

        # 2. 平衡性检验结果
        if self.balance_df is not None:
            self.balance_df.to_excel(output_dir / 'balance_test.xlsx', index=False, engine='openpyxl')
            print(f"  [OK] balance_test.xlsx")

        # 3. 匹配统计
        stats_df = pd.DataFrame([self.matching_stats])
        stats_df.to_excel(output_dir / 'matching_stats.xlsx', index=False, engine='openpyxl')
        print(f"  [OK] matching_stats.xlsx")

        # 4. 倾向得分
        if self.baseline_df is not None:
            ps_df = self.baseline_df[['city_name', 'Treat', 'propensity_score']].copy()
            ps_df.to_excel(output_dir / 'propensity_scores.xlsx', index=False, engine='openpyxl')
            print(f"  [OK] propensity_scores.xlsx")

        print(f"\n结果保存至: {output_dir}")

    def generate_report(self):
        """生成文本报告"""
        report = f"""
{'='*70}
基期倾向得分匹配（PSM）分析报告
{'='*70}

一、研究设计
{'='*70}
基期年份：{self.baseline_year}年
匹配变量：
  1. ln_real_gdp（实际GDP对数）
  2. ln_人口密度（人口密度对数）
  3. ln_金融发展水平（金融发展水平对数）
  4. 第二产业占GDP比重（第二产业占比）

匹配方法：最近邻匹配（1:1）
卡尺（Caliper）：{self.caliper}

二、样本统计
{'='*70}
"""

        for key, value in self.matching_stats.items():
            if key not in ['caliper', 'baseline_year']:
                label = {
                    'n_treated': '处理组数量',
                    'n_matched': '成功匹配数量',
                    'n_unmatched': '未匹配数量',
                    'match_rate': '匹配成功率'
                }.get(key, key)
                if key == 'match_rate':
                    report += f"{label}：{value:.2f}%\n"
                else:
                    report += f"{label}：{value}个\n"

        if self.balance_df is not None:
            report += f"""
三、协变量平衡性检验
{'='*70}
{'变量':<25} {'匹配前StdDiff(%)':<18} {'匹配后StdDiff(%)':<18} {'偏差减少(%)':<15} {'平衡性':<10}
{'-'*90}
"""
            for _, row in self.balance_df.iterrows():
                report += f"{row['变量']:<25} {row['匹配前-StdDiff(%)']:<18.2f} {row['匹配后-StdDiff(%)']:<18.2f} {row['偏差减少(%)']:<15.2f} {row['平衡性']:<10}\n"

            report += f"\n满足平衡性(<10%)的变量: {(self.balance_df['平衡性']=='[OK]').sum()}/{len(self.balance_df)}\n"
            report += f"平均偏差减少: {self.balance_df['偏差减少(%)'].mean():.2f}%\n"

        report += f"""
四、匹配质量评估
{'='*70}
"""

        match_rate = self.matching_stats['match_rate']
        if match_rate > 95:
            report += f"[OK] 匹配成功率优秀（{match_rate:.2f}% > 95%）\n"
        elif match_rate > 80:
            report += f"[OK] 匹配成功率良好（{match_rate:.2f}%）\n"
        else:
            report += f"[!] 匹配成功率较低（{match_rate:.2f}%），建议调整参数\n"

        if self.balance_df is not None:
            n_balanced = (self.balance_df['平衡性'] == '[OK]').sum()
            if n_balanced >= len(self.balance_df) * 0.5:
                report += f"[OK] 平衡性检验通过（{n_balanced}/{len(self.balance_df)}个变量满足要求）\n"
            else:
                report += f"[!] 部分变量不平衡（{n_balanced}/{len(self.balance_df)}个变量满足要求）\n"

        report += f"""
五、结论与建议
{'='*70}
"""

        if match_rate > 95 and self.balance_df is not None and (self.balance_df['平衡性']=='[OK]').sum() >= 2:
            report += """[OK] 匹配质量良好，可以使用匹配后的样本进行DID分析。

建议：
1. 使用匹配后的样本进行多期DID估计
2. 检查平行趋势假设
3. 考虑进行安慰剂检验
"""
        else:
            report += """匹配质量基本合格，建议进一步优化。

建议：
1. 调整卡尺范围（如0.03或0.1）
2. 尝试其他匹配方法（如核匹配、局部线性回归匹配）
3. 增加匹配比例（1:2或1:3）
"""

        report += f"""
{'='*70}
生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}
"""

        return report


def main():
    """主函数"""
    # 路径设置
    base_dir = Path(r"c:\Users\HP\Desktop\did-0206")
    data_file = base_dir / "总数据集_已合并_含碳排放_new.xlsx"
    output_dir = base_dir / "psm_baseline_2009" / "results"

    print("="*70)
    print("基期倾向得分匹配（PSM）分析")
    print("基期：2009年 | 卡尺：0.05")
    print("="*70)

    # 创建分析器
    analyzer = PSMAnalyzer(
        data_path=data_file,
        baseline_year=2009,
        caliper=0.05
    )

    # 执行分析
    analyzer.load_data()
    analyzer.prepare_baseline_data()
    analyzer.calculate_propensity_scores()
    analyzer.perform_matching()
    analyzer.check_balance()
    analyzer.visualize(output_dir)
    analyzer.save_results(output_dir)

    # 生成报告
    report = analyzer.generate_report()
    print(report)

    # 保存报告
    with open(output_dir / 'PSM分析报告.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n报告已保存至: {output_dir / 'PSM分析报告.txt'}")

    print("\n" + "="*70)
    print("分析完成！")
    print("="*70)

    return analyzer


if __name__ == "__main__":
    analyzer = main()
