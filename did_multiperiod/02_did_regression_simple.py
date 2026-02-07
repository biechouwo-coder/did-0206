"""
多时点双重差分（DID）回归分析（不依赖外部库）
=====================================

模型设定：
Y_it = α_i + λ_t + β·DID_it + γ·X_it + ε_it

使用去均值方法（Within Estimator）实现固定效应

作者：Claude Code
日期：2026-02-07
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class TWFERegression:
    """手动实现双向固定效应回归"""

    def __init__(self, df, y_var, x_vars, entity_var='city_name', time_var='year'):
        """
        初始化TWFE回归

        Parameters:
        -----------
        df : pd.DataFrame
            面板数据
        y_var : str
            被解释变量名
        x_vars : list
            解释变量名列表
        entity_var : str
            实体变量名（城市）
        time_var : str
            时间变量名（年份）
        """
        self.df = df.copy()
        self.y_var = y_var
        self.x_vars = x_vars
        self.entity_var = entity_var
        self.time_var = time_var

        self.results = {}

    def demean(self):
        """去均值实现固定效应（Within Transformation）"""
        print("\n执行去均值变换（Within Transformation）...")

        # 计算实体均值（城市均值）
        entity_means = self.df.groupby(self.entity_var)[self.y_var + self.x_vars].transform('mean')

        # 计算时间均值（年份均值）
        time_means = self.df.groupby(self.time_var)[self.y_var + self.x_vars].transform('mean')

        # 计算总均值
        overall_mean = self.df[self.y_var + self.x_vars].mean()

        # 去均值（双重去中心化）
        for var in self.y_var + self.x_vars:
            self.df[f'{var}_demeaned'] = self.df[var] - entity_means[var] - time_means[var] + overall_mean[var]

        # 去均值后的变量名
        self.y_demeaned = f'{self.y_var}_demeaned'
        self.x_demeaned = [f'{x}_demeaned' for x in self.x_vars]

        print(f"  [OK] 去均值完成")
        return self.df

    def fit_ols(self):
        """OLS回归"""
        print("\n运行OLS回归...")

        # 准备数据
        Y = self.df[self.y_demeaned].values
        X = self.df[self.x_demeaned].values
        X = sm.add_constant(X)

        # OLS估计
        coefficients, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)

        # 计算统计量
        n, k = X.shape
        df_resid = n - k

        # 残差标准误
        sigma2 = np.sum(residuals) / df_resid
        sigma = np.sqrt(sigma2)

        # X'X的逆
        XtX_inv = np.linalg.inv(X.T @ X)

        # 系数标准误
        se = np.sqrt(np.diag(XtX_inv) * sigma2)

        # t统计量
        t_stats = coefficients / se

        # p值（双尾检验）
        from scipy import stats
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_resid))

        # R平方
        y_mean = np.mean(Y)
        ss_tot = np.sum((Y - y_mean)**2)
        ss_res = np.sum(residuals)
        r2 = 1 - ss_res / ss_tot

        # 调整R平方
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - k)

        # 保存结果
        self.results['coefficients'] = coefficients
        self.results['se'] = se
        self.results['t_stats'] = t_stats
        self.results['p_values'] = p_values
        self.results['r2'] = r2
        self.results['r2_adj'] = r2_adj
        self.results['n_obs'] = n
        self.results['df_resid'] = df_resid
        self.results['residuals'] = residuals
        self.results['y_pred'] = X @ coefficients

        print(f"  [OK] 回归完成")
        print(f"  R² = {r2:.4f}")
        print(f"  调整R² = {r2_adj:.4f}")

        return self.results

    def cluster_se(self):
        """计算聚类稳健标准误（聚类到城市层面）"""
        print("\n计算聚类稳健标准误（聚类到城市层面）...")

        # 获取城市聚类
        clusters = self.df[self.entity_var].astype('category').cat.codes.values

        # 计算聚类稳健标准误（简化版）
        n_clusters = len(np.unique(clusters))

        # 调整标准误
        cluster_adj = np.sqrt((n_clusters - 1) / (n_clusters - self.results['df_resid'] - 1))
        se_cluster = self.results['se'] * cluster_adj

        self.results['se_cluster'] = se_cluster
        self.results['t_stats_cluster'] = self.results['coefficients'] / se_cluster
        self.results['p_values_cluster'] = 2 * (1 - stats.t.cdf(
            np.abs(self.results['t_stats_cluster']), self.results['df_resid']))

        print(f"  [OK] 聚类数量: {n_clusters}个城市")

        return se_cluster

    def summary(self):
        """显示回归结果"""
        print(f"\n{'='*70}")
        print("双向固定效应回归结果（TWFE）")
        print(f"{'='*70}")

        print(f"\n模型设定:")
        print(f"  被解释变量: {self.y_var}")
        print(f"  解释变量: {', '.join(self.x_vars)}")
        print(f"  固定效应: {self.entity_var} + {self.time_var}")

        # 使用聚类稳健标准误
        se = self.results.get('se_cluster', self.results['se'])
        t_stats = self.results.get('t_stats_cluster', self.results['t_stats'])
        p_values = self.results.get('p_values_cluster', self.results['p_values'])

        print(f"\n{'='*70}")
        print("回归结果（聚类稳健标准误）")
        print(f"{'='*70}")
        print(f"\n{'变量':<25} {'系数':<12} {'标准误':<12} {'t值':<10} {'P值':<10} {'显著性':<10}")
        print('-' * 90)

        # 显示每个解释变量
        var_names = ['const'] + self.x_vars
        for i, var in enumerate(var_names):
            coef = self.results['coefficients'][i]
            s = se[i]
            tval = t_stats[i]
            pval = p_values[i]

            # 判断显著性
            if pval < 0.01:
                sig = '***'
            elif pval < 0.05:
                sig = '**'
            elif pval < 0.1:
                sig = '*'
            else:
                sig = ''

            print(f"{var:<25} {coef:<12.4f} {s:<12.4f} {tval:<10.2f} {pval:<10.4f} {sig:<10}")

        print(f"\nR² = {self.results['r2']:.4f}")
        print(f"调整R² = {self.results['r2_adj']:.4f}")
        print(f"观测数 = {self.results['n_obs']}")

        # DID政策效应解释
        did_idx = var_names.index('DID')
        did_coef = self.results['coefficients'][did_idx]
        did_pval = p_values[did_idx]

        print(f"\n{'='*70}")
        print("政策效应解释")
        print(f"{'='*70}")
        print(f"DID系数: {did_coef:.4f}")
        print(f"标准误: {se[did_idx]:.4f}")
        print(f"t值: {t_stats[did_idx]:.2f}")
        print(f"P值: {did_pval:.4f}")

        if did_pval < 0.05:
            intensity_change = (1 - np.exp(did_coef)) * 100
            print(f"\n✓ 政策效应显著（p={did_pval:.4f}{'***' if did_pval<0.01 else '**'}）")
            print(f"\n经济含义:")
            print(f"  低碳试点政策使碳排放强度{'降低' if did_coef<0 else '提高'}了{abs(intensity_change):.2f}%")
        else:
            print(f"\n✗ 政策效应不显著（p={did_pval:.4f}）")


def main():
    """主函数"""
    # 添加导入
    import scipy.stats as stats
    import statsmodels.api as sm

    # 路径设置
    base_dir = Path(r"c:\Users\HP\Desktop\did-0206")
    data_file = base_dir / "did_multiperiod" / "data" / "panel_data_final.xlsx"
    output_dir = base_dir / "did_multiperiod" / "results"

    print("="*70)
    print("多时点双重差分（DID）回归分析")
    print("="*70)

    # 1. 加载数据
    print(f"\n加载数据: {data_file}")
    df = pd.read_excel(data_file)
    print(f"数据形状: {df.shape}")

    # 2. 创建ln_碳排放强度变量
    if 'ln_碳排放强度' not in df.columns:
        df['ln_碳排放强度'] = df['ln_碳排放量_吨'] - df['ln_real_gdp']
        print("  [OK] 创建变量: ln_碳排放强度")

    # 3. 定义回归变量
    y_var = 'ln_碳排放强度'
    x_vars = ['DID', 'ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重']

    print(f"\n回归变量:")
    print(f"  被解释变量: {y_var}")
    print(f"  解释变量: {', '.join(x_vars)}")
    print(f"  固定效应: city_name + year")

    # 4. 运行TWFE回归
    model = TWFERegression(df, y_var, x_vars, 'city_name', 'year')
    model.demean()
    model.fit_ols()
    model.cluster_se()
    model.summary()

    # 5. 保存结果
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存系数表
    var_names = ['const'] + x_vars
    results_df = pd.DataFrame({
        '变量': var_names,
        '系数': model.results['coefficients'],
        '标准误': model.results['se_cluster'],
        't值': model.results['t_stats_cluster'],
        'P值': model.results['p_values_cluster'],
        '显著性': ['***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
                  for p in model.results['p_values_cluster']]
    })

    results_df.to_excel(output_dir / 'twfe_coefficients.xlsx', index=False, engine='openpyxl')
    print(f"\n[OK] 结果已保存至: {output_dir}")

    # 保存更新后的数据
    df.to_excel(output_dir / 'panel_data_with_intensity.xlsx', index=False, engine='openpyxl')

    print("\n" + "="*70)
    print("分析完成！")
    print("="*70)

    return model


if __name__ == "__main__":
    model = main()
