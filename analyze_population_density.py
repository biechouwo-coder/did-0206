import pandas as pd
import numpy as np
from scipy import stats

# 读取数据
file_path = '总数据集_已合并_含碳排放_new.xlsx'
df = pd.read_excel(file_path)

col_name = '人口密度'
print("=" * 80)
print(f"【{col_name}】变量详细描述")
print("=" * 80)

# 1. 基本统计信息
print("\n【1. 基本统计信息】")
print("-" * 80)
desc_df = pd.DataFrame({
    '统计量': ['观测数', '有效观测数', '缺失值数量', '缺失比例(%)',
              '均值', '中位数', '标准差', '方差',
              '最小值', '最大值', '极差', '变异系数'],
    '值': [
        len(df[col_name]),
        df[col_name].notna().sum(),
        df[col_name].isnull().sum(),
        f"{df[col_name].isnull().sum() / len(df) * 100:.2f}",
        f"{df[col_name].mean():.2f}",
        f"{df[col_name].median():.2f}",
        f"{df[col_name].std():.2f}",
        f"{df[col_name].var():.2f}",
        f"{df[col_name].min():.2f}",
        f"{df[col_name].max():.2f}",
        f"{df[col_name].max() - df[col_name].min():.2f}",
        f"{df[col_name].std() / df[col_name].mean():.4f}"
    ]
})
print(desc_df.to_string(index=False))

# 2. 分位数信息
print("\n【2. 分位数信息】")
print("-" * 80)
quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
quantile_df = pd.DataFrame({
    '分位数': [f"{q*100:.0f}%" for q in quantiles],
    '值': [f"{df[col_name].quantile(q):.2f}" for q in quantiles]
})
print(quantile_df.to_string(index=False))

# 3. 分布形状特征
print("\n【3. 分布形状特征】")
print("-" * 80)
skewness = df[col_name].skew()
kurtosis = df[col_name].kurtosis()
dist_df = pd.DataFrame({
    '统计量': ['偏度 (Skewness)', '峰度 (Kurtosis)'],
    '值': [f"{skewness:.4f}", f"{kurtosis:.4f}"],
    '解释': [
        '明显右偏' if skewness > 0.5 else '明显左偏' if skewness < -0.5 else '近似对称分布',
        '尖峰分布（厚尾）' if kurtosis > 1 else '平峰分布（薄尾）' if kurtosis < -1 else '近似正态峰度'
    ]
})
print(dist_df.to_string(index=False))

# 4. 正态性检验
print("\n【4. 正态性检验】")
print("-" * 80)
valid_data = df[col_name].dropna()
if len(valid_data) >= 3:
    statistic, p_value = stats.shapiro(valid_data.sample(min(5000, len(valid_data))))
    normality_df = pd.DataFrame({
        '检验方法': ['Shapiro-Wilk检验'],
        '统计量': [f"{statistic:.4f}"],
        'p值': [f"{p_value:.4f}"],
        '结论': ['拒绝正态分布假设' if p_value < 0.05 else '不能拒绝正态分布假设']
    })
    print(normality_df.to_string(index=False))

# 5. 按年份统计
print("\n【5. 按年份统计】")
print("-" * 80)
year_stats = df.groupby('year')[col_name].agg([
    ('观测数', 'count'),
    ('缺失数', lambda x: x.isnull().sum()),
    ('均值', 'mean'),
    ('标准差', 'std'),
    ('最小值', 'min'),
    ('最大值', 'max')
]).round(2)
print(year_stats)

# 6. 时间趋势分析
print("\n【6. 时间趋势分析】")
print("-" * 80)
yearly_mean = df.groupby('year')[col_name].mean()
yearly_change = yearly_mean.diff()

trend_df = pd.DataFrame({
    '年份': yearly_mean.index,
    '年均值': yearly_mean.values,
    '同比变化': yearly_change.values
})
trend_df['同比变化'] = trend_df['同比变化'].apply(lambda x: f"{x:+.2f}" if not np.isnan(x) else "N/A")
print(trend_df.to_string(index=False))

# 计算总体趋势
first_year = yearly_mean.index[0]
last_year = yearly_mean.index[-1]
first_val = yearly_mean.iloc[0]
last_val = yearly_mean.iloc[-1]
total_change = last_val - first_val
total_change_pct = (total_change / first_val) * 100

print(f"\n总体趋势:")
print(f"  {first_year}年: {first_val:.2f}")
print(f"  {last_year}年: {last_val:.2f}")
print(f"  总变化: {total_change:+.2f} ({total_change_pct:+.2f}%)")

# 7. 按省份统计
print("\n【7. 按省份统计 (Top 10 & Bottom 5)】")
print("-" * 80)
province_stats = df.groupby('province')[col_name].agg([
    ('观测数', 'count'),
    ('均值', 'mean'),
    ('标准差', 'std'),
    ('最小值', 'min'),
    ('最大值', 'max')
]).round(2)
province_stats_sorted = province_stats.sort_values('均值', ascending=False)

print("\n人口密度最高的10个省份:")
print(province_stats_sorted.head(10))

print("\n人口密度最低的5个省份:")
print(province_stats_sorted.tail(5))

# 8. 城市排名
print("\n【8. 城市排名 (Top 10 & Bottom 10)】")
print("-" * 80)
city_mean = df.groupby('city_name')[col_name].mean().sort_values(ascending=False)

print("\n人口密度最高的10个城市:")
print(city_mean.head(10))

print("\n人口密度最低的10个城市:")
print(city_mean.tail(10))

# 9. 人口密度分级
print("\n【9. 人口密度分级分布】")
print("-" * 80)
def classify_density(x):
    if pd.isna(x):
        return '缺失'
    elif x < 100:
        return '低密度(<100)'
    elif x < 300:
        return '中低密度(100-300)'
    elif x < 500:
        return '中等密度(300-500)'
    elif x < 1000:
        return '中高密度(500-1000)'
    else:
        return '高密度(≥1000)'

df['人口密度等级'] = df[col_name].apply(classify_density)
density_dist = df['人口密度等级'].value_counts()
density_pct = df['人口密度等级'].value_counts(normalize=True) * 100

density_df = pd.DataFrame({
    '人口密度等级': density_dist.index,
    '观测数': density_dist.values,
    '比例(%)': density_pct.values
})
print(density_df.to_string(index=False))

# 10. 与其他变量的相关性
print("\n【10. 与其他主要变量的相关性 (Top 15)】")
print("-" * 80)
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlations = []
for col in numeric_cols:
    if col != col_name:
        corr = df[col_name].corr(df[col])
        if not np.isnan(corr):
            correlations.append({'变量': col, '相关系数': corr})
corr_df = pd.DataFrame(correlations).sort_values('相关系数', key=lambda x: abs(x), ascending=False)
corr_df['相关系数'] = corr_df['相关系数'].apply(lambda x: f"{x:.4f}")
print(corr_df.head(15).to_string(index=False))

# 11. DID分组对比
print("\n【11. 按DID分组对比】")
print("-" * 80)
if 'did' in df.columns:
    did_stats = df.groupby('did')[col_name].agg([
        ('观测数', 'count'),
        ('均值', 'mean'),
        ('标准差', 'std'),
        ('最小值', 'min'),
        ('最大值', 'max')
    ]).round(2)
    did_stats.index = ['控制组 (DID=0)', '处理组 (DID=1)']
    print(did_stats)

    # t检验
    from scipy.stats import ttest_ind
    control = df[df['did'] == 0][col_name].dropna()
    treat = df[df['did'] == 1][col_name].dropna()

    if len(control) > 0 and len(treat) > 0:
        t_stat, p_val = ttest_ind(treat, control)
        print(f"\n独立样本t检验:")
        print(f"  t统计量: {t_stat:.4f}")
        print(f"  p值: {p_val:.4f}")
        print(f"  结论: {'处理组和控制组存在显著差异' if p_val < 0.05 else '处理组和控制组无显著差异'}")

# 12. 数据质量检查
print("\n【12. 数据质量检查】")
print("-" * 80)
quality_df = pd.DataFrame({
    '检查项': [
        '缺失值数量',
        '缺失比例',
        '零值数量',
        '负值数量',
        '无穷值数量'
    ],
    '结果': [
        df[col_name].isnull().sum(),
        f"{df[col_name].isnull().sum() / len(df) * 100:.2f}%",
        (df[col_name] == 0).sum(),
        (df[col_name] < 0).sum(),
        np.isinf(df[col_name]).sum()
    ]
})
print(quality_df.to_string(index=False))

# 13. 经济学解释
print("\n【13. 经济学解释与意义】")
print("-" * 80)
print(f"人口密度均值: {df[col_name].mean():.2f} 人/平方公里")
print(f"人口密度中位数: {df[col_name].median():.2f} 人/平方公里")
print(f"变异系数: {df[col_name].std() / df[col_name].mean():.4f}")
print("\n人口密度的经济意义:")
print("• 反映地区人口聚集程度和城市化水平")
print("• 高密度地区通常具有更高的经济活动强度")
print("• 影响劳动力供给、市场规模和集聚效应")
print("• 与经济发展水平、产业结构密切相关")

print("\n中国人口密度特点:")
print("• 东部沿海地区密度高，西部内陆地区密度低")
print("• 省会城市和直辖市密度最高")
print("• 随着城市化进程，人口向高密度地区集中")

print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)
