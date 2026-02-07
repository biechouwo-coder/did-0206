# 基期倾向得分匹配（PSM）分析

## 📁 目录结构

```
psm_analysis/
├── baseline_psm.py          # PSM主脚本
├── generate_report.py       # 报告生成脚本
├── README.md                # 本文件
└── results/                 # 结果输出目录
    ├── matched_data.xlsx          # 匹配后的数据（240个样本）
    ├── balance_test.xlsx          # 平衡性检验结果
    ├── matching_stats.xlsx        # 匹配统计信息
    ├── propensity_scores.xlsx     # 倾向得分
    └── PSM报告.txt                # 详细分析报告
```

## 🎯 分析目的

在进行双重差分（DID）分析前，使用倾向得分匹配（PSM）方法解决处理组和对照组之间的选择偏差问题，确保满足"平行趋势"假设。

## 📊 研究设计

### 基期设定
- **基期年份**: 2009年
- **理由**: 低碳城市试点政策于2010年启动，2009年为政策实施前一年

### 匹配变量（协变量）
1. **ln_real_gdp**: 实际GDP对数（衡量经济发展水平）
2. **ln_人口密度**: 人口密度对数（衡量城市化程度）
3. **ln_金融发展水平**: 金融发展水平对数（衡量金融发展程度）
4. **第二产业占GDP比重**: 第二产业占比（衡量产业结构）

### 匹配参数
- **匹配方法**: 最近邻匹配（Nearest Neighbor Matching）
- **匹配比例**: 1:1（每个处理组匹配1个对照组）
- **卡尺（Caliper）**: 0.05（倾向得分差异不超过0.05）
- **是否放回**: 无放回抽样

## 📈 主要结果

### 1. 样本统计
```
处理组（试点城市）:    122个
成功匹配数量:          120个
未匹配数量:            2个
匹配成功率:            98.36%
```

### 2. 倾向得分统计
```
处理组均值:  0.4843 ± 0.1969
对照组均值:  0.3616 ± 0.1255
得分范围:    [0.1778, 0.9287]
```

### 3. 平衡性检验

| 变量 | 匹配前StdDiff(%) | 匹配后StdDiff(%) | 偏差减少(%) | 平衡性 |
|------|------------------|------------------|-------------|--------|
| ln_real_gdp | 53.75 | 21.70 | 59.63 | ✗ |
| ln_人口密度 | 39.20 | 10.44 | 73.36 | ✗ |
| ln_金融发展水平 | 57.26 | 9.43 | 83.52 | ✓ |
| 第二产业占GDP比重 | 4.99 | 2.42 | 51.43 | ✓ |

**判断标准**: 标准化差异（Standardized Difference）< 10% 认为满足平衡性

**结果**: 4个变量中2个满足平衡性要求，平均偏差减少66.99%

### 4. 匹配质量评估
- ✓ **匹配成功率优秀**: 98.36% (>95%)
- ✓ **偏差显著减少**: 所有变量的匹配后偏差都大幅降低
- ⚠ **部分变量不平衡**: ln_real_gdp和ln_人口密度匹配后仍>10%

## 🚀 使用方法

### 运行PSM分析
```bash
# 确保在项目根目录
cd c:\Users\HP\Desktop\did-0206

# 运行PSM分析脚本
py -3.8 psm_analysis\baseline_psm.py

# 生成报告
py -3.8 psm_analysis\generate_report.py
```

### 输入数据要求
脚本会自动读取 `总数据集_已合并_含碳排放_new.xlsx`，该文件需包含以下变量：
- `city_name`: 城市名称
- `year`: 年份
- `Treat`: 处理组标识（1=试点城市，0=非试点城市）
- 匹配变量：ln_real_gdp, ln_人口密度, ln_金融发展水平, 第二产业占GDP比重

### 输出文件
运行完成后，在 `psm_analysis/results/` 目录下会生成：
1. **matched_data.xlsx**: 匹配后的数据（可用于后续DID分析）
2. **balance_test.xlsx**: 平衡性检验详细结果
3. **matching_stats.xlsx**: 匹配统计信息
4. **propensity_scores.xlsx**: 所有样本的倾向得分
5. **PSM报告.txt**: 完整的分析报告

## 💡 下一步建议

### 1. 使用匹配样本进行DID分析
使用 `matched_data.xlsx` 中的120对匹配样本进行多期DID估计，可以得到更可靠的因果效应。

### 2. 优化匹配策略（可选）
如果需要更严格的平衡性，可以尝试：
- **调整卡尺**: 尝试0.03或0.1
- **改变匹配方法**: 核匹配（Kernel Matching）或局部线性回归匹配
- **增加匹配比例**: 1:2或1:3匹配

### 3. 敏感性分析
- 使用不同的协变量组合进行匹配
- 检查匹配结果的稳健性

## 📝 注意事项

1. **基期选择**: 使用2009年作为基期，确保所有匹配变量在政策实施前
2. **共同支撑假设**: 建议检查处理组和对照组的倾向得分分布是否有足够的重叠
3. **平行趋势检验**: PSM后仍需进行平行趋势检验，确保DID的有效性

## 📚 参考文献

1. Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70(1), 41-55.
2. Dehejia, R. H., & Wahba, S. (1999). Causal effects in nonexperimental studies: Reevaluating the evaluation of training programs. *Journal of the American Statistical Association*, 94(448), 1053-1062.

## 🔧 技术细节

### 倾向得分计算
使用Logistic回归模型：
```
P(Treat=1|X) = 1 / (1 + exp(-βX))
```
其中X包含4个协变量。

### 匹配算法
对每个处理组单位i，在对照组中找到倾向得分最接近的单位j，满足：
```
|PS_i - PS_j| ≤ caliper (0.05)
```

### 平衡性检验
标准化差异计算：
```
StdDiff = |X̄_treated - X̄_control| / SD_pooled × 100%
```

---

**最后更新**: 2026-02-07
**分析者**: Claude Code
**数据来源**: 总数据集_已合并_含碳排放_new.xlsx
