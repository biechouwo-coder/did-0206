# 多时点DID分析（Multi-period Difference-in-Differences）

## 📁 目录结构

```
did_multiperiod/
├── 01_data_preparation.py            # 数据准备脚本
├── 02_did_regression.py               # DID回归(numpy实现,详细版)
├── 02_did_regression_simple.py       # DID回归(numpy实现,简化版)
├── 03_did_regression_linearmodels.py # DID回归(linearmodels实现) ⭐推荐
├── 04_compare_methods.py             # 对比numpy和linearmodels结果
├── verify_did_construction.py        # DID变量构造验证
├── README.md                          # 本文件
├── data/                              # 数据文件夹
│   ├── panel_data_final.xlsx         # 最终面板数据（3264个观测）
│   ├── baseline_2009_matched.xlsx    # 基期匹配数据（192个城市）
│   ├── city_list.xlsx                # 城市名单（192个）
│   └── data_preparation_report.txt   # 数据准备报告
└── results/                           # 结果文件夹
    ├── twfe_coefficients.xlsx        # numpy回归系数
    ├── linearmodels_coefficients.xlsx # linearmodels回归系数 ⭐推荐
    └── method_comparison.xlsx        # 方法对比结果
```

## 🎯 分析目标

**为什么要做多时点DID？**

传统的"一刀切"DID假设所有处理组在同一时间接受政策处理。但低碳城市试点政策是**分三批**实施的：
- 第一批：2010年（81个城市）
- 第二批：2012年（26个城市）
- 第三批：2017年（28个城市）

因此，必须使用**多时点DID（Multi-period DID）**方法，也称为：
- 交叠DID（Staggered DID）
- 逐步推进DID（Roll-out DID）

## 📊 数据准备过程

### 第一阶段：数据"回捞"与组装

#### 步骤1：确定入围名单
从PSM匹配结果（`matched_data.xlsx`）中提取所有城市名称：
- **处理组**：96个试点城市（匹配成功）
- **对照组**：96个非试点城市（匹配上的对照）
- **总计**：192个城市

#### 步骤2：筛选面板数据
从原始数据集（`总数据集_已合并_含碳排放_new.xlsx`）中：
- 保留这192个入围城市
- 保留2007-2023年的所有数据
- **最终数据集**：192个城市 × 17年 = **3,264个观测**

## 📈 最终数据集统计

### 基本信息
| 指标 | 数值 |
|------|------|
| 观测数量 | 3,264 |
| 城市数量 | 192个 |
| 年份范围 | 2007-2023年 |
| 时间跨度 | 17年 |
| 处理组城市 | 96个 |
| 对照组城市 | 96个 |

### 各年份数据分布
每个年份都有192个观测（96个处理组 + 96个对照组）：
```
2007-2023年: 每年192个观测
总计: 17年 × 192 = 3,264个观测
```

### DID变量统计
| 变量 | 观测数 | 说明 |
|------|--------|------|
| Treat=1 | 1,632 | 试点城市观测（96个 × 17年） |
| Treat=0 | 1,632 | 对照城市观测（96个 × 17年） |
| Post=1 | 1,172 | 政策实施后观测 |
| DID=1 | 1,172 | 政策变量为1的观测 |

### 关键变量完整性
✅ **所有关键变量均无缺失值**：
- Treat, Post, DID: 0%缺失
- ln_碳排放量_吨: 0%缺失
- ln_real_gdp, ln_人口密度, ln_金融发展水平: 0%缺失
- 第二产业占GDP比重: 0%缺失

## 🔑 DID变量说明

### Treat（分组变量）
- **定义**：城市是否为试点城市
- **取值**：
  - 1 = 试点城市（处理组）
  - 0 = 非试点城市（对照组）
- **特点**：不随时间变化，每个城市固定

### Post（时间变量）
- **定义**：年份是否≥该城市的政策实施年份
- **取值**：
  - 对于试点城市i：如果 year ≥ policy_year_i，则Post=1
  - 对于非试点城市：始终Post=0
- **特点**：随城市和年份变化

**举例**：
- 天津市（2010年试点）：2010年及以后Post=1，2009年及以前Post=0
- 北京市（2012年试点）：2012年及以后Post=1，2011年及以前Post=0
- 南京市（2017年试点）：2017年及以后Post=1，2016年及以前Post=0
- 非试点城市：所有年份Post=0

### DID（政策变量）
- **定义**：DID = Treat × Post
- **取值**：
  - 1 = 试点城市且在政策实施年份及之后
  - 0 = 其他情况
- **含义**：真正暴露于政策的观测

**DID=1的观测分布**：
- 2010-2011年：仅第一批城市（2010年试点）
- 2012-2016年：第一批+第二批城市（2010+2012年试点）
- 2017-2023年：所有三批城市（2010+2012+2017年试点）

## 💡 多时点DID的特点

### 与传统DID的区别

| 特征 | 传统"一刀切"DID | 多时点DID |
|------|-----------------|-----------|
| 政策实施时间 | 所有人同一时间 | 不同人不同时间 |
| 处理效应 | 单一平均效应 | 多个时期效应 |
| 对照组 | 始终未处理 | 尚未处理的人 |
| 估计方法 | 简单双重差分 | 需要特殊方法（如TWFE） |

### 分析挑战

1. **异质性处理效应**：
   - 不同批次的处理效应可能不同
   - 早期处理可能影响后期处理的效果

2. **平行趋势假设**：
   - 需要检验"事件研究"平行趋势
   - 不同批次可能有不同的趋势

3. **估计方法选择**：
   - **TWFE（双向固定效应）**：传统方法，但可能有偏
   - **Callaway & Sant'Anna (2021)**：推荐新方法
   - **Sun & Abraham (2021)**：处理异质性问题
   - **Wooldridge (2021)**：增强回归方法

## 🚀 使用方法

### 数据准备脚本
```bash
# 重新运行数据准备
py -3.8 did_multiperiod\01_data_preparation.py
```

### 输入数据
1. **PSM匹配结果**：`psm_baseline_2009/results/matched_data.xlsx`
   - 96对匹配的试点城市和对照城市（基期2009年）

2. **原始面板数据**：`总数据集_已合并_含碳排放_new.xlsx`
   - 包含所有城市2007-2023年的完整数据
   - 已包含DID变量（Treat, Post, DID）

### 输出数据
所有输出保存在 `did_multiperiod/data/` 目录：

1. **panel_data_final.xlsx** ⭐（最重要）
   - 最终的面板数据集
   - 3,264个观测 × 24个变量
   - 用于多时点DID回归分析

2. **baseline_2009_matched.xlsx**
   - 基期匹配数据（2009年）
   - 供参考和验证使用

3. **city_list.xlsx**
   - 192个入围城市名单
   - 包含是否为试点城市的标识

4. **data_preparation_report.txt**
   - 数据准备完整报告
   - 包含统计信息和下一步建议

## 📝 下一步分析

### 第一阶段：描述性分析
1. 绘制处理组和对照组的时间趋势图
2. 计算各批次的处理效应
3. 检查数据的平衡性和协变量分布

## 🔧 回归分析方法选择

### 为什么推荐使用 linearmodels?

本项目提供了两种DID回归实现方式:

#### 1. **linearmodels 实现** ⭐ **推荐使用**

**优点:**
- ✅ 专业的计量经济学库,结果更可靠
- ✅ 准确的聚类稳健标准误(三明治估计量)
- ✅ 完整的固定效应F检验
- ✅ 多维度R²统计量(Within/Between/Overall)
- ✅ 符合学术发表标准
- ✅ 自动处理面板数据结构

**使用方法:**
```bash
# 安装linearmodels
pip install linearmodels

# 运行回归
py -3.8 did_multiperiod\03_did_regression_linearmodels.py
```

**代码示例:**
```python
from linearmodels.panel import PanelOLS

# 设置面板索引
df = df.set_index(['city_name', 'year'])

# 一行代码完成TWFE回归
model = PanelOLS.from_formula(
    'ln_碳排放强度 ~ DID + ln_real_gdp + ln_人口密度 + ln_金融发展水平 + 第二产业占GDP比重 + EntityEffects + TimeEffects',
    data=df
)
result = model.fit(cov_type='clustered', cluster_entity=True)
print(result)
```

#### 2. **numpy 实现** (用于学习和验证)

**优点:**
- ✅ 完全透明,易于理解Within Transformation原理
- ✅ 无需额外依赖(只需pandas/numpy/scipy)
- ✅ 可以灵活修改每一步
- ✅ 适合教学和理解计量经济学原理

**缺点:**
- ❌ 聚类标准误使用简化算法,可能不准确
- ❌ 缺少完整的统计检验
- ❌ 需要手动编写所有功能

**使用方法:**
```bash
# 运行回归(numpy实现)
py -3.8 did_multiperiod\02_did_regression.py
```

### 两种方法的差异

| 特性 | numpy实现 | linearmodels实现 |
|------|----------|-----------------|
| 回归系数 | ✓ 一致 | ✓ 一致 |
| 标准误准确性 | ⚠️ 简化算法 | ✓ 三明治估计量 |
| 固定效应F检验 | ❌ 需手动计算 | ✓ 自动提供 |
| R²统计量 | 仅Within | Within/Between/Overall |
| 学术认可度 | ⚠️ 需说明 | ✓ 广泛认可 |
| 依赖库 | 基础库 | 需安装linearmodels |
| 代码复杂度 | 高 | 低 |

### 对比两种方法的结果

我们提供了对比脚本来验证两种方法的一致性:

```bash
# 对比numpy和linearmodels的结果
py -3.8 did_multiperiod\04_compare_methods.py
```

**对比内容:**
- 回归系数差异
- 标准误差异
- t统计量和P值
- R平方统计量

**预期结果:**
- 回归系数应该几乎完全一致(差异<0.01%)
- 标准误可能有少量差异(因为使用不同的计算方法)
- linearmodels的标准误更准确

### 使用建议

**对于学术研究:**
1. 主要使用 **linearmodels** 实现的结果
2. 在论文中说明使用 `linearmodels.panel.PanelOLS` 方法
3. 报告聚类稳健标准误(使用三明治估计量)
4. 报告固定效应F检验结果

**对于学习理解:**
1. 先阅读 **numpy实现** 的代码,理解Within Transformation
2. 对比两种方法的输出,加深理解
3. 使用 **linearmodels** 进行实际分析

### 技术说明

#### numpy实现的聚类标准误(简化版):
```python
# 简化调整
SE_cluster = SE_ols * sqrt(n_clusters / (n_clusters - 1))
```
这种方法只考虑了聚类数量,忽略了残差的相关性结构。

#### linearmodels的聚类标准误(三明治估计量):
```python
# 完整的三明治估计量
VCE = (X'X)^(-1) * X' * Ω * X * (X'X)^(-1)
```
其中 Ω 是聚类调整的残差外积矩阵,更准确地处理了聚类内的相关性。

**三明治估计量**是计量经济学中的标准方法,被Stata、R等主流软件采用。

## 📝 下一步分析
1. **事件研究法（Event Study）**
   - 绘制相对时间的系数趋势
   - 检验政策前的系数是否为0
   - 可视化平行趋势

2. **预处理趋势检验**
   - 对每批试点城市分别检验
   - 确保处理组与对照组在政策前有共同趋势

### 第三阶段：多时点DID估计
1. **TWFE估计（基准）**
   ```
   Y_it = α_i + λ_t + β·DID_it + γ·X_it + ε_it
   ```
   其中：
   - α_i：城市固定效应
   - λ_t：年份固定效应
   - β：平均处理效应（ATE）
   - X_it：控制变量

2. **异质性处理效应**
   - Callaway & Sant'Anna方法
   - 分组-时间平均处理效应（GT-A TE）

3. **稳健性检验**
   - 安慰剂检验
   - 排除特定年份或批次
   - 使用不同的控制变量组合

### 第四阶段：异质性分析
1. 按批次分析：第一批 vs 第二批 vs 第三批
2. 按地区分析：东部 vs 中部 vs 西部
3. 按城市规模分析：大城市 vs 中小城市
4. 按产业结构分析：高碳 vs 低碳行业

## 🔧 技术说明

### 数据格式
- **面板数据格式**：每个观测对应一个城市一年
- **平衡面板**：每个城市都有17年的数据
- **变量类型**：
  - 城市标识符：city_name
  - 时间标识符：year
  - 处理变量：Treat, Post, DID
  - 结果变量：ln_碳排放量_吨
  - 控制变量：ln_real_gdp, ln_人口密度, ln_金融发展水平, 第二产业占GDP比重

### 固定效应说明
多时点DID模型需要包含：
1. **城市固定效应（α_i）**：控制不随时间变化的城市特征
2. **年份固定效应（λ_t）**：控制不随城市变化的时间趋势

### 聚类标准误
- **建议**：在城市层面聚类（Cluster by city）
- **原因**：同一城市不同年份的观测可能相关

## 📚 参考文献

### 多时点DID方法论文
1. **Callaway & Sant'Anna (2021)**: "Difference-in-differences with multiple time periods"
2. **Sun & Abraham (2021)**: "Estimating dynamic treatment effects in event studies with heterogeneous treatment effects"
3. **Goodman-Bacon (2021)**: "Difference-in-differences with variation in treatment timing"
4. **Wooldridge (2021)**: "Two-way fixed effects, the two-way Mundlak regression, and difference-in-differences estimators"

### Stata/R/Python包
- **Stata**: `csdid`, `eventstudyinteract`, `jwdid`
- **R**: `did`, `fixest`
- **Python**: `linearmodels.panel.PanelOLS` ⭐ **本项目推荐使用**

**Python linearmodels优势:**
- 与Stata的`reghdfe`类似的功能
- 支持高维固定效应
- 准确的聚类稳健标准误
- 活跃维护和更新

## ⚠️ 注意事项

### 数据质量
✅ **已完成**：
- PSM匹配保证可比性
- 所有变量无缺失值
- 平衡面板数据

⚠️ **需要注意**：
- 确保DID变量定义正确
- 检查政策实施年份准确性
- 验证城市名称一致性

### 模型选择
1. 如果处理效应同质：TWFE是合适的
2. 如果处理效应异质：使用CS-DID或类似方法
3. 建议同时报告多种方法的结果

### 平行趋势
- 必须检验平行趋势假设
- 如果不满足，考虑使用合成控制方法
- 或使用倾向得分匹配+DID（已完成）

## 📞 文件信息

**创建时间**: 2026-02-07
**数据范围**: 2007-2023年（17年）
**城市数量**: 192个（96个处理组 + 96个对照组）
**观测数量**: 3,264个
**分析工具**: Python + pandas + linearmodels ⭐
**推荐方法**: linearmodels.panel.PanelOLS
**下一步**: 多时点DID回归分析

---

**数据准备完成！可以开始多时点DID分析。**
