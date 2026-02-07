# Scripts 文件夹

本文件夹包含项目开发过程中使用的所有Python脚本。

## 📁 脚本分类

### 🔧 变量生成脚本

#### add_ln_carbon_v2.py
- **功能**：生成 `ln_碳排放量_吨` 变量
- **输入**：总数据集_已合并_含碳排放_new.xlsx
- **输出**：更新总数据集（添加对数碳排放变量）
- **状态**：✅ 已使用

#### add_ln_carbon_intensity.py
- **功能**：生成 `ln_碳排放强度_名义GDP` 变量
- **输入**：总数据集_已合并_含碳排放_new.xlsx
- **输出**：更新总数据集（添加对数碳排放强度变量）
- **状态**：✅ 已使用

### 📊 数据检查脚本

#### check_columns.py
- **功能**：检查数据集列名
- **用途**：调试和验证数据结构

#### read_with_openpyxl.py
- **功能**：使用openpyxl读取Excel文件
- **用途**：替代pandas读取，测试不同读取方式

### 🏙️ 大连市数据分析脚本

#### find_dalian.py
- **功能**：在数据集中查找大连市
- **输出**：大连市的基本信息

#### read_dalian_data.py
- **功能**：读取大连市的完整数据
- **用途**：查看大连市的DID变量

#### read_dalian_fixed.py
- **功能**：修正版本的大连市数据读取
- **改进**：修复了之前版本的问题

#### extract_dalian.py
- **功能**：提取大连市数据并导出
- **输出**：大连市的完整数据集

#### format_dalian_table.py
- **功能**：格式化大连市数据表
- **输出**：美观的表格格式

#### to_csv.py
- **功能**：将大连市数据导出为CSV
- **用途**：便于在其他软件中使用

### 📝 报告生成脚本

#### create_report.py
- **功能**：生成大连市分析报告
- **输出**：文本格式的分析报告

#### create_markdown_report.py
- **功能**：生成Markdown格式的报告
- **输出**：Dalian_DID_Report.md

## 🗂️ 项目结构

```
did-0206/
├── scripts/                          # Python脚本文件夹
│   ├── add_ln_carbon_v2.py          # 生成ln_碳排放量_吨
│   ├── add_ln_carbon_intensity.py   # 生成ln_碳排放强度_名义GDP
│   ├── find_dalian.py               # 查找大连市
│   ├── read_dalian_data.py          # 读取大连市数据
│   ├── create_markdown_report.py    # 生成MD报告
│   └── ... (其他辅助脚本)
├── psm_analysis_2009/               # PSM分析结果
├── did_analysis/                    # DID分析（水平值）
├── did_analysis_ln_carbon/          # DID分析（ln_碳排放量）
├── 总数据集_已合并_含碳排放_new.xlsx
└── README.md
```

## 📌 使用说明

### 运行脚本

所有脚本都需要从项目根目录运行：

```bash
# 错误方式
cd scripts
python add_ln_carbon_v2.py

# 正确方式
cd c:\Users\HP\Desktop\did-0206
python scripts/add_ln_carbon_v2.py
```

或者使用相对路径：

```bash
cd c:\Users\HP\Desktop\id-0206
python scripts/add_ln_carbon_intensity.py
```

### 脚本依赖

所有脚本依赖以下库：
- pandas
- numpy
- matplotlib
- scipy
- openpyxl

安装依赖：
```bash
pip install pandas numpy matplotlib scipy openpyxl
```

## 🗑️ 清理说明

这些脚本主要用于数据准备和调试，核心分析结果在对应的文件夹中：
- **PSM分析**：`psm_analysis_2009/`
- **DID分析**：`did_analysis/` 和 `did_analysis_ln_carbon/`

脚本文件保留在 `scripts/` 文件夹中，便于：
1. 追踪数据处理过程
2. 复用代码
3. 调试和验证

## 📊 脚本执行历史

| 日期 | 脚本 | 功能 | 状态 |
|------|------|------|------|
| 2026-02-06 | add_ln_carbon_v2.py | 生成ln_碳排放量_吨 | ✅ 完成 |
| 2026-02-07 | add_ln_carbon_intensity.py | 生成ln_碳排放强度_名义GDP | ✅ 完成 |
| 2026-02-07 | extract_dalian.py | 提取大连市数据 | ✅ 完成 |

---

**注意**：这些是辅助脚本，主要的分析脚本在各自的分析文件夹中。
