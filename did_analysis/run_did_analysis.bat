@echo off
cd /d c:\Users\HP\Desktop\did-0206\did_analysis

echo ============================================
echo 基于PSM的DID分析（修正版）
echo ============================================
echo.
echo 说明：
echo   本脚本从PSM匹配结果中显式生成DID变量
echo   确保treat、post、did变量的逻辑正确性
echo.

echo [1/2] 读取PSM匹配结果...
echo 处理组城市数: 73个
echo 对照组城市数: 73个
echo.

echo [2/2] 构造面板数据并执行DID分析...
py 02_reconstruct_did_vars.py

echo.
echo ============================================
echo 分析完成!
echo ============================================
echo.
echo 输出文件:
echo   - panel_data_2007_2023_corrected.xlsx
echo   - parallel_trend_plot_psm_based.png
echo   - parallel_trend_pre_policy_psm_based.png
echo   - did_variable_verification_report.txt
echo.
pause
