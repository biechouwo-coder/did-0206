@echo off
cd /d c:\Users\HP\Desktop\did-0206\did_analysis

echo ============================================
echo 基于PSM的DID分析
echo ============================================
echo.

echo 正在运行DID分析...
py 01_construct_panel_and_did.py

echo.
echo ============================================
echo 分析完成!
echo ============================================
echo.
echo 输出文件:
echo   - panel_data_2007_2023.xlsx
echo   - parallel_trend_plot.png
echo   - event_study_plot.png
echo   - did_analysis_report.txt
echo.

pause
