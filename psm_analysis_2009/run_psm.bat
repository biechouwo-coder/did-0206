@echo off
cd /d c:\Users\HP\Desktop\did-0206\psm_analysis_2009

echo ============================================
echo 基期PSM分析 (2009年)
echo ============================================
echo.

echo [1/2] 探索数据...
python 01_explore_data.py
echo.
echo 已生成 explore_output.txt
echo.

echo [2/2] 执行PSM分析...
python psm_analysis.py
echo.

echo ============================================
echo 分析完成!
echo ============================================
echo.
echo 输出文件:
echo   - matched_data_2009.xlsx
echo   - matched_pairs_2009.xlsx
echo   - psm_report_2009.txt
echo   - psm_score_distribution.png
echo   - psm_balance_check.png
echo   - psm_std_bias.png
echo.

pause
