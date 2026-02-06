@echo off
cd /d c:\Users\HP\Desktop\did-0206\psm_analysis_2009
python 01_explore_data.py > explore_output.txt 2>&1
type explore_output.txt
pause
