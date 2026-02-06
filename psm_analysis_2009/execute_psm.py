import subprocess
import sys

print("开始运行PSM分析...")
print("="*80)

# 运行数据探索
print("\n[步骤 1/2] 运行数据探索...")
result1 = subprocess.run([sys.executable, "01_explore_data.py"],
                       capture_output=True, text=True)
print(result1.stdout)
if result1.stderr:
    print("错误:", result1.stderr)

# 运行PSM分析
print("\n[步骤 2/2] 运行PSM分析...")
result2 = subprocess.run([sys.executable, "psm_analysis.py"],
                       capture_output=True, text=True)
print(result2.stdout)
if result2.stderr:
    print("错误:", result2.stderr)

print("\n" + "="*80)
print("分析完成!")
print("="*80)

input("\n按回车键退出...")
