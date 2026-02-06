import subprocess
import sys

# Run the TWFE script
result = subprocess.run([sys.executable, '05_did_twfe_numpy.py'],
                       capture_output=True,
                       text=True,
                       encoding='utf-8')

# Print output
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

# Save output
with open('did_twfe_numpy_output.txt', 'w', encoding='utf-8') as f:
    f.write(result.stdout)
    if result.stderr:
        f.write("\nSTDERR:\n" + result.stderr)

print(f"\nExit code: {result.returncode}")
