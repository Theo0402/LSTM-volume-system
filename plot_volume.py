import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ymin', type=float, default=1.5, help='Y-axis minimum (default: 0)')
parser.add_argument('--ymax', type=float, default=3, help='Y-axis maximum (default: 3)')
args = parser.parse_args()

# Load data
df = pd.read_csv('data.csv')

# Calculate statistics
avg_vol = df['volume_measured'].mean()
std_vol = df['volume_measured'].std()
var_vol = df['volume_measured'].var()

print(f"Average Volume Measured: {avg_vol:.4f}")
print(f"Standard Deviation:      {std_vol:.4f}")
print(f"Variance:                {var_vol:.6f}")

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(df['avg_speed'], df['volume_measured'], color='royalblue', edgecolors='black', alpha=0.7)
plt.xlabel('Average Speed')
plt.ylabel('Volume Measured')
plt.title('Average Speed vs Volume Measured')
plt.ylim(args.ymin, args.ymax)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('speed_vs_volume.png', dpi=150)
plt.show()
print("\nPlot saved as 'speed_vs_volume.png'")