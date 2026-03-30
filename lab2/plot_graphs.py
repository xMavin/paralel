import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")
df["Time"] = df["Time"].astype(float)

plt.figure(figsize=(10, 6))
for t in sorted(df["Threads"].unique()):
    subset = df[df["Threads"] == t]
    plt.plot(subset["N"], subset["Time"], marker='o', label=f"{t} threads")

plt.xlabel("Matrix size (N)")
plt.ylabel("Time (seconds)")
plt.title("Execution time vs matrix size")
plt.legend()
plt.grid()
plt.savefig("time_vs_size.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
for n in sorted(df["N"].unique()):
    subset = df[df["N"] == n]
    plt.plot(subset["Threads"], subset["Time"], marker='o', label=f"N={n}")

plt.xlabel("Number of threads")
plt.ylabel("Time (seconds)")
plt.title("Execution time vs threads count")
plt.legend()
plt.grid()
plt.savefig("time_vs_threads.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
base = df[df["Threads"] == 1].set_index("N")["Time"]
for t in sorted(df["Threads"].unique()):
    if t == 1:
        continue
    subset = df[df["Threads"] == t].set_index("N")
    speedup = base / subset["Time"]
    plt.plot(subset.index, speedup, marker='o', label=f"{t} threads")

plt.xlabel("Matrix size (N)")
plt.ylabel("Speedup")
plt.title("Speedup vs matrix size")
plt.legend()
plt.grid()
plt.savefig("speedup.png", dpi=300, bbox_inches='tight')
plt.show()

print("Graphs saved: time_vs_size.png, time_vs_threads.png, speedup.png")