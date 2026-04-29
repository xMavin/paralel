import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results_mpi.csv")
df["Time"] = df["Time"].astype(float)

plt.figure(figsize=(10, 6))
for p in sorted(df["Processes"].unique()):
    subset = df[df["Processes"] == p]
    plt.plot(subset["N"], subset["Time"], marker='o', label=f"{p} processes")

plt.xlabel("Matrix size (N)")
plt.ylabel("Time (seconds)")
plt.title("Execution time vs matrix size (MPI)")
plt.legend()
plt.grid()
plt.savefig("time_vs_size_mpi.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
for n in sorted(df["N"].unique()):
    subset = df[df["N"] == n]
    plt.plot(subset["Processes"], subset["Time"], marker='o', label=f"N={n}")

plt.xlabel("Number of processes")
plt.ylabel("Time (seconds)")
plt.title("Execution time vs processes count (MPI)")
plt.legend()
plt.grid()
plt.savefig("time_vs_processes_mpi.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
base = df[df["Processes"] == 1].set_index("N")["Time"]
for p in sorted(df["Processes"].unique()):
    if p == 1:
        continue
    subset = df[df["Processes"] == p].set_index("N")
    speedup = base / subset["Time"]
    plt.plot(subset.index, speedup, marker='o', label=f"{p} processes")

plt.xlabel("Matrix size (N)")
plt.ylabel("Speedup")
plt.title("Speedup vs matrix size (MPI)")
plt.legend()
plt.grid()
plt.savefig("speedup_mpi.png", dpi=300, bbox_inches='tight')
plt.show()

print("Graphs saved: time_vs_size_mpi.png, time_vs_processes_mpi.png, speedup_mpi.png")