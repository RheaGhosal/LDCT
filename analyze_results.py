import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
csv_file = "results_classifier.csv"
df = pd.read_csv(csv_file)

# Group by dose level
grouped = df.groupby("dose")

# Compute mean ± std for each metric
summary = grouped.agg(["mean", "std"])[["acc", "auc", "sens", "spec", "prec", "f1"]]

# Save summary to CSV for paper
summary.to_csv("summary_stats.csv")
print("✅ Saved summary statistics to summary_stats.csv")
print(summary)

# ----------- PLOTS -----------

# 1. Bar plots with error bars (mean ± std)
metrics_to_plot = ["acc", "auc", "sens", "spec", "f1"]

for metric in metrics_to_plot:
    means = grouped[metric].mean()
    stds = grouped[metric].std()

    plt.figure(figsize=(6,4))
    plt.bar(means.index, means.values, yerr=stds.values, capsize=5, alpha=0.7)
    plt.xlabel("Dose Level")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} across dose levels")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{metric}_barplot.png", dpi=300)
    plt.close()
    print(f"📊 Saved {metric}_barplot.png")

# 2. Boxplots across seeds (distribution per dose)
for metric in metrics_to_plot:
    plt.figure(figsize=(6,4))
    df.boxplot(column=metric, by="dose", grid=False)
    plt.xlabel("Dose Level")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} distribution across seeds")
    plt.suptitle("")  # remove default pandas title
    plt.tight_layout()
    plt.savefig(f"{metric}_boxplot.png", dpi=300)
    plt.close()
    print(f"📊 Saved {metric}_boxplot.png")

