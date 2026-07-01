#!/usr/bin/env python3
import argparse
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Regenerate MPC grid sweep heatmap plots from CSV data.")
    parser.add_argument("csv_file", type=str, help="Path to the sweep results CSV file")
    parser.add_argument("--no-annotate", action="store_true", help="Do not print numbers inside the heatmap cells")
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: File {args.csv_file} does not exist.")
        return
        
    h_set = set()
    l_set = set()
    mu_set = set()
    results = {}
    
    # We will read args from the CSV header or file details if possible, but we can also infer
    # the horizon and max force from the filename or the CSV content if present.
    # In our CSV structure, the horizon N is not explicitly written in every row,
    # so we will look up the rows to reconstruct the grid.
    with open(args.csv_file, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = float(row["h"])
            l = float(row["l"])
            mu = float(row["mu"])
            
            h_set.add(h)
            l_set.add(l)
            mu_set.add(mu)
            
            stab_time_str = row["stabilization_time"]
            if stab_time_str in ("nan", "", None):
                stab_time = np.nan
            else:
                stab_time = float(stab_time_str)
                
            results[(h, l, mu)] = stab_time

    h_vals = sorted(list(h_set))
    l_vals = sorted(list(l_set))
    mu_vals = sorted(list(mu_set))
    
    print(f"Inferred Grid Dimensions from CSV:")
    print(f"  h_mpc (step size):  {len(h_vals)} values in {h_vals[0]:.3f}s - {h_vals[-1]:.3f}s")
    print(f"  l (pendulum len):   {len(l_vals)} values in {l_vals[0]:.2f}m - {l_vals[-1]:.2f}m")
    print(f"  m/M (mass ratio):   {len(mu_vals)} values in {mu_vals[0]:.2f} - {mu_vals[-1]:.2f}")
    
    n_subplots = len(mu_vals)
    n_cols = min(3, n_subplots)
    n_rows = (n_subplots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5.0 * n_rows), squeeze=False)
    
    for idx, mu in enumerate(mu_vals):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        grid_data = np.zeros((len(h_vals), len(l_vals)))
        for i, h in enumerate(h_vals):
            for j, l in enumerate(l_vals):
                val = results.get((h, l, mu), np.nan)
                grid_data[i, j] = val
                
        cax = ax.imshow(grid_data, interpolation="nearest", cmap="viridis", origin="lower")
        ax.set_title(f"Mass ratio m/M = {mu:.2f}")
        
        ax.set_xticks(np.arange(len(l_vals)))
        ax.set_xticklabels([f"{l:.2f}" for l in l_vals], rotation=45)
        ax.set_yticks(np.arange(len(h_vals)))
        ax.set_yticklabels([f"{h:.3f}" for h in h_vals])
        
        ax.set_xlabel("Pendulum length l [m]")
        ax.set_ylabel("Step size h_mpc [s]")
        
        # Annotate cells if --no-annotate is not specified
        if not args.no_annotate:
            for i in range(len(h_vals)):
                for j in range(len(l_vals)):
                    val = grid_data[i, j]
                    text_val = "NaN" if np.isnan(val) else f"{val:.2f}"
                    ax.text(j, i, text_val, ha="center", va="center", 
                            color="w" if np.isnan(val) or val > (12.0/2) else "black")
                    
        fig.colorbar(cax, ax=ax, label="Stabilization Time [s]")
        
    for idx in range(n_subplots, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")
        
    # Infer filename for output
    base_name, _ = os.path.splitext(args.csv_file)
    suffix = "_no_annotate" if args.no_annotate else ""
    plot_filename = f"{base_name}{suffix}.png"
    
    plt.suptitle(f"MPC Sweep: Stabilization Time (Regenerated from CSV)", y=0.98, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    plt.savefig(plot_filename, dpi=150)
    print(f"Heatmap visualization saved to {plot_filename}.")

if __name__ == "__main__":
    main()
