#!/usr/bin/env python3
"""
04_tumor_blood_correlation.py
=============================
Analyze correlations between tumor GDF15 expression and circulating GDF15 levels.

Analyses:
1. Baseline correlation (tumor RNA vs T1 blood)
2. On-treatment correlation (tumor RNA vs T3 blood)
3. Change correlation (delta tumor vs delta blood)
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR
RESULTS_DIR = BASE_DIR / "GDF15_Analysis" / "results"
FIGURES_DIR = BASE_DIR / "GDF15_Analysis" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_tumor_gdf15():
    """Load tumor GDF15 expression from RNA-seq."""
    xls = pd.ExcelFile(DATA_DIR / "43018_2022_467_MOESM2_ESM.xlsx")
    tumor_rna = pd.read_excel(xls, sheet_name='Supplementary Table 8', header=1)

    sample_cols = [c for c in tumor_rna.columns if c.startswith('SP_')]

    # Get GDF15 row
    gdf15_row = tumor_rna[tumor_rna['Gene ID'] == 'ENSG00000130513']
    gdf15_expr = gdf15_row[sample_cols].iloc[0]

    # Create dataframe
    patient_ids = [int(c.replace('SP_', '')) for c in sample_cols]
    tumor_gdf15 = pd.DataFrame({
        'id': patient_ids,
        'tumor_GDF15': gdf15_expr.values
    })

    return tumor_gdf15


def analyze_correlations():
    """Analyze tumor-blood GDF15 correlations at different timepoints."""
    print("=" * 70)
    print("TUMOR-BLOOD GDF15 CORRELATION ANALYSIS")
    print("=" * 70)

    # Load data
    cosinr = pd.read_csv(DATA_DIR / "regression_ml_inputs.csv")
    tumor_gdf15 = load_tumor_gdf15()

    # Merge
    merged = pd.merge(cosinr, tumor_gdf15, on='id', how='inner')
    print(f"\nPatients with tumor RNA-seq data: {len(merged)}")

    results = []

    # 1. Baseline correlation (tumor vs T1 blood)
    print("\n" + "-" * 50)
    print("1. BASELINE CORRELATION (Tumor RNA vs T1 Blood)")
    print("-" * 50)

    mask_t1 = merged['p.GDF15.T1'].notna() & merged['tumor_GDF15'].notna()
    n_t1 = mask_t1.sum()

    if n_t1 > 5:
        rho_t1, p_t1 = stats.spearmanr(
            merged.loc[mask_t1, 'tumor_GDF15'],
            merged.loc[mask_t1, 'p.GDF15.T1']
        )
        print(f"  n = {n_t1}")
        print(f"  Spearman ρ = {rho_t1:.2f}")
        print(f"  p-value = {p_t1:.4f}")

        results.append({
            'timepoint': 'Baseline',
            'n': n_t1,
            'spearman_rho': rho_t1,
            'p_value': p_t1
        })

    # 2. On-treatment correlation (tumor vs T3 blood)
    print("\n" + "-" * 50)
    print("2. ON-TREATMENT CORRELATION (Tumor RNA vs T3 Blood)")
    print("-" * 50)

    mask_t3 = merged['p.GDF15.T3'].notna() & merged['tumor_GDF15'].notna()
    n_t3 = mask_t3.sum()

    if n_t3 > 5:
        rho_t3, p_t3 = stats.spearmanr(
            merged.loc[mask_t3, 'tumor_GDF15'],
            merged.loc[mask_t3, 'p.GDF15.T3']
        )
        print(f"  n = {n_t3}")
        print(f"  Spearman ρ = {rho_t3:.2f}")
        print(f"  p-value = {p_t3:.4f}")

        results.append({
            'timepoint': 'On-treatment',
            'n': n_t3,
            'spearman_rho': rho_t3,
            'p_value': p_t3
        })

    # 3. Change correlation
    print("\n" + "-" * 50)
    print("3. CHANGE CORRELATION (Delta Tumor vs Delta Blood)")
    print("-" * 50)

    # For change, we need patients with both T1 and T3 blood AND tumor data
    # Note: The tumor data is from a single timepoint, so "change" analysis
    # typically uses blood change vs tumor expression level

    mask_change = (merged['p.GDF15.T1'].notna() &
                   merged['p.GDF15.T3'].notna() &
                   merged['tumor_GDF15'].notna())
    n_change = mask_change.sum()

    if n_change > 5:
        blood_change = merged.loc[mask_change, 'p.GDF15.T3'] - merged.loc[mask_change, 'p.GDF15.T1']

        rho_change, p_change = stats.spearmanr(
            merged.loc[mask_change, 'tumor_GDF15'],
            blood_change
        )
        print(f"  n = {n_change}")
        print(f"  Spearman ρ = {rho_change:.2f}")
        print(f"  p-value = {p_change:.4f}")

        results.append({
            'timepoint': 'Change',
            'n': n_change,
            'spearman_rho': rho_change,
            'p_value': p_change
        })

    return pd.DataFrame(results), merged


def create_correlation_figure(merged):
    """Create correlation scatter plots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Baseline
    ax1 = axes[0]
    mask_t1 = merged['p.GDF15.T1'].notna() & merged['tumor_GDF15'].notna()
    if mask_t1.sum() > 0:
        x = merged.loc[mask_t1, 'tumor_GDF15']
        y = merged.loc[mask_t1, 'p.GDF15.T1']
        ax1.scatter(x, y, alpha=0.7, s=50)

        # Add regression line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax1.plot(x.sort_values(), p(x.sort_values()), "r--", alpha=0.8)

        rho, pval = stats.spearmanr(x, y)
        ax1.set_xlabel('Tumor GDF15 (FPKM)')
        ax1.set_ylabel('Blood GDF15 (NPX)')
        ax1.set_title(f'Baseline\nn={mask_t1.sum()}, ρ={rho:.2f}, p={pval:.3f}')

    # 2. On-treatment
    ax2 = axes[1]
    mask_t3 = merged['p.GDF15.T3'].notna() & merged['tumor_GDF15'].notna()
    if mask_t3.sum() > 0:
        x = merged.loc[mask_t3, 'tumor_GDF15']
        y = merged.loc[mask_t3, 'p.GDF15.T3']
        ax2.scatter(x, y, alpha=0.7, s=50, color='green')

        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax2.plot(x.sort_values(), p(x.sort_values()), "r--", alpha=0.8)

        rho, pval = stats.spearmanr(x, y)
        ax2.set_xlabel('Tumor GDF15 (FPKM)')
        ax2.set_ylabel('Blood GDF15 (NPX)')
        ax2.set_title(f'On-treatment\nn={mask_t3.sum()}, ρ={rho:.2f}, p={pval:.3f}')

    # 3. Change
    ax3 = axes[2]
    mask_change = (merged['p.GDF15.T1'].notna() &
                   merged['p.GDF15.T3'].notna() &
                   merged['tumor_GDF15'].notna())
    if mask_change.sum() > 0:
        x = merged.loc[mask_change, 'tumor_GDF15']
        y = merged.loc[mask_change, 'p.GDF15.T3'] - merged.loc[mask_change, 'p.GDF15.T1']
        ax3.scatter(x, y, alpha=0.7, s=50, color='purple')

        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax3.plot(x.sort_values(), p(x.sort_values()), "r--", alpha=0.8)

        rho, pval = stats.spearmanr(x, y)
        ax3.set_xlabel('Tumor GDF15 (FPKM)')
        ax3.set_ylabel('Blood GDF15 Change (NPX)')
        ax3.set_title(f'Blood Change\nn={mask_change.sum()}, ρ={rho:.2f}, p={pval:.3f}')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'tumor_blood_correlation.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'tumor_blood_correlation.pdf', bbox_inches='tight')
    plt.close()

    print(f"\nFigure saved to: {FIGURES_DIR / 'tumor_blood_correlation.png'}")


def main():
    """Main function."""
    results_df, merged = analyze_correlations()

    # Create figure
    create_correlation_figure(merged)

    # Save results
    results_df.to_csv(RESULTS_DIR / 'tumor_blood_correlations.csv', index=False)
    print(f"\nResults saved to: {RESULTS_DIR / 'tumor_blood_correlations.csv'}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))

    return results_df


if __name__ == "__main__":
    results = main()
