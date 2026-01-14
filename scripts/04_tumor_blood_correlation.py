#!/usr/bin/env python3
"""
04_tumor_blood_correlation.py
=============================
Analyze correlations between tumor GDF15 expression and circulating GDF15 levels.

Analyses:
1. PRE tumor vs T1 blood (baseline correlation)
2. POST tumor vs T3 blood (on-treatment correlation)
3. Tumor change vs blood change correlation
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


def load_tumor_gdf15_by_timepoint():
    """Load tumor GDF15 expression from RNA-seq with PRE/POST timepoint separation."""
    xls = pd.ExcelFile(DATA_DIR / "43018_2022_467_MOESM2_ESM.xlsx")

    # Load sample metadata to get PRE/POST designation
    sample_meta = pd.read_excel(xls, sheet_name='Supplementary Table 2', header=1)
    sample_meta = sample_meta[['Study_ID', 'Sample_ID', 'Timepoint']].copy()

    # Load tumor RNA-seq data
    tumor_rna = pd.read_excel(xls, sheet_name='Supplementary Table 8', header=1)
    sample_cols = [c for c in tumor_rna.columns if c.startswith('SP_')]

    # Get GDF15 row
    gdf15_row = tumor_rna[tumor_rna['Gene ID'] == 'ENSG00000130513']

    if len(gdf15_row) == 0:
        print("Warning: GDF15 not found in tumor RNA-seq data")
        return pd.DataFrame()

    # Create long-format dataframe with sample IDs
    tumor_data = []
    for col in sample_cols:
        tumor_data.append({
            'Sample_ID': col,
            'tumor_GDF15': gdf15_row[col].values[0]
        })
    tumor_df = pd.DataFrame(tumor_data)

    # Merge with sample metadata to get Study_ID and Timepoint
    tumor_df = pd.merge(tumor_df, sample_meta, on='Sample_ID', how='left')

    # Pivot to get PRE and POST columns per patient
    tumor_wide = tumor_df.pivot(index='Study_ID', columns='Timepoint', values='tumor_GDF15').reset_index()
    tumor_wide.columns = ['id', 'POST_tumor_GDF15', 'PRE_tumor_GDF15']

    # Reorder columns
    tumor_wide = tumor_wide[['id', 'PRE_tumor_GDF15', 'POST_tumor_GDF15']]

    return tumor_wide


def analyze_correlations():
    """Analyze tumor-blood GDF15 correlations with proper PRE/POST matching."""
    print("=" * 70)
    print("TUMOR-BLOOD GDF15 CORRELATION ANALYSIS")
    print("=" * 70)

    # Load data
    cosinr = pd.read_csv(DATA_DIR / "regression_ml_inputs.csv")
    tumor_gdf15 = load_tumor_gdf15_by_timepoint()

    if tumor_gdf15.empty:
        print("No tumor data available")
        return pd.DataFrame(), pd.DataFrame()

    # Merge
    merged = pd.merge(cosinr, tumor_gdf15, on='id', how='inner')
    print(f"\nPatients with tumor RNA-seq data: {len(merged)}")
    print(f"Patients with PRE tumor samples: {merged['PRE_tumor_GDF15'].notna().sum()}")
    print(f"Patients with POST tumor samples: {merged['POST_tumor_GDF15'].notna().sum()}")

    results = []

    # 1. PRE tumor vs T1 blood (baseline correlation)
    print("\n" + "-" * 50)
    print("1. BASELINE CORRELATION (PRE Tumor vs T1 Blood)")
    print("-" * 50)

    mask_pre = merged['p.GDF15.T1'].notna() & merged['PRE_tumor_GDF15'].notna()
    n_pre = mask_pre.sum()

    if n_pre >= 5:
        rho_pre, p_pre = stats.spearmanr(
            merged.loc[mask_pre, 'PRE_tumor_GDF15'],
            merged.loc[mask_pre, 'p.GDF15.T1']
        )
        print(f"  n = {n_pre}")
        print(f"  Spearman rho = {rho_pre:.2f}")
        print(f"  p-value = {p_pre:.4f}")

        results.append({
            'comparison': 'PRE tumor vs T1 blood',
            'n': n_pre,
            'spearman_rho': rho_pre,
            'p_value': p_pre
        })
    else:
        print(f"  Insufficient samples (n={n_pre})")

    # 2. POST tumor vs T3 blood (on-treatment correlation)
    print("\n" + "-" * 50)
    print("2. ON-TREATMENT CORRELATION (POST Tumor vs T3 Blood)")
    print("-" * 50)

    mask_post = merged['p.GDF15.T3'].notna() & merged['POST_tumor_GDF15'].notna()
    n_post = mask_post.sum()

    if n_post >= 5:
        rho_post, p_post = stats.spearmanr(
            merged.loc[mask_post, 'POST_tumor_GDF15'],
            merged.loc[mask_post, 'p.GDF15.T3']
        )
        print(f"  n = {n_post}")
        print(f"  Spearman rho = {rho_post:.2f}")
        print(f"  p-value = {p_post:.4f}")

        results.append({
            'comparison': 'POST tumor vs T3 blood',
            'n': n_post,
            'spearman_rho': rho_post,
            'p_value': p_post
        })
    else:
        print(f"  Insufficient samples (n={n_post})")

    # 3. Change correlation (tumor log2FC vs blood NPX change)
    # Note: Blood NPX is already on log2 scale, so NPX change = log2 fold change
    # Tumor FPKM is on linear scale, so we use log2(POST/PRE) for comparability
    print("\n" + "-" * 50)
    print("3. CHANGE CORRELATION (Tumor log2FC vs Blood NPX Change)")
    print("-" * 50)

    mask_change = (merged['p.GDF15.T1'].notna() &
                   merged['p.GDF15.T3'].notna() &
                   merged['PRE_tumor_GDF15'].notna() &
                   merged['POST_tumor_GDF15'].notna())
    n_change = mask_change.sum()

    if n_change >= 5:
        # Blood change in NPX (already log2 scale)
        blood_change = merged.loc[mask_change, 'p.GDF15.T3'] - merged.loc[mask_change, 'p.GDF15.T1']
        # Tumor log2 fold change (convert linear FPKM to log2 scale)
        tumor_log2fc = np.log2(merged.loc[mask_change, 'POST_tumor_GDF15'] /
                               merged.loc[mask_change, 'PRE_tumor_GDF15'])

        rho_change, p_change = stats.spearmanr(tumor_log2fc, blood_change)
        print(f"  n = {n_change}")
        print(f"  Spearman rho = {rho_change:.2f}")
        print(f"  p-value = {p_change:.4f}")

        results.append({
            'comparison': 'Tumor log2FC vs blood NPX change',
            'n': n_change,
            'spearman_rho': rho_change,
            'p_value': p_change
        })
    else:
        print(f"  Insufficient samples (n={n_change})")

    return pd.DataFrame(results), merged


def create_correlation_figure(merged):
    """Create correlation scatter plots with proper PRE/POST matching."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. PRE tumor vs T1 blood (baseline)
    ax1 = axes[0]
    mask_pre = merged['p.GDF15.T1'].notna() & merged['PRE_tumor_GDF15'].notna()
    if mask_pre.sum() > 0:
        x = merged.loc[mask_pre, 'PRE_tumor_GDF15']
        y = merged.loc[mask_pre, 'p.GDF15.T1']
        ax1.scatter(x, y, alpha=0.7, s=50)

        # Add regression line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax1.plot(x.sort_values(), p(x.sort_values()), "r--", alpha=0.8)

        rho, pval = stats.spearmanr(x, y)
        ax1.set_xlabel('PRE Tumor GDF15 (FPKM)')
        ax1.set_ylabel('T1 Blood GDF15 (NPX)')
        ax1.set_title(f'Baseline (PRE tumor vs T1 blood)\nn={mask_pre.sum()}, rho={rho:.2f}, p={pval:.3f}')

    # 2. POST tumor vs T3 blood (on-treatment)
    ax2 = axes[1]
    mask_post = merged['p.GDF15.T3'].notna() & merged['POST_tumor_GDF15'].notna()
    if mask_post.sum() > 0:
        x = merged.loc[mask_post, 'POST_tumor_GDF15']
        y = merged.loc[mask_post, 'p.GDF15.T3']
        ax2.scatter(x, y, alpha=0.7, s=50, color='green')

        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax2.plot(x.sort_values(), p(x.sort_values()), "r--", alpha=0.8)

        rho, pval = stats.spearmanr(x, y)
        ax2.set_xlabel('POST Tumor GDF15 (FPKM)')
        ax2.set_ylabel('T3 Blood GDF15 (NPX)')
        ax2.set_title(f'On-treatment (POST tumor vs T3 blood)\nn={mask_post.sum()}, rho={rho:.2f}, p={pval:.3f}')

    # 3. Change correlation (tumor log2FC vs blood NPX change)
    ax3 = axes[2]
    mask_change = (merged['p.GDF15.T1'].notna() &
                   merged['p.GDF15.T3'].notna() &
                   merged['PRE_tumor_GDF15'].notna() &
                   merged['POST_tumor_GDF15'].notna())
    if mask_change.sum() > 0:
        tumor_log2fc = np.log2(merged.loc[mask_change, 'POST_tumor_GDF15'] /
                               merged.loc[mask_change, 'PRE_tumor_GDF15'])
        blood_change = merged.loc[mask_change, 'p.GDF15.T3'] - merged.loc[mask_change, 'p.GDF15.T1']
        ax3.scatter(tumor_log2fc, blood_change, alpha=0.7, s=50, color='purple')

        z = np.polyfit(tumor_log2fc, blood_change, 1)
        p = np.poly1d(z)
        ax3.plot(tumor_log2fc.sort_values(), p(tumor_log2fc.sort_values()), "r--", alpha=0.8)

        rho, pval = stats.spearmanr(tumor_log2fc, blood_change)
        ax3.set_xlabel('Tumor GDF15 log2FC')
        ax3.set_ylabel('Blood GDF15 Change (NPX)')
        ax3.set_title(f'Change Correlation\nn={mask_change.sum()}, rho={rho:.2f}, p={pval:.3f}')

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
