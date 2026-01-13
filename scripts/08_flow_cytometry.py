#!/usr/bin/env python3
"""
08_flow_cytometry.py
====================
Analyze correlations between GDF15 and immune cell populations from flow cytometry.

Analyses:
1. Baseline GDF15 vs baseline immune cell frequencies
2. GDF15 change vs immune cell frequency changes
3. GDF15 vs TCR sequencing metrics

All correlations use Spearman rank correlation with FDR correction.
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


def fdr_correction(p_values):
    """Apply Benjamini-Hochberg FDR correction."""
    p_arr = np.array(p_values)
    n = len(p_arr)
    sorted_idx = np.argsort(p_arr)
    fdr = np.zeros(n)

    for i, idx in enumerate(sorted_idx):
        rank = i + 1
        fdr[idx] = p_arr[idx] * n / rank

    fdr_sorted = fdr[sorted_idx]
    for i in range(n - 2, -1, -1):
        fdr_sorted[i] = min(fdr_sorted[i], fdr_sorted[i + 1])
    fdr[sorted_idx] = fdr_sorted

    return np.minimum(fdr, 1.0)


def analyze_baseline_flow(cosinr):
    """Analyze baseline GDF15 vs flow cytometry immune populations."""
    print("=" * 70)
    print("BASELINE GDF15 vs FLOW CYTOMETRY IMMUNE POPULATIONS")
    print("=" * 70)

    gdf15_col = 'p.GDF15.T1'

    # Main immune populations
    flow_markers = {
        'CD8+ T cells': 'fc.CD3p_CD56n_CD8p.T1',
        'CD4+ T cells': 'fc.CD3p_CD56n_CD4p.T1',
        'Tregs': 'fc.CD3p_CD56n_CD4p_Treg.T1',
        'NK cells': 'fc.NK.T1',
        'B cells (CD20+)': 'fc.CD20.T1',
        'Classical monocytes': 'fc.classical_mono.T1',
        'Intermediate monocytes': 'fc.intermediate_mono.T1',
        'Non-classical monocytes': 'fc.non_classical_mono.T1',
        'mDC': 'fc.mDC.T1',
        'pDC': 'fc.pDC.T1',
        'Total DC': 'fc.DC.T1'
    }

    results = []
    for name, col in flow_markers.items():
        if col in cosinr.columns:
            mask = cosinr[gdf15_col].notna() & cosinr[col].notna()
            n = mask.sum()
            if n > 5:
                rho, p = stats.spearmanr(cosinr.loc[mask, gdf15_col], cosinr.loc[mask, col])
                results.append({
                    'cell_type': name,
                    'column': col,
                    'n': n,
                    'spearman_rho': rho,
                    'p_value': p
                })

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        results_df['fdr_q'] = fdr_correction(results_df['p_value'].values)
        results_df = results_df.sort_values('p_value')

        print(f"\nPatients with matched GDF15 and flow data: n = {results_df['n'].min()}-{results_df['n'].max()}")
        print("\nBaseline GDF15 vs immune cell frequencies:")
        print("-" * 70)

        for _, row in results_df.iterrows():
            sig = "*" if row['fdr_q'] < 0.05 else ""
            print(f"  {row['cell_type']:25s}: ρ = {row['spearman_rho']:+.2f}, "
                  f"raw p = {row['p_value']:.4f}, FDR p = {row['fdr_q']:.4f} {sig}")

        # Summary
        n_sig = (results_df['fdr_q'] < 0.05).sum()
        print(f"\nSignificant after FDR correction: {n_sig}/{len(results_df)}")

    return results_df


def analyze_gdf15_change_flow(cosinr):
    """Analyze GDF15 change vs flow cytometry changes."""
    print("\n" + "=" * 70)
    print("GDF15 CHANGE vs FLOW CYTOMETRY CHANGES")
    print("=" * 70)

    gdf15_change = 'p.GDF15.dif1v3'

    flow_markers = {
        'CD8+ T cells': 'fc.CD3p_CD56n_CD8p.dif1v3',
        'CD4+ T cells': 'fc.CD3p_CD56n_CD4p.dif1v3',
        'Tregs': 'fc.CD3p_CD56n_CD4p_Treg.dif1v3',
        'NK cells': 'fc.NK.dif1v3',
        'Total DC': 'fc.DC.dif1v3',
        'Classical monocytes': 'fc.classical_mono.dif1v3'
    }

    results = []
    for name, col in flow_markers.items():
        if col in cosinr.columns and gdf15_change in cosinr.columns:
            mask = cosinr[gdf15_change].notna() & cosinr[col].notna()
            n = mask.sum()
            if n > 5:
                rho, p = stats.spearmanr(cosinr.loc[mask, gdf15_change], cosinr.loc[mask, col])
                results.append({
                    'cell_type': name,
                    'n': n,
                    'spearman_rho': rho,
                    'p_value': p
                })

    if results:
        results_df = pd.DataFrame(results)
        results_df['fdr_q'] = fdr_correction(results_df['p_value'].values)
        results_df = results_df.sort_values('p_value')

        print("\nGDF15 change vs immune cell changes:")
        print("-" * 70)

        for _, row in results_df.iterrows():
            sig = "*" if row['fdr_q'] < 0.05 else ""
            print(f"  {row['cell_type']:25s}: ρ = {row['spearman_rho']:+.2f}, "
                  f"raw p = {row['p_value']:.4f}, FDR p = {row['fdr_q']:.4f} {sig}")

        return results_df

    return pd.DataFrame()


def analyze_tcr_metrics(cosinr):
    """Analyze GDF15 vs TCR sequencing metrics."""
    print("\n" + "=" * 70)
    print("BASELINE GDF15 vs TCR METRICS")
    print("=" * 70)

    gdf15_col = 'p.GDF15.T1'

    tcr_metrics = {
        'TCR richness': 'tcr.obs_richness.T1',
        'TCR clonality': 'tcr.all_simpson_clonality.T1',
        'TCR evenness': 'tcr.all_simpson_evenness.T1',
        'Total T cells': 'tcr.total_t_cells.T1',
        'Productive clonality': 'tcr.productive_simpson_clonality.T1'
    }

    results = []
    for name, col in tcr_metrics.items():
        if col in cosinr.columns:
            mask = cosinr[gdf15_col].notna() & cosinr[col].notna()
            n = mask.sum()
            if n > 5:
                rho, p = stats.spearmanr(cosinr.loc[mask, gdf15_col], cosinr.loc[mask, col])
                results.append({
                    'metric': name,
                    'n': n,
                    'spearman_rho': rho,
                    'p_value': p
                })

    if results:
        results_df = pd.DataFrame(results)
        results_df['fdr_q'] = fdr_correction(results_df['p_value'].values)
        results_df = results_df.sort_values('p_value')

        print(f"\nPatients with TCR data: n = {results_df['n'].min()}-{results_df['n'].max()}")
        print("\nBaseline GDF15 vs TCR metrics:")
        print("-" * 70)

        for _, row in results_df.iterrows():
            sig = "*" if row['fdr_q'] < 0.05 else ""
            print(f"  {row['metric']:25s}: ρ = {row['spearman_rho']:+.2f}, "
                  f"raw p = {row['p_value']:.4f}, FDR p = {row['fdr_q']:.4f} {sig}")

        return results_df

    return pd.DataFrame()


def create_flow_correlation_figure(cosinr):
    """Create figure showing flow cytometry correlations."""
    gdf15_col = 'p.GDF15.T1'

    flow_markers = {
        'CD8+ T cells': 'fc.CD3p_CD56n_CD8p.T1',
        'CD4+ T cells': 'fc.CD3p_CD56n_CD4p.T1',
        'NK cells': 'fc.NK.T1',
        'Tregs': 'fc.CD3p_CD56n_CD4p_Treg.T1',
        'DC': 'fc.DC.T1',
        'Monocytes': 'fc.classical_mono.T1'
    }

    # Filter to available markers
    available = {k: v for k, v in flow_markers.items() if v in cosinr.columns}

    if len(available) < 2:
        print("Insufficient flow cytometry data for figure")
        return

    n_plots = len(available)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    for i, (name, col) in enumerate(available.items()):
        ax = axes[i]
        mask = cosinr[gdf15_col].notna() & cosinr[col].notna()

        if mask.sum() > 5:
            x = cosinr.loc[mask, gdf15_col]
            y = cosinr.loc[mask, col]

            ax.scatter(x, y, alpha=0.6, s=40)

            # Add regression line
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x.sort_values(), p(x.sort_values()), "r--", alpha=0.8)

            rho, pval = stats.spearmanr(x, y)
            ax.set_xlabel('Baseline GDF15 (NPX)')
            ax.set_ylabel(f'{name} (%)')
            ax.set_title(f'{name}\nρ={rho:.2f}, p={pval:.3f}')

    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'flow_cytometry_correlations.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'flow_cytometry_correlations.pdf', bbox_inches='tight')
    plt.close()

    print(f"\nFigure saved to: {FIGURES_DIR / 'flow_cytometry_correlations.png'}")


def main():
    """Main function."""
    print("=" * 70)
    print("FLOW CYTOMETRY AND TCR CORRELATION ANALYSIS")
    print("=" * 70)

    # Load data
    cosinr = pd.read_csv(DATA_DIR / "regression_ml_inputs.csv")

    # Run analyses
    baseline_flow = analyze_baseline_flow(cosinr)
    change_flow = analyze_gdf15_change_flow(cosinr)
    tcr_results = analyze_tcr_metrics(cosinr)

    # Create figure
    create_flow_correlation_figure(cosinr)

    # Save results
    if len(baseline_flow) > 0:
        baseline_flow.to_csv(RESULTS_DIR / 'flow_correlations_baseline.csv', index=False)
    if len(change_flow) > 0:
        change_flow.to_csv(RESULTS_DIR / 'flow_correlations_change.csv', index=False)
    if len(tcr_results) > 0:
        tcr_results.to_csv(RESULTS_DIR / 'tcr_correlations.csv', index=False)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Analysis complete. See output above for:
- GDF15 correlations with immune cell frequencies
- Statistical significance after FDR correction
""")

    return {
        'baseline_flow': baseline_flow,
        'change_flow': change_flow,
        'tcr': tcr_results
    }


if __name__ == "__main__":
    results = main()
