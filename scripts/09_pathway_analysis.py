#!/usr/bin/env python3
"""
09_pathway_analysis.py
======================
Analyze correlations between GDF15 and ssGSEA pathway enrichment scores.

Analyses:
1. GDF15 change vs pathway score changes
2. Baseline GDF15 vs baseline pathway scores
3. Comprehensive analysis of all Hallmark pathways
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


def analyze_gdf15_change_vs_pathway_change(merged):
    """Analyze GDF15 change vs pathway changes (the key analysis)."""
    print("=" * 70)
    print("GDF15 CHANGE vs PATHWAY CHANGES")
    print("(This is the analysis used in the manuscript)")
    print("=" * 70)

    gdf15_change = 'p.GDF15.dif1v3'

    # Key pathways of interest
    key_pathways = {
        'IFN-stimulated genes': 'gs.Antiviral_mechanism_by_IFN_stimulated_genes.dif1v3',
        'TCR signaling': 'gs.TCR_signaling.dif1v3',
        'MHC Class I processing': 'gs.Class_I_MHC_mediated_antigen_processing_n_presentation.dif1v3',
        'IL-4/IL-13 signaling': 'gs.Interleukin_4_and_Interleukin_13_signaling.dif1v3',
        'IL-10 signaling': 'gs.Interleukin_10_signaling.dif1v3',
        'IFN alpha response': 'gs.HALLMARK_INTERFERON_ALPHA_RESPONSE.dif1v3',
        'IFN gamma response': 'gs.HALLMARK_INTERFERON_GAMMA_RESPONSE.dif1v3',
        'EMT': 'gs.HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION.dif1v3',
        'Angiogenesis': 'gs.HALLMARK_ANGIOGENESIS.dif1v3',
        'Inflammatory response': 'gs.HALLMARK_INFLAMMATORY_RESPONSE.dif1v3'
    }

    results = []
    for name, col in key_pathways.items():
        if col in merged.columns and gdf15_change in merged.columns:
            mask = merged[gdf15_change].notna() & merged[col].notna()
            n = mask.sum()
            if n > 5:
                rho, p = stats.spearmanr(merged.loc[mask, gdf15_change], merged.loc[mask, col])
                results.append({
                    'pathway': name,
                    'column': col,
                    'n': n,
                    'spearman_rho': rho,
                    'p_value': p
                })

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        results_df['fdr_q'] = fdr_correction(results_df['p_value'].values)
        results_df = results_df.sort_values('p_value')

        print("\nGDF15 CHANGE vs PATHWAY CHANGE correlations:")
        print("-" * 70)

        for _, row in results_df.iterrows():
            direction = "↓" if row['spearman_rho'] < 0 else "↑"
            sig = "**" if row['fdr_q'] < 0.01 else ("*" if row['fdr_q'] < 0.05 else "")
            print(f"  {row['pathway']:30s}: ρ = {row['spearman_rho']:+.2f}, "
                  f"p = {row['p_value']:.4f} {direction} {sig}")

    return results_df


def analyze_baseline_gdf15_vs_pathways(merged):
    """Analyze baseline GDF15 vs baseline pathway scores."""
    print("\n" + "=" * 70)
    print("BASELINE GDF15 vs BASELINE PATHWAY SCORES")
    print("(For comparison - not the main analysis)")
    print("=" * 70)

    gdf15_t1 = 'p.GDF15.T1'

    key_pathways = {
        'IFN-stimulated genes': 'gs.Antiviral_mechanism_by_IFN_stimulated_genes.T1',
        'TCR signaling': 'gs.TCR_signaling.T1',
        'MHC Class I processing': 'gs.Class_I_MHC_mediated_antigen_processing_n_presentation.T1',
        'IL-4/IL-13 signaling': 'gs.Interleukin_4_and_Interleukin_13_signaling.T1',
        'IL-10 signaling': 'gs.Interleukin_10_signaling.T1',
        'EMT': 'gs.HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION.T1',
        'Angiogenesis': 'gs.HALLMARK_ANGIOGENESIS.T1'
    }

    results = []
    for name, col in key_pathways.items():
        if col in merged.columns and gdf15_t1 in merged.columns:
            mask = merged[gdf15_t1].notna() & merged[col].notna()
            n = mask.sum()
            if n > 5:
                rho, p = stats.spearmanr(merged.loc[mask, gdf15_t1], merged.loc[mask, col])
                results.append({
                    'pathway': name,
                    'n': n,
                    'spearman_rho': rho,
                    'p_value': p
                })

    if results:
        results_df = pd.DataFrame(results)
        results_df['fdr_q'] = fdr_correction(results_df['p_value'].values)
        results_df = results_df.sort_values('p_value')

        print("\nBASELINE GDF15 vs BASELINE PATHWAY correlations:")
        print("-" * 70)

        for _, row in results_df.iterrows():
            sig = "*" if row['fdr_q'] < 0.05 else ""
            print(f"  {row['pathway']:30s}: ρ = {row['spearman_rho']:+.2f}, "
                  f"p = {row['p_value']:.4f} {sig}")

        return results_df

    return pd.DataFrame()


def analyze_all_hallmark_pathways(merged):
    """Comprehensive analysis of all Hallmark pathways."""
    print("\n" + "=" * 70)
    print("ALL HALLMARK PATHWAYS vs GDF15 CHANGE")
    print("=" * 70)

    gdf15_change = 'p.GDF15.dif1v3'

    # Find all Hallmark pathway change columns
    hallmark_cols = [c for c in merged.columns if 'HALLMARK' in c and '.dif1v3' in c]

    results = []
    for col in hallmark_cols:
        if gdf15_change in merged.columns:
            mask = merged[gdf15_change].notna() & merged[col].notna()
            n = mask.sum()
            if n > 5:
                rho, p = stats.spearmanr(merged.loc[mask, gdf15_change], merged.loc[mask, col])
                pathway_name = col.replace('gs.HALLMARK_', '').replace('.dif1v3', '')
                results.append({
                    'pathway': pathway_name,
                    'column': col,
                    'n': n,
                    'spearman_rho': rho,
                    'p_value': p
                })

    if results:
        results_df = pd.DataFrame(results)
        results_df['fdr_q'] = fdr_correction(results_df['p_value'].values)
        results_df = results_df.sort_values('p_value')

        print(f"\nTotal Hallmark pathways analyzed: {len(results_df)}")
        print(f"Significant after FDR (q < 0.05): {(results_df['fdr_q'] < 0.05).sum()}")

        print("\nTop 15 significant pathways:")
        print("-" * 70)

        for _, row in results_df.head(15).iterrows():
            direction = "↓" if row['spearman_rho'] < 0 else "↑"
            sig = "**" if row['fdr_q'] < 0.01 else ("*" if row['fdr_q'] < 0.05 else "")
            print(f"  {row['pathway'][:35]:35s}: ρ = {row['spearman_rho']:+.2f}, "
                  f"FDR = {row['fdr_q']:.4f} {direction} {sig}")

        return results_df

    return pd.DataFrame()


def create_pathway_figure(merged):
    """Create figure showing key pathway correlations."""
    gdf15_change = 'p.GDF15.dif1v3'

    key_pathways = {
        'MHC Class I': 'gs.Class_I_MHC_mediated_antigen_processing_n_presentation.dif1v3',
        'TCR signaling': 'gs.TCR_signaling.dif1v3',
        'IL-10 signaling': 'gs.Interleukin_10_signaling.dif1v3',
        'IL-4/13 signaling': 'gs.Interleukin_4_and_Interleukin_13_signaling.dif1v3',
        'EMT': 'gs.HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION.dif1v3',
        'Angiogenesis': 'gs.HALLMARK_ANGIOGENESIS.dif1v3'
    }

    available = {k: v for k, v in key_pathways.items() if v in merged.columns}

    if len(available) < 2:
        print("Insufficient pathway data for figure")
        return

    n_plots = len(available)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    axes = axes.flatten()

    colors = {'positive': 'red', 'negative': 'blue'}

    for i, (name, col) in enumerate(available.items()):
        ax = axes[i]
        mask = merged[gdf15_change].notna() & merged[col].notna()

        if mask.sum() > 5:
            x = merged.loc[mask, gdf15_change]
            y = merged.loc[mask, col]

            rho, pval = stats.spearmanr(x, y)
            color = colors['positive'] if rho > 0 else colors['negative']

            ax.scatter(x, y, alpha=0.6, s=40, c=color)

            # Add regression line
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x.sort_values(), p(x.sort_values()), "--", color=color, alpha=0.8)

            ax.set_xlabel('GDF15 Change (NPX)')
            ax.set_ylabel('Pathway Change')
            ax.set_title(f'{name}\nρ={rho:.2f}, p={pval:.4f}')

    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'pathway_correlations.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'pathway_correlations.pdf', bbox_inches='tight')
    plt.close()

    print(f"\nFigure saved to: {FIGURES_DIR / 'pathway_correlations.png'}")


def main():
    """Main function."""
    print("=" * 70)
    print("ssGSEA PATHWAY CORRELATION ANALYSIS")
    print("=" * 70)

    # Load and merge data
    cosinr = pd.read_csv(DATA_DIR / "regression_ml_inputs.csv")
    hallmark = pd.read_csv(DATA_DIR / "hallmark_ssGSEA.csv")
    reactome = pd.read_csv(DATA_DIR / "reactome_immune_only_ssGSEA.csv")

    merged = pd.merge(cosinr, hallmark, on='id', how='inner')
    merged = pd.merge(merged, reactome, on='id', how='inner')

    print(f"Merged dataset: {len(merged)} patients")

    # Run analyses
    change_results = analyze_gdf15_change_vs_pathway_change(merged)
    baseline_results = analyze_baseline_gdf15_vs_pathways(merged)
    hallmark_results = analyze_all_hallmark_pathways(merged)

    # Create figure
    create_pathway_figure(merged)

    # Save results
    if len(change_results) > 0:
        change_results.to_csv(RESULTS_DIR / 'pathway_correlations_change.csv', index=False)
    if len(baseline_results) > 0:
        baseline_results.to_csv(RESULTS_DIR / 'pathway_correlations_baseline.csv', index=False)
    if len(hallmark_results) > 0:
        hallmark_results.to_csv(RESULTS_DIR / 'hallmark_pathway_correlations.csv', index=False)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Analysis complete. See output above for:
- GDF15 change vs pathway change correlations
- Pathways with significant positive and negative correlations
- Statistical significance after FDR correction
""")

    return {
        'change': change_results,
        'baseline': baseline_results,
        'hallmark': hallmark_results
    }


if __name__ == "__main__":
    results = main()
