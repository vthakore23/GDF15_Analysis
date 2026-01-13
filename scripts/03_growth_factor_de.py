#!/usr/bin/env python3
"""
03_growth_factor_de.py
======================
Analyze growth factor differential expression in both cohorts.

Key question: Which growth factors are significantly upregulated after SBRT?

Analyses:
1. COSINR cohort (tumor RNA-seq): DESeq2 results for growth factors
2. Early-stage cohort (blood proteomics): Paired analysis for growth factors
3. Comparison between cohorts to identify SBRT-specific vs ICB-related effects
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


# Comprehensive list of growth factors
GROWTH_FACTORS = [
    'AREG', 'BDNF', 'BMP2', 'BMP4', 'BMP6', 'BMP7', 'BTC', 'CCL2', 'CCL5',
    'CSF1', 'CSF2', 'CSF3', 'CTGF', 'CXCL12', 'EGF', 'EREG', 'FGF1', 'FGF2',
    'FGF7', 'FGF9', 'FGF10', 'FGF19', 'FGF21', 'FGF23', 'FLT3LG', 'GDF15',
    'GDF11', 'GH1', 'HGF', 'IGF1', 'IGF2', 'IL1A', 'IL1B', 'IL2', 'IL3',
    'IL4', 'IL5', 'IL6', 'IL7', 'IL10', 'IL11', 'IL12A', 'IL12B', 'IL13',
    'IL15', 'IL17A', 'IL18', 'IL21', 'IL22', 'IL23A', 'IL27', 'IL33', 'IL34',
    'INHA', 'INHBA', 'INHBB', 'KIT', 'KITLG', 'LIF', 'MDK', 'MSTN', 'NGF',
    'NRG1', 'NRG2', 'NRG3', 'NRG4', 'NODAL', 'NTF3', 'NTF4', 'OSM', 'PDGFA',
    'PDGFB', 'PDGFC', 'PDGFD', 'PGF', 'PTN', 'SPP1', 'TGFA', 'TGFB1',
    'TGFB2', 'TGFB3', 'THPO', 'TNF', 'TNFSF10', 'TNFSF11', 'VEGFA', 'VEGFB',
    'VEGFC', 'VEGFD', 'WNT1', 'WNT3A', 'WNT5A', 'WNT7A'
]


def fdr_correction(p_values):
    """Apply Benjamini-Hochberg FDR correction."""
    p_arr = np.array(p_values)
    n = len(p_arr)
    sorted_idx = np.argsort(p_arr)
    fdr = np.zeros(n)

    for i, idx in enumerate(sorted_idx):
        rank = i + 1
        fdr[idx] = p_arr[idx] * n / rank

    # Make monotonic
    fdr_sorted = fdr[sorted_idx]
    for i in range(n - 2, -1, -1):
        fdr_sorted[i] = min(fdr_sorted[i], fdr_sorted[i + 1])
    fdr[sorted_idx] = fdr_sorted

    return np.minimum(fdr, 1.0)


def analyze_cosinr_tumor_rnaseq():
    """Analyze growth factor DE in COSINR tumor RNA-seq (DESeq2)."""
    print("=" * 70)
    print("COSINR TUMOR RNA-SEQ: GROWTH FACTOR DIFFERENTIAL EXPRESSION")
    print("=" * 70)

    # Load DESeq2 results
    deseq = pd.read_csv(DATA_DIR / "01_DESeq2_Combined_AllGenes.csv")

    # Filter for growth factors
    gf_results = deseq[deseq['symbol'].isin(GROWTH_FACTORS)].copy()
    gf_results = gf_results.sort_values('padj')

    print(f"\nGrowth factors found in DESeq2: {len(gf_results)}")

    # Significant upregulated
    sig_up = gf_results[(gf_results['padj'] < 0.05) & (gf_results['log2FoldChange'] > 0)]
    sig_down = gf_results[(gf_results['padj'] < 0.05) & (gf_results['log2FoldChange'] < 0)]

    print(f"\nSignificantly upregulated (padj < 0.05, log2FC > 0): {len(sig_up)}")
    if len(sig_up) > 0:
        print(sig_up[['symbol', 'log2FoldChange', 'pvalue', 'padj']].to_string(index=False))

    print(f"\nSignificantly downregulated (padj < 0.05, log2FC < 0): {len(sig_down)}")
    if len(sig_down) > 0:
        print(sig_down[['symbol', 'log2FoldChange', 'pvalue', 'padj']].to_string(index=False))

    # Check GDF15 specifically
    gdf15 = gf_results[gf_results['symbol'] == 'GDF15']
    if len(gdf15) > 0:
        print(f"\nGDF15 details:")
        print(f"  log2FC: {gdf15['log2FoldChange'].values[0]:.3f}")
        print(f"  p-value: {gdf15['pvalue'].values[0]:.2e}")
        print(f"  padj: {gdf15['padj'].values[0]:.2e}")

    # Check FGF7 specifically
    fgf7 = gf_results[gf_results['symbol'] == 'FGF7']
    if len(fgf7) > 0:
        print(f"\nFGF7 details:")
        print(f"  log2FC: {fgf7['log2FoldChange'].values[0]:.3f}")
        print(f"  p-value: {fgf7['pvalue'].values[0]:.2e}")
        print(f"  padj: {fgf7['padj'].values[0]:.2e}")

    return gf_results


def analyze_early_stage_blood():
    """Analyze growth factor changes in early-stage blood proteomics."""
    print("\n" + "=" * 70)
    print("EARLY-STAGE BLOOD PROTEOMICS: GROWTH FACTOR ANALYSIS")
    print("=" * 70)

    # Load data
    early = pd.read_parquet(DATA_DIR / "Q-12622_Zha_NPX_2024-08-21.parquet")
    manifest = pd.read_excel(DATA_DIR / "Q-12622_Zha - Olink_-_Sample_Manifest.xlsx")

    # Pivot to wide
    early_wide = early.pivot_table(index='SampleID', columns='Assay', values='NPX', aggfunc='first').reset_index()
    manifest['SampleID'] = manifest['SampleID'].astype(str)
    manifest_map = manifest[['SampleID', 'TP', 'Subj ID']].drop_duplicates()
    early_wide = pd.merge(early_wide, manifest_map, on='SampleID', how='left')

    # Get pre and post
    pre = early_wide[early_wide['TP'] == 'pre'].copy()
    post = early_wide[early_wide['TP'] == 'post'].copy()

    # Find growth factors in the data
    available_gf = [gf for gf in GROWTH_FACTORS if gf in early_wide.columns]
    print(f"\nGrowth factors available in blood proteomics: {len(available_gf)}")

    # Perform paired t-test for each growth factor
    results = []
    for gf in available_gf:
        # Get paired data
        pre_vals = pre.set_index('Subj ID')[gf]
        post_vals = post.set_index('Subj ID')[gf]

        common_ids = pre_vals.index.intersection(post_vals.index)
        pre_paired = pre_vals.loc[common_ids].dropna()
        post_paired = post_vals.loc[common_ids].dropna()

        common_ids = pre_paired.index.intersection(post_paired.index)
        if len(common_ids) < 5:
            continue

        pre_final = pre_paired.loc[common_ids].values
        post_final = post_paired.loc[common_ids].values

        # Paired t-test
        t_stat, p_val = stats.ttest_rel(post_final, pre_final)
        mean_change = (post_final - pre_final).mean()

        results.append({
            'gene': gf,
            'n': len(common_ids),
            'mean_pre': pre_final.mean(),
            'mean_post': post_final.mean(),
            'mean_change': mean_change,
            't_statistic': t_stat,
            'p_value': p_val
        })

    results_df = pd.DataFrame(results)

    # FDR correction
    results_df['fdr_q'] = fdr_correction(results_df['p_value'].values)
    results_df = results_df.sort_values('p_value')

    # Significant results
    sig = results_df[results_df['fdr_q'] < 0.05]
    sig_up = sig[sig['mean_change'] > 0]
    sig_down = sig[sig['mean_change'] < 0]

    print(f"\nTotal growth factors tested: {len(results_df)}")
    print(f"Significantly changed (FDR < 0.05): {len(sig)}")
    print(f"  Upregulated: {len(sig_up)}")
    print(f"  Downregulated: {len(sig_down)}")

    if len(sig_up) > 0:
        print(f"\nSignificantly UPREGULATED growth factors:")
        print(sig_up[['gene', 'mean_change', 'p_value', 'fdr_q']].to_string(index=False))

    # Check GDF15
    gdf15 = results_df[results_df['gene'] == 'GDF15']
    if len(gdf15) > 0:
        print(f"\nGDF15 details:")
        print(f"  Mean change: {gdf15['mean_change'].values[0]:.3f}")
        print(f"  p-value: {gdf15['p_value'].values[0]:.2e}")
        print(f"  FDR q: {gdf15['fdr_q'].values[0]:.4f}")

    # Check FGF7
    fgf7 = results_df[results_df['gene'] == 'FGF7']
    if len(fgf7) > 0:
        print(f"\nFGF7 details:")
        print(f"  Mean change: {fgf7['mean_change'].values[0]:.3f}")
        print(f"  p-value: {fgf7['p_value'].values[0]:.2e}")
        print(f"  FDR q: {fgf7['fdr_q'].values[0]:.4f}")

    return results_df


def create_volcano_plot(cosinr_results, early_results):
    """Create volcano plot comparing both cohorts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # COSINR tumor RNA-seq
    ax1 = axes[0]
    cosinr_results = cosinr_results.dropna(subset=['log2FoldChange', 'padj'])
    cosinr_results['neg_log10_padj'] = -np.log10(cosinr_results['padj'].clip(lower=1e-50))

    # Color by significance
    colors = ['red' if (row['padj'] < 0.05 and row['log2FoldChange'] > 0)
              else 'blue' if (row['padj'] < 0.05 and row['log2FoldChange'] < 0)
              else 'gray' for _, row in cosinr_results.iterrows()]

    ax1.scatter(cosinr_results['log2FoldChange'], cosinr_results['neg_log10_padj'],
                c=colors, alpha=0.6, s=30)

    # Label GDF15 and FGF7
    for gene in ['GDF15', 'FGF7']:
        gene_data = cosinr_results[cosinr_results['symbol'] == gene]
        if len(gene_data) > 0:
            ax1.annotate(gene,
                        xy=(gene_data['log2FoldChange'].values[0],
                            gene_data['neg_log10_padj'].values[0]),
                        fontsize=10, fontweight='bold')

    ax1.axhline(-np.log10(0.05), color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('log2 Fold Change')
    ax1.set_ylabel('-log10(FDR-adjusted p)')
    ax1.set_title('COSINR (SBRT + ICB)\nTumor RNA-seq')

    # Early-stage blood
    ax2 = axes[1]
    early_results = early_results.dropna(subset=['mean_change', 'fdr_q'])
    early_results['neg_log10_fdr'] = -np.log10(early_results['fdr_q'].clip(lower=1e-50))

    colors2 = ['red' if (row['fdr_q'] < 0.05 and row['mean_change'] > 0)
               else 'blue' if (row['fdr_q'] < 0.05 and row['mean_change'] < 0)
               else 'gray' for _, row in early_results.iterrows()]

    ax2.scatter(early_results['mean_change'], early_results['neg_log10_fdr'],
                c=colors2, alpha=0.6, s=30)

    # Label GDF15 and FGF7
    for gene in ['GDF15', 'FGF7']:
        gene_data = early_results[early_results['gene'] == gene]
        if len(gene_data) > 0:
            ax2.annotate(gene,
                        xy=(gene_data['mean_change'].values[0],
                            gene_data['neg_log10_fdr'].values[0]),
                        fontsize=10, fontweight='bold')

    ax2.axhline(-np.log10(0.05), color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Mean Change (NPX)')
    ax2.set_ylabel('-log10(FDR-adjusted p)')
    ax2.set_title('Early-Stage (SBRT Alone)\nBlood Proteomics')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'growth_factor_volcano.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'growth_factor_volcano.pdf', bbox_inches='tight')
    plt.close()

    print(f"\nFigure saved to: {FIGURES_DIR / 'growth_factor_volcano.png'}")


def main():
    """Main function."""
    print("=" * 70)
    print("GROWTH FACTOR DIFFERENTIAL EXPRESSION ANALYSIS")
    print("=" * 70)

    # Analyze COSINR tumor RNA-seq
    cosinr_results = analyze_cosinr_tumor_rnaseq()

    # Analyze early-stage blood proteomics
    early_results = analyze_early_stage_blood()

    # Create volcano plots
    create_volcano_plot(cosinr_results, early_results)

    # Save results
    cosinr_results.to_csv(RESULTS_DIR / 'growth_factors_cosinr_tumor.csv', index=False)
    early_results.to_csv(RESULTS_DIR / 'growth_factors_early_blood.csv', index=False)

    print("\n" + "=" * 70)
    print("SUMMARY: SBRT-SPECIFIC vs ICB-RELATED GROWTH FACTORS")
    print("=" * 70)

    print("""
Analysis complete. See output above for:
- Significantly upregulated growth factors in each cohort
- Comparison of SBRT-alone vs SBRT+ICB effects
""")

    return cosinr_results, early_results


if __name__ == "__main__":
    results = main()
