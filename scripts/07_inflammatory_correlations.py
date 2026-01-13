#!/usr/bin/env python3
"""
07_inflammatory_correlations.py
===============================
Analyze correlations between GDF15 and inflammatory markers.

Analyses:
1. Baseline GDF15 vs baseline inflammatory markers (both cohorts)
2. On-treatment GDF15 vs on-treatment inflammatory markers
3. GDF15 change vs inflammatory marker changes
4. GDF15 vs TNF receptor superfamily proteins

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


def analyze_cosinr_baseline(cosinr):
    """Analyze baseline GDF15 vs inflammatory markers in COSINR."""
    print("=" * 70)
    print("COSINR: BASELINE GDF15 vs INFLAMMATORY MARKERS")
    print("=" * 70)

    gdf15_col = 'p.GDF15.T1'

    # Key inflammatory markers
    markers = {
        'TNF-α': 'p.TNF.T1',
        'CXCL9': 'p.CXCL9.T1',
        'CXCL10': 'p.CXCL10.T1',
        'IL-6': 'p.IL6.T1',
        'IL-15': 'p.IL15.T1',
        'IL-10': 'p.IL10.T1',
        'CD274 (PD-L1)': 'p.CD274.T1',
        'PDCD1 (PD-1)': 'p.PDCD1.T1'
    }

    results = []
    for name, col in markers.items():
        if col in cosinr.columns:
            mask = cosinr[gdf15_col].notna() & cosinr[col].notna()
            n = mask.sum()
            if n > 5:
                rho, p = stats.spearmanr(cosinr.loc[mask, gdf15_col], cosinr.loc[mask, col])
                results.append({'marker': name, 'rho': rho, 'p_value': p, 'n': n})

    results_df = pd.DataFrame(results)
    results_df['fdr_q'] = fdr_correction(results_df['p_value'].values)
    results_df = results_df.sort_values('p_value')

    print("\nBaseline correlations:")
    for _, row in results_df.iterrows():
        sig = "*" if row['fdr_q'] < 0.05 else ""
        print(f"  {row['marker']:20s}: ρ = {row['rho']:.2f}, FDR p = {row['fdr_q']:.4f} {sig}")

    return results_df


def analyze_cosinr_on_treatment(cosinr):
    """Analyze on-treatment GDF15 vs inflammatory markers in COSINR."""
    print("\n" + "=" * 70)
    print("COSINR: ON-TREATMENT GDF15 vs INFLAMMATORY MARKERS")
    print("=" * 70)

    gdf15_col = 'p.GDF15.T3'

    markers = {
        'TNF-α': 'p.TNF.T3',
        'CXCL9': 'p.CXCL9.T3',
        'CXCL10': 'p.CXCL10.T3',
        'IL-6': 'p.IL6.T3',
        'IL-15': 'p.IL15.T3',
        'IL-10': 'p.IL10.T3',
        'CD274 (PD-L1)': 'p.CD274.T3',
        'PDCD1 (PD-1)': 'p.PDCD1.T3'
    }

    results = []
    for name, col in markers.items():
        if col in cosinr.columns:
            mask = cosinr[gdf15_col].notna() & cosinr[col].notna()
            n = mask.sum()
            if n > 5:
                rho, p = stats.spearmanr(cosinr.loc[mask, gdf15_col], cosinr.loc[mask, col])
                results.append({'marker': name, 'rho': rho, 'p_value': p, 'n': n})

    results_df = pd.DataFrame(results)
    results_df['fdr_q'] = fdr_correction(results_df['p_value'].values)
    results_df = results_df.sort_values('p_value')

    print("\nOn-treatment correlations:")
    for _, row in results_df.iterrows():
        sig = "*" if row['fdr_q'] < 0.05 else ""
        print(f"  {row['marker']:20s}: ρ = {row['rho']:.2f}, FDR p = {row['fdr_q']:.4f} {sig}")

    return results_df


def analyze_tnf_superfamily(cosinr):
    """Analyze GDF15 vs TNF receptor superfamily proteins."""
    print("\n" + "=" * 70)
    print("COSINR: GDF15 vs TNF RECEPTOR SUPERFAMILY")
    print("=" * 70)

    # Find TNFRSF columns
    tnfrsf_cols_t1 = [c for c in cosinr.columns if 'TNFRSF' in c and '.T1' in c]
    tnfrsf_cols_t3 = [c for c in cosinr.columns if 'TNFRSF' in c and '.T3' in c]

    print(f"\nTNFRSF proteins found: {len(tnfrsf_cols_t1)}")

    # Baseline correlations
    print("\n--- BASELINE (T1) ---")
    results_t1 = []
    for col in tnfrsf_cols_t1:
        mask = cosinr['p.GDF15.T1'].notna() & cosinr[col].notna()
        n = mask.sum()
        if n > 5:
            rho, p = stats.spearmanr(cosinr.loc[mask, 'p.GDF15.T1'], cosinr.loc[mask, col])
            name = col.replace('p.', '').replace('.T1', '')
            results_t1.append({'marker': name, 'rho': rho, 'p_value': p, 'n': n})

    results_t1_df = pd.DataFrame(results_t1).sort_values('rho', ascending=False)

    print("\nTop TNFRSF correlations (baseline):")
    for _, row in results_t1_df.head(10).iterrows():
        strong = "**" if abs(row['rho']) > 0.6 else ""
        print(f"  {row['marker']:15s}: ρ = {row['rho']:.2f}, p = {row['p_value']:.2e} {strong}")

    # On-treatment correlations
    print("\n--- ON-TREATMENT (T3) ---")
    results_t3 = []
    for col in tnfrsf_cols_t3:
        mask = cosinr['p.GDF15.T3'].notna() & cosinr[col].notna()
        n = mask.sum()
        if n > 5:
            rho, p = stats.spearmanr(cosinr.loc[mask, 'p.GDF15.T3'], cosinr.loc[mask, col])
            name = col.replace('p.', '').replace('.T3', '')
            results_t3.append({'marker': name, 'rho': rho, 'p_value': p, 'n': n})

    results_t3_df = pd.DataFrame(results_t3).sort_values('rho', ascending=False)

    print("\nTop TNFRSF correlations (on-treatment):")
    for _, row in results_t3_df.head(10).iterrows():
        strong = "**" if abs(row['rho']) > 0.6 else ""
        print(f"  {row['marker']:15s}: ρ = {row['rho']:.2f}, p = {row['p_value']:.2e} {strong}")

    return results_t1_df, results_t3_df


def analyze_early_stage(early_stage):
    """Analyze GDF15 vs inflammatory markers in early-stage cohort."""
    print("\n" + "=" * 70)
    print("EARLY-STAGE: BASELINE GDF15 vs INFLAMMATORY MARKERS")
    print("=" * 70)

    # Filter for baseline (pre)
    early_t1 = early_stage[early_stage['TP'] == 'pre'].copy()

    markers = {
        'TNF-α': 'TNF',
        'CXCL9': 'CXCL9',
        'CXCL10': 'CXCL10',
        'IL-6': 'IL6',
        'IL-15': 'IL15',
        'IL-10': 'IL10',
        'CD274 (PD-L1)': 'CD274'
    }

    results = []
    for name, col in markers.items():
        if col in early_t1.columns and 'GDF15' in early_t1.columns:
            mask = early_t1['GDF15'].notna() & early_t1[col].notna()
            n = mask.sum()
            if n > 5:
                rho, p = stats.spearmanr(early_t1.loc[mask, 'GDF15'], early_t1.loc[mask, col])
                results.append({'marker': name, 'rho': rho, 'p_value': p, 'n': n})

    if results:
        results_df = pd.DataFrame(results)
        results_df['fdr_q'] = fdr_correction(results_df['p_value'].values)
        results_df = results_df.sort_values('p_value')

        print("\nBaseline correlations (early-stage):")
        for _, row in results_df.iterrows():
            sig = "*" if row['fdr_q'] < 0.05 else ""
            print(f"  {row['marker']:20s}: ρ = {row['rho']:.2f}, FDR p = {row['fdr_q']:.4f} {sig}")

        return results_df

    return pd.DataFrame()


def analyze_gdf15_change_correlations(cosinr):
    """Analyze GDF15 change vs inflammatory marker changes."""
    print("\n" + "=" * 70)
    print("COSINR: GDF15 CHANGE vs INFLAMMATORY MARKER CHANGES")
    print("=" * 70)

    gdf15_change = 'p.GDF15.dif1v3'

    markers = {
        'TNF-α': 'p.TNF.dif1v3',
        'CXCL9': 'p.CXCL9.dif1v3',
        'CXCL10': 'p.CXCL10.dif1v3',
        'IL-6': 'p.IL6.dif1v3',
        'IL-15': 'p.IL15.dif1v3',
        'IL-10': 'p.IL10.dif1v3',
        'CD274 (PD-L1)': 'p.CD274.dif1v3'
    }

    results = []
    for name, col in markers.items():
        if col in cosinr.columns and gdf15_change in cosinr.columns:
            mask = cosinr[gdf15_change].notna() & cosinr[col].notna()
            n = mask.sum()
            if n > 5:
                rho, p = stats.spearmanr(cosinr.loc[mask, gdf15_change], cosinr.loc[mask, col])
                results.append({'marker': name, 'rho': rho, 'p_value': p, 'n': n})

    if results:
        results_df = pd.DataFrame(results)
        results_df['fdr_q'] = fdr_correction(results_df['p_value'].values)
        results_df = results_df.sort_values('p_value')

        print("\nChange correlations:")
        for _, row in results_df.iterrows():
            sig = "*" if row['fdr_q'] < 0.05 else ""
            print(f"  {row['marker']:20s}: ρ = {row['rho']:.2f}, FDR p = {row['fdr_q']:.4f} {sig}")

        return results_df

    return pd.DataFrame()


def create_correlation_heatmap(cosinr):
    """Create correlation heatmap for inflammatory markers."""
    markers_t1 = {
        'GDF15': 'p.GDF15.T1',
        'TNF-α': 'p.TNF.T1',
        'CXCL9': 'p.CXCL9.T1',
        'CXCL10': 'p.CXCL10.T1',
        'IL-6': 'p.IL6.T1',
        'IL-15': 'p.IL15.T1',
        'IL-10': 'p.IL10.T1',
        'PD-L1': 'p.CD274.T1'
    }

    # Build correlation matrix
    available = {k: v for k, v in markers_t1.items() if v in cosinr.columns}
    data = cosinr[[v for v in available.values()]].dropna()

    if len(data) > 10:
        corr_matrix = data.corr(method='spearman')
        corr_matrix.columns = list(available.keys())
        corr_matrix.index = list(available.keys())

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, ax=ax)
        ax.set_title('Spearman Correlations: GDF15 and Inflammatory Markers (Baseline)')

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'inflammatory_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig(FIGURES_DIR / 'inflammatory_correlation_heatmap.pdf', bbox_inches='tight')
        plt.close()

        print(f"\nHeatmap saved to: {FIGURES_DIR / 'inflammatory_correlation_heatmap.png'}")


def main():
    """Main function."""
    print("=" * 70)
    print("GDF15 INFLAMMATORY MARKER CORRELATION ANALYSIS")
    print("=" * 70)

    # Load COSINR data
    cosinr = pd.read_csv(DATA_DIR / "regression_ml_inputs.csv")

    # Load early-stage data
    early = pd.read_parquet(DATA_DIR / "Q-12622_Zha_NPX_2024-08-21.parquet")
    manifest = pd.read_excel(DATA_DIR / "Q-12622_Zha - Olink_-_Sample_Manifest.xlsx")
    early_wide = early.pivot_table(index='SampleID', columns='Assay', values='NPX', aggfunc='first').reset_index()
    manifest['SampleID'] = manifest['SampleID'].astype(str)
    manifest_map = manifest[['SampleID', 'TP', 'Subj ID']].drop_duplicates()
    early_stage = pd.merge(early_wide, manifest_map, on='SampleID', how='left')

    # Run analyses
    cosinr_baseline = analyze_cosinr_baseline(cosinr)
    cosinr_ontreat = analyze_cosinr_on_treatment(cosinr)
    tnfrsf_t1, tnfrsf_t3 = analyze_tnf_superfamily(cosinr)
    early_results = analyze_early_stage(early_stage)
    change_results = analyze_gdf15_change_correlations(cosinr)

    # Create heatmap
    create_correlation_heatmap(cosinr)

    # Save all results
    cosinr_baseline.to_csv(RESULTS_DIR / 'inflammatory_correlations_cosinr_baseline.csv', index=False)
    cosinr_ontreat.to_csv(RESULTS_DIR / 'inflammatory_correlations_cosinr_ontreat.csv', index=False)
    tnfrsf_t1.to_csv(RESULTS_DIR / 'tnfrsf_correlations_baseline.csv', index=False)
    tnfrsf_t3.to_csv(RESULTS_DIR / 'tnfrsf_correlations_ontreat.csv', index=False)
    if len(early_results) > 0:
        early_results.to_csv(RESULTS_DIR / 'inflammatory_correlations_early_stage.csv', index=False)
    if len(change_results) > 0:
        change_results.to_csv(RESULTS_DIR / 'inflammatory_correlations_change.csv', index=False)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return {
        'cosinr_baseline': cosinr_baseline,
        'cosinr_ontreat': cosinr_ontreat,
        'tnfrsf_t1': tnfrsf_t1,
        'tnfrsf_t3': tnfrsf_t3,
        'early_stage': early_results,
        'change': change_results
    }


if __name__ == "__main__":
    results = main()
