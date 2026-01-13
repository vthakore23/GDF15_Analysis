#!/usr/bin/env python3
"""
02_gdf15_induction.py
=====================
Analyze GDF15 induction by SBRT in both cohorts.

Analyses:
1. COSINR cohort: Paired t-test comparing T1 vs T3 GDF15 levels
2. Early-stage cohort: Paired t-test comparing pre vs post GDF15 levels
3. Visualization of GDF15 changes
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


def analyze_cosinr_induction(cosinr):
    """Analyze GDF15 induction in COSINR cohort (SBRT + ICB)."""
    print("=" * 70)
    print("COSINR COHORT: GDF15 INDUCTION ANALYSIS")
    print("=" * 70)

    # Get paired samples
    mask = cosinr['p.GDF15.T1'].notna() & cosinr['p.GDF15.T3'].notna()
    paired = cosinr[mask].copy()

    t1_values = paired['p.GDF15.T1'].values
    t3_values = paired['p.GDF15.T3'].values
    change = t3_values - t1_values

    n = len(paired)

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(t3_values, t1_values)

    # Summary statistics
    print(f"\nSample size: n = {n}")
    print(f"\nBaseline (T1):")
    print(f"  Mean ± SD: {t1_values.mean():.2f} ± {t1_values.std():.2f} NPX")
    print(f"  Median (IQR): {np.median(t1_values):.2f} ({np.percentile(t1_values, 25):.2f} - {np.percentile(t1_values, 75):.2f})")

    print(f"\nOn-treatment (T3):")
    print(f"  Mean ± SD: {t3_values.mean():.2f} ± {t3_values.std():.2f} NPX")
    print(f"  Median (IQR): {np.median(t3_values):.2f} ({np.percentile(t3_values, 25):.2f} - {np.percentile(t3_values, 75):.2f})")

    print(f"\nChange (T3 - T1):")
    print(f"  Mean ± SD: {change.mean():.2f} ± {change.std():.2f} NPX")
    print(f"  Patients with increase: {(change > 0).sum()} ({100*(change > 0).sum()/n:.1f}%)")
    print(f"  Patients with decrease: {(change <= 0).sum()} ({100*(change <= 0).sum()/n:.1f}%)")

    print(f"\nPaired t-test:")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.2e}")

    # Effect size (Cohen's d for paired samples)
    cohens_d = change.mean() / change.std()
    print(f"  Cohen's d: {cohens_d:.3f}")

    results = {
        'cohort': 'COSINR',
        'n': n,
        't1_mean': t1_values.mean(),
        't1_sd': t1_values.std(),
        't3_mean': t3_values.mean(),
        't3_sd': t3_values.std(),
        'change_mean': change.mean(),
        'change_sd': change.std(),
        'pct_increase': 100 * (change > 0).sum() / n,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d
    }

    return results, paired


def analyze_early_stage_induction(early_stage):
    """Analyze GDF15 induction in early-stage cohort (SBRT alone)."""
    print("\n" + "=" * 70)
    print("EARLY-STAGE COHORT: GDF15 INDUCTION ANALYSIS")
    print("=" * 70)

    # Separate pre and post
    pre = early_stage[early_stage['TP'] == 'pre'][['Subj ID', 'GDF15']].copy()
    post = early_stage[early_stage['TP'] == 'post'][['Subj ID', 'GDF15']].copy()

    pre.columns = ['Subj ID', 'GDF15_pre']
    post.columns = ['Subj ID', 'GDF15_post']

    # Merge to get paired samples
    paired = pd.merge(pre, post, on='Subj ID')
    paired = paired.dropna()

    pre_values = paired['GDF15_pre'].values
    post_values = paired['GDF15_post'].values
    change = post_values - pre_values

    n = len(paired)

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(post_values, pre_values)

    # Summary statistics
    print(f"\nSample size: n = {n}")
    print(f"\nPre-treatment:")
    print(f"  Mean ± SD: {pre_values.mean():.2f} ± {pre_values.std():.2f} NPX")
    print(f"  Median (IQR): {np.median(pre_values):.2f} ({np.percentile(pre_values, 25):.2f} - {np.percentile(pre_values, 75):.2f})")

    print(f"\nPost-treatment:")
    print(f"  Mean ± SD: {post_values.mean():.2f} ± {post_values.std():.2f} NPX")
    print(f"  Median (IQR): {np.median(post_values):.2f} ({np.percentile(post_values, 25):.2f} - {np.percentile(post_values, 75):.2f})")

    print(f"\nChange (Post - Pre):")
    print(f"  Mean ± SD: {change.mean():.2f} ± {change.std():.2f} NPX")
    print(f"  Patients with increase: {(change > 0).sum()} ({100*(change > 0).sum()/n:.1f}%)")
    print(f"  Patients with decrease: {(change <= 0).sum()} ({100*(change <= 0).sum()/n:.1f}%)")

    print(f"\nPaired t-test:")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.2e}")

    # Effect size
    cohens_d = change.mean() / change.std()
    print(f"  Cohen's d: {cohens_d:.3f}")

    results = {
        'cohort': 'Early-Stage',
        'n': n,
        'pre_mean': pre_values.mean(),
        'pre_sd': pre_values.std(),
        'post_mean': post_values.mean(),
        'post_sd': post_values.std(),
        'change_mean': change.mean(),
        'change_sd': change.std(),
        'pct_increase': 100 * (change > 0).sum() / n,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d
    }

    return results, paired


def plot_gdf15_induction(cosinr_paired, early_paired):
    """Create visualization of GDF15 induction."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # COSINR cohort
    ax1 = axes[0]
    for _, row in cosinr_paired.iterrows():
        ax1.plot([0, 1], [row['p.GDF15.T1'], row['p.GDF15.T3']],
                 'o-', color='gray', alpha=0.5, linewidth=0.5, markersize=4)

    # Add mean line
    ax1.plot([0, 1],
             [cosinr_paired['p.GDF15.T1'].mean(), cosinr_paired['p.GDF15.T3'].mean()],
             'o-', color='red', linewidth=3, markersize=10, label='Mean')

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Baseline (T1)', 'On-treatment (T3)'])
    ax1.set_ylabel('GDF15 (NPX)')
    ax1.set_title(f'COSINR Cohort (n={len(cosinr_paired)})\nSBRT + Immunotherapy')
    ax1.legend()

    # Add p-value annotation
    t_stat, p_val = stats.ttest_rel(cosinr_paired['p.GDF15.T3'], cosinr_paired['p.GDF15.T1'])
    ax1.annotate(f'p = {p_val:.2e}', xy=(0.5, 0.95), xycoords='axes fraction',
                 ha='center', fontsize=12, fontweight='bold')

    # Early-stage cohort
    ax2 = axes[1]
    for _, row in early_paired.iterrows():
        ax2.plot([0, 1], [row['GDF15_pre'], row['GDF15_post']],
                 'o-', color='gray', alpha=0.5, linewidth=0.5, markersize=4)

    # Add mean line
    ax2.plot([0, 1],
             [early_paired['GDF15_pre'].mean(), early_paired['GDF15_post'].mean()],
             'o-', color='blue', linewidth=3, markersize=10, label='Mean')

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Pre-SBRT', 'Post-SBRT'])
    ax2.set_ylabel('GDF15 (NPX)')
    ax2.set_title(f'Early-Stage Cohort (n={len(early_paired)})\nSBRT Alone')
    ax2.legend()

    # Add p-value annotation
    t_stat, p_val = stats.ttest_rel(early_paired['GDF15_post'], early_paired['GDF15_pre'])
    ax2.annotate(f'p = {p_val:.2e}', xy=(0.5, 0.95), xycoords='axes fraction',
                 ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'GDF15_induction_paired.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'GDF15_induction_paired.pdf', bbox_inches='tight')
    plt.close()

    print(f"\nFigure saved to: {FIGURES_DIR / 'GDF15_induction_paired.png'}")


def main():
    """Main function."""
    print("=" * 70)
    print("GDF15 INDUCTION ANALYSIS")
    print("=" * 70)

    # Load data
    cosinr = pd.read_csv(DATA_DIR / "regression_ml_inputs.csv")

    early = pd.read_parquet(DATA_DIR / "Q-12622_Zha_NPX_2024-08-21.parquet")
    manifest = pd.read_excel(DATA_DIR / "Q-12622_Zha - Olink_-_Sample_Manifest.xlsx")
    early_wide = early.pivot_table(index='SampleID', columns='Assay', values='NPX', aggfunc='first').reset_index()
    manifest['SampleID'] = manifest['SampleID'].astype(str)
    manifest_map = manifest[['SampleID', 'TP', 'Subj ID']].drop_duplicates()
    early_stage = pd.merge(early_wide, manifest_map, on='SampleID', how='left')

    # Run analyses
    cosinr_results, cosinr_paired = analyze_cosinr_induction(cosinr)
    early_results, early_paired = analyze_early_stage_induction(early_stage)

    # Create figures
    plot_gdf15_induction(cosinr_paired, early_paired)

    # Save results
    results_df = pd.DataFrame([cosinr_results, early_results])
    results_df.to_csv(RESULTS_DIR / 'gdf15_induction_results.csv', index=False)
    print(f"\nResults saved to: {RESULTS_DIR / 'gdf15_induction_results.csv'}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nGDF15 induction results:")
    print(f"  COSINR (SBRT+ICB): p = {cosinr_results['p_value']:.2e}, Cohen's d = {cosinr_results['cohens_d']:.2f}")
    print(f"  Early-stage (SBRT alone): p = {early_results['p_value']:.2e}, Cohen's d = {early_results['cohens_d']:.2f}")

    return cosinr_results, early_results


if __name__ == "__main__":
    results = main()
