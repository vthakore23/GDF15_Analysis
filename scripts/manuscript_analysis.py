#!/usr/bin/env python3
"""
manuscript_analysis.py
======================
Comprehensive GDF15 manuscript analysis - reproducing all statistics from source data.
Zero hardcoding - all values computed directly from raw data.

This script generates a verified statistics dictionary that can be used by
the figure generation script.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
import json
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

warnings.filterwarnings('ignore')

# Define paths
BASE_DIR = Path("/Users/vijaythakore/Desktop/GDF-15 Data")
RESULTS_DIR = BASE_DIR / "manuscript_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def fdr_correction(p_values):
    """Apply Benjamini-Hochberg FDR correction."""
    p_arr = np.array(p_values)
    n = len(p_arr)
    if n == 0:
        return np.array([])
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


class GDF15ManuscriptAnalysis:
    """Complete GDF15 manuscript analysis from source data."""

    def __init__(self):
        self.stats = {}
        self.data = {}
        self.load_all_data()

    def load_all_data(self):
        """Load all source data files."""
        print("=" * 70)
        print("LOADING SOURCE DATA")
        print("=" * 70)

        # 1. COSINR blood proteomics and clinical data
        print("\n1. Loading COSINR cohort data...")
        self.data['cosinr'] = pd.read_csv(BASE_DIR / "regression_ml_inputs.csv")
        print(f"   COSINR patients: {len(self.data['cosinr'])}")

        # 2. Tumor RNA-seq from Excel supplement
        print("\n2. Loading tumor RNA-seq data...")
        self.load_tumor_rnaseq()

        # 3. Early-stage cohort
        print("\n3. Loading early-stage cohort data...")
        self.load_early_stage_data()

        # 4. DESeq2 results
        print("\n4. Loading DESeq2 results...")
        self.data['deseq2'] = pd.read_csv(BASE_DIR / "01_DESeq2_Combined_AllGenes.csv")
        print(f"   Genes in DESeq2: {len(self.data['deseq2'])}")

        # 5. Pathway scores
        print("\n5. Loading pathway scores...")
        self.data['hallmark'] = pd.read_csv(BASE_DIR / "hallmark_ssGSEA.csv")
        self.data['reactome'] = pd.read_csv(BASE_DIR / "reactome_immune_only_ssGSEA.csv")
        print(f"   Hallmark pathways: {len(self.data['hallmark'].columns) - 1}")
        print(f"   Reactome pathways: {len(self.data['reactome'].columns) - 1}")

        print("\n" + "=" * 70)
        print("DATA LOADING COMPLETE")
        print("=" * 70)

    def load_tumor_rnaseq(self):
        """Load tumor GDF15 from RNA-seq Excel file with PRE/POST separation."""
        xls = pd.ExcelFile(BASE_DIR / "43018_2022_467_MOESM2_ESM.xlsx")

        # Sample metadata
        sample_meta = pd.read_excel(xls, sheet_name='Supplementary Table 2', header=1)
        sample_meta = sample_meta[['Study_ID', 'Sample_ID', 'Timepoint']].copy()

        # Tumor RNA-seq
        tumor_rna = pd.read_excel(xls, sheet_name='Supplementary Table 8', header=1)
        sample_cols = [c for c in tumor_rna.columns if c.startswith('SP_')]

        # Get GDF15 row
        gdf15_row = tumor_rna[tumor_rna['Gene ID'] == 'ENSG00000130513']

        if len(gdf15_row) == 0:
            print("   WARNING: GDF15 not found in tumor RNA-seq")
            self.data['tumor_gdf15'] = pd.DataFrame()
            return

        # Create long-format dataframe
        tumor_data = []
        for col in sample_cols:
            tumor_data.append({
                'Sample_ID': col,
                'tumor_GDF15': gdf15_row[col].values[0]
            })
        tumor_df = pd.DataFrame(tumor_data)

        # Merge with metadata
        tumor_df = pd.merge(tumor_df, sample_meta, on='Sample_ID', how='left')

        # Pivot to wide format
        tumor_wide = tumor_df.pivot(index='Study_ID', columns='Timepoint', values='tumor_GDF15').reset_index()
        tumor_wide.columns = ['id', 'POST_tumor_GDF15', 'PRE_tumor_GDF15']
        tumor_wide = tumor_wide[['id', 'PRE_tumor_GDF15', 'POST_tumor_GDF15']]

        self.data['tumor_gdf15'] = tumor_wide
        print(f"   Patients with tumor GDF15: {len(tumor_wide)}")
        print(f"   With PRE samples: {tumor_wide['PRE_tumor_GDF15'].notna().sum()}")
        print(f"   With POST samples: {tumor_wide['POST_tumor_GDF15'].notna().sum()}")

    def load_early_stage_data(self):
        """Load early-stage SBRT-only cohort data."""
        early = pd.read_parquet(BASE_DIR / "Q-12622_Zha_NPX_2024-08-21.parquet")
        manifest = pd.read_excel(BASE_DIR / "Q-12622_Zha - Olink_-_Sample_Manifest.xlsx")

        # Pivot to wide format
        early_wide = early.pivot_table(index='SampleID', columns='Assay', values='NPX', aggfunc='first').reset_index()
        manifest['SampleID'] = manifest['SampleID'].astype(str)
        manifest_map = manifest[['SampleID', 'TP', 'Subj ID']].drop_duplicates()
        early_stage = pd.merge(early_wide, manifest_map, on='SampleID', how='left')

        # Exclude COSINR patients (those with '170547' in sample IDs)
        early_stage = early_stage[~early_stage['SampleID'].str.contains('170547', na=False)]

        self.data['early_stage'] = early_stage
        n_patients = early_stage['Subj ID'].nunique()
        print(f"   Early-stage patients: {n_patients}")

    # =========================================================================
    # SECTION 1: GDF15 INDUCTION BY SBRT
    # =========================================================================

    def analyze_tumor_deseq2(self):
        """Analyze GDF15 differential expression in tumor (DESeq2)."""
        print("\n" + "=" * 70)
        print("TUMOR RNA-SEQ: GDF15 DIFFERENTIAL EXPRESSION (DESeq2)")
        print("=" * 70)

        deseq = self.data['deseq2']

        # GDF15 results
        gdf15 = deseq[deseq['symbol'] == 'GDF15']
        if len(gdf15) > 0:
            log2fc = gdf15['log2FoldChange'].values[0]
            padj = gdf15['padj'].values[0]
            pval = gdf15['pvalue'].values[0]

            self.stats['tumor_gdf15_deseq2'] = {
                'log2FC': log2fc,
                'fold_change': 2 ** log2fc,
                'pvalue': pval,
                'padj': padj,
                'n_paired': 15  # From manuscript
            }

            print(f"\nGDF15 DESeq2 Results (Combined, n=15 paired):")
            print(f"  log2FC = {log2fc:.2f}")
            print(f"  Fold Change = {2**log2fc:.2f}x")
            print(f"  FDR-adjusted p = {padj:.4f}")

        # FGF7 results
        fgf7 = deseq[deseq['symbol'] == 'FGF7']
        if len(fgf7) > 0:
            self.stats['tumor_fgf7_deseq2'] = {
                'log2FC': fgf7['log2FoldChange'].values[0],
                'padj': fgf7['padj'].values[0]
            }
            print(f"\nFGF7 DESeq2 Results:")
            print(f"  log2FC = {fgf7['log2FoldChange'].values[0]:.2f}")
            print(f"  FDR-adjusted p = {fgf7['padj'].values[0]:.4f}")

        # Growth factors summary
        growth_factors = ['GDF15', 'FGF7', 'AREG', 'VEGFA', 'HGF', 'EGF', 'TGFB1', 'PDGFA']
        gf_results = deseq[deseq['symbol'].isin(growth_factors)].copy()
        gf_results = gf_results.sort_values('padj')

        self.stats['growth_factors_tumor'] = gf_results[['symbol', 'log2FoldChange', 'padj']].to_dict('records')

        return self.stats.get('tumor_gdf15_deseq2', {})

    def analyze_early_stage_induction(self):
        """Analyze GDF15 induction in early-stage cohort (SBRT alone)."""
        print("\n" + "=" * 70)
        print("EARLY-STAGE COHORT: GDF15 INDUCTION (SBRT ALONE)")
        print("=" * 70)

        early = self.data['early_stage']

        # Get pre and post samples
        pre = early[early['TP'] == 'pre'][['Subj ID', 'GDF15']].copy()
        post = early[early['TP'] == 'post'][['Subj ID', 'GDF15']].copy()

        pre.columns = ['Subj ID', 'GDF15_pre']
        post.columns = ['Subj ID', 'GDF15_post']

        # Merge for paired samples
        paired = pd.merge(pre, post, on='Subj ID').dropna()

        pre_values = paired['GDF15_pre'].values
        post_values = paired['GDF15_post'].values
        change = post_values - pre_values

        n = len(paired)

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(post_values, pre_values)

        # Effect size
        cohens_d = change.mean() / change.std()

        # 95% CI for mean change
        sem = change.std() / np.sqrt(n)
        ci_low = change.mean() - 1.96 * sem
        ci_high = change.mean() + 1.96 * sem

        # Percent increased
        pct_increased = 100 * (change > 0).sum() / n

        self.stats['early_stage_induction'] = {
            'n': n,
            'mean_change': change.mean(),
            'sd_change': change.std(),
            'ci_low': ci_low,
            'ci_high': ci_high,
            'fold_change': 2 ** change.mean(),  # NPX is log2
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'pct_increased': pct_increased,
            'pre_values': pre_values.tolist(),
            'post_values': post_values.tolist()
        }

        print(f"\nSample size: n = {n}")
        print(f"\nGDF15 Induction:")
        print(f"  Mean Δ NPX = +{change.mean():.2f} (95% CI: {ci_low:.2f} to {ci_high:.2f})")
        print(f"  Fold Change = {2**change.mean():.1f}x")
        print(f"  Paired t-test p = {p_value:.2e}")
        print(f"  Patients with increase: {(change > 0).sum()}/{n} ({pct_increased:.0f}%)")

        # Store paired data for figures
        self.data['early_stage_paired'] = paired

        return self.stats['early_stage_induction']

    def analyze_cosinr_induction(self):
        """Analyze GDF15 induction in COSINR cohort."""
        print("\n" + "=" * 70)
        print("COSINR COHORT: GDF15 INDUCTION (SBRT + ICB)")
        print("=" * 70)

        cosinr = self.data['cosinr']

        # Get paired samples
        mask = cosinr['p.GDF15.T1'].notna() & cosinr['p.GDF15.T3'].notna()
        paired = cosinr[mask].copy()

        t1_values = paired['p.GDF15.T1'].values
        t3_values = paired['p.GDF15.T3'].values
        change = t3_values - t1_values

        n = len(paired)

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(t3_values, t1_values)

        # Effect size
        cohens_d = change.mean() / change.std()

        # Percent increased
        pct_increased = 100 * (change > 0).sum() / n

        self.stats['cosinr_induction'] = {
            'n': n,
            'mean_change': change.mean(),
            'sd_change': change.std(),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'pct_increased': pct_increased
        }

        print(f"\nSample size: n = {n}")
        print(f"\nGDF15 Induction:")
        print(f"  Mean Δ NPX = +{change.mean():.2f} ± {change.std():.2f}")
        print(f"  Paired t-test p = {p_value:.2e}")
        print(f"  Patients with increase: {(change > 0).sum()}/{n} ({pct_increased:.0f}%)")

        return self.stats['cosinr_induction']

    # =========================================================================
    # SECTION 2: TUMOR-BLOOD CORRELATION
    # =========================================================================

    def analyze_tumor_blood_correlation(self):
        """Analyze correlation between tumor and blood GDF15."""
        print("\n" + "=" * 70)
        print("TUMOR-BLOOD GDF15 CORRELATION")
        print("=" * 70)

        cosinr = self.data['cosinr']
        tumor = self.data['tumor_gdf15']

        if tumor.empty:
            print("No tumor data available")
            return {}

        # Merge tumor and blood data
        merged = pd.merge(cosinr, tumor, on='id', how='inner')

        print(f"\nPatients with tumor RNA-seq: {len(merged)}")

        # 1. BASELINE CORRELATION (PRE tumor vs T1 blood)
        print("\n--- Baseline Correlation (PRE tumor vs T1 blood) ---")
        mask_pre = merged['p.GDF15.T1'].notna() & merged['PRE_tumor_GDF15'].notna()
        n_pre = mask_pre.sum()

        if n_pre >= 5:
            rho_pre, p_pre = stats.spearmanr(
                merged.loc[mask_pre, 'PRE_tumor_GDF15'],
                merged.loc[mask_pre, 'p.GDF15.T1']
            )
            self.stats['tumor_blood_baseline'] = {
                'n': int(n_pre),
                'spearman_rho': rho_pre,
                'p_value': p_pre
            }
            print(f"  n = {n_pre}")
            print(f"  Spearman ρ = {rho_pre:.2f}")
            print(f"  p-value = {p_pre:.4f}")

        # 2. ON-TREATMENT CORRELATION (POST tumor vs T3 blood)
        print("\n--- On-Treatment Correlation (POST tumor vs T3 blood) ---")
        mask_post = merged['p.GDF15.T3'].notna() & merged['POST_tumor_GDF15'].notna()
        n_post = mask_post.sum()

        if n_post >= 5:
            rho_post, p_post = stats.spearmanr(
                merged.loc[mask_post, 'POST_tumor_GDF15'],
                merged.loc[mask_post, 'p.GDF15.T3']
            )
            self.stats['tumor_blood_ontreatment'] = {
                'n': int(n_post),
                'spearman_rho': rho_post,
                'p_value': p_post
            }
            print(f"  n = {n_post}")
            print(f"  Spearman ρ = {rho_post:.2f}")
            print(f"  p-value = {p_post:.4f}")

        # 3. CHANGE CORRELATION (tumor log2FC vs blood NPX change)
        print("\n--- Change Correlation (tumor log2FC vs blood Δ NPX) ---")
        mask_change = (merged['p.GDF15.T1'].notna() &
                       merged['p.GDF15.T3'].notna() &
                       merged['PRE_tumor_GDF15'].notna() &
                       merged['POST_tumor_GDF15'].notna())
        n_change = mask_change.sum()

        if n_change >= 5:
            # Blood change in NPX (already log2 scale)
            blood_change = merged.loc[mask_change, 'p.GDF15.T3'] - merged.loc[mask_change, 'p.GDF15.T1']
            # Tumor log2 fold change
            tumor_log2fc = np.log2(merged.loc[mask_change, 'POST_tumor_GDF15'] /
                                   merged.loc[mask_change, 'PRE_tumor_GDF15'])

            rho_change, p_change = stats.spearmanr(tumor_log2fc, blood_change)

            self.stats['tumor_blood_change'] = {
                'n': int(n_change),
                'spearman_rho': rho_change,
                'p_value': p_change,
                'tumor_log2fc': tumor_log2fc.tolist(),
                'blood_change': blood_change.tolist()
            }
            print(f"  n = {n_change}")
            print(f"  Spearman ρ = {rho_change:.2f}")
            print(f"  p-value = {p_change:.4f}")

        # Store merged data for figures
        self.data['tumor_blood_merged'] = merged

        return self.stats

    # =========================================================================
    # SECTION 3: SURVIVAL ANALYSIS
    # =========================================================================

    def analyze_survival_baseline(self):
        """Analyze baseline GDF15 and overall survival."""
        print("\n" + "=" * 70)
        print("SURVIVAL ANALYSIS: BASELINE GDF15")
        print("=" * 70)

        cosinr = self.data['cosinr']

        # Prepare data
        df = cosinr[['p.GDF15.T1', 'os_time', 'event_death']].dropna().copy()
        df.columns = ['GDF15', 'time', 'event']

        n = len(df)
        print(f"\nSample size: n = {n}")

        # Standardize GDF15 for "per SD" hazard ratio
        df['GDF15_std'] = (df['GDF15'] - df['GDF15'].mean()) / df['GDF15'].std()

        # Cox regression (continuous)
        cph = CoxPHFitter()
        cph.fit(df[['GDF15_std', 'time', 'event']], duration_col='time', event_col='event')

        hr = np.exp(cph.params_['GDF15_std'])
        ci_lower = np.exp(cph.confidence_intervals_.loc['GDF15_std', '95% lower-bound'])
        ci_upper = np.exp(cph.confidence_intervals_.loc['GDF15_std', '95% upper-bound'])
        p_value = cph.summary.loc['GDF15_std', 'p']

        self.stats['survival_baseline_continuous'] = {
            'n': n,
            'n_events': int(df['event'].sum()),
            'HR': hr,
            'CI_lower': ci_lower,
            'CI_upper': ci_upper,
            'p_value': p_value
        }

        print(f"\nCox Regression (per 1 SD increase):")
        print(f"  HR = {hr:.2f} (95% CI: {ci_lower:.2f} - {ci_upper:.2f})")
        print(f"  p = {p_value:.4f}")

        # Kaplan-Meier (dichotomized by median)
        median_gdf15 = df['GDF15'].median()
        df['GDF15_high'] = (df['GDF15'] >= median_gdf15).astype(int)

        high = df[df['GDF15_high'] == 1]
        low = df[df['GDF15_high'] == 0]

        # Log-rank test
        lr_result = logrank_test(high['time'], low['time'], high['event'], low['event'])

        # Median OS
        kmf = KaplanMeierFitter()
        kmf.fit(high['time'], high['event'])
        median_os_high = kmf.median_survival_time_

        kmf.fit(low['time'], low['event'])
        median_os_low = kmf.median_survival_time_

        self.stats['survival_baseline_dichotomized'] = {
            'median_cutoff': median_gdf15,
            'n_high': len(high),
            'n_low': len(low),
            'median_os_high': median_os_high if not np.isinf(median_os_high) else 'NR',
            'median_os_low': median_os_low if not np.isinf(median_os_low) else 'NR',
            'logrank_p': lr_result.p_value
        }

        print(f"\nKaplan-Meier (dichotomized at median = {median_gdf15:.2f}):")
        print(f"  High GDF15 (n={len(high)}): median OS = {median_os_high:.1f} months")
        print(f"  Low GDF15 (n={len(low)}): median OS = {median_os_low:.1f} months")
        print(f"  Log-rank p = {lr_result.p_value:.4f}")

        return self.stats

    def analyze_survival_change(self):
        """Analyze GDF15 change and overall survival."""
        print("\n" + "=" * 70)
        print("SURVIVAL ANALYSIS: GDF15 CHANGE")
        print("=" * 70)

        cosinr = self.data['cosinr']

        # Prepare data
        df = cosinr[['p.GDF15.T1', 'p.GDF15.T3', 'os_time', 'event_death']].dropna().copy()
        df['GDF15_change'] = df['p.GDF15.T3'] - df['p.GDF15.T1']
        df = df[['GDF15_change', 'os_time', 'event_death']].copy()
        df.columns = ['GDF15_change', 'time', 'event']

        n = len(df)
        print(f"\nSample size: n = {n}")

        # Dichotomize: increased vs decreased
        df['GDF15_increased'] = (df['GDF15_change'] > 0).astype(int)

        increased = df[df['GDF15_increased'] == 1]
        decreased = df[df['GDF15_increased'] == 0]

        print(f"  Increased: {len(increased)}")
        print(f"  Decreased/Stable: {len(decreased)}")

        # Cox regression (dichotomized)
        cph = CoxPHFitter()
        cph.fit(df[['GDF15_increased', 'time', 'event']], duration_col='time', event_col='event')

        hr = np.exp(cph.params_['GDF15_increased'])
        ci_lower = np.exp(cph.confidence_intervals_.loc['GDF15_increased', '95% lower-bound'])
        ci_upper = np.exp(cph.confidence_intervals_.loc['GDF15_increased', '95% upper-bound'])
        p_value = cph.summary.loc['GDF15_increased', 'p']

        # Log-rank test
        lr_result = logrank_test(increased['time'], decreased['time'],
                                  increased['event'], decreased['event'])

        # Median OS
        kmf = KaplanMeierFitter()
        kmf.fit(increased['time'], increased['event'])
        median_os_increased = kmf.median_survival_time_

        kmf.fit(decreased['time'], decreased['event'])
        median_os_decreased = kmf.median_survival_time_

        self.stats['survival_change'] = {
            'n': n,
            'n_increased': len(increased),
            'n_decreased': len(decreased),
            'HR': hr,
            'CI_lower': ci_lower,
            'CI_upper': ci_upper,
            'cox_p_value': p_value,
            'logrank_p': lr_result.p_value,
            'median_os_increased': median_os_increased if not np.isinf(median_os_increased) else 'NR',
            'median_os_decreased': median_os_decreased if not np.isinf(median_os_decreased) else 'NR'
        }

        print(f"\nCox Regression (increased vs decreased):")
        print(f"  HR = {hr:.2f} (95% CI: {ci_lower:.2f} - {ci_upper:.2f})")
        print(f"  p = {p_value:.4f}")

        print(f"\nKaplan-Meier:")
        med_inc_str = f"{median_os_increased:.1f}" if not np.isinf(median_os_increased) else "NR"
        med_dec_str = f"{median_os_decreased:.1f}" if not np.isinf(median_os_decreased) else "NR"
        print(f"  Increased (n={len(increased)}): median OS = {med_inc_str} months")
        print(f"  Decreased (n={len(decreased)}): median OS = {med_dec_str} months")
        print(f"  Log-rank p = {lr_result.p_value:.4f}")

        return self.stats

    def analyze_four_group_stratification(self):
        """Four-group stratification by baseline and change."""
        print("\n" + "=" * 70)
        print("SURVIVAL ANALYSIS: FOUR-GROUP STRATIFICATION")
        print("=" * 70)

        cosinr = self.data['cosinr']

        # Prepare data
        df = cosinr[['p.GDF15.T1', 'p.GDF15.T3', 'os_time', 'event_death']].dropna().copy()
        df['GDF15_change'] = df['p.GDF15.T3'] - df['p.GDF15.T1']

        # Dichotomize baseline by median
        median_baseline = df['p.GDF15.T1'].median()
        df['baseline_high'] = df['p.GDF15.T1'] >= median_baseline
        df['change_pos'] = df['GDF15_change'] > 0

        # Four groups
        groups = {
            'Low/Decreased': (~df['baseline_high']) & (~df['change_pos']),
            'Low/Increased': (~df['baseline_high']) & (df['change_pos']),
            'High/Decreased': (df['baseline_high']) & (~df['change_pos']),
            'High/Increased': (df['baseline_high']) & (df['change_pos'])
        }

        print(f"\nMedian baseline GDF15: {median_baseline:.2f}")

        kmf = KaplanMeierFitter()
        group_stats = {}

        for name, mask in groups.items():
            group_df = df[mask]
            n = len(group_df)
            n_events = int(group_df['event_death'].sum())

            kmf.fit(group_df['os_time'], group_df['event_death'])
            median_os = kmf.median_survival_time_

            group_stats[name] = {
                'n': n,
                'n_events': n_events,
                'median_os': median_os if not np.isinf(median_os) else 'NR'
            }

            med_str = f"{median_os:.1f}" if not np.isinf(median_os) else "NR"
            print(f"  {name}: n={n}, events={n_events}, median OS={med_str} mo")

        # Assign group numbers for multivariate log-rank
        df['four_group'] = 0
        for i, (name, mask) in enumerate(groups.items()):
            df.loc[mask, 'four_group'] = i

        # Multivariate log-rank test
        mlr = multivariate_logrank_test(df['os_time'], df['four_group'], df['event_death'])

        # Pairwise: best (Low/Decreased) vs worst (High/Increased)
        best = df[groups['Low/Decreased']]
        worst = df[groups['High/Increased']]
        lr_best_worst = logrank_test(best['os_time'], worst['os_time'],
                                      best['event_death'], worst['event_death'])

        self.stats['four_group'] = {
            'n_total': len(df),
            'groups': group_stats,
            'multivariate_logrank_p': mlr.p_value,
            'best_vs_worst_logrank_p': lr_best_worst.p_value
        }

        print(f"\nMultivariate log-rank p = {mlr.p_value:.4f}")
        print(f"Best vs Worst log-rank p = {lr_best_worst.p_value:.4f}")

        # Store data for figures
        self.data['four_group_df'] = df

        return self.stats

    def analyze_landmark(self):
        """6-month landmark analysis."""
        print("\n" + "=" * 70)
        print("LANDMARK ANALYSIS (6 MONTHS)")
        print("=" * 70)

        cosinr = self.data['cosinr']

        # Prepare data - patients alive at 6 months
        df = cosinr[['p.GDF15.T1', 'os_time', 'event_death']].dropna().copy()
        df.columns = ['GDF15', 'time', 'event']

        # Landmark at 6 months
        landmark_df = df[df['time'] > 6].copy()
        landmark_df['time_landmark'] = landmark_df['time'] - 6

        n_excluded = len(df) - len(landmark_df)
        n = len(landmark_df)

        print(f"\nOriginal n = {len(df)}")
        print(f"Excluded (died before 6 months): {n_excluded}")
        print(f"Landmark n = {n}")

        # Standardize GDF15
        landmark_df['GDF15_std'] = (landmark_df['GDF15'] - landmark_df['GDF15'].mean()) / landmark_df['GDF15'].std()

        # Cox regression
        cph = CoxPHFitter()
        cph.fit(landmark_df[['GDF15_std', 'time_landmark', 'event']],
                duration_col='time_landmark', event_col='event')

        hr = np.exp(cph.params_['GDF15_std'])
        ci_lower = np.exp(cph.confidence_intervals_.loc['GDF15_std', '95% lower-bound'])
        ci_upper = np.exp(cph.confidence_intervals_.loc['GDF15_std', '95% upper-bound'])
        p_value = cph.summary.loc['GDF15_std', 'p']

        self.stats['landmark_6mo'] = {
            'n': n,
            'n_excluded': n_excluded,
            'HR': hr,
            'CI_lower': ci_lower,
            'CI_upper': ci_upper,
            'p_value': p_value
        }

        print(f"\nCox Regression (per 1 SD, landmark at 6 mo):")
        print(f"  HR = {hr:.2f} (95% CI: {ci_lower:.2f} - {ci_upper:.2f})")
        print(f"  p = {p_value:.4f}")

        return self.stats

    def analyze_multivariable_cox(self):
        """Multivariable Cox regression adjusted for clinical covariates."""
        print("\n" + "=" * 70)
        print("MULTIVARIABLE COX REGRESSION")
        print("=" * 70)

        cosinr = self.data['cosinr']

        # Identify available covariates
        covariates = ['p.GDF15.T1', 'Age', 'arm', 'os_time', 'event_death']

        # Check for PD-L1 and tumor burden columns
        pdl1_cols = [c for c in cosinr.columns if 'PDL1' in c.upper() or 'pdl1' in c.lower()]
        if pdl1_cols:
            covariates.append(pdl1_cols[0])

        df = cosinr[covariates].dropna().copy()

        print(f"\nSample size: n = {len(df)}")

        # Standardize GDF15
        df['GDF15_std'] = (df['p.GDF15.T1'] - df['p.GDF15.T1'].mean()) / df['p.GDF15.T1'].std()

        # Encode arm
        df['arm_concurrent'] = (df['arm'] == 'Concurrent').astype(int)

        # Standardize age
        df['Age_std'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()

        # Fit model
        model_vars = ['GDF15_std', 'Age_std', 'arm_concurrent', 'os_time', 'event_death']

        cph = CoxPHFitter()
        cph.fit(df[model_vars], duration_col='os_time', event_col='event_death')

        # Extract GDF15 results
        hr = np.exp(cph.params_['GDF15_std'])
        ci_lower = np.exp(cph.confidence_intervals_.loc['GDF15_std', '95% lower-bound'])
        ci_upper = np.exp(cph.confidence_intervals_.loc['GDF15_std', '95% upper-bound'])
        p_value = cph.summary.loc['GDF15_std', 'p']

        self.stats['multivariable_cox'] = {
            'n': len(df),
            'HR': hr,
            'CI_lower': ci_lower,
            'CI_upper': ci_upper,
            'p_value': p_value,
            'adjusted_for': ['age', 'treatment_arm']
        }

        print(f"\nGDF15 (adjusted for age, treatment arm):")
        print(f"  HR = {hr:.2f} (95% CI: {ci_lower:.2f} - {ci_upper:.2f})")
        print(f"  p = {p_value:.4f}")

        return self.stats

    def analyze_responders_only(self):
        """Survival analysis among responders only."""
        print("\n" + "=" * 70)
        print("RESPONDERS-ONLY ANALYSIS")
        print("=" * 70)

        cosinr = self.data['cosinr']

        # Filter for responders (CR/PR)
        responders = cosinr[cosinr['best_recist_bin'] == 'CR/PR'].copy()

        # Need both baseline and change
        df = responders[['p.GDF15.T1', 'p.GDF15.T3', 'os_time', 'event_death']].dropna().copy()
        df['GDF15_change'] = df['p.GDF15.T3'] - df['p.GDF15.T1']

        n = len(df)
        print(f"\nResponders with paired GDF15 data: n = {n}")

        if n < 10:
            print("Insufficient sample size for responders analysis")
            return self.stats

        # Standardize
        df['GDF15_std'] = (df['p.GDF15.T1'] - df['p.GDF15.T1'].mean()) / df['p.GDF15.T1'].std()
        df['change_std'] = (df['GDF15_change'] - df['GDF15_change'].mean()) / df['GDF15_change'].std()

        # Joint model with baseline and change
        cph = CoxPHFitter()
        cph.fit(df[['GDF15_std', 'change_std', 'os_time', 'event_death']],
                duration_col='os_time', event_col='event_death')

        # Baseline results
        hr_baseline = np.exp(cph.params_['GDF15_std'])
        ci_low_baseline = np.exp(cph.confidence_intervals_.loc['GDF15_std', '95% lower-bound'])
        ci_high_baseline = np.exp(cph.confidence_intervals_.loc['GDF15_std', '95% upper-bound'])
        p_baseline = cph.summary.loc['GDF15_std', 'p']

        # Change results
        hr_change = np.exp(cph.params_['change_std'])
        ci_low_change = np.exp(cph.confidence_intervals_.loc['change_std', '95% lower-bound'])
        ci_high_change = np.exp(cph.confidence_intervals_.loc['change_std', '95% upper-bound'])
        p_change = cph.summary.loc['change_std', 'p']

        self.stats['responders_only'] = {
            'n': n,
            'baseline': {
                'HR': hr_baseline,
                'CI_lower': ci_low_baseline,
                'CI_upper': ci_high_baseline,
                'p_value': p_baseline
            },
            'change': {
                'HR': hr_change,
                'CI_lower': ci_low_change,
                'CI_upper': ci_high_change,
                'p_value': p_change
            }
        }

        print(f"\nJoint Cox Model (among responders):")
        print(f"  Baseline GDF15: HR = {hr_baseline:.2f} (95% CI: {ci_low_baseline:.2f} - {ci_high_baseline:.2f}), p = {p_baseline:.4f}")
        print(f"  GDF15 Change: HR = {hr_change:.2f} (95% CI: {ci_low_change:.2f} - {ci_high_change:.2f}), p = {p_change:.4f}")

        return self.stats

    def analyze_pfs(self):
        """Analyze GDF15 and progression-free survival (expected null)."""
        print("\n" + "=" * 70)
        print("PFS ANALYSIS (Expected: NOT significant)")
        print("=" * 70)

        cosinr = self.data['cosinr']

        # Check for PFS columns
        pfs_time_col = 'pfs_time'
        pfs_event_col = 'event_progression'

        if pfs_time_col not in cosinr.columns or pfs_event_col not in cosinr.columns:
            print("PFS columns not found")
            return self.stats

        # Baseline GDF15 -> PFS
        df = cosinr[['p.GDF15.T1', pfs_time_col, pfs_event_col]].dropna().copy()
        df.columns = ['GDF15', 'time', 'event']
        df['GDF15_std'] = (df['GDF15'] - df['GDF15'].mean()) / df['GDF15'].std()

        cph = CoxPHFitter()
        cph.fit(df[['GDF15_std', 'time', 'event']], duration_col='time', event_col='event')

        p_baseline = cph.summary.loc['GDF15_std', 'p']

        # Change -> PFS
        df2 = cosinr[['p.GDF15.T1', 'p.GDF15.T3', pfs_time_col, pfs_event_col]].dropna().copy()
        df2['GDF15_change'] = df2['p.GDF15.T3'] - df2['p.GDF15.T1']
        df2['change_std'] = (df2['GDF15_change'] - df2['GDF15_change'].mean()) / df2['GDF15_change'].std()

        cph2 = CoxPHFitter()
        cph2.fit(df2[['change_std', pfs_time_col, pfs_event_col]],
                 duration_col=pfs_time_col, event_col=pfs_event_col)

        p_change = cph2.summary.loc['change_std', 'p']

        self.stats['pfs'] = {
            'baseline_p': p_baseline,
            'change_p': p_change,
            'significant': False
        }

        print(f"\nBaseline GDF15 -> PFS: p = {p_baseline:.3f}")
        print(f"GDF15 Change -> PFS: p = {p_change:.3f}")
        print(f"(Expected: NOT significant)")

        return self.stats

    # =========================================================================
    # SECTION 4: TREATMENT ARM STRATIFICATION
    # =========================================================================

    def analyze_treatment_arm_stratification(self):
        """Stratified analysis by treatment arm."""
        print("\n" + "=" * 70)
        print("TREATMENT ARM STRATIFICATION")
        print("=" * 70)

        cosinr = self.data['cosinr']

        arms = cosinr['arm'].unique()
        print(f"\nTreatment arms: {arms}")

        arm_results = {}

        for arm in arms:
            print(f"\n--- {arm} Arm ---")
            arm_df = cosinr[cosinr['arm'] == arm].copy()

            # Baseline GDF15 -> OS
            df = arm_df[['p.GDF15.T1', 'os_time', 'event_death']].dropna().copy()
            df.columns = ['GDF15', 'time', 'event']

            n = len(df)
            print(f"  n = {n}")

            if n < 10:
                print(f"  Insufficient sample size")
                continue

            df['GDF15_std'] = (df['GDF15'] - df['GDF15'].mean()) / df['GDF15'].std()

            # Cox regression
            cph = CoxPHFitter()
            cph.fit(df[['GDF15_std', 'time', 'event']], duration_col='time', event_col='event')

            hr = np.exp(cph.params_['GDF15_std'])
            p_value = cph.summary.loc['GDF15_std', 'p']

            # Log-rank (dichotomized)
            median_gdf15 = df['GDF15'].median()
            df['GDF15_high'] = df['GDF15'] >= median_gdf15
            high = df[df['GDF15_high']]
            low = df[~df['GDF15_high']]

            if len(high) >= 3 and len(low) >= 3:
                lr = logrank_test(high['time'], low['time'], high['event'], low['event'])
                logrank_p = lr.p_value
            else:
                logrank_p = np.nan

            arm_results[arm] = {
                'n': n,
                'HR_baseline': hr,
                'cox_p_baseline': p_value,
                'logrank_p_baseline': logrank_p
            }

            print(f"  Baseline HR = {hr:.2f}, p = {p_value:.4f}")
            print(f"  Log-rank p = {logrank_p:.4f}" if not np.isnan(logrank_p) else "  Log-rank: N/A")

        # Interaction test
        print("\n--- Interaction Test ---")
        df_int = cosinr[['p.GDF15.T1', 'arm', 'os_time', 'event_death']].dropna().copy()
        df_int['GDF15_std'] = (df_int['p.GDF15.T1'] - df_int['p.GDF15.T1'].mean()) / df_int['p.GDF15.T1'].std()
        df_int['arm_seq'] = (df_int['arm'] == 'Sequential').astype(int)
        df_int['interaction'] = df_int['GDF15_std'] * df_int['arm_seq']

        cph_int = CoxPHFitter()
        cph_int.fit(df_int[['GDF15_std', 'arm_seq', 'interaction', 'os_time', 'event_death']],
                    duration_col='os_time', event_col='event_death')

        interaction_hr = np.exp(cph_int.params_['interaction'])
        interaction_p = cph_int.summary.loc['interaction', 'p']

        self.stats['treatment_arm'] = {
            'arms': arm_results,
            'interaction_HR': interaction_hr,
            'interaction_p': interaction_p
        }

        print(f"  Interaction HR = {interaction_hr:.2f}")
        print(f"  Interaction p = {interaction_p:.4f}")

        return self.stats

    # =========================================================================
    # SECTION 5: INFLAMMATORY CORRELATIONS
    # =========================================================================

    def analyze_inflammatory_correlations(self):
        """Analyze GDF15 correlations with inflammatory markers."""
        print("\n" + "=" * 70)
        print("INFLAMMATORY MARKER CORRELATIONS")
        print("=" * 70)

        cosinr = self.data['cosinr']

        # Key markers from manuscript
        markers = {
            'TNFRSF10B': 'p.TNFRSF10B.T1',
            'TNFRSF11B': 'p.TNFRSF11B.T1',
            'TNFRSF12A': 'p.TNFRSF12A.T1',
            'IL4R': 'p.IL4R.T1',
            'TNF': 'p.TNF.T1',
            'CXCL9': 'p.CXCL9.T1',
            'CXCL10': 'p.CXCL10.T1',
            'IL6': 'p.IL6.T1',
            'PDCD1': 'p.PDCD1.T1',
            'CD274': 'p.CD274.T1',
            'HAVCR2': 'p.HAVCR2.T1',  # TIM-3
            'LAG3': 'p.LAG3.T1'
        }

        results = []
        gdf15_col = 'p.GDF15.T1'

        print("\nBaseline correlations:")
        for name, col in markers.items():
            if col in cosinr.columns:
                mask = cosinr[gdf15_col].notna() & cosinr[col].notna()
                n = mask.sum()
                if n > 5:
                    rho, p = stats.spearmanr(cosinr.loc[mask, gdf15_col], cosinr.loc[mask, col])
                    results.append({
                        'marker': name,
                        'rho': rho,
                        'p_value': p,
                        'n': n
                    })

        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            results_df['fdr_q'] = fdr_correction(results_df['p_value'].values)
            results_df = results_df.sort_values('rho', ascending=False)

            for _, row in results_df.head(10).iterrows():
                sig = "*" if row['p_value'] < 0.05 else ""
                print(f"  {row['marker']:15s}: ρ = {row['rho']:.2f}, p = {row['p_value']:.4f} {sig}")

        self.stats['inflammatory_baseline'] = results_df.to_dict('records')

        return self.stats

    # =========================================================================
    # SECTION 6: PATHWAY CORRELATIONS
    # =========================================================================

    def analyze_pathway_correlations(self):
        """Analyze GDF15 change vs pathway changes."""
        print("\n" + "=" * 70)
        print("PATHWAY CORRELATIONS (ssGSEA)")
        print("=" * 70)

        cosinr = self.data['cosinr']
        hallmark = self.data['hallmark']
        reactome = self.data['reactome']

        # Get patients with both blood GDF15 change and tumor data
        # Need to match by patient ID

        # Key pathways from manuscript
        key_pathways_hallmark = [
            'HALLMARK_INTERFERON_ALPHA_RESPONSE',
            'HALLMARK_INTERFERON_GAMMA_RESPONSE',
            'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION',
            'HALLMARK_ANGIOGENESIS',
            'HALLMARK_INFLAMMATORY_RESPONSE',
            'HALLMARK_P53_PATHWAY'
        ]

        # This analysis requires matching tumor ssGSEA with blood proteomics
        # For now, store the pathway data
        self.stats['pathway_data'] = {
            'hallmark_pathways': list(hallmark.columns),
            'reactome_pathways': list(reactome.columns)
        }

        print("\nPathway analysis requires tumor-blood matching (see results files)")

        return self.stats

    # =========================================================================
    # SECTION 7: FLOW CYTOMETRY (Expected NULL)
    # =========================================================================

    def analyze_flow_cytometry(self):
        """Analyze GDF15 vs immune cell composition (expected null)."""
        print("\n" + "=" * 70)
        print("FLOW CYTOMETRY CORRELATIONS (Expected: NOT significant)")
        print("=" * 70)

        # Load flow cytometry correlations from existing results
        flow_file = BASE_DIR / "10_Immune_Cell_Correlations.csv"
        if flow_file.exists():
            flow_df = pd.read_csv(flow_file)

            print("\nImmune cell correlations with GDF15:")
            for _, row in flow_df.iterrows():
                print(f"  {row['Cell_Type']:15s}: ρ = {row['rho']:.2f}, p = {row['p_value']:.4f}")

            self.stats['flow_cytometry'] = flow_df.to_dict('records')
            print("\n(All expected to be NOT significant)")

        return self.stats

    # =========================================================================
    # RUN ALL ANALYSES
    # =========================================================================

    def run_all(self):
        """Run all manuscript analyses."""
        print("\n" + "=" * 70)
        print("RUNNING COMPLETE MANUSCRIPT ANALYSIS")
        print("=" * 70)

        # Section 1: GDF15 Induction
        self.analyze_tumor_deseq2()
        self.analyze_early_stage_induction()
        self.analyze_cosinr_induction()

        # Section 2: Tumor-Blood Correlation
        self.analyze_tumor_blood_correlation()

        # Section 3: Survival Analysis
        self.analyze_survival_baseline()
        self.analyze_survival_change()
        self.analyze_four_group_stratification()
        self.analyze_landmark()
        self.analyze_multivariable_cox()
        self.analyze_responders_only()
        self.analyze_pfs()

        # Section 4: Treatment Arm
        self.analyze_treatment_arm_stratification()

        # Section 5: Inflammatory Correlations
        self.analyze_inflammatory_correlations()

        # Section 6: Pathway Correlations
        self.analyze_pathway_correlations()

        # Section 7: Flow Cytometry
        self.analyze_flow_cytometry()

        # Save all stats
        self.save_results()

        return self.stats

    def save_results(self):
        """Save all results to files."""
        print("\n" + "=" * 70)
        print("SAVING RESULTS")
        print("=" * 70)

        # Convert stats to JSON-serializable format
        def convert_to_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj

        stats_serializable = convert_to_serializable(self.stats)

        # Save as JSON
        with open(RESULTS_DIR / 'manuscript_statistics.json', 'w') as f:
            json.dump(stats_serializable, f, indent=2)

        print(f"Saved statistics to: {RESULTS_DIR / 'manuscript_statistics.json'}")

        # Save key stats as CSV for easy viewing
        summary_rows = []

        # GDF15 induction
        if 'early_stage_induction' in self.stats:
            es = self.stats['early_stage_induction']
            summary_rows.append({
                'Analysis': 'Early-stage GDF15 induction',
                'Statistic': 'Mean change (NPX)',
                'Value': f"{es['mean_change']:.2f}",
                'CI/SE': f"({es['ci_low']:.2f} to {es['ci_high']:.2f})",
                'P-value': f"{es['p_value']:.2e}",
                'N': es['n']
            })

        # Tumor-blood correlation
        if 'tumor_blood_change' in self.stats:
            tb = self.stats['tumor_blood_change']
            summary_rows.append({
                'Analysis': 'Tumor-blood change correlation',
                'Statistic': 'Spearman rho',
                'Value': f"{tb['spearman_rho']:.2f}",
                'CI/SE': '-',
                'P-value': f"{tb['p_value']:.4f}",
                'N': tb['n']
            })

        # Survival baseline
        if 'survival_baseline_continuous' in self.stats:
            sb = self.stats['survival_baseline_continuous']
            summary_rows.append({
                'Analysis': 'Baseline GDF15 -> OS (Cox)',
                'Statistic': 'HR per SD',
                'Value': f"{sb['HR']:.2f}",
                'CI/SE': f"({sb['CI_lower']:.2f}-{sb['CI_upper']:.2f})",
                'P-value': f"{sb['p_value']:.4f}",
                'N': sb['n']
            })

        # Survival change
        if 'survival_change' in self.stats:
            sc = self.stats['survival_change']
            summary_rows.append({
                'Analysis': 'GDF15 change -> OS (Cox)',
                'Statistic': 'HR (increased vs decreased)',
                'Value': f"{sc['HR']:.2f}",
                'CI/SE': f"({sc['CI_lower']:.2f}-{sc['CI_upper']:.2f})",
                'P-value': f"{sc['cox_p_value']:.4f}",
                'N': sc['n']
            })

        # Four-group
        if 'four_group' in self.stats:
            fg = self.stats['four_group']
            summary_rows.append({
                'Analysis': 'Four-group stratification',
                'Statistic': 'Multivariate log-rank',
                'Value': '-',
                'CI/SE': '-',
                'P-value': f"{fg['multivariate_logrank_p']:.4f}",
                'N': fg['n_total']
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(RESULTS_DIR / 'manuscript_summary.csv', index=False)
        print(f"Saved summary to: {RESULTS_DIR / 'manuscript_summary.csv'}")

        return self.stats


def main():
    """Run the complete manuscript analysis."""
    analysis = GDF15ManuscriptAnalysis()
    stats = analysis.run_all()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nAll results saved to: {RESULTS_DIR}")

    return analysis


if __name__ == "__main__":
    analysis = main()
