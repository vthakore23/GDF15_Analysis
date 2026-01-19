#!/usr/bin/env python3
"""
compute_all_statistics.py
=========================
Compute ALL manuscript statistics and save to comprehensive CSV files.

This script computes every statistic mentioned in the manuscript and saves them
to CSV files for verification and figure generation.

Statistics computed:
1. DESeq2 results (GDF15, FGF7)
2. Early-stage cohort (GDF15 induction)
3. Tumor-blood correlations (baseline, on-treatment, change)
4. Survival analyses (baseline, change, 4-group, landmark, multivariable)
5. Treatment arm stratification (interaction, arm-specific)
6. Immune correlations (inflammatory markers)
7. Pathway correlations (including p53)
8. Responders-only analyses
9. Flow cytometry (null results)
10. PFS analyses (null results)
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path("/Users/vijaythakore/Desktop/GDF-15 Data")

# Dictionary to store all computed statistics
all_stats = {}

print("=" * 70)
print("COMPUTING ALL MANUSCRIPT STATISTICS")
print("=" * 70)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n--- Loading data ---")

# COSINR cohort
cosinr = pd.read_csv(BASE_DIR / "regression_ml_inputs.csv")
print(f"COSINR patients: {len(cosinr)}")

# DESeq2 results
deseq = pd.read_csv(BASE_DIR / "01_DESeq2_Combined_AllGenes.csv")

# Early-stage cohort
early_parquet = BASE_DIR / "Q-12622_Zha_NPX_2024-08-21.parquet"
early_manifest = BASE_DIR / "Q-12622_Zha_-_Olink_-_Sample_Manifest.xlsx"

# Pathway scores
hallmark_path = BASE_DIR / "hallmark_ssGSEA.csv"
reactome_path = BASE_DIR / "reactome_immune_only_ssGSEA.csv"

# =============================================================================
# 1. DESeq2 RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("1. DESeq2 RESULTS (Tumor RNA-seq)")
print("=" * 70)

gdf15_deseq = deseq[deseq['symbol'] == 'GDF15'].iloc[0]
fgf7_deseq = deseq[deseq['symbol'] == 'FGF7']
ret_deseq = deseq[deseq['symbol'] == 'RET']

# Store key DESeq2 results for export
deseq2_key_genes = []

all_stats['deseq2_gdf15_log2fc'] = gdf15_deseq['log2FoldChange']
all_stats['deseq2_gdf15_padj'] = gdf15_deseq['padj']
all_stats['deseq2_gdf15_foldchange'] = 2 ** gdf15_deseq['log2FoldChange']
deseq2_key_genes.append({'Gene': 'GDF15', 'log2FoldChange': gdf15_deseq['log2FoldChange'],
                         'pvalue': gdf15_deseq['pvalue'], 'padj': gdf15_deseq['padj']})

print(f"GDF15: log2FC = {all_stats['deseq2_gdf15_log2fc']:.3f}, padj = {all_stats['deseq2_gdf15_padj']:.4f}")
print(f"       Fold change = {all_stats['deseq2_gdf15_foldchange']:.2f}x")

if len(fgf7_deseq) > 0:
    fgf7_row = fgf7_deseq.iloc[0]
    all_stats['deseq2_fgf7_log2fc'] = fgf7_row['log2FoldChange']
    all_stats['deseq2_fgf7_padj'] = fgf7_row['padj']
    deseq2_key_genes.append({'Gene': 'FGF7', 'log2FoldChange': fgf7_row['log2FoldChange'],
                             'pvalue': fgf7_row['pvalue'], 'padj': fgf7_row['padj']})
    print(f"FGF7:  log2FC = {all_stats['deseq2_fgf7_log2fc']:.3f}, padj = {all_stats['deseq2_fgf7_padj']:.4f}")

# RET (GDF15 receptor ligand - expected NOT induced)
if len(ret_deseq) > 0:
    ret_row = ret_deseq.iloc[0]
    all_stats['deseq2_ret_log2fc'] = ret_row['log2FoldChange']
    all_stats['deseq2_ret_padj'] = ret_row['padj']
    deseq2_key_genes.append({'Gene': 'RET', 'log2FoldChange': ret_row['log2FoldChange'],
                             'pvalue': ret_row['pvalue'], 'padj': ret_row['padj']})
    print(f"RET:   log2FC = {all_stats['deseq2_ret_log2fc']:.3f}, padj = {all_stats['deseq2_ret_padj']:.4f} (expected ~0.997)")

# Manuscript expected: log2FC = 1.22, q = 0.007
print(f"\nManuscript expected: log2FC = 1.22, q = 0.007")
print(f"Computed:            log2FC = {all_stats['deseq2_gdf15_log2fc']:.2f}, q = {all_stats['deseq2_gdf15_padj']:.4f}")

# =============================================================================
# 2. EARLY-STAGE COHORT (GDF15 INDUCTION)
# =============================================================================
print("\n" + "=" * 70)
print("2. EARLY-STAGE COHORT (SBRT alone)")
print("=" * 70)

if early_parquet.exists() and early_manifest.exists():
    early_npx = pd.read_parquet(early_parquet)
    manifest = pd.read_excel(early_manifest)

    # Exclude COSINR patients
    early_npx = early_npx[~early_npx['SampleID'].str.contains('170547', na=False)]

    # Get GDF15 column
    gdf15_col = [c for c in early_npx.columns if 'GDF15' in c.upper() or c == 'GDF15']
    if gdf15_col:
        gdf15_col = gdf15_col[0]

        # Match to manifest for timepoints
        merged = pd.merge(early_npx, manifest, left_on='SampleID', right_on='SampleID', how='inner')

        # Get pre and post values per patient
        if 'Timepoint' in merged.columns or 'Visit' in merged.columns:
            time_col = 'Timepoint' if 'Timepoint' in merged.columns else 'Visit'
            patient_col = 'SubjectID' if 'SubjectID' in merged.columns else 'PatientID'

            if patient_col not in merged.columns:
                for c in merged.columns:
                    if 'subject' in c.lower() or 'patient' in c.lower():
                        patient_col = c
                        break

            # Pivot to get pre/post
            pre_mask = merged[time_col].astype(str).str.contains('1|pre|baseline', case=False, na=False)
            post_mask = merged[time_col].astype(str).str.contains('2|post|follow', case=False, na=False)

            pre_data = merged[pre_mask].groupby(patient_col)[gdf15_col].first()
            post_data = merged[post_mask].groupby(patient_col)[gdf15_col].first()

            # Match patients with both timepoints
            common_patients = pre_data.index.intersection(post_data.index)
            pre_vals = pre_data.loc[common_patients]
            post_vals = post_data.loc[common_patients]

            if len(common_patients) > 5:
                change = post_vals - pre_vals
                t_stat, p_val = stats.ttest_rel(post_vals, pre_vals)

                all_stats['early_n'] = len(common_patients)
                all_stats['early_delta_npx'] = change.mean()
                all_stats['early_delta_npx_ci_low'] = change.mean() - 1.96 * change.std() / np.sqrt(len(change))
                all_stats['early_delta_npx_ci_high'] = change.mean() + 1.96 * change.std() / np.sqrt(len(change))
                all_stats['early_p_value'] = p_val
                all_stats['early_pct_increased'] = (change > 0).mean() * 100
                all_stats['early_fold_change'] = 2 ** change.mean()

                print(f"n = {all_stats['early_n']}")
                print(f"ΔNPX = {all_stats['early_delta_npx']:.2f} (95% CI: {all_stats['early_delta_npx_ci_low']:.2f} to {all_stats['early_delta_npx_ci_high']:.2f})")
                print(f"p = {all_stats['early_p_value']:.2e}")
                print(f"Patients increased: {all_stats['early_pct_increased']:.0f}%")

                print(f"\nManuscript expected: ΔNPX = +0.52, p = 1.5e-05, 76% increased")

# =============================================================================
# 3. TUMOR-BLOOD CORRELATIONS
# =============================================================================
print("\n" + "=" * 70)
print("3. TUMOR-BLOOD CORRELATIONS")
print("=" * 70)

# Load tumor-blood correlation file if exists (check both possible locations)
tb_corr_file = BASE_DIR / "GDF15_Analysis" / "results" / "tumor_blood_correlations.csv"
if not tb_corr_file.exists():
    tb_corr_file = BASE_DIR / "tumor_blood_gdf15_correlation.csv"

tumor_blood_results = []
if tb_corr_file.exists():
    tb_corr = pd.read_csv(tb_corr_file)

    for _, row in tb_corr.iterrows():
        comparison = row['comparison'].replace(' ', '_').replace('/', '_')
        all_stats[f'tumor_blood_{comparison}_n'] = row['n']
        all_stats[f'tumor_blood_{comparison}_rho'] = row['spearman_rho']
        all_stats[f'tumor_blood_{comparison}_p'] = row['p_value']
        tumor_blood_results.append({
            'Comparison': row['comparison'],
            'n': row['n'],
            'Spearman_rho': row['spearman_rho'],
            'p_value': row['p_value']
        })
        print(f"{row['comparison']}: ρ = {row['spearman_rho']:.2f}, p = {row['p_value']:.4f}, n = {row['n']}")

    print(f"\nManuscript expected:")
    print(f"  Baseline:     ρ = 0.11, p = 0.65, n = 20")
    print(f"  On-treatment: ρ = 0.70, p = 0.025, n = 10")
    print(f"  Change:       ρ = 0.64, p = 0.048, n = 10")
else:
    print("Tumor-blood correlation file not found")

# =============================================================================
# 4. SURVIVAL ANALYSES
# =============================================================================
print("\n" + "=" * 70)
print("4. SURVIVAL ANALYSES")
print("=" * 70)

# Prepare survival data
surv_cols = ['id', 'p.GDF15.T1', 'p.GDF15.T3', 'p.GDF15.dif1v3',
             'os_time', 'event_death', 'arm', 'best_recist_bin']
# Add optional columns if they exist
if 'Age' in cosinr.columns:
    surv_cols.append('Age')
if 'bcf.PDL1_bin2' in cosinr.columns:
    surv_cols.append('bcf.PDL1_bin2')

surv_df = cosinr[[c for c in surv_cols if c in cosinr.columns]].copy()
if 'Age' in surv_df.columns:
    surv_df = surv_df.rename(columns={'Age': 'age'})
if 'bcf.PDL1_bin2' in surv_df.columns:
    surv_df = surv_df.rename(columns={'bcf.PDL1_bin2': 'pdl1'})
surv_df = surv_df.dropna(subset=['os_time', 'event_death'])

# --- Baseline GDF15 ---
baseline_df = surv_df.dropna(subset=['p.GDF15.T1']).copy()
baseline_df['gdf15_std'] = (baseline_df['p.GDF15.T1'] - baseline_df['p.GDF15.T1'].mean()) / baseline_df['p.GDF15.T1'].std()

cph = CoxPHFitter()
cph.fit(baseline_df[['os_time', 'event_death', 'gdf15_std']], duration_col='os_time', event_col='event_death')

all_stats['baseline_n'] = len(baseline_df)
all_stats['baseline_events'] = baseline_df['event_death'].sum()
all_stats['baseline_hr'] = np.exp(cph.params_['gdf15_std'])
all_stats['baseline_hr_ci_low'] = np.exp(cph.confidence_intervals_.iloc[0, 0])
all_stats['baseline_hr_ci_high'] = np.exp(cph.confidence_intervals_.iloc[0, 1])
all_stats['baseline_p'] = cph.summary.loc['gdf15_std', 'p']

print(f"\nBaseline GDF15 (n={all_stats['baseline_n']}, events={all_stats['baseline_events']}):")
print(f"  HR = {all_stats['baseline_hr']:.2f} (95% CI: {all_stats['baseline_hr_ci_low']:.2f}-{all_stats['baseline_hr_ci_high']:.2f})")
print(f"  p = {all_stats['baseline_p']:.4f}")

# Dichotomized at median
median_gdf15 = baseline_df['p.GDF15.T1'].median()
all_stats['baseline_median_cutoff'] = median_gdf15

high = baseline_df[baseline_df['p.GDF15.T1'] > median_gdf15]
low = baseline_df[baseline_df['p.GDF15.T1'] <= median_gdf15]

kmf_high = KaplanMeierFitter()
kmf_low = KaplanMeierFitter()
kmf_high.fit(high['os_time'], high['event_death'])
kmf_low.fit(low['os_time'], low['event_death'])

lr = logrank_test(high['os_time'], low['os_time'], high['event_death'], low['event_death'])

all_stats['baseline_high_n'] = len(high)
all_stats['baseline_low_n'] = len(low)
all_stats['baseline_high_median_os'] = kmf_high.median_survival_time_
all_stats['baseline_low_median_os'] = kmf_low.median_survival_time_
all_stats['baseline_logrank_p'] = lr.p_value

print(f"\nDichotomized (median = {median_gdf15:.2f}):")
print(f"  High (n={all_stats['baseline_high_n']}): median OS = {all_stats['baseline_high_median_os']:.1f} months")
print(f"  Low (n={all_stats['baseline_low_n']}): median OS = {all_stats['baseline_low_median_os']:.1f} months")
print(f"  Log-rank p = {all_stats['baseline_logrank_p']:.4f}")

print(f"\nManuscript expected: HR = 1.59, p = 0.008; High vs Low: 19.0 vs 42.5 mo, p = 0.007")

# --- GDF15 Change ---
print("\n--- GDF15 Change ---")
change_df = surv_df.dropna(subset=['p.GDF15.dif1v3']).copy()
change_df['change_std'] = (change_df['p.GDF15.dif1v3'] - change_df['p.GDF15.dif1v3'].mean()) / change_df['p.GDF15.dif1v3'].std()
change_df['increased'] = (change_df['p.GDF15.dif1v3'] > 0).astype(int)

cph = CoxPHFitter()
cph.fit(change_df[['os_time', 'event_death', 'increased']], duration_col='os_time', event_col='event_death')

all_stats['change_n'] = len(change_df)
all_stats['change_events'] = change_df['event_death'].sum()
all_stats['change_hr'] = np.exp(cph.params_['increased'])
all_stats['change_hr_ci_low'] = np.exp(cph.confidence_intervals_.iloc[0, 0])
all_stats['change_hr_ci_high'] = np.exp(cph.confidence_intervals_.iloc[0, 1])
all_stats['change_p'] = cph.summary.loc['increased', 'p']

increased = change_df[change_df['increased'] == 1]
decreased = change_df[change_df['increased'] == 0]

kmf_inc = KaplanMeierFitter()
kmf_dec = KaplanMeierFitter()
kmf_inc.fit(increased['os_time'], increased['event_death'])
kmf_dec.fit(decreased['os_time'], decreased['event_death'])

lr_change = logrank_test(increased['os_time'], decreased['os_time'],
                         increased['event_death'], decreased['event_death'])

all_stats['change_increased_n'] = len(increased)
all_stats['change_decreased_n'] = len(decreased)
all_stats['change_increased_median_os'] = kmf_inc.median_survival_time_
all_stats['change_decreased_median_os'] = kmf_dec.median_survival_time_
all_stats['change_logrank_p'] = lr_change.p_value

print(f"GDF15 Change (n={all_stats['change_n']}, events={all_stats['change_events']}):")
print(f"  HR (increased vs decreased) = {all_stats['change_hr']:.2f} (95% CI: {all_stats['change_hr_ci_low']:.2f}-{all_stats['change_hr_ci_high']:.2f})")
print(f"  p = {all_stats['change_p']:.4f}")
print(f"  Increased (n={all_stats['change_increased_n']}): median OS = {all_stats['change_increased_median_os']:.1f} months")
print(f"  Decreased (n={all_stats['change_decreased_n']}): median OS = {all_stats['change_decreased_median_os']}")
print(f"  Log-rank p = {all_stats['change_logrank_p']:.4f}")

print(f"\nManuscript expected: HR = 2.40, p = 0.036; 19.9 vs NR months, p = 0.033")

# --- Four-Group Stratification ---
print("\n--- Four-Group Stratification ---")
four_df = surv_df.dropna(subset=['p.GDF15.T1', 'p.GDF15.dif1v3']).copy()
median_baseline = four_df['p.GDF15.T1'].median()

four_df['baseline_high'] = four_df['p.GDF15.T1'] > median_baseline
four_df['change_pos'] = four_df['p.GDF15.dif1v3'] > 0

four_df['four_group'] = 0
four_df.loc[(~four_df['baseline_high']) & (~four_df['change_pos']), 'four_group'] = 1  # Low/Decreased (best)
four_df.loc[(~four_df['baseline_high']) & (four_df['change_pos']), 'four_group'] = 2   # Low/Increased
four_df.loc[(four_df['baseline_high']) & (~four_df['change_pos']), 'four_group'] = 3   # High/Decreased
four_df.loc[(four_df['baseline_high']) & (four_df['change_pos']), 'four_group'] = 4    # High/Increased (worst)

mlr = multivariate_logrank_test(four_df['os_time'], four_df['four_group'], four_df['event_death'])
all_stats['fourgroup_n'] = len(four_df)
all_stats['fourgroup_logrank_p'] = mlr.p_value

# Best vs worst
best = four_df[four_df['four_group'] == 1]
worst = four_df[four_df['four_group'] == 4]

kmf_best = KaplanMeierFitter()
kmf_worst = KaplanMeierFitter()
kmf_best.fit(best['os_time'], best['event_death'])
kmf_worst.fit(worst['os_time'], worst['event_death'])

lr_bw = logrank_test(best['os_time'], worst['os_time'], best['event_death'], worst['event_death'])

all_stats['fourgroup_best_n'] = len(best)
all_stats['fourgroup_worst_n'] = len(worst)
all_stats['fourgroup_best_median_os'] = kmf_best.median_survival_time_
all_stats['fourgroup_worst_median_os'] = kmf_worst.median_survival_time_
all_stats['fourgroup_best_vs_worst_p'] = lr_bw.p_value

for g in [1, 2, 3, 4]:
    grp = four_df[four_df['four_group'] == g]
    kmf = KaplanMeierFitter()
    kmf.fit(grp['os_time'], grp['event_death'])
    all_stats[f'fourgroup_g{g}_n'] = len(grp)
    all_stats[f'fourgroup_g{g}_events'] = grp['event_death'].sum()
    all_stats[f'fourgroup_g{g}_median_os'] = kmf.median_survival_time_

print(f"Four-group (n={all_stats['fourgroup_n']}):")
print(f"  G1 (Low/Decreased): n={all_stats['fourgroup_g1_n']}, median OS = {all_stats['fourgroup_g1_median_os']}")
print(f"  G2 (Low/Increased): n={all_stats['fourgroup_g2_n']}, median OS = {all_stats['fourgroup_g2_median_os']:.1f}")
print(f"  G3 (High/Decreased): n={all_stats['fourgroup_g3_n']}, median OS = {all_stats['fourgroup_g3_median_os']:.1f}")
print(f"  G4 (High/Increased): n={all_stats['fourgroup_g4_n']}, median OS = {all_stats['fourgroup_g4_median_os']:.1f}")
print(f"  Overall log-rank p = {all_stats['fourgroup_logrank_p']:.4f}")
print(f"  Best vs Worst p = {all_stats['fourgroup_best_vs_worst_p']:.4f}")

print(f"\nManuscript expected: 61 vs 11 months, p = 0.0015")

# --- Landmark Analysis (6 months) ---
print("\n--- Landmark Analysis (6 months) ---")
landmark_df = baseline_df[baseline_df['os_time'] > 6].copy()
landmark_df['os_time_landmark'] = landmark_df['os_time'] - 6

cph = CoxPHFitter()
cph.fit(landmark_df[['os_time_landmark', 'event_death', 'gdf15_std']],
        duration_col='os_time_landmark', event_col='event_death')

all_stats['landmark_n'] = len(landmark_df)
all_stats['landmark_excluded'] = len(baseline_df) - len(landmark_df)
all_stats['landmark_hr'] = np.exp(cph.params_['gdf15_std'])
all_stats['landmark_hr_ci_low'] = np.exp(cph.confidence_intervals_.iloc[0, 0])
all_stats['landmark_hr_ci_high'] = np.exp(cph.confidence_intervals_.iloc[0, 1])
all_stats['landmark_p'] = cph.summary.loc['gdf15_std', 'p']

print(f"Landmark (n={all_stats['landmark_n']}, excluded {all_stats['landmark_excluded']}):")
print(f"  HR = {all_stats['landmark_hr']:.2f} (95% CI: {all_stats['landmark_hr_ci_low']:.2f}-{all_stats['landmark_hr_ci_high']:.2f})")
print(f"  p = {all_stats['landmark_p']:.4f}")

print(f"\nManuscript expected: HR = 1.61, p = 0.049, n = 46")

# --- Multivariable Cox ---
print("\n--- Multivariable Cox ---")
mv_df = baseline_df.dropna(subset=['age', 'arm']).copy()
mv_df['arm_binary'] = (mv_df['arm'] == 'Sequential').astype(int)

cph = CoxPHFitter()
cph.fit(mv_df[['os_time', 'event_death', 'gdf15_std', 'age', 'arm_binary']],
        duration_col='os_time', event_col='event_death')

all_stats['multivariable_n'] = len(mv_df)
all_stats['multivariable_hr'] = np.exp(cph.params_['gdf15_std'])
all_stats['multivariable_hr_ci_low'] = np.exp(cph.confidence_intervals_.loc['gdf15_std'].iloc[0])
all_stats['multivariable_hr_ci_high'] = np.exp(cph.confidence_intervals_.loc['gdf15_std'].iloc[1])
all_stats['multivariable_p'] = cph.summary.loc['gdf15_std', 'p']

print(f"Multivariable (n={all_stats['multivariable_n']}):")
print(f"  HR (adjusted for age, arm) = {all_stats['multivariable_hr']:.2f} (95% CI: {all_stats['multivariable_hr_ci_low']:.2f}-{all_stats['multivariable_hr_ci_high']:.2f})")
print(f"  p = {all_stats['multivariable_p']:.4f}")

print(f"\nManuscript expected: HR = 1.71, p = 0.016")

# =============================================================================
# 5. TREATMENT ARM STRATIFICATION
# =============================================================================
print("\n" + "=" * 70)
print("5. TREATMENT ARM STRATIFICATION")
print("=" * 70)

forest_results = []

for arm_name in ['Sequential', 'Concurrent']:
    arm_df = baseline_df[baseline_df['arm'] == arm_name].copy()
    arm_df['gdf15_std'] = (arm_df['p.GDF15.T1'] - arm_df['p.GDF15.T1'].mean()) / arm_df['p.GDF15.T1'].std()

    cph = CoxPHFitter()
    cph.fit(arm_df[['os_time', 'event_death', 'gdf15_std']], duration_col='os_time', event_col='event_death')

    hr = np.exp(cph.params_['gdf15_std'])
    ci_low = np.exp(cph.confidence_intervals_.iloc[0, 0])
    ci_high = np.exp(cph.confidence_intervals_.iloc[0, 1])
    p = cph.summary.loc['gdf15_std', 'p']

    all_stats[f'{arm_name.lower()}_baseline_n'] = len(arm_df)
    all_stats[f'{arm_name.lower()}_baseline_hr'] = hr
    all_stats[f'{arm_name.lower()}_baseline_hr_ci_low'] = ci_low
    all_stats[f'{arm_name.lower()}_baseline_hr_ci_high'] = ci_high
    all_stats[f'{arm_name.lower()}_baseline_p'] = p

    forest_results.append((f'{arm_name} Baseline', hr, ci_low, ci_high, p, None))

    print(f"\n{arm_name} Baseline (n={len(arm_df)}):")
    print(f"  HR = {hr:.2f} (95% CI: {ci_low:.2f}-{ci_high:.2f}), p = {p:.4f}")

# Change analysis by arm
for arm_name in ['Sequential', 'Concurrent']:
    arm_change_df = change_df[change_df['arm'] == arm_name].copy()
    if len(arm_change_df) > 5:
        arm_change_df['change_std'] = (arm_change_df['p.GDF15.dif1v3'] - arm_change_df['p.GDF15.dif1v3'].mean()) / arm_change_df['p.GDF15.dif1v3'].std()

        cph = CoxPHFitter()
        cph.fit(arm_change_df[['os_time', 'event_death', 'change_std']], duration_col='os_time', event_col='event_death')

        hr = np.exp(cph.params_['change_std'])
        ci_low = np.exp(cph.confidence_intervals_.iloc[0, 0])
        ci_high = np.exp(cph.confidence_intervals_.iloc[0, 1])
        p = cph.summary.loc['change_std', 'p']

        all_stats[f'{arm_name.lower()}_change_n'] = len(arm_change_df)
        all_stats[f'{arm_name.lower()}_change_hr'] = hr
        all_stats[f'{arm_name.lower()}_change_hr_ci_low'] = ci_low
        all_stats[f'{arm_name.lower()}_change_hr_ci_high'] = ci_high
        all_stats[f'{arm_name.lower()}_change_p'] = p

        forest_results.append((f'{arm_name} Change', hr, ci_low, ci_high, p, None))

        print(f"\n{arm_name} Change (n={len(arm_change_df)}):")
        print(f"  HR = {hr:.2f} (95% CI: {ci_low:.2f}-{ci_high:.2f}), p = {p:.4f}")

# Interaction test
print("\n--- Interaction Test ---")
int_df = baseline_df.dropna(subset=['arm']).copy()
int_df['arm_binary'] = (int_df['arm'] == 'Sequential').astype(int)
int_df['interaction'] = int_df['gdf15_std'] * int_df['arm_binary']

cph = CoxPHFitter()
cph.fit(int_df[['os_time', 'event_death', 'gdf15_std', 'arm_binary', 'interaction']],
        duration_col='os_time', event_col='event_death')

hr_int = np.exp(cph.params_['interaction'])
ci_int = cph.confidence_intervals_.loc['interaction']
ci_low_int = np.exp(ci_int.iloc[0])
ci_high_int = np.exp(ci_int.iloc[1])
p_two = cph.summary.loc['interaction', 'p']
z_stat = cph.summary.loc['interaction', 'z']
p_one = p_two / 2 if z_stat > 0 else 1 - (p_two / 2)

all_stats['interaction_hr'] = hr_int
all_stats['interaction_hr_ci_low'] = ci_low_int
all_stats['interaction_hr_ci_high'] = ci_high_int
all_stats['interaction_p_two_tailed'] = p_two
all_stats['interaction_p_one_tailed'] = p_one

forest_results.append(('Interaction', hr_int, ci_low_int, ci_high_int, p_two, p_one))

print(f"Interaction (GDF15 x Arm):")
print(f"  HR = {hr_int:.2f} (95% CI: {ci_low_int:.2f}-{ci_high_int:.2f})")
print(f"  Two-tailed p = {p_two:.4f}, One-tailed p = {p_one:.4f}")

print(f"\nManuscript expected: HR = 2.29, p = 0.045")

# Sequential 4-group
print("\n--- Sequential Arm 4-Group ---")
seq_four_df = four_df[four_df['arm'] == 'Sequential'].copy()
if len(seq_four_df) > 10:
    mlr_seq = multivariate_logrank_test(seq_four_df['os_time'], seq_four_df['four_group'], seq_four_df['event_death'])
    all_stats['sequential_fourgroup_n'] = len(seq_four_df)
    all_stats['sequential_fourgroup_p'] = mlr_seq.p_value
    print(f"Sequential 4-group (n={len(seq_four_df)}): p = {mlr_seq.p_value:.4f}")
    print(f"Manuscript expected: p = 0.009")

# =============================================================================
# 6. IMMUNE CORRELATIONS
# =============================================================================
print("\n" + "=" * 70)
print("6. IMMUNE CORRELATIONS")
print("=" * 70)

immune_markers = {
    'TNFRSF10B': 'p.TNFRSF10B.T1',
    'IL4R': 'p.IL4R.T1',
    'TIM3': 'p.HAVCR2.T1',
    'LAG3': 'p.LAG3.T1',
    'TNF': 'p.TNF.T1',
    'IL6': 'p.IL6.T1',
    'CXCL9': 'p.CXCL9.T1',
    'CXCL10': 'p.CXCL10.T1',
    'PD1': 'p.PDCD1.T1',
    'PDL1': 'p.CD274.T1',
}

immune_results = []

for name, col in immune_markers.items():
    if col in cosinr.columns:
        valid = cosinr[['p.GDF15.T1', col]].dropna()
        if len(valid) > 10:
            rho, p = stats.spearmanr(valid['p.GDF15.T1'], valid[col])
            all_stats[f'immune_{name}_rho'] = rho
            all_stats[f'immune_{name}_p'] = p
            all_stats[f'immune_{name}_n'] = len(valid)
            immune_results.append((name, rho, p, len(valid)))
            print(f"{name:12s}: ρ = {rho:.2f}, p = {p:.4f}, n = {len(valid)}")

print(f"\nManuscript expected: TNFRSF10B ρ = 0.74, IL4R ρ = 0.56-0.57, TIM-3 ρ = 0.41-0.59")

# =============================================================================
# 7. PATHWAY CORRELATIONS (including p53)
# =============================================================================
print("\n" + "=" * 70)
print("7. PATHWAY CORRELATIONS (including p53)")
print("=" * 70)

pathway_results = []

if hallmark_path.exists():
    hallmark = pd.read_csv(hallmark_path)

    # Merge with GDF15 change data
    merged = pd.merge(hallmark, cosinr[['id', 'p.GDF15.dif1v3']], on='id', how='inner')
    merged = merged.dropna(subset=['p.GDF15.dif1v3'])

    # Hallmark pathways
    hallmark_pathways = {
        'Angiogenesis': 'gs.HALLMARK_ANGIOGENESIS.dif1v3',
        'EMT': 'gs.HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION.dif1v3',
        'IFN-alpha Response': 'gs.HALLMARK_INTERFERON_ALPHA_RESPONSE.dif1v3',
        'IFN-gamma Response': 'gs.HALLMARK_INTERFERON_GAMMA_RESPONSE.dif1v3',
        'Inflammatory Response': 'gs.HALLMARK_INFLAMMATORY_RESPONSE.dif1v3',
        'IL2-STAT5 Signaling': 'gs.HALLMARK_IL2_STAT5_SIGNALING.dif1v3',
        'IL6-JAK-STAT3 Signaling': 'gs.HALLMARK_IL6_JAK_STAT3_SIGNALING.dif1v3',
        'p53 Pathway': 'gs.HALLMARK_P53_PATHWAY.dif1v3',
    }

    print("\nHallmark pathways (ΔGDF15 vs Δpathway):")
    for name, col in hallmark_pathways.items():
        if col in merged.columns:
            valid = merged[['p.GDF15.dif1v3', col]].dropna()
            if len(valid) > 5:
                rho, p = stats.spearmanr(valid['p.GDF15.dif1v3'], valid[col])
                all_stats[f'pathway_{name.replace(" ", "_").replace("-", "_")}_rho'] = rho
                all_stats[f'pathway_{name.replace(" ", "_").replace("-", "_")}_p'] = p
                pathway_results.append((name, rho, p))
                print(f"  {name:25s}: ρ = {rho:+.3f}, p = {p:.4f}")

    # p53 baseline correlation (blood GDF15 vs tumor p53 ssGSEA)
    print("\n--- p53 Baseline Correlation ---")
    p53_col = 'gs.HALLMARK_P53_PATHWAY.T1'
    if p53_col in hallmark.columns:
        merged_p53 = pd.merge(hallmark[['id', p53_col]], cosinr[['id', 'p.GDF15.T1']], on='id', how='inner')
        merged_p53 = merged_p53.dropna()
        if len(merged_p53) > 10:
            rho, p = stats.spearmanr(merged_p53['p.GDF15.T1'], merged_p53[p53_col])
            all_stats['p53_baseline_rho'] = rho
            all_stats['p53_baseline_p'] = p
            all_stats['p53_baseline_n'] = len(merged_p53)
            pathway_results.append(('p53 Baseline', rho, p))
            print(f"  Blood GDF15 (T1) vs Tumor p53 ssGSEA: ρ = {rho:.2f}, p = {p:.4f}, n = {len(merged_p53)}")
            print(f"  Manuscript expected: ρ = 0.38, p = 0.005")

if reactome_path.exists():
    reactome = pd.read_csv(reactome_path)
    merged_r = pd.merge(reactome, cosinr[['id', 'p.GDF15.dif1v3']], on='id', how='inner')
    merged_r = merged_r.dropna(subset=['p.GDF15.dif1v3'])

    reactome_pathways = {
        'IFN-stimulated genes': 'gs.Antiviral_mechanism_by_IFN_stimulated_genes.dif1v3',
        'TCR Signaling': 'gs.TCR_signaling.dif1v3',
        'MHC-I Processing': 'gs.Class_I_MHC_mediated_antigen_processing_n_presentation.dif1v3',
        'B Cell Receptor': 'gs.Signaling_by_the_B_Cell_Receptor_BCR.dif1v3',
        'IL-10 Signaling': 'gs.Interleukin_10_signaling.dif1v3',
        'IL-4/IL-13 Signaling': 'gs.Interleukin_4_and_Interleukin_13_signaling.dif1v3',
    }

    print("\nReactome immune pathways (ΔGDF15 vs Δpathway):")
    for name, col in reactome_pathways.items():
        if col in merged_r.columns:
            valid = merged_r[['p.GDF15.dif1v3', col]].dropna()
            if len(valid) > 5:
                rho, p = stats.spearmanr(valid['p.GDF15.dif1v3'], valid[col])
                all_stats[f'pathway_{name.replace(" ", "_").replace("-", "_").replace("/", "_")}_rho'] = rho
                all_stats[f'pathway_{name.replace(" ", "_").replace("-", "_").replace("/", "_")}_p'] = p
                pathway_results.append((name, rho, p))
                print(f"  {name:25s}: ρ = {rho:+.3f}, p = {p:.4f}")

print(f"\nManuscript expected:")
print(f"  MHC-I Processing: ρ = -0.56, p = 0.0001")
print(f"  IL-10 Signaling:  ρ = +0.56, p = 0.0001")
print(f"  p53 Pathway:      ρ = +0.38, p = 0.005")

# =============================================================================
# 8. RESPONDERS-ONLY ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("8. RESPONDERS-ONLY ANALYSIS")
print("=" * 70)

resp_df = surv_df[surv_df['best_recist_bin'] == 'CR/PR'].copy()
resp_df = resp_df.dropna(subset=['p.GDF15.T1', 'os_time', 'event_death'])
resp_df['gdf15_std'] = (resp_df['p.GDF15.T1'] - resp_df['p.GDF15.T1'].mean()) / resp_df['p.GDF15.T1'].std()

if len(resp_df) > 10:
    cph = CoxPHFitter()
    cph.fit(resp_df[['os_time', 'event_death', 'gdf15_std']], duration_col='os_time', event_col='event_death')

    all_stats['responders_baseline_n'] = len(resp_df)
    all_stats['responders_baseline_hr'] = np.exp(cph.params_['gdf15_std'])
    all_stats['responders_baseline_hr_ci_low'] = np.exp(cph.confidence_intervals_.iloc[0, 0])
    all_stats['responders_baseline_hr_ci_high'] = np.exp(cph.confidence_intervals_.iloc[0, 1])
    all_stats['responders_baseline_p'] = cph.summary.loc['gdf15_std', 'p']

    print(f"Responders Baseline (n={all_stats['responders_baseline_n']}):")
    print(f"  HR = {all_stats['responders_baseline_hr']:.2f} (95% CI: {all_stats['responders_baseline_hr_ci_low']:.2f}-{all_stats['responders_baseline_hr_ci_high']:.2f})")
    print(f"  p = {all_stats['responders_baseline_p']:.4f}")

# Responders change
resp_change_df = surv_df[surv_df['best_recist_bin'] == 'CR/PR'].copy()
resp_change_df = resp_change_df.dropna(subset=['p.GDF15.dif1v3', 'os_time', 'event_death'])
resp_change_df['change_std'] = (resp_change_df['p.GDF15.dif1v3'] - resp_change_df['p.GDF15.dif1v3'].mean()) / resp_change_df['p.GDF15.dif1v3'].std()

if len(resp_change_df) > 10:
    cph = CoxPHFitter()
    cph.fit(resp_change_df[['os_time', 'event_death', 'change_std']], duration_col='os_time', event_col='event_death')

    all_stats['responders_change_n'] = len(resp_change_df)
    all_stats['responders_change_hr'] = np.exp(cph.params_['change_std'])
    all_stats['responders_change_hr_ci_low'] = np.exp(cph.confidence_intervals_.iloc[0, 0])
    all_stats['responders_change_hr_ci_high'] = np.exp(cph.confidence_intervals_.iloc[0, 1])
    all_stats['responders_change_p'] = cph.summary.loc['change_std', 'p']

    print(f"\nResponders Change (n={all_stats['responders_change_n']}):")
    print(f"  HR = {all_stats['responders_change_hr']:.2f} (95% CI: {all_stats['responders_change_hr_ci_low']:.2f}-{all_stats['responders_change_hr_ci_high']:.2f})")
    print(f"  p = {all_stats['responders_change_p']:.4f}")

print(f"\nManuscript expected: Baseline HR = 2.73, p = 0.016; Change HR = 2.12, p = 0.032")

# =============================================================================
# 9. FLOW CYTOMETRY (Expected NULL)
# =============================================================================
print("\n" + "=" * 70)
print("9. FLOW CYTOMETRY CORRELATIONS (Expected: NOT significant)")
print("=" * 70)

# Correct column names for flow cytometry cell populations
# Using BASELINE GDF15 (T1) vs immune cell frequencies at baseline
flow_cols = {
    'CD8+ T': 'fc.CD3p_CD56n_CD8p.T1',
    'CD4+ T': 'fc.CD3p_CD56n_CD4p.T1',
    'Treg': 'fc.CD3p_CD56n_CD4p_Treg.T1',
    'M-MDSC': 'fc.M_MDSC.T1',
    'Monocytes': 'fc.classical_mono.T1',
    'DC': 'fc.DC.T1',
    'NK': 'fc.NK.T1',
}

# Use COMPLETE CASES to ensure uniform sample size across all cell types
# This matches manuscript claim of n ≈ 38-40 uniformly
all_flow_cols_list = list(flow_cols.values())
existing_flow_cols = [c for c in all_flow_cols_list if c in cosinr.columns]
complete_flow_df = cosinr[['p.GDF15.T1'] + existing_flow_cols].dropna()
print(f"Using complete cases: n = {len(complete_flow_df)} (uniform across all cell types)")

flow_results = []
for name, col in flow_cols.items():
    if col in complete_flow_df.columns:
        rho, p = stats.spearmanr(complete_flow_df['p.GDF15.T1'], complete_flow_df[col])
        all_stats[f'flow_{name.replace(" ", "_").replace("+", "")}_rho'] = rho
        all_stats[f'flow_{name.replace(" ", "_").replace("+", "")}_p'] = p
        flow_results.append({'Cell_Type': name, 'n': len(complete_flow_df), 'Spearman_rho': rho, 'p_value': p})
        print(f"{name:12s}: ρ = {rho:.2f}, p = {p:.4f}, n = {len(complete_flow_df)}")

# Apply FDR correction
if flow_results:
    from statsmodels.stats.multitest import fdrcorrection
    flow_df = pd.DataFrame(flow_results)
    _, fdr_q = fdrcorrection(flow_df['p_value'].values)
    flow_df['FDR_q'] = fdr_q

    print(f"\nWith FDR correction:")
    for _, row in flow_df.iterrows():
        print(f"  {row['Cell_Type']:12s}: FDR q = {row['FDR_q']:.4f}")

    print(f"\nAll FDR q > 0.49? {all(flow_df['FDR_q'] > 0.49)}")
    all_stats['flow_min_fdr_q'] = flow_df['FDR_q'].min()

print(f"\nManuscript expected: all FDR-adjusted p > 0.49")

# =============================================================================
# 10. PFS ANALYSIS (Expected NULL)
# =============================================================================
print("\n" + "=" * 70)
print("10. PFS ANALYSIS (Expected: NOT significant)")
print("=" * 70)

pfs_cols = ['pfs_time', 'event_prog']  # event_prog not event_progression
if all(c in cosinr.columns for c in pfs_cols):
    pfs_df = cosinr[['p.GDF15.T1', 'p.GDF15.dif1v3', 'pfs_time', 'event_prog']].dropna(subset=['pfs_time', 'event_prog', 'p.GDF15.T1'])
    pfs_df['gdf15_std'] = (pfs_df['p.GDF15.T1'] - pfs_df['p.GDF15.T1'].mean()) / pfs_df['p.GDF15.T1'].std()

    if len(pfs_df) > 10:
        cph = CoxPHFitter()
        cph.fit(pfs_df[['pfs_time', 'event_prog', 'gdf15_std']],
                duration_col='pfs_time', event_col='event_prog')

        all_stats['pfs_baseline_hr'] = np.exp(cph.params_['gdf15_std'])
        all_stats['pfs_baseline_p'] = cph.summary.loc['gdf15_std', 'p']

        print(f"Baseline GDF15 → PFS: HR = {all_stats['pfs_baseline_hr']:.2f}, p = {all_stats['pfs_baseline_p']:.4f}")

    # Change
    pfs_change_df = pfs_df.dropna(subset=['p.GDF15.dif1v3'])
    if len(pfs_change_df) > 10:
        pfs_change_df['change_std'] = (pfs_change_df['p.GDF15.dif1v3'] - pfs_change_df['p.GDF15.dif1v3'].mean()) / pfs_change_df['p.GDF15.dif1v3'].std()

        cph = CoxPHFitter()
        cph.fit(pfs_change_df[['pfs_time', 'event_prog', 'change_std']],
                duration_col='pfs_time', event_col='event_prog')

        all_stats['pfs_change_hr'] = np.exp(cph.params_['change_std'])
        all_stats['pfs_change_p'] = cph.summary.loc['change_std', 'p']

        print(f"ΔGDF15 → PFS: HR = {all_stats['pfs_change_hr']:.2f}, p = {all_stats['pfs_change_p']:.4f}")

    print(f"\nManuscript expected: both p > 0.35")
else:
    print("PFS columns not found")

# =============================================================================
# SAVE ALL STATISTICS
# =============================================================================
print("\n" + "=" * 70)
print("SAVING ALL STATISTICS")
print("=" * 70)

# Save main statistics to CSV
stats_df = pd.DataFrame(list(all_stats.items()), columns=['Statistic', 'Value'])
stats_df.to_csv(BASE_DIR / 'all_computed_statistics.csv', index=False)
print(f"Saved {len(stats_df)} statistics to: {BASE_DIR / 'all_computed_statistics.csv'}")

# Save forest plot values
forest_df = pd.DataFrame(forest_results, columns=['Analysis', 'HR', 'CI_Low', 'CI_High', 'p_value', 'p_one_tailed'])
forest_df.to_csv(BASE_DIR / 'verified_forest_plot_values.csv', index=False)
print(f"Saved forest plot values to: {BASE_DIR / 'verified_forest_plot_values.csv'}")

# Save pathway correlations
pathway_df = pd.DataFrame(pathway_results, columns=['Pathway', 'Spearman_rho', 'p_value'])
pathway_df.to_csv(BASE_DIR / 'verified_pathway_correlations.csv', index=False)
print(f"Saved pathway correlations to: {BASE_DIR / 'verified_pathway_correlations.csv'}")

# Save immune correlations
immune_df = pd.DataFrame(immune_results, columns=['Marker', 'Spearman_rho', 'p_value', 'n'])
immune_df.to_csv(BASE_DIR / 'verified_immune_correlations.csv', index=False)
print(f"Saved immune correlations to: {BASE_DIR / 'verified_immune_correlations.csv'}")

# Save flow cytometry results (flow_df already created with FDR correction above)
if flow_results:
    # flow_df already has: Cell_Type, n, Spearman_rho, p_value, FDR_q
    flow_df.to_csv(BASE_DIR / 'verified_flow_cytometry.csv', index=False)
    print(f"Saved flow cytometry results to: {BASE_DIR / 'verified_flow_cytometry.csv'}")

# Save tumor-blood correlations
if tumor_blood_results:
    tb_df = pd.DataFrame(tumor_blood_results)
    tb_df.to_csv(BASE_DIR / 'verified_tumor_blood_correlations.csv', index=False)
    print(f"Saved tumor-blood correlations to: {BASE_DIR / 'verified_tumor_blood_correlations.csv'}")

# Save key DESeq2 genes (GDF15, FGF7, RET)
if deseq2_key_genes:
    deseq_df = pd.DataFrame(deseq2_key_genes)
    deseq_df.to_csv(BASE_DIR / 'verified_deseq2_key_genes.csv', index=False)
    print(f"Saved DESeq2 key genes to: {BASE_DIR / 'verified_deseq2_key_genes.csv'}")

print("\n" + "=" * 70)
print("ALL STATISTICS COMPUTED AND SAVED")
print("=" * 70)
