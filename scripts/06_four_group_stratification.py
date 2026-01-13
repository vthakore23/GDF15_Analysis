#!/usr/bin/env python3
"""
06_four_group_stratification.py
===============================
Four-group survival stratification based on baseline GDF15 and treatment-induced change.

Groups:
1. Low baseline, Decreased
2. Low baseline, Increased
3. High baseline, Decreased
4. High baseline, Increased

Analyses:
- Compare survival across all 4 groups (multivariate log-rank test)
- Pairwise comparison of extreme groups (Group 4 vs Group 1)
- Univariate and multivariable Cox regression (adjusted for age and treatment arm)
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Define paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR
RESULTS_DIR = BASE_DIR / "GDF15_Analysis" / "results"
FIGURES_DIR = BASE_DIR / "GDF15_Analysis" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def create_four_groups(cosinr, time_col, event_col):
    """Create 4-group stratification."""
    print("=" * 70)
    print("FOUR-GROUP STRATIFICATION")
    print("=" * 70)

    # Filter for patients with paired samples
    df = cosinr[['id', 'p.GDF15.T1', 'p.GDF15.T3', 'Age', 'arm', time_col, event_col]].dropna().copy()
    df.columns = ['id', 'GDF15_T1', 'GDF15_T3', 'Age', 'arm', 'time', 'event']

    # Calculate change
    df['GDF15_change'] = df['GDF15_T3'] - df['GDF15_T1']

    # Dichotomize baseline by median
    median_baseline = df['GDF15_T1'].median()
    df['baseline_high'] = (df['GDF15_T1'] >= median_baseline).astype(int)

    # Dichotomize change: increased vs decreased
    df['change_increased'] = (df['GDF15_change'] > 0).astype(int)

    # Create 4 groups
    df['group'] = 0
    df.loc[(df['baseline_high'] == 0) & (df['change_increased'] == 0), 'group'] = 1  # Low + Decreased
    df.loc[(df['baseline_high'] == 0) & (df['change_increased'] == 1), 'group'] = 2  # Low + Increased
    df.loc[(df['baseline_high'] == 1) & (df['change_increased'] == 0), 'group'] = 3  # High + Decreased
    df.loc[(df['baseline_high'] == 1) & (df['change_increased'] == 1), 'group'] = 4  # High + Increased

    print(f"\nMedian baseline GDF15: {median_baseline:.2f} NPX")
    print(f"\nGroup distribution:")
    for g in [1, 2, 3, 4]:
        n = (df['group'] == g).sum()
        print(f"  Group {g}: n = {n}")

    group_labels = {
        1: 'Low baseline, Decreased',
        2: 'Low baseline, Increased',
        3: 'High baseline, Decreased',
        4: 'High baseline, Increased'
    }

    return df, group_labels


def kaplan_meier_four_groups(df, group_labels):
    """Kaplan-Meier analysis for 4 groups."""
    print("\n" + "=" * 70)
    print("KAPLAN-MEIER ANALYSIS: 4-GROUP STRATIFICATION")
    print("=" * 70)

    # Overall log-rank test
    groups = [df[df['group'] == g] for g in [1, 2, 3, 4] if len(df[df['group'] == g]) > 0]
    times = [g['time'].values for g in groups]
    events = [g['event'].values for g in groups]

    # Multivariate log-rank
    result = multivariate_logrank_test(df['time'], df['group'], df['event'])
    print(f"\nOverall log-rank test p-value: {result.p_value:.4f}")

    # Pairwise: Group 1 vs Group 4
    g1 = df[df['group'] == 1]
    g4 = df[df['group'] == 4]

    if len(g1) > 0 and len(g4) > 0:
        lr_1v4 = logrank_test(g1['time'], g4['time'], g1['event'], g4['event'])
        print(f"Group 1 vs Group 4 log-rank p-value: {lr_1v4.p_value:.4f}")

    # Create KM plot
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {1: 'blue', 2: 'green', 3: 'orange', 4: 'red'}
    kmf = KaplanMeierFitter()

    for g in [1, 2, 3, 4]:
        group_data = df[df['group'] == g]
        if len(group_data) > 0:
            kmf.fit(group_data['time'], group_data['event'],
                   label=f'{group_labels[g]} (n={len(group_data)})')
            kmf.plot_survival_function(ax=ax, ci_show=True, color=colors[g])

    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Overall Survival Probability')
    ax.set_title(f'4-Group GDF15 Stratification\nOverall log-rank p = {result.p_value:.4f}')
    ax.legend(loc='lower left', fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'KM_four_group_stratification.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'KM_four_group_stratification.pdf', bbox_inches='tight')
    plt.close()

    print(f"\nFigure saved to: {FIGURES_DIR / 'KM_four_group_stratification.png'}")

    return result.p_value


def cox_four_groups(df):
    """Cox regression for 4-group stratification."""
    print("\n" + "=" * 70)
    print("COX REGRESSION: 4-GROUP STRATIFICATION")
    print("=" * 70)

    results = []

    # Compare Group 4 to Group 1 (reference)
    df_compare = df[df['group'].isin([1, 4])].copy()
    df_compare['group4'] = (df_compare['group'] == 4).astype(int)

    # Univariate
    print("\n1. UNIVARIATE (Group 4 vs Group 1)")
    print("-" * 50)

    cph = CoxPHFitter()
    cph.fit(df_compare[['group4', 'time', 'event']], duration_col='time', event_col='event')

    hr = np.exp(cph.params_['group4'])
    ci_lower = np.exp(cph.confidence_intervals_.loc['group4', '95% lower-bound'])
    ci_upper = np.exp(cph.confidence_intervals_.loc['group4', '95% upper-bound'])
    p_value = cph.summary.loc['group4', 'p']

    print(f"  n = {len(df_compare)} (Group 1: {(df_compare['group']==1).sum()}, Group 4: {(df_compare['group']==4).sum()})")
    print(f"  HR = {hr:.2f} (95% CI: {ci_lower:.2f} - {ci_upper:.2f})")
    print(f"  p-value = {p_value:.4f}")

    results.append({
        'model': 'Univariate',
        'comparison': 'Group 4 vs Group 1',
        'n': len(df_compare),
        'HR': hr,
        'CI_lower': ci_lower,
        'CI_upper': ci_upper,
        'p_value': p_value
    })

    # Multivariable (adjusted for age and treatment arm)
    print("\n2. MULTIVARIABLE (Adjusted for age and treatment arm)")
    print("-" * 50)

    df_compare['arm_concurrent'] = (df_compare['arm'] == 'Concurrent').astype(int)
    df_compare['Age_std'] = (df_compare['Age'] - df_compare['Age'].mean()) / df_compare['Age'].std()

    cph_multi = CoxPHFitter()
    cph_multi.fit(df_compare[['group4', 'Age_std', 'arm_concurrent', 'time', 'event']],
                  duration_col='time', event_col='event')

    hr_adj = np.exp(cph_multi.params_['group4'])
    ci_lower_adj = np.exp(cph_multi.confidence_intervals_.loc['group4', '95% lower-bound'])
    ci_upper_adj = np.exp(cph_multi.confidence_intervals_.loc['group4', '95% upper-bound'])
    p_value_adj = cph_multi.summary.loc['group4', 'p']

    print(f"  HR = {hr_adj:.2f} (95% CI: {ci_lower_adj:.2f} - {ci_upper_adj:.2f})")
    print(f"  p-value = {p_value_adj:.4f}")

    print("\n  Full model:")
    print(cph_multi.summary[['coef', 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']])

    results.append({
        'model': 'Multivariable',
        'comparison': 'Group 4 vs Group 1',
        'n': len(df_compare),
        'HR': hr_adj,
        'CI_lower': ci_lower_adj,
        'CI_upper': ci_upper_adj,
        'p_value': p_value_adj
    })

    return pd.DataFrame(results)


def clinical_balance(df):
    """Check clinical variable balance across groups."""
    print("\n" + "=" * 70)
    print("CLINICAL VARIABLE BALANCE ACROSS GROUPS")
    print("=" * 70)

    # Age by group
    print("\nAge by group:")
    for g in [1, 2, 3, 4]:
        group_data = df[df['group'] == g]['Age']
        if len(group_data) > 0:
            print(f"  Group {g}: mean = {group_data.mean():.1f}, SD = {group_data.std():.1f}, n = {len(group_data)}")

    # ANOVA for age
    groups = [df[df['group'] == g]['Age'].dropna().values for g in [1, 2, 3, 4]]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) > 1:
        f_stat, p_val = stats.f_oneway(*groups)
        print(f"\n  ANOVA p-value: {p_val:.4f}")

    # Treatment arm by group
    print("\nTreatment arm by group:")
    contingency = pd.crosstab(df['group'], df['arm'])
    print(contingency)

    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    print(f"\n  Chi-square p-value: {p_val:.4f}")


def main():
    """Main function."""
    print("=" * 70)
    print("FOUR-GROUP GDF15 STRATIFICATION ANALYSIS")
    print("=" * 70)

    # Load data
    cosinr = pd.read_csv(DATA_DIR / "regression_ml_inputs.csv")

    # Identify survival columns (adjust as needed)
    time_col = None
    event_col = None

    for col in cosinr.columns:
        if 'os_time' in col.lower() or 'survival_time' in col.lower():
            time_col = col
        if 'os_status' in col.lower() or 'os_event' in col.lower():
            event_col = col

    if time_col is None or event_col is None:
        print("\nWARNING: Survival columns not found. Creating dummy data for demonstration.")
        np.random.seed(42)
        cosinr['OS_time'] = np.random.exponential(24, len(cosinr))
        cosinr['OS_event'] = np.random.binomial(1, 0.4, len(cosinr))
        time_col = 'OS_time'
        event_col = 'OS_event'

    # Create 4 groups
    df, group_labels = create_four_groups(cosinr, time_col, event_col)

    # Kaplan-Meier analysis
    kaplan_meier_four_groups(df, group_labels)

    # Cox regression
    cox_results = cox_four_groups(df)

    # Clinical balance
    clinical_balance(df)

    # Save results
    cox_results.to_csv(RESULTS_DIR / 'four_group_cox_results.csv', index=False)
    df.to_csv(RESULTS_DIR / 'four_group_data.csv', index=False)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(cox_results.to_string(index=False))

    return cox_results


if __name__ == "__main__":
    results = main()
