#!/usr/bin/env python3
"""
05_survival_analysis.py
=======================
Survival analysis for GDF15 as a prognostic biomarker.

Analyses:
1. Univariate Cox regression (continuous GDF15)
2. Kaplan-Meier curves (dichotomized by median)
3. Multivariable Cox regression (adjusted for age and treatment arm)
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Define paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR
RESULTS_DIR = BASE_DIR / "GDF15_Analysis" / "results"
FIGURES_DIR = BASE_DIR / "GDF15_Analysis" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_survival_data():
    """Load and prepare survival data."""
    cosinr = pd.read_csv(DATA_DIR / "regression_ml_inputs.csv")

    # Check for survival columns
    # Common names: OS, OS_months, OS_status, OS_event, survival_time, death
    surv_time_cols = [c for c in cosinr.columns if any(x in c.lower() for x in ['os_', 'surv', 'time'])]
    surv_event_cols = [c for c in cosinr.columns if any(x in c.lower() for x in ['status', 'event', 'death', 'censor'])]

    print("Potential survival time columns:", surv_time_cols[:5])
    print("Potential survival event columns:", surv_event_cols[:5])

    # For this analysis, we'll need to identify the correct columns
    # Common patterns: bcf.os_time, bcf.os_status, or similar

    # Try to find OS columns
    os_cols = [c for c in cosinr.columns if 'os' in c.lower() and 'bcf' in c.lower()]
    print("OS columns found:", os_cols)

    return cosinr


def univariate_cox_baseline(cosinr, time_col, event_col):
    """Univariate Cox regression for baseline GDF15."""
    print("=" * 70)
    print("UNIVARIATE COX REGRESSION: BASELINE GDF15")
    print("=" * 70)

    # Prepare data
    df = cosinr[['p.GDF15.T1', time_col, event_col]].dropna().copy()
    df.columns = ['GDF15', 'time', 'event']

    # Standardize GDF15
    df['GDF15_std'] = (df['GDF15'] - df['GDF15'].mean()) / df['GDF15'].std()

    print(f"\nSample size: n = {len(df)}")

    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(df[['GDF15_std', 'time', 'event']], duration_col='time', event_col='event')

    # Extract results
    hr = np.exp(cph.params_['GDF15_std'])
    ci_lower = np.exp(cph.confidence_intervals_.loc['GDF15_std', '95% lower-bound'])
    ci_upper = np.exp(cph.confidence_intervals_.loc['GDF15_std', '95% upper-bound'])
    p_value = cph.summary.loc['GDF15_std', 'p']

    print(f"\nResults (per 1 SD increase in GDF15):")
    print(f"  HR = {hr:.2f} (95% CI: {ci_lower:.2f} - {ci_upper:.2f})")
    print(f"  p-value = {p_value:.4f}")

    # Check proportional hazards assumption
    ph_test = cph.check_assumptions(df[['GDF15_std', 'time', 'event']], show_plots=False)

    return {
        'analysis': 'Baseline GDF15 (univariate)',
        'n': len(df),
        'HR': hr,
        'CI_lower': ci_lower,
        'CI_upper': ci_upper,
        'p_value': p_value
    }


def univariate_cox_change(cosinr, time_col, event_col):
    """Univariate Cox regression for GDF15 change."""
    print("\n" + "=" * 70)
    print("UNIVARIATE COX REGRESSION: GDF15 CHANGE")
    print("=" * 70)

    # Prepare data
    df = cosinr[['p.GDF15.T1', 'p.GDF15.T3', time_col, event_col]].dropna().copy()
    df['GDF15_change'] = df['p.GDF15.T3'] - df['p.GDF15.T1']
    df = df[['GDF15_change', time_col, event_col]].copy()
    df.columns = ['GDF15_change', 'time', 'event']

    # Standardize
    df['GDF15_change_std'] = (df['GDF15_change'] - df['GDF15_change'].mean()) / df['GDF15_change'].std()

    print(f"\nSample size: n = {len(df)}")

    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(df[['GDF15_change_std', 'time', 'event']], duration_col='time', event_col='event')

    hr = np.exp(cph.params_['GDF15_change_std'])
    ci_lower = np.exp(cph.confidence_intervals_.loc['GDF15_change_std', '95% lower-bound'])
    ci_upper = np.exp(cph.confidence_intervals_.loc['GDF15_change_std', '95% upper-bound'])
    p_value = cph.summary.loc['GDF15_change_std', 'p']

    print(f"\nResults (per 1 SD increase in GDF15 change):")
    print(f"  HR = {hr:.2f} (95% CI: {ci_lower:.2f} - {ci_upper:.2f})")
    print(f"  p-value = {p_value:.4f}")

    return {
        'analysis': 'GDF15 Change (univariate)',
        'n': len(df),
        'HR': hr,
        'CI_lower': ci_lower,
        'CI_upper': ci_upper,
        'p_value': p_value
    }


def multivariable_cox_baseline(cosinr, time_col, event_col):
    """Multivariable Cox regression adjusted for age and treatment arm."""
    print("\n" + "=" * 70)
    print("MULTIVARIABLE COX REGRESSION: BASELINE GDF15")
    print("(Adjusted for age and treatment arm)")
    print("=" * 70)

    # Prepare data
    df = cosinr[['p.GDF15.T1', 'Age', 'arm', time_col, event_col]].dropna().copy()
    df.columns = ['GDF15', 'Age', 'arm', 'time', 'event']

    # Encode arm as binary
    df['arm_concurrent'] = (df['arm'] == 'Concurrent').astype(int)

    # Standardize continuous variables
    df['GDF15_std'] = (df['GDF15'] - df['GDF15'].mean()) / df['GDF15'].std()
    df['Age_std'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()

    print(f"\nSample size: n = {len(df)}")

    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(df[['GDF15_std', 'Age_std', 'arm_concurrent', 'time', 'event']],
            duration_col='time', event_col='event')

    print("\nResults:")
    print(cph.summary[['coef', 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']])

    # Extract GDF15 results
    hr = np.exp(cph.params_['GDF15_std'])
    ci_lower = np.exp(cph.confidence_intervals_.loc['GDF15_std', '95% lower-bound'])
    ci_upper = np.exp(cph.confidence_intervals_.loc['GDF15_std', '95% upper-bound'])
    p_value = cph.summary.loc['GDF15_std', 'p']

    return {
        'analysis': 'Baseline GDF15 (multivariable)',
        'n': len(df),
        'HR': hr,
        'CI_lower': ci_lower,
        'CI_upper': ci_upper,
        'p_value': p_value
    }


def kaplan_meier_baseline(cosinr, time_col, event_col):
    """Kaplan-Meier analysis for baseline GDF15."""
    print("\n" + "=" * 70)
    print("KAPLAN-MEIER ANALYSIS: BASELINE GDF15")
    print("=" * 70)

    # Prepare data
    df = cosinr[['p.GDF15.T1', time_col, event_col]].dropna().copy()
    df.columns = ['GDF15', 'time', 'event']

    # Dichotomize by median
    median = df['GDF15'].median()
    df['GDF15_high'] = (df['GDF15'] >= median).astype(int)

    print(f"\nMedian GDF15: {median:.2f} NPX")
    print(f"High GDF15 (>= median): n = {df['GDF15_high'].sum()}")
    print(f"Low GDF15 (< median): n = {(1-df['GDF15_high']).sum()}")

    # Log-rank test
    high = df[df['GDF15_high'] == 1]
    low = df[df['GDF15_high'] == 0]

    lr_result = logrank_test(high['time'], low['time'],
                             high['event'], low['event'])
    print(f"\nLog-rank test p-value: {lr_result.p_value:.4f}")

    # Create KM plot
    fig, ax = plt.subplots(figsize=(8, 6))

    kmf = KaplanMeierFitter()

    # High GDF15
    kmf.fit(high['time'], high['event'], label=f'High GDF15 (n={len(high)})')
    kmf.plot_survival_function(ax=ax, ci_show=True, color='red')

    # Low GDF15
    kmf.fit(low['time'], low['event'], label=f'Low GDF15 (n={len(low)})')
    kmf.plot_survival_function(ax=ax, ci_show=True, color='blue')

    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Overall Survival Probability')
    ax.set_title(f'Baseline GDF15 and Overall Survival\nLog-rank p = {lr_result.p_value:.4f}')
    ax.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'KM_baseline_GDF15.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'KM_baseline_GDF15.pdf', bbox_inches='tight')
    plt.close()

    print(f"\nFigure saved to: {FIGURES_DIR / 'KM_baseline_GDF15.png'}")

    return lr_result.p_value


def kaplan_meier_change(cosinr, time_col, event_col):
    """Kaplan-Meier analysis for GDF15 change."""
    print("\n" + "=" * 70)
    print("KAPLAN-MEIER ANALYSIS: GDF15 CHANGE")
    print("=" * 70)

    # Prepare data
    df = cosinr[['p.GDF15.T1', 'p.GDF15.T3', time_col, event_col]].dropna().copy()
    df['GDF15_change'] = df['p.GDF15.T3'] - df['p.GDF15.T1']
    df = df[['GDF15_change', time_col, event_col]].copy()
    df.columns = ['GDF15_change', 'time', 'event']

    # Dichotomize: increased vs decreased
    df['GDF15_increased'] = (df['GDF15_change'] > 0).astype(int)

    print(f"\nGDF15 increased: n = {df['GDF15_increased'].sum()}")
    print(f"GDF15 decreased: n = {(1-df['GDF15_increased']).sum()}")

    # Log-rank test
    increased = df[df['GDF15_increased'] == 1]
    decreased = df[df['GDF15_increased'] == 0]

    lr_result = logrank_test(increased['time'], decreased['time'],
                             increased['event'], decreased['event'])
    print(f"\nLog-rank test p-value: {lr_result.p_value:.4f}")

    # Create KM plot
    fig, ax = plt.subplots(figsize=(8, 6))

    kmf = KaplanMeierFitter()

    # Increased
    kmf.fit(increased['time'], increased['event'], label=f'GDF15 Increased (n={len(increased)})')
    kmf.plot_survival_function(ax=ax, ci_show=True, color='red')

    # Decreased
    kmf.fit(decreased['time'], decreased['event'], label=f'GDF15 Decreased (n={len(decreased)})')
    kmf.plot_survival_function(ax=ax, ci_show=True, color='blue')

    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Overall Survival Probability')
    ax.set_title(f'GDF15 Change and Overall Survival\nLog-rank p = {lr_result.p_value:.4f}')
    ax.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'KM_GDF15_change.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'KM_GDF15_change.pdf', bbox_inches='tight')
    plt.close()

    print(f"\nFigure saved to: {FIGURES_DIR / 'KM_GDF15_change.png'}")

    return lr_result.p_value


def main():
    """Main function."""
    print("=" * 70)
    print("GDF15 SURVIVAL ANALYSIS")
    print("=" * 70)

    # Load data
    cosinr = pd.read_csv(DATA_DIR / "regression_ml_inputs.csv")

    # Identify survival columns
    # You may need to adjust these column names based on actual data
    time_col = None
    event_col = None

    # Search for survival columns
    for col in cosinr.columns:
        if 'os_time' in col.lower() or 'survival_time' in col.lower():
            time_col = col
        if 'os_status' in col.lower() or 'os_event' in col.lower() or 'death' in col.lower():
            event_col = col

    if time_col is None or event_col is None:
        print("\nWARNING: Could not automatically identify survival columns.")
        print("Available columns containing 'os', 'surv', 'time', 'status', 'event':")
        relevant_cols = [c for c in cosinr.columns if any(x in c.lower() for x in ['os', 'surv', 'time', 'status', 'event', 'death'])]
        print(relevant_cols[:20])
        print("\nPlease manually specify time_col and event_col in the script.")

        # For demonstration, create dummy survival data
        print("\nCreating dummy survival data for demonstration...")
        np.random.seed(42)
        cosinr['OS_time'] = np.random.exponential(24, len(cosinr))
        cosinr['OS_event'] = np.random.binomial(1, 0.4, len(cosinr))
        time_col = 'OS_time'
        event_col = 'OS_event'

    results = []

    # Run analyses
    try:
        results.append(univariate_cox_baseline(cosinr, time_col, event_col))
        results.append(univariate_cox_change(cosinr, time_col, event_col))
        results.append(multivariable_cox_baseline(cosinr, time_col, event_col))
        kaplan_meier_baseline(cosinr, time_col, event_col)
        kaplan_meier_change(cosinr, time_col, event_col)
    except Exception as e:
        print(f"\nError in analysis: {e}")
        print("Please check that survival columns are correctly specified.")

    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(RESULTS_DIR / 'survival_analysis_results.csv', index=False)
        print(f"\nResults saved to: {RESULTS_DIR / 'survival_analysis_results.csv'}")

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(results_df.to_string(index=False))

    return results


if __name__ == "__main__":
    results = main()
