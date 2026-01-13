#!/usr/bin/env python3
"""
10_covariate_analysis.py
========================
Analyze clinical covariate balance across GDF15 groups and potential confounders.

Analyses:
1. Age distribution across GDF15 high/low groups
2. Treatment arm distribution across GDF15 groups
3. Histology and PD-L1 distribution
4. 4-group stratification covariate balance

Statistical tests:
- Continuous variables: t-test (2 groups) or ANOVA (4 groups)
- Categorical variables: Chi-square test
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


def analyze_baseline_groups(cosinr):
    """Analyze clinical variables by baseline GDF15 groups."""
    print("=" * 70)
    print("CLINICAL VARIABLES BY BASELINE GDF15 GROUP")
    print("=" * 70)

    # Create high/low groups based on median
    df = cosinr[cosinr['p.GDF15.T1'].notna()].copy()
    median = df['p.GDF15.T1'].median()
    df['GDF15_high'] = (df['p.GDF15.T1'] >= median).astype(int)

    print(f"\nMedian baseline GDF15: {median:.2f} NPX")
    print(f"High GDF15 (>= median): n = {df['GDF15_high'].sum()}")
    print(f"Low GDF15 (< median): n = {(1-df['GDF15_high']).sum()}")

    results = []

    # 1. Age
    print("\n" + "-" * 50)
    print("1. AGE")
    print("-" * 50)

    if 'Age' in df.columns:
        high_age = df[df['GDF15_high'] == 1]['Age']
        low_age = df[df['GDF15_high'] == 0]['Age']

        print(f"  High GDF15: mean = {high_age.mean():.1f} ± {high_age.std():.1f}")
        print(f"  Low GDF15: mean = {low_age.mean():.1f} ± {low_age.std():.1f}")

        # t-test
        t_stat, p_val = stats.ttest_ind(high_age.dropna(), low_age.dropna())
        print(f"  t-test p-value: {p_val:.4f}")

        results.append({
            'variable': 'Age',
            'high_mean': high_age.mean(),
            'high_sd': high_age.std(),
            'low_mean': low_age.mean(),
            'low_sd': low_age.std(),
            'test': 't-test',
            'p_value': p_val,
            'significant': p_val < 0.05
        })

        if p_val < 0.05:
            print("  ** AGE IS A POTENTIAL CONFOUNDER **")

    # 2. Treatment arm
    print("\n" + "-" * 50)
    print("2. TREATMENT ARM")
    print("-" * 50)

    if 'arm' in df.columns:
        contingency = pd.crosstab(df['GDF15_high'], df['arm'])
        print(contingency)

        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
        print(f"\n  Chi-square p-value: {p_val:.4f}")

        results.append({
            'variable': 'Treatment arm',
            'high_mean': None,
            'low_mean': None,
            'test': 'Chi-square',
            'p_value': p_val,
            'significant': p_val < 0.05
        })

        if p_val >= 0.05:
            print("  Treatment arm is BALANCED across GDF15 groups")

    # 3. Histology
    print("\n" + "-" * 50)
    print("3. HISTOLOGY")
    print("-" * 50)

    if 'bcf.histology_bin' in df.columns:
        contingency = pd.crosstab(df['GDF15_high'], df['bcf.histology_bin'])
        print(contingency)

        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
        print(f"\n  Chi-square p-value: {p_val:.4f}")

        results.append({
            'variable': 'Histology',
            'test': 'Chi-square',
            'p_value': p_val,
            'significant': p_val < 0.05
        })

    # 4. PD-L1 expression
    print("\n" + "-" * 50)
    print("4. PD-L1 EXPRESSION")
    print("-" * 50)

    if 'bcf.PDL1_bin' in df.columns:
        contingency = pd.crosstab(df['GDF15_high'], df['bcf.PDL1_bin'])
        print(contingency)

        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
        print(f"\n  Chi-square p-value: {p_val:.4f}")

        results.append({
            'variable': 'PD-L1 expression',
            'test': 'Chi-square',
            'p_value': p_val,
            'significant': p_val < 0.05
        })

    return pd.DataFrame(results)


def analyze_four_group_balance(cosinr):
    """Analyze clinical variables by 4-group stratification."""
    print("\n" + "=" * 70)
    print("CLINICAL VARIABLES BY 4-GROUP STRATIFICATION")
    print("=" * 70)

    # Filter for patients with paired samples
    df = cosinr[cosinr['p.GDF15.T1'].notna() & cosinr['p.GDF15.T3'].notna()].copy()

    # Create groups
    median = df['p.GDF15.T1'].median()
    df['baseline_high'] = (df['p.GDF15.T1'] >= median).astype(int)
    df['change_increased'] = (df['p.GDF15.T3'] > df['p.GDF15.T1']).astype(int)

    df['group'] = 0
    df.loc[(df['baseline_high'] == 0) & (df['change_increased'] == 0), 'group'] = 1
    df.loc[(df['baseline_high'] == 0) & (df['change_increased'] == 1), 'group'] = 2
    df.loc[(df['baseline_high'] == 1) & (df['change_increased'] == 0), 'group'] = 3
    df.loc[(df['baseline_high'] == 1) & (df['change_increased'] == 1), 'group'] = 4

    group_labels = {
        1: 'Low+Decreased',
        2: 'Low+Increased',
        3: 'High+Decreased',
        4: 'High+Increased'
    }

    print("\nGroup sizes:")
    for g in [1, 2, 3, 4]:
        n = (df['group'] == g).sum()
        print(f"  Group {g} ({group_labels[g]}): n = {n}")

    results = []

    # 1. Age by group
    print("\n" + "-" * 50)
    print("AGE BY GROUP")
    print("-" * 50)

    if 'Age' in df.columns:
        for g in [1, 2, 3, 4]:
            group_age = df[df['group'] == g]['Age']
            if len(group_age) > 0:
                print(f"  Group {g}: mean = {group_age.mean():.1f} ± {group_age.std():.1f}, n = {len(group_age)}")

        # ANOVA
        groups = [df[df['group'] == g]['Age'].dropna().values for g in [1, 2, 3, 4]]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) > 1:
            f_stat, p_val = stats.f_oneway(*groups)
            print(f"\n  ANOVA p-value: {p_val:.4f}")

            results.append({
                'variable': 'Age',
                'test': 'ANOVA',
                'p_value': p_val,
                'significant': p_val < 0.05
            })

            if p_val < 0.05:
                print("  ** AGE IS A POTENTIAL CONFOUNDER FOR 4-GROUP ANALYSIS **")

    # 2. Treatment arm by group
    print("\n" + "-" * 50)
    print("TREATMENT ARM BY GROUP")
    print("-" * 50)

    if 'arm' in df.columns:
        contingency = pd.crosstab(df['group'], df['arm'])
        print(contingency)

        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
        print(f"\n  Chi-square p-value: {p_val:.4f}")

        results.append({
            'variable': 'Treatment arm',
            'test': 'Chi-square',
            'p_value': p_val,
            'significant': p_val < 0.05
        })

    return pd.DataFrame(results), df


def create_covariate_figure(cosinr):
    """Create figure showing covariate distributions."""
    df = cosinr[cosinr['p.GDF15.T1'].notna()].copy()
    median = df['p.GDF15.T1'].median()
    df['GDF15_group'] = df['p.GDF15.T1'].apply(lambda x: 'High' if x >= median else 'Low')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Age distribution
    ax1 = axes[0]
    if 'Age' in df.columns:
        high = df[df['GDF15_group'] == 'High']['Age']
        low = df[df['GDF15_group'] == 'Low']['Age']

        ax1.boxplot([low.dropna(), high.dropna()], labels=['Low GDF15', 'High GDF15'])
        ax1.set_ylabel('Age (years)')
        ax1.set_title('Age Distribution by Baseline GDF15')

        t_stat, p_val = stats.ttest_ind(high.dropna(), low.dropna())
        ax1.annotate(f'p = {p_val:.3f}', xy=(0.5, 0.95), xycoords='axes fraction',
                     ha='center', fontsize=12)

    # 2. Treatment arm distribution
    ax2 = axes[1]
    if 'arm' in df.columns:
        contingency = pd.crosstab(df['GDF15_group'], df['arm'], normalize='index') * 100
        contingency.plot(kind='bar', ax=ax2, rot=0)
        ax2.set_ylabel('Percentage')
        ax2.set_xlabel('GDF15 Group')
        ax2.set_title('Treatment Arm Distribution by Baseline GDF15')
        ax2.legend(title='Treatment Arm')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'covariate_balance.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'covariate_balance.pdf', bbox_inches='tight')
    plt.close()

    print(f"\nFigure saved to: {FIGURES_DIR / 'covariate_balance.png'}")


def main():
    """Main function."""
    print("=" * 70)
    print("CLINICAL COVARIATE BALANCE ANALYSIS")
    print("=" * 70)

    # Load data
    cosinr = pd.read_csv(DATA_DIR / "regression_ml_inputs.csv")

    # Print available clinical variables
    print("\nAvailable clinical variables:")
    bcf_cols = [c for c in cosinr.columns if c.startswith('bcf.')]
    print(f"  {bcf_cols}")

    # Run analyses
    baseline_results = analyze_baseline_groups(cosinr)
    four_group_results, four_group_data = analyze_four_group_balance(cosinr)

    # Create figure
    create_covariate_figure(cosinr)

    # Save results
    if len(baseline_results) > 0:
        baseline_results.to_csv(RESULTS_DIR / 'covariate_balance_baseline.csv', index=False)
    if len(four_group_results) > 0:
        four_group_results.to_csv(RESULTS_DIR / 'covariate_balance_four_group.csv', index=False)

    print("\n" + "=" * 70)
    print("SUMMARY: CONFOUNDERS FOR MULTIVARIABLE MODELS")
    print("=" * 70)

    print("""
Analysis complete. See output above for:
- Covariate balance across GDF15 groups
- Potential confounders identified (p < 0.05)
- Variables that are balanced across groups

Note: Variables not in the dataset (e.g., ECOG, smoking status)
cannot be included in multivariable models.
""")

    return {
        'baseline': baseline_results,
        'four_group': four_group_results
    }


if __name__ == "__main__":
    results = main()
