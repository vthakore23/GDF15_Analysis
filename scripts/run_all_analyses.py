#!/usr/bin/env python3
"""
run_all_analyses.py
===================
Master script to run all GDF15 biomarker analyses.

Usage:
    python run_all_analyses.py

This will run all analysis scripts in order and generate all results and figures.
"""

import subprocess
import sys
from pathlib import Path

# Define paths
SCRIPT_DIR = Path(__file__).parent
SCRIPTS = [
    '01_data_loading.py',
    '02_gdf15_induction.py',
    '03_growth_factor_de.py',
    '04_tumor_blood_correlation.py',
    '05_survival_analysis.py',
    '06_four_group_stratification.py',
    '07_inflammatory_correlations.py',
    '08_flow_cytometry.py',
    '09_pathway_analysis.py',
    '10_covariate_analysis.py'
]


def run_script(script_name):
    """Run a single analysis script."""
    script_path = SCRIPT_DIR / script_name
    print(f"\n{'='*70}")
    print(f"RUNNING: {script_name}")
    print('='*70)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(result.stdout)
            print(f"\n✓ {script_name} completed successfully")
            return True
        else:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            print(f"\n✗ {script_name} failed with return code {result.returncode}")
            return False

    except Exception as e:
        print(f"\n✗ Error running {script_name}: {e}")
        return False


def main():
    """Run all analysis scripts."""
    print("="*70)
    print("GDF15 BIOMARKER ANALYSIS PIPELINE")
    print("="*70)

    results = {}

    for script in SCRIPTS:
        success = run_script(script)
        results[script] = success

        if not success:
            print(f"\nWARNING: {script} failed. Continuing with remaining scripts...")

    # Summary
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)

    n_success = sum(results.values())
    n_total = len(results)

    print(f"\nCompleted: {n_success}/{n_total} scripts")

    for script, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {script}")

    if n_success == n_total:
        print("\n✓ All analyses completed successfully!")
    else:
        print(f"\n⚠ {n_total - n_success} script(s) failed. Check output above for errors.")

    return results


if __name__ == "__main__":
    main()
