#!/usr/bin/env python3
"""
01_data_loading.py
==================
Load and preprocess all datasets for GDF15 biomarker analysis.

Datasets:
- COSINR blood proteomics (regression_ml_inputs.csv)
- COSINR tumor RNA-seq (from Nature Cancer supplement)
- Early-stage blood proteomics (Olink parquet file)
- Hallmark ssGSEA scores
- Reactome immune pathway ssGSEA scores

Output:
- Preprocessed dataframes saved as pickle files for downstream analysis
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR
OUTPUT_DIR = BASE_DIR / "GDF15_Analysis" / "results"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_cosinr_proteomics():
    """Load COSINR blood proteomics data."""
    print("Loading COSINR blood proteomics...")
    df = pd.read_csv(DATA_DIR / "regression_ml_inputs.csv")

    print(f"  Total patients: {len(df)}")
    print(f"  Baseline (T1) samples: {df['p.GDF15.T1'].notna().sum()}")
    print(f"  On-treatment (T3) samples: {df['p.GDF15.T3'].notna().sum()}")
    print(f"  Paired samples: {(df['p.GDF15.T1'].notna() & df['p.GDF15.T3'].notna()).sum()}")

    return df


def load_early_stage_proteomics():
    """Load early-stage NSCLC blood proteomics data."""
    print("\nLoading early-stage blood proteomics...")

    # Load Olink data
    early = pd.read_parquet(DATA_DIR / "Q-12622_Zha_NPX_2024-08-21.parquet")
    manifest = pd.read_excel(DATA_DIR / "Q-12622_Zha - Olink_-_Sample_Manifest.xlsx")

    # Pivot to wide format
    early_wide = early.pivot_table(
        index='SampleID',
        columns='Assay',
        values='NPX',
        aggfunc='first'
    ).reset_index()

    # Map timepoints
    manifest['SampleID'] = manifest['SampleID'].astype(str)
    manifest_map = manifest[['SampleID', 'TP', 'Subj ID']].drop_duplicates()
    early_wide = pd.merge(early_wide, manifest_map, on='SampleID', how='left')

    # Count paired samples
    pre_patients = set(early_wide[early_wide['TP'] == 'pre']['Subj ID'].dropna())
    post_patients = set(early_wide[early_wide['TP'] == 'post']['Subj ID'].dropna())
    paired = pre_patients & post_patients

    print(f"  Pre-treatment samples: {len(pre_patients)}")
    print(f"  Post-treatment samples: {len(post_patients)}")
    print(f"  Paired samples: {len(paired)}")

    return early_wide


def load_tumor_rnaseq():
    """Load tumor RNA-seq data from Nature Cancer supplement."""
    print("\nLoading tumor RNA-seq data...")

    xls = pd.ExcelFile(DATA_DIR / "43018_2022_467_MOESM2_ESM.xlsx")
    tumor_rna = pd.read_excel(xls, sheet_name='Supplementary Table 8', header=1)

    # Get sample columns
    sample_cols = [c for c in tumor_rna.columns if c.startswith('SP_')]
    print(f"  Tumor samples: {len(sample_cols)}")

    # Extract GDF15 expression
    gdf15_row = tumor_rna[tumor_rna['Gene ID'] == 'ENSG00000130513']
    if len(gdf15_row) > 0:
        gdf15_expr = gdf15_row[sample_cols].iloc[0]
        patient_ids = [int(c.replace('SP_', '')) for c in sample_cols]
        tumor_gdf15 = pd.DataFrame({
            'id': patient_ids,
            'tumor_GDF15': gdf15_expr.values
        })
        print(f"  GDF15 expression range: {gdf15_expr.min():.1f} - {gdf15_expr.max():.1f}")
    else:
        tumor_gdf15 = None
        print("  WARNING: GDF15 not found in tumor RNA-seq")

    return tumor_rna, tumor_gdf15


def load_deseq2_results():
    """Load DESeq2 differential expression results."""
    print("\nLoading DESeq2 results...")

    deseq = pd.read_csv(DATA_DIR / "01_DESeq2_Combined_AllGenes.csv")
    print(f"  Total genes: {len(deseq)}")
    print(f"  Significant (padj < 0.05): {(deseq['padj'] < 0.05).sum()}")

    return deseq


def load_ssgsea_data():
    """Load ssGSEA pathway enrichment scores."""
    print("\nLoading ssGSEA pathway data...")

    hallmark = pd.read_csv(DATA_DIR / "hallmark_ssGSEA.csv")
    reactome = pd.read_csv(DATA_DIR / "reactome_immune_only_ssGSEA.csv")

    # Count pathways (each has T1, T3, dif1v3 columns)
    hallmark_pathways = len([c for c in hallmark.columns if 'HALLMARK' in c]) // 3
    reactome_pathways = len([c for c in reactome.columns if 'gs.' in c]) // 3

    print(f"  Hallmark pathways: {hallmark_pathways}")
    print(f"  Reactome immune pathways: {reactome_pathways}")

    return hallmark, reactome


def load_clinical_supplement():
    """Load clinical data from Nature Cancer supplement."""
    print("\nLoading clinical supplement data...")

    xls = pd.ExcelFile(DATA_DIR / "43018_2022_467_MOESM2_ESM.xlsx")
    clinical = pd.read_excel(xls, sheet_name='Supplementary Table 1', header=1)

    print(f"  Patients in supplement: {len(clinical)}")
    print(f"  Columns: {clinical.columns.tolist()[:10]}...")

    return clinical


def define_growth_factors():
    """Return list of growth factor gene symbols for analysis."""
    growth_factors = [
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
    return growth_factors


def main():
    """Main function to load all data."""
    print("=" * 70)
    print("GDF15 BIOMARKER ANALYSIS - DATA LOADING")
    print("=" * 70)

    # Load all datasets
    cosinr = load_cosinr_proteomics()
    early_stage = load_early_stage_proteomics()
    tumor_rna, tumor_gdf15 = load_tumor_rnaseq()
    deseq = load_deseq2_results()
    hallmark, reactome = load_ssgsea_data()
    clinical = load_clinical_supplement()

    # Save processed data
    print("\n" + "=" * 70)
    print("Saving processed data...")

    cosinr.to_pickle(OUTPUT_DIR / "cosinr_proteomics.pkl")
    early_stage.to_pickle(OUTPUT_DIR / "early_stage_proteomics.pkl")
    if tumor_gdf15 is not None:
        tumor_gdf15.to_pickle(OUTPUT_DIR / "tumor_gdf15.pkl")
    deseq.to_pickle(OUTPUT_DIR / "deseq2_results.pkl")
    hallmark.to_pickle(OUTPUT_DIR / "hallmark_ssgsea.pkl")
    reactome.to_pickle(OUTPUT_DIR / "reactome_ssgsea.pkl")
    clinical.to_pickle(OUTPUT_DIR / "clinical_data.pkl")

    print("Data loading complete!")
    print(f"Output saved to: {OUTPUT_DIR}")

    return {
        'cosinr': cosinr,
        'early_stage': early_stage,
        'tumor_rna': tumor_rna,
        'tumor_gdf15': tumor_gdf15,
        'deseq': deseq,
        'hallmark': hallmark,
        'reactome': reactome,
        'clinical': clinical
    }


if __name__ == "__main__":
    data = main()
