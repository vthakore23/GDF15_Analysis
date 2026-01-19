# GDF15 Biomarker Analysis in SBRT + Immunotherapy for NSCLC

This repository contains analysis scripts for investigating GDF15 as a biomarker in non-small cell lung cancer (NSCLC) patients treated with stereotactic body radiation therapy (SBRT) with or without immunotherapy.

## Study Cohorts

### COSINR Trial (Metastatic NSCLC, SBRT + Immunotherapy)
- **Treatment arms**: Concurrent vs Sequential SBRT + immunotherapy
- **Data types**: Blood proteomics (Olink), flow cytometry, TCR sequencing, tumor RNA-seq

### Early-Stage NSCLC (SBRT Alone - Validation Cohort)
- **Treatment**: SBRT alone (no immunotherapy)
- **Data types**: Blood proteomics (Olink)

## Requirements

```
python >= 3.9
pandas >= 1.4.0
numpy >= 1.21.0
scipy >= 1.7.0
matplotlib >= 3.5.0
seaborn >= 0.11.0
lifelines >= 0.27.0
statsmodels >= 0.13.0
openpyxl >= 3.0.0
pyarrow >= 6.0.0
```

## Required Data Files

The following data files are required (not included in repository):

- `regression_ml_inputs.csv` - COSINR blood proteomics and clinical data
- `Q-12622_Zha_NPX_2024-08-21.parquet` - Early-stage blood proteomics
- `Q-12622_Zha - Olink_-_Sample_Manifest.xlsx` - Early-stage sample manifest
- `43018_2022_467_MOESM2_ESM.xlsx` - Nature Cancer supplementary data
- `01_DESeq2_Combined_AllGenes.csv` - Tumor RNA-seq differential expression
- `hallmark_ssGSEA.csv` - Hallmark pathway enrichment scores
- `reactome_immune_only_ssGSEA.csv` - Reactome immune pathway scores

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run individual scripts
python scripts/01_data_loading.py
python scripts/02_gdf15_induction.py
# ... etc

# Or run all analyses
python scripts/run_all_analyses.py
```

## Statistical Methods

- **Paired comparisons**: Paired t-test with Cohen's d effect size
- **Correlations**: Spearman rank correlation (robust to outliers and non-normality)
- **Multiple testing**: Benjamini-Hochberg FDR correction
- **Survival analysis**: Kaplan-Meier estimator, log-rank test, Cox proportional hazards regression
- **Group comparisons**: t-test (2 groups), ANOVA (>2 groups), Chi-square test (categorical)

## Output

Each script generates:
- Console output with statistical results
- CSV files saved to `results/` directory
- Figures saved to `figures/` directory (PNG and PDF formats)


