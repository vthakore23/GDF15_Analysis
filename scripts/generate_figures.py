#!/usr/bin/env python3
"""
generate_figures.py
===================

Generate all manuscript figures for GDF15 biomarker analysis.

Figure Layout (6 Main Figures):

Figure 1: GDF15 is Selectively Induced by SBRT (Discovery)
    - A: Tumor RNA volcano (109 growth factors)
    - B: Blood proteomics validation (early-stage cohort volcano)
    - C: Tumor-blood concordance (3 panels: baseline, on-treatment, change)
    - D: Induction kinetics (paired plot showing pre->post change)
    - E: SBRT fractionation effect (dose-response)

Figure 2: GDF15 Predicts Overall Survival Independent of Tumor Response
    - A: Baseline GDF15 survival (KM)
    - B: Change in GDF15 survival
    - C: Four-group stratification
    - D: Forest plot: GDF15 vs tumor metrics
    - E: Responder-only analysis

Figure 3: Sequential Therapy Amplifies GDF15's Prognostic Effect
    - A: Sequential arm KM curves
    - B: Concurrent arm KM curves
    - C: Interaction analysis forest plot

Figure 4: GDF15 Correlates with Inflammatory Stress and Checkpoint Exhaustion
    - A: Correlation heatmap
    - B-E: Scatter plots for key markers
    - F: Cross-cohort validation

Figure 5: GDF15 Associates with Functional Immune Suppression
    - A: Flow cytometry null results
    - B: Pathway correlations

Figure 6: GDF15 Reflects Unresolved Inflammation
    - A: Acute vs chronic inflammation
    - B: Module correlations

Usage:
    python generate_figures.py --data-dir /path/to/data --output-dir /path/to/figures
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path
import warnings
import argparse
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

warnings.filterwarnings('ignore')

# Publication-quality settings
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0

# Color palette - colorblind-friendly
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'highlight': '#d62728',
    'low': '#2ca02c',
    'high': '#d62728',
    'ns': '#7f7f7f',
    'sequential': '#9467bd',
    'concurrent': '#8c564b',
    'low_dec': '#2166AC',
    'low_inc': '#67A9CF',
    'high_dec': '#F4A582',
    'high_inc': '#B2182B',
    'increased': '#d62728',
    'decreased': '#2ca02c',
    'positive': '#d62728',
    'negative': '#1f77b4',
}

# Growth factors list (109 total)
GROWTH_FACTORS = [
    "BDNF", "NTRK2", "BAMBI", "BMPER", "BMP10", "BMP2KL", "BMP4", "BMP6", "BMP7", "BMPR1B",
    "CNTF", "CSF1", "CSF1R", "CSF2", "CSF2RA", "CSF2RB", "CSF3", "CSF3R",
    "CELSR2", "CCBE1", "DNER",
    "EGF", "EFEMP1", "EFEMP2", "EDIL3", "EGFL6", "EGFL7", "EGFLAM", "EGFR", "EPS8L2",
    "FGF1", "FGF12", "FGF16", "FGF19", "FGF2", "FGF20", "FGF21", "FGF23",
    "FGF3", "FGF5", "FGF6", "FGF7", "FGF9",
    "FGFBP2", "FGFBP3", "FGFR2", "FGFR4",
    "GDF15", "GDF2", "GDNF", "GFRA1", "GFRA3", "GFRAL",
    "HBEGF", "HGF", "HGFAC", "MET", "MST1", "HGS", "HDGF", "HDGFL1", "HDGFL2",
    "NTRK1", "IGF1R", "IGF2-AS", "IGF2BP3", "IGF2R",
    "IGFBP1", "IGFBP2", "IGFBP3", "IGFBP4", "IGFBP6", "IGFBP7", "IGFBPL1", "IGFL3", "IGFL4",
    "LTBP2", "LTBP3",
    "MEGF10", "MEGF11", "MEGF9", "KIT", "MYDGF", "MDK", "NGFR", "NTRK3", "OGFR",
    "PDGFA", "PDGFB", "PDGFC", "PDGFD", "PDGFRA", "PDGFRB", "PGF",
    "SCUBE3", "SNEG1",
    "TGFBR1", "TGFBR2", "TAB2", "TGFA", "TGFB1", "TGFB2", "TGFBI", "TGFBR3", "TDGF1",
    "VEGFA", "VEGFB", "VEGFC", "VEGFD", "FLT1", "KDR", "FLT4"
]


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


class GDF15FigureGenerator:
    """Generate all manuscript figures."""

    def __init__(self, data_dir, output_dir):
        """Initialize with data and output directories."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("LOADING DATA")
        print("=" * 70)

        self.load_data()

    def load_data(self):
        """Load all required datasets."""
        # COSINR cohort
        self.cosinr = pd.read_csv(self.data_dir / "regression_ml_inputs.csv")
        print(f"COSINR patients: {len(self.cosinr)}")

        # DESeq2 results
        deseq_file = self.data_dir / "01_DESeq2_Combined_AllGenes.csv"
        if deseq_file.exists():
            self.deseq = pd.read_csv(deseq_file)
            print(f"DESeq2 genes: {len(self.deseq)}")
        else:
            self.deseq = None
            print("DESeq2 file not found")

        # Early-stage cohort
        early_file = self.data_dir / "Q-12622_Zha_NPX_2024-08-21.parquet"
        manifest_file = self.data_dir / "Q-12622_Zha - Olink_-_Sample_Manifest.xlsx"

        if early_file.exists() and manifest_file.exists():
            early = pd.read_parquet(early_file)
            manifest = pd.read_excel(manifest_file)

            early_wide = early.pivot_table(index='SampleID', columns='Assay',
                                           values='NPX', aggfunc='first').reset_index()
            manifest['SampleID'] = manifest['SampleID'].astype(str)
            manifest_map = manifest[['SampleID', 'TP', 'Subj ID']].drop_duplicates()
            self.early_stage = pd.merge(early_wide, manifest_map, on='SampleID', how='left')
            self.early_stage = self.early_stage[~self.early_stage['SampleID'].str.contains('170547', na=False)]

            # Create paired data
            pre = self.early_stage[self.early_stage['TP'] == 'pre'][['Subj ID', 'GDF15']].copy()
            post = self.early_stage[self.early_stage['TP'] == 'post'][['Subj ID', 'GDF15']].copy()
            pre.columns = ['Subj ID', 'GDF15_pre']
            post.columns = ['Subj ID', 'GDF15_post']
            self.early_paired = pd.merge(pre, post, on='Subj ID').dropna()
            self.early_paired['delta_GDF15'] = self.early_paired['GDF15_post'] - self.early_paired['GDF15_pre']

            print(f"Early-stage paired samples: {len(self.early_paired)}")
        else:
            self.early_stage = None
            self.early_paired = None
            print("Early-stage files not found")

        # Early-stage clinical data (for fractionation)
        clinical_file = self.data_dir / "Early Stage SBRT Clinical Data-sp 9.13.24.xlsx"
        if clinical_file.exists() and self.early_paired is not None:
            clinical = pd.read_excel(clinical_file)
            clinical['Subject_ID'] = clinical['Subject_ID'].astype(float)
            self.early_paired['Subj ID'] = self.early_paired['Subj ID'].astype(float)
            self.early_fx = pd.merge(self.early_paired,
                                     clinical[['Subject_ID', 'SBRT_fx', 'SBRT_dose']],
                                     left_on='Subj ID', right_on='Subject_ID', how='left')
            self.early_fx = self.early_fx.dropna(subset=['SBRT_fx'])
            print(f"Early-stage with fraction data: {len(self.early_fx)}")
        else:
            self.early_fx = None

        # Growth factor results
        gf_file = self.data_dir / "GDF15_Analysis" / "results" / "growth_factors_early_blood_109list.csv"
        if gf_file.exists():
            self.gf_results = pd.read_csv(gf_file)
        else:
            self.gf_results = None

        # Pathway scores
        hallmark_file = self.data_dir / "hallmark_ssGSEA.csv"
        reactome_file = self.data_dir / "reactome_immune_only_ssGSEA.csv"

        if hallmark_file.exists():
            self.hallmark = pd.read_csv(hallmark_file)
        else:
            self.hallmark = None

        if reactome_file.exists():
            self.reactome = pd.read_csv(reactome_file)
        else:
            self.reactome = None

    def figure1_discovery(self):
        """
        Figure 1: GDF15 is selectively induced by SBRT
        Panels: A (tumor volcano), B (blood volcano), C1-C3 (correlations),
                D (kinetics), E (fractionation)
        """
        print("\nCreating Figure 1: GDF15 Selective Induction...")

        fig = plt.figure(figsize=(16, 11))
        gs = fig.add_gridspec(2, 10, hspace=0.4, wspace=0.6,
                              height_ratios=[1, 1],
                              left=0.05, right=0.95, top=0.92, bottom=0.08)

        # Panel A: Tumor volcano
        ax_a = fig.add_subplot(gs[0, 0:5])
        self._plot_tumor_volcano(ax_a)

        # Panel B: Blood volcano
        ax_b = fig.add_subplot(gs[0, 5:10])
        self._plot_blood_volcano(ax_b)

        # Panel C1-C3: Tumor-blood correlations
        ax_c1 = fig.add_subplot(gs[1, 0:2])
        ax_c2 = fig.add_subplot(gs[1, 2:4])
        ax_c3 = fig.add_subplot(gs[1, 4:6])
        self._plot_tumor_blood_correlations(ax_c1, ax_c2, ax_c3)

        # Panel D: Induction kinetics
        ax_d = fig.add_subplot(gs[1, 6:8])
        self._plot_induction_kinetics(ax_d)

        # Panel E: Fractionation
        ax_e = fig.add_subplot(gs[1, 8:10])
        self._plot_fractionation(ax_e)

        plt.suptitle('Figure 1: GDF15 is Selectively Induced by SBRT',
                     fontsize=14, fontweight='bold', y=0.97)

        plt.savefig(self.output_dir / 'Figure1_Discovery.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'Figure1_Discovery.pdf', bbox_inches='tight')
        plt.close()

        print(f"  Saved: {self.output_dir / 'Figure1_Discovery.png'}")

    def _plot_tumor_volcano(self, ax):
        """Panel A: Tumor RNA-seq volcano plot."""
        if self.deseq is None:
            ax.text(0.5, 0.5, 'DESeq2 data not available', ha='center', va='center')
            ax.set_title('A. Tumor RNA-seq', fontsize=11, fontweight='bold')
            return

        gf_data = self.deseq[self.deseq['symbol'].isin(GROWTH_FACTORS)].copy()
        gf_data = gf_data.dropna(subset=['log2FoldChange', 'padj'])
        gf_data['neg_log10_padj'] = -np.log10(gf_data['padj'].clip(lower=1e-50))

        colors = []
        for _, row in gf_data.iterrows():
            if row['padj'] < 0.05 and row['log2FoldChange'] > 0:
                colors.append(COLORS['highlight'])
            elif row['padj'] < 0.05 and row['log2FoldChange'] < 0:
                colors.append(COLORS['primary'])
            else:
                colors.append(COLORS['ns'])

        ax.scatter(gf_data['log2FoldChange'], gf_data['neg_log10_padj'],
                   c=colors, s=50, alpha=0.7, edgecolors='black', linewidth=0.3)

        # Label significant genes
        sig_genes = gf_data[gf_data['padj'] < 0.05]
        for _, row in sig_genes.iterrows():
            gene = row['symbol']
            fontweight = 'bold' if gene == 'GDF15' else 'normal'
            ax.annotate(gene, xy=(row['log2FoldChange'], row['neg_log10_padj']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, fontweight=fontweight)

        ax.axhline(-np.log10(0.05), color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

        # Stats box
        gdf15 = gf_data[gf_data['symbol'] == 'GDF15']
        if len(gdf15) > 0:
            ax.text(0.98, 0.02, f"GDF15: log2FC = {gdf15['log2FoldChange'].values[0]:.2f}\n"
                    f"FDR q = {gdf15['padj'].values[0]:.3f}",
                    transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_xlabel('log2 Fold Change', fontsize=10)
        ax.set_ylabel('-log10(FDR q)', fontsize=10)
        ax.set_title('A. Tumor RNA-seq (n=15 paired)', fontsize=11, fontweight='bold')

    def _plot_blood_volcano(self, ax):
        """Panel B: Blood proteomics volcano plot."""
        if self.gf_results is None:
            ax.text(0.5, 0.5, 'Growth factor results not available', ha='center', va='center')
            ax.set_title('B. Blood Proteomics', fontsize=11, fontweight='bold')
            return

        gf_data = self.gf_results.dropna(subset=['mean_change', 'fdr_q'])
        gf_data['neg_log10_fdr'] = -np.log10(gf_data['fdr_q'].clip(lower=1e-50))

        gene_col = 'gene' if 'gene' in gf_data.columns else 'symbol'

        colors = []
        for _, row in gf_data.iterrows():
            if row['fdr_q'] < 0.05 and row['mean_change'] > 0:
                colors.append(COLORS['highlight'])
            elif row['fdr_q'] < 0.05 and row['mean_change'] < 0:
                colors.append(COLORS['primary'])
            else:
                colors.append(COLORS['ns'])

        ax.scatter(gf_data['mean_change'], gf_data['neg_log10_fdr'],
                   c=colors, s=50, alpha=0.7, edgecolors='black', linewidth=0.3)

        # Label significant
        sig_genes = gf_data[gf_data['fdr_q'] < 0.05]
        for _, row in sig_genes.iterrows():
            gene = row[gene_col]
            fontweight = 'bold' if gene == 'GDF15' else 'normal'
            ax.annotate(gene, xy=(row['mean_change'], row['neg_log10_fdr']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, fontweight=fontweight)

        ax.axhline(-np.log10(0.05), color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

        # Stats box
        gdf15 = gf_data[gf_data[gene_col] == 'GDF15']
        if len(gdf15) > 0:
            ax.text(0.98, 0.02, f"GDF15: dNPX = +{gdf15['mean_change'].values[0]:.2f}\n"
                    f"FDR q = {gdf15['fdr_q'].values[0]:.4f}",
                    transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_xlabel('Mean Change (dNPX)', fontsize=10)
        ax.set_ylabel('-log10(FDR q)', fontsize=10)
        ax.set_title('B. Blood Proteomics (n=33, SBRT alone)', fontsize=11, fontweight='bold')

    def _plot_tumor_blood_correlations(self, ax1, ax2, ax3):
        """Panels C1-C3: Tumor-blood GDF15 correlations."""
        # This requires tumor-blood merged data - placeholder for now
        for ax, title in [(ax1, 'C1. Baseline (NS)'),
                          (ax2, 'C2. On-Treatment'),
                          (ax3, 'C3. Change')]:
            ax.text(0.5, 0.5, 'Data required', ha='center', va='center', fontsize=9)
            ax.set_title(title, fontsize=10, fontweight='bold')

    def _plot_induction_kinetics(self, ax):
        """Panel D: Paired GDF15 induction plot."""
        if self.early_paired is None:
            ax.text(0.5, 0.5, 'Early-stage data not available', ha='center', va='center')
            ax.set_title('D. Induction Kinetics', fontsize=10, fontweight='bold')
            return

        for _, row in self.early_paired.iterrows():
            change = row['delta_GDF15']
            color = COLORS['increased'] if change > 0 else COLORS['decreased']
            alpha = 0.5 if change > 0 else 0.3
            ax.plot([0, 1], [row['GDF15_pre'], row['GDF15_post']],
                    'o-', color=color, alpha=alpha, linewidth=1, markersize=4)

        # Mean trajectory
        pre_mean = self.early_paired['GDF15_pre'].mean()
        post_mean = self.early_paired['GDF15_post'].mean()
        ax.plot([0, 1], [pre_mean, post_mean], 'o-', color='black',
                linewidth=3, markersize=10, zorder=10)

        # Stats
        n = len(self.early_paired)
        mean_change = self.early_paired['delta_GDF15'].mean()
        t_stat, p_val = stats.ttest_rel(self.early_paired['GDF15_post'],
                                         self.early_paired['GDF15_pre'])
        pct_inc = (self.early_paired['delta_GDF15'] > 0).mean() * 100

        ax.text(0.02, 0.98, f'n = {n}\ndNPX = +{mean_change:.2f}\np = {p_val:.1e}\n{pct_inc:.0f}% increased',
                transform=ax.transAxes, fontsize=9, ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Pre', 'Post'], fontsize=10)
        ax.set_ylabel('GDF15 (NPX)', fontsize=10)
        ax.set_title('D. Induction Kinetics', fontsize=10, fontweight='bold')
        ax.set_xlim(-0.3, 1.3)

    def _plot_fractionation(self, ax):
        """Panel E: SBRT fractionation effect."""
        if self.early_fx is None or len(self.early_fx) == 0:
            ax.text(0.5, 0.5, 'Fractionation data not available', ha='center', va='center')
            ax.set_title('E. Fractionation', fontsize=10, fontweight='bold')
            return

        fx_colors = {1: '#E57373', 3: '#64B5F6', 5: '#81C784'}

        for fx in [1, 3, 5]:
            fx_data = self.early_fx[self.early_fx['SBRT_fx'] == fx]['delta_GDF15']
            if len(fx_data) == 0:
                continue

            bp = ax.boxplot([fx_data], positions=[fx], widths=0.8, patch_artist=True,
                            showfliers=False)
            bp['boxes'][0].set_facecolor(fx_colors.get(fx, 'gray'))
            bp['boxes'][0].set_alpha(0.4)
            bp['boxes'][0].set_edgecolor('black')
            bp['medians'][0].set_color('black')
            bp['medians'][0].set_linewidth(2)

            # Jittered points
            np.random.seed(42)
            jitter = np.random.uniform(-0.15, 0.15, len(fx_data))
            ax.scatter([fx] * len(fx_data) + jitter, fx_data,
                       c=fx_colors.get(fx, 'gray'), s=40, alpha=0.7,
                       edgecolor='white', linewidth=0.5, zorder=3)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # Spearman correlation
        rho, p = stats.spearmanr(self.early_fx['SBRT_fx'], self.early_fx['delta_GDF15'])
        ax.text(0.05, 0.95, f'rho = {rho:.2f}\np = {p:.3f}',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel('SBRT Fractions', fontsize=10)
        ax.set_ylabel('dGDF15 (NPX)', fontsize=10)
        ax.set_title('E. Fractionation', fontsize=10, fontweight='bold')
        ax.set_xticks([1, 3, 5])
        ax.set_xlim(0, 6)

        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=fx_colors[1], alpha=0.6, edgecolor='black',
                          label=f'1fx (n={len(self.early_fx[self.early_fx["SBRT_fx"]==1])})'),
            mpatches.Patch(facecolor=fx_colors[3], alpha=0.6, edgecolor='black',
                          label=f'3fx (n={len(self.early_fx[self.early_fx["SBRT_fx"]==3])})'),
            mpatches.Patch(facecolor=fx_colors[5], alpha=0.6, edgecolor='black',
                          label=f'5fx (n={len(self.early_fx[self.early_fx["SBRT_fx"]==5])})')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=7)

    def figure2_survival(self):
        """Figure 2: GDF15 predicts overall survival."""
        print("\nCreating Figure 2: Survival Analysis...")

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)

        # Prepare survival data
        surv_df = self.cosinr[['p.GDF15.T1', 'p.GDF15.T3', 'os_time', 'event_death', 'arm']].copy()
        surv_df = surv_df.dropna(subset=['os_time', 'event_death'])
        surv_df['p.GDF15.dif1v3'] = surv_df['p.GDF15.T3'] - surv_df['p.GDF15.T1']

        # Panel A: Baseline KM
        ax_a = fig.add_subplot(gs[0, 0])
        self._plot_baseline_km(ax_a, surv_df)

        # Panel B: Change KM
        ax_b = fig.add_subplot(gs[0, 1])
        self._plot_change_km(ax_b, surv_df)

        # Panel C: Four-group KM
        ax_c = fig.add_subplot(gs[0, 2])
        self._plot_four_group_km(ax_c, surv_df)

        # Panel D: Forest plot
        ax_d = fig.add_subplot(gs[1, 0:2])
        self._plot_forest(ax_d, surv_df)

        # Panel E: Responders only
        ax_e = fig.add_subplot(gs[1, 2])
        self._plot_responders(ax_e)

        plt.suptitle('Figure 2: GDF15 Predicts Overall Survival',
                     fontsize=14, fontweight='bold', y=0.98)

        plt.savefig(self.output_dir / 'Figure2_Survival.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'Figure2_Survival.pdf', bbox_inches='tight')
        plt.close()

        print(f"  Saved: {self.output_dir / 'Figure2_Survival.png'}")

    def _plot_baseline_km(self, ax, surv_df):
        """Panel A: Baseline GDF15 Kaplan-Meier."""
        df = surv_df.dropna(subset=['p.GDF15.T1'])
        median_gdf15 = df['p.GDF15.T1'].median()

        high = df[df['p.GDF15.T1'] >= median_gdf15]
        low = df[df['p.GDF15.T1'] < median_gdf15]

        kmf = KaplanMeierFitter()

        kmf.fit(high['os_time'], high['event_death'], label=f'High (n={len(high)})')
        kmf.plot_survival_function(ax=ax, ci_show=True, color=COLORS['high'])

        kmf.fit(low['os_time'], low['event_death'], label=f'Low (n={len(low)})')
        kmf.plot_survival_function(ax=ax, ci_show=True, color=COLORS['low'])

        lr = logrank_test(high['os_time'], low['os_time'],
                          high['event_death'], low['event_death'])

        ax.text(0.98, 0.02, f'Log-rank p = {lr.p_value:.3f}',
                transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_xlabel('Time (months)', fontsize=10)
        ax.set_ylabel('Overall Survival', fontsize=10)
        ax.set_title('A. Baseline GDF15', fontsize=11, fontweight='bold')
        ax.legend(loc='lower left', fontsize=9)

    def _plot_change_km(self, ax, surv_df):
        """Panel B: GDF15 change Kaplan-Meier."""
        df = surv_df.dropna(subset=['p.GDF15.dif1v3'])

        increased = df[df['p.GDF15.dif1v3'] > 0]
        decreased = df[df['p.GDF15.dif1v3'] <= 0]

        kmf = KaplanMeierFitter()

        kmf.fit(increased['os_time'], increased['event_death'],
                label=f'Increased (n={len(increased)})')
        kmf.plot_survival_function(ax=ax, ci_show=True, color=COLORS['increased'])

        kmf.fit(decreased['os_time'], decreased['event_death'],
                label=f'Decreased (n={len(decreased)})')
        kmf.plot_survival_function(ax=ax, ci_show=True, color=COLORS['decreased'])

        lr = logrank_test(increased['os_time'], decreased['os_time'],
                          increased['event_death'], decreased['event_death'])

        ax.text(0.98, 0.02, f'Log-rank p = {lr.p_value:.3f}',
                transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_xlabel('Time (months)', fontsize=10)
        ax.set_ylabel('Overall Survival', fontsize=10)
        ax.set_title('B. GDF15 Change', fontsize=11, fontweight='bold')
        ax.legend(loc='lower left', fontsize=9)

    def _plot_four_group_km(self, ax, surv_df):
        """Panel C: Four-group stratification."""
        df = surv_df.dropna(subset=['p.GDF15.T1', 'p.GDF15.dif1v3'])

        median_baseline = df['p.GDF15.T1'].median()
        df['baseline_high'] = df['p.GDF15.T1'] >= median_baseline
        df['change_pos'] = df['p.GDF15.dif1v3'] > 0

        groups = {
            'Low/Dec': (~df['baseline_high']) & (~df['change_pos']),
            'Low/Inc': (~df['baseline_high']) & (df['change_pos']),
            'High/Dec': (df['baseline_high']) & (~df['change_pos']),
            'High/Inc': (df['baseline_high']) & (df['change_pos'])
        }

        colors = [COLORS['low_dec'], COLORS['low_inc'],
                  COLORS['high_dec'], COLORS['high_inc']]

        kmf = KaplanMeierFitter()
        for (name, mask), color in zip(groups.items(), colors):
            grp = df[mask]
            if len(grp) > 0:
                kmf.fit(grp['os_time'], grp['event_death'], label=f'{name} (n={len(grp)})')
                kmf.plot_survival_function(ax=ax, ci_show=False, color=color)

        mlr = multivariate_logrank_test(df['os_time'],
                                        df['baseline_high'].astype(int) * 2 + df['change_pos'].astype(int),
                                        df['event_death'])

        ax.text(0.98, 0.02, f'Log-rank p = {mlr.p_value:.4f}',
                transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_xlabel('Time (months)', fontsize=10)
        ax.set_ylabel('Overall Survival', fontsize=10)
        ax.set_title('C. Four-Group Stratification', fontsize=11, fontweight='bold')
        ax.legend(loc='lower left', fontsize=8)

    def _plot_forest(self, ax, surv_df):
        """Panel D: Forest plot placeholder."""
        ax.text(0.5, 0.5, 'Forest plot - see compute_all_statistics.py for values',
                ha='center', va='center', fontsize=10)
        ax.set_title('D. Forest Plot: GDF15 vs Tumor Metrics', fontsize=11, fontweight='bold')
        ax.axis('off')

    def _plot_responders(self, ax):
        """Panel E: Responders-only analysis placeholder."""
        ax.text(0.5, 0.5, 'Responders analysis\n(CR/PR subgroup)',
                ha='center', va='center', fontsize=10)
        ax.set_title('E. Responders Only', fontsize=11, fontweight='bold')
        ax.axis('off')

    def generate_all_figures(self):
        """Generate all manuscript figures."""
        print("\n" + "=" * 70)
        print("GENERATING ALL FIGURES")
        print("=" * 70)

        self.figure1_discovery()
        self.figure2_survival()

        print("\n" + "=" * 70)
        print("FIGURE GENERATION COMPLETE")
        print(f"Figures saved to: {self.output_dir}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Generate GDF15 manuscript figures')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default='./figures',
                        help='Path to output directory (default: ./figures)')

    args = parser.parse_args()

    generator = GDF15FigureGenerator(args.data_dir, args.output_dir)
    generator.generate_all_figures()


if __name__ == "__main__":
    main()
