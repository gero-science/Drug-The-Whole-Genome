#!/usr/bin/env python3
"""Analyze performance differences between template-based and Fpocket pockets.

Classifies each pocket as 'template' (PDBbind structural alignment) or
'fpocket' (Fpocket + GenPack refinement) based on directory naming, then
compares AUROC distributions between groups.

Output:
    reproducibility/results/pocket_type_comparison.tsv
    Printed summary statistics
"""

import os
import sys
import re
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def classify_pocket(dirname):
    """Template-based dirs have no 'pocket' in name; Fpocket dirs do."""
    return 'template' if 'pocket' not in dirname else 'fpocket'


def main():
    data_dir = os.path.join(PROJECT_ROOT, "data")

    # Load annotations
    annot = pd.read_csv(os.path.join(data_dir, "pocket_annotations_full.tsv"), sep="\t")
    annot['pocket_type'] = annot['pocket_dir'].apply(classify_pocket)

    print("=== Pocket Type Distribution ===")
    print(annot['pocket_type'].value_counts())
    print()

    # Per-protein classification
    per_prot = annot.groupby('uniprot_id').agg(
        n_total=('pocket_type', 'count'),
        n_template=('pocket_type', lambda x: (x == 'template').sum()),
        n_fpocket=('pocket_type', lambda x: (x == 'fpocket').sum()),
    ).reset_index()

    per_prot['category'] = 'fpocket_only'
    per_prot.loc[(per_prot['n_template'] > 0) & (per_prot['n_fpocket'] > 0), 'category'] = 'both'
    per_prot.loc[(per_prot['n_template'] > 0) & (per_prot['n_fpocket'] == 0), 'category'] = 'template_only'

    print("=== Per-Protein Coverage ===")
    print(per_prot['category'].value_counts())
    print(f"Total proteins: {len(per_prot)}")
    print()

    # Merge with benchmark metrics
    bench_path = os.path.join(data_dir, "benchmark_metrics.tsv")
    if os.path.exists(bench_path):
        bench = pd.read_csv(bench_path, sep="\t")
        merged = bench.merge(per_prot[['uniprot_id', 'category', 'n_template', 'n_fpocket']],
                             on='uniprot_id', how='left')

        print("=== Benchmark Performance by Pocket Type ===")
        for cat in ['template_only', 'fpocket_only', 'both']:
            subset = merged[merged['category'] == cat]
            if len(subset) == 0:
                continue
            print(f"\n  {cat} (n={len(subset)}):")
            print(f"    Median AUROC: {subset['auroc'].median():.4f}")
            print(f"    Mean AUROC:   {subset['auroc'].mean():.4f}")
            print(f"    Median EF1%:  {subset['ef1'].median():.4f}")
            print(f"    Mean EF1%:    {subset['ef1'].mean():.4f}")
            print(f"    % AUROC>0.6:  {100*(subset['auroc']>0.6).mean():.1f}%")
            print(f"    % AUROC>0.7:  {100*(subset['auroc']>0.7).mean():.1f}%")

        # Statistical test
        from scipy.stats import mannwhitneyu
        template_auc = merged[merged['category'].isin(['template_only', 'both'])]['auroc'].dropna()
        fpocket_auc = merged[merged['category'] == 'fpocket_only']['auroc'].dropna()
        if len(template_auc) > 0 and len(fpocket_auc) > 0:
            U, p = mannwhitneyu(template_auc, fpocket_auc, alternative='two-sided')
            print(f"\n  Mann-Whitney U test (template vs fpocket): U={U:.0f}, p={p:.4f}")

        # Save
        out_path = os.path.join(PROJECT_ROOT, "reproducibility/results/pocket_type_comparison.tsv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        merged[['uniprot_id', 'auroc', 'ef1', 'category', 'n_template', 'n_fpocket']].to_csv(
            out_path, sep="\t", index=False)
        print(f"\n  Saved to {out_path}")

    # Extract PDB template IDs from screen_results
    screen_dir = os.path.join(data_dir, "screen_results")
    if os.path.isdir(screen_dir):
        template_dirs = annot[annot['pocket_type'] == 'template']['pocket_dir'].unique()
        pdb_template_counts = defaultdict(int)
        for d in template_dirs[:1000]:
            full = os.path.join(screen_dir, d)
            if not os.path.isdir(full):
                continue
            for f in os.listdir(full):
                m = re.search(r'_([a-z0-9]{4})_complex_refined', f)
                if m:
                    pdb_template_counts[m.group(1)] += 1

        print(f"\n=== PDB Template Sources (from {min(1000, len(template_dirs))} dirs) ===")
        print(f"Unique PDB templates: {len(pdb_template_counts)}")
        for pdb, count in sorted(pdb_template_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {pdb}: {count} uses")


if __name__ == "__main__":
    main()
