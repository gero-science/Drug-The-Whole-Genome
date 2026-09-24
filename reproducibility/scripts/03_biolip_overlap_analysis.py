#!/usr/bin/env python3
"""Test hypothesis: training data leakage from BioLiP2 explains performance.

Downloads BioLiP2 database, maps UniProt IDs to PDB structures,
cross-references with benchmark proteins, and tests correlation
between BioLiP presence/count and AUROC.

Output:
    reproducibility/results/biolip_overlap.tsv
    Printed correlation statistics
"""

import os
import sys
import gzip
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import spearmanr, mannwhitneyu

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_biolip(biolip_path):
    """Parse BioLiP2 database to extract UniProt -> PDB ID mapping."""
    uniprot_to_pdbs = defaultdict(set)
    n_entries = 0

    opener = gzip.open if biolip_path.endswith('.gz') else open
    with opener(biolip_path, 'rt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 18:
                continue
            pdb_id = parts[0]
            uniprot_id = parts[17].strip()
            if uniprot_id and len(uniprot_id) >= 4:
                uniprot_to_pdbs[uniprot_id].add(pdb_id)
            n_entries += 1

    print(f"BioLiP2: {n_entries} entries, {len(uniprot_to_pdbs)} unique UniProt IDs")
    return uniprot_to_pdbs


def main():
    data_dir = os.path.join(PROJECT_ROOT, "data")
    results_dir = os.path.join(PROJECT_ROOT, "reproducibility/results")
    os.makedirs(results_dir, exist_ok=True)

    # Load benchmark metrics
    bench = pd.read_csv(os.path.join(data_dir, "benchmark_metrics.tsv"), sep="\t")
    print(f"Benchmark proteins: {len(bench)}")

    # Load or parse BioLiP
    biolip_cache = os.path.join(results_dir, "biolip_uniprot_to_pdbs.pkl")
    biolip_path = None

    # Try common locations
    for candidate in [
        os.path.join(data_dir, "BioLiP.txt.gz"),
        os.path.join(data_dir, "BioLiP.txt"),
        os.path.expanduser("~/data/BioLiP.txt.gz"),
    ]:
        if os.path.exists(candidate):
            biolip_path = candidate
            break

    if biolip_path and not os.path.exists(biolip_cache):
        uniprot_to_pdbs = parse_biolip(biolip_path)
        with open(biolip_cache, 'wb') as f:
            pickle.dump(dict(uniprot_to_pdbs), f)
    elif os.path.exists(biolip_cache):
        with open(biolip_cache, 'rb') as f:
            uniprot_to_pdbs = pickle.load(f)
        print(f"Loaded BioLiP cache: {len(uniprot_to_pdbs)} UniProt IDs")
    else:
        print("ERROR: BioLiP database not found. Download from https://zhanggroup.org/BioLiP/")
        sys.exit(1)

    # Cross-reference
    bench['in_biolip'] = bench['uniprot_id'].isin(uniprot_to_pdbs)
    bench['biolip_structures'] = bench['uniprot_id'].map(
        lambda x: len(uniprot_to_pdbs.get(x, set()))
    )

    n_in = bench['in_biolip'].sum()
    n_out = (~bench['in_biolip']).sum()
    print(f"\nIn BioLiP: {n_in} ({100*n_in/len(bench):.1f}%)")
    print(f"Not in BioLiP: {n_out} ({100*n_out/len(bench):.1f}%)")

    # Correlation test
    rho, p = spearmanr(bench['biolip_structures'], bench['auroc'])
    print(f"\nSpearman ρ(BioLiP structures, AUROC): {rho:.4f}, p={p:.4f}")

    # Group comparison
    in_auc = bench[bench['in_biolip']]['auroc']
    out_auc = bench[~bench['in_biolip']]['auroc']
    U, p_mw = mannwhitneyu(in_auc, out_auc, alternative='two-sided')
    print(f"\nMedian AUROC (in BioLiP): {in_auc.median():.4f}")
    print(f"Median AUROC (not in BioLiP): {out_auc.median():.4f}")
    print(f"Mann-Whitney p: {p_mw:.4f}")

    # Proteins with many BioLiP structures
    heavy = bench[bench['biolip_structures'] > 100]
    if len(heavy) > 0:
        print(f"\nProteins with >100 BioLiP structures: {len(heavy)}")
        print(f"  Median AUROC: {heavy['auroc'].median():.4f}")

    # Save
    out_path = os.path.join(results_dir, "biolip_overlap.tsv")
    bench[['uniprot_id', 'auroc', 'ef1', 'in_biolip', 'biolip_structures']].to_csv(
        out_path, sep="\t", index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
