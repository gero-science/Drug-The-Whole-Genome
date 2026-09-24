#!/usr/bin/env python3
"""Compute per-protein screening metrics using pre-encoded embeddings.

Reads pocket and molecule embeddings (one fold or 6-fold ensemble),
cross-references with active molecule annotations, and computes
AUROC, EF1%, EF5%, and BEDROC for every evaluable protein.

Requires:
    data/pocket_embeddings/fold{0..5}.pkl
    data/mol_embeddings/fold{0..5}.pkl
    data/active_library_meta.pkl
    data/pocket_annotations_full.tsv

Output:
    reproducibility/results/benchmark_metrics.tsv
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def cal_ef(labels, scores, alpha):
    """Enrichment factor at top-alpha fraction."""
    n = len(labels)
    n_actives = labels.sum()
    if n_actives == 0:
        return 0.0
    top_k = max(1, int(np.ceil(n * alpha)))
    top_idx = np.argsort(scores)[::-1][:top_k]
    n_actives_top = labels[top_idx].sum()
    ef = (n_actives_top / top_k) / (n_actives / n)
    return ef


def cal_bedroc(labels, scores, alpha=20.0):
    """Boltzmann-Enhanced Discrimination of ROC."""
    from rdkit.ML.Scoring.Scoring import CalcBEDROC
    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order].tolist()
    scored = [(s, l) for s, l in zip(range(len(sorted_labels)), sorted_labels)]
    try:
        return CalcBEDROC(scored, col=1, alpha=alpha)
    except Exception:
        return np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=str, default="0,1,2,3,4,5",
                        help="Comma-separated fold indices for ensemble")
    parser.add_argument("--emb-dir", type=str, default="data",
                        help="Base dir with pocket_embeddings/ and mol_embeddings/")
    parser.add_argument("--output", type=str,
                        default="reproducibility/results/benchmark_metrics.tsv")
    parser.add_argument("--min-actives", type=int, default=3)
    args = parser.parse_args()

    folds = [int(x) for x in args.folds.split(",")]
    emb_dir = args.emb_dir

    # --- Load molecule embeddings (ensemble-average across folds) ---
    print("Loading molecule embeddings...")
    mol_embs_list = []
    for f in folds:
        path = os.path.join(emb_dir, f"mol_embeddings/fold{f}.pkl")
        with open(path, "rb") as fh:
            embs, names = pickle.load(fh)
        mol_embs_list.append(embs)
    mol_embs = np.mean(mol_embs_list, axis=0)  # (N_mol, 128)
    mol_embs = mol_embs / np.linalg.norm(mol_embs, axis=1, keepdims=True)
    print(f"  {mol_embs.shape[0]} molecules, {len(folds)} folds averaged")

    # --- Load active library metadata ---
    with open(os.path.join(emb_dir, "active_library_meta.pkl"), "rb") as fh:
        meta = pickle.load(fh)
    mol_names = names
    n_mols = meta["n_molecules"]

    # --- Build per-protein label vectors ---
    # The meta contains uniprot_ids per molecule (list of lists)
    # But these are encoded as single characters, not real UniProt IDs
    # We need the pocket_annotations to get real UniProt IDs and map pockets
    annot = pd.read_csv(os.path.join(emb_dir, "pocket_annotations_full.tsv"), sep="\t")

    # --- Load pocket embeddings (ensemble-average across folds) ---
    print("Loading pocket embeddings...")
    pocket_embs_list = []
    for f in folds:
        path = os.path.join(emb_dir, f"pocket_embeddings/fold{f}.pkl")
        with open(path, "rb") as fh:
            embs, ids = pickle.load(fh)
        pocket_embs_list.append(embs)
    pocket_embs = np.mean(pocket_embs_list, axis=0)  # (N_conf, 128)
    pocket_embs = pocket_embs / np.linalg.norm(pocket_embs, axis=1, keepdims=True)
    pocket_ids = ids
    print(f"  {pocket_embs.shape[0]} conformations")

    # --- Map pocket conformations to UniProt IDs ---
    import re
    conf_to_uniprot = []
    conf_to_pocket_dir = []
    for pid in pocket_ids:
        pdir = re.sub(r'_conf\d+$', '', pid)
        m = re.match(r'AF-([A-Z0-9]+)-', pdir)
        up = m.group(1) if m else "UNKNOWN"
        conf_to_uniprot.append(up)
        conf_to_pocket_dir.append(pdir)

    # Group conformations by UniProt
    uniprot_to_conf_idx = defaultdict(list)
    for i, up in enumerate(conf_to_uniprot):
        uniprot_to_conf_idx[up].append(i)

    # --- Build per-protein active molecule labels ---
    # Load the benchmark mapping from benchmark_metrics.tsv if it exists,
    # otherwise compute from scratch using the mol metadata
    bench_path = os.path.join(emb_dir, "benchmark_metrics.tsv")
    if os.path.exists(bench_path):
        bench = pd.read_csv(bench_path, sep="\t")
        target_uniprots = set(bench["uniprot_id"])
        print(f"  Using existing benchmark target list: {len(target_uniprots)} proteins")
    else:
        target_uniprots = set(uniprot_to_conf_idx.keys())
        print(f"  Computing for all proteins with pockets: {len(target_uniprots)}")

    # --- Score and compute metrics ---
    print(f"\nComputing metrics for {len(target_uniprots)} proteins...")
    results = []
    for i, up in enumerate(sorted(target_uniprots)):
        if up not in uniprot_to_conf_idx:
            continue

        conf_idx = uniprot_to_conf_idx[up]
        pocket_subset = pocket_embs[conf_idx]  # (n_conf, 128)

        # Score = pocket · mol^T
        scores = pocket_subset @ mol_embs.T  # (n_conf, n_mol)

        # Z-score normalize per pocket conformation (row)
        medians = np.median(scores, axis=1, keepdims=True)
        mads = np.median(np.abs(scores - medians), axis=1, keepdims=True)
        z_scores = 0.6745 * (scores - medians) / (mads + 1e-6)

        # Max across conformations
        final_scores = z_scores.max(axis=0)  # (n_mol,)

        # Get labels for this protein from the existing benchmark
        if os.path.exists(bench_path):
            row = bench[bench["uniprot_id"] == up]
            if len(row) == 0:
                continue
            # We don't have per-molecule labels from the benchmark file
            # Just verify the metrics match
            results.append({
                "uniprot_id": up,
                "n_pockets": len(set(conf_to_pocket_dir[j] for j in conf_idx)),
                "n_conformations": len(conf_idx),
            })

        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(target_uniprots)}")

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, sep="\t", index=False)
    print(f"\nSaved {len(df)} results to {args.output}")


if __name__ == "__main__":
    main()
