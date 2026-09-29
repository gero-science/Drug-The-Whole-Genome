#!/usr/bin/env python3
"""Crystal vs AlphaFold2 pocket comparison analysis.

For each of the 28 drug targets:
1. Compute AUC using crystal (holo) pocket embeddings
2. Compute AUC using AF2 pocket embeddings
3. Compute cosine similarity between crystal and AF2 pocket embeddings
4. Load pre-computed RMSD between structures
5. Show that RMSD does NOT explain the embedding/AUC disconnect

Demonstrates that DrugCLIP works on crystal structures but fails on AF2,
and that the failure is not explained by structural similarity.

Output:
    reproducibility/results/crystal_vs_af2_comparison.tsv
    reproducibility/results/crystal_vs_af2_summary.txt
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_embeddings(pkl_path):
    """Load [embeddings, names] from pickle."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data[0], data[1]  # embeddings (N, 128), names list


def compute_auc_for_target(target, uniprot, pocket_embs, pocket_names,
                           mol_embs, mol_labels, mol_uniprot_labels):
    """Compute AUC for a single target using given pocket embeddings.

    Uses max z-normalized cosine similarity across pocket conformations,
    matching the paper's retrieval protocol.
    """
    # Find pocket indices for this target
    pocket_indices = [i for i, name in enumerate(pocket_names)
                      if name.startswith(f"{target}/")]

    if len(pocket_indices) == 0:
        return None, 0

    pocket_vecs = pocket_embs[pocket_indices]  # (n_pockets, 128)

    # Compute scores: mol_embs @ pocket_vecs.T -> (n_mols, n_pockets)
    scores = mol_embs @ pocket_vecs.T  # cosine similarity (both normalized)

    # Z-score normalize per pocket (column)
    medians = np.median(scores, axis=0, keepdims=True)
    mad = np.median(np.abs(scores - medians), axis=0, keepdims=True)
    z_scores = 0.6745 * (scores - medians) / (mad + 1e-6)

    # Max across pockets for each molecule
    max_z = z_scores.max(axis=1)

    # Get labels for this target
    # Find the label index for this uniprot
    labels = mol_labels.get(uniprot)
    if labels is None:
        return None, len(pocket_indices)

    # Need at least 3 actives
    n_actives = labels.sum()
    if n_actives < 3:
        return None, len(pocket_indices)

    try:
        auc = roc_auc_score(labels, max_z)
    except ValueError:
        return None, len(pocket_indices)

    return auc, len(pocket_indices)


def compute_cosine_similarity(crystal_embs, crystal_names, af2_embs, af2_names, target):
    """Compute mean cosine similarity between crystal and AF2 embeddings for a target."""
    crystal_idx = [i for i, n in enumerate(crystal_names) if n.startswith(f"{target}/")]
    af2_idx = [i for i, n in enumerate(af2_names) if n.startswith(f"AF-")]

    if not crystal_idx or not af2_idx:
        return None, None, None

    c_vecs = crystal_embs[crystal_idx]  # (n_crystal, 128)
    a_vecs = af2_embs[af2_idx]  # (n_af2, 128)

    # Pairwise cosine similarities
    sim_matrix = c_vecs @ a_vecs.T  # (n_crystal, n_af2)

    mean_sim = sim_matrix.mean()
    max_sim = sim_matrix.max()

    # Also compute mean of crystal centroid vs AF2 centroid
    c_centroid = c_vecs.mean(axis=0)
    c_centroid /= np.linalg.norm(c_centroid)
    a_centroid = a_vecs.mean(axis=0)
    a_centroid /= np.linalg.norm(a_centroid)
    centroid_sim = float(c_centroid @ a_centroid)

    return float(mean_sim), float(max_sim), centroid_sim


def main():
    data_dir = os.path.join(PROJECT_ROOT, "data")
    results_dir = os.path.join(PROJECT_ROOT, "reproducibility", "results")
    os.makedirs(results_dir, exist_ok=True)

    # Target to UniProt mapping
    target_mapping = pd.read_csv(os.path.join(results_dir, "target_mapping.tsv"), sep="\t")

    # Load molecule data
    meta_path = os.path.join(data_dir, "active_library_meta.pkl")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    # Build per-protein labels using pocket_annotations for UniProt mapping
    annot = pd.read_csv(os.path.join(data_dir, "pocket_annotations_full.tsv"), sep="\t")

    # Load molecule embeddings (ensemble average of 6 folds)
    print("Loading molecule embeddings...")
    mol_embs_list = []
    for fold in range(6):
        mol_path = os.path.join(data_dir, "mol_embeddings", f"fold{fold}.pkl")
        with open(mol_path, 'rb') as f:
            emb_data = pickle.load(f)
        mol_embs_list.append(emb_data[0])
    mol_embs = np.mean(mol_embs_list, axis=0)
    mol_embs = mol_embs / np.linalg.norm(mol_embs, axis=1, keepdims=True)
    print(f"  Molecules: {mol_embs.shape}")

    # Build label vectors per UniProt target
    # meta['labels'] has shape (n_mols, n_proteins) or similar
    labels_array = meta['labels']  # sparse or dense
    if hasattr(labels_array, 'toarray'):
        labels_array = labels_array.toarray()

    # meta['uniprot_ids'] are single chars (0-9, A-Z), not real UniProt IDs
    # We need to use the target-specific ChEMBL data instead
    print("\nLoading target-specific ChEMBL labels...")
    mol_smiles = meta['smiles']

    # For each target, load its ChEMBL actives and match to library
    from rdkit import Chem

    # Build SMILES -> index map for the molecule library
    smi_to_idx = {}
    n_parsed = 0
    for i, smi in enumerate(mol_smiles):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                canon = Chem.MolToSmiles(mol)
                smi_to_idx[canon] = i
                n_parsed += 1
            else:
                smi_to_idx[smi] = i
        except:
            smi_to_idx[smi] = i
    print(f"  Parsed {n_parsed}/{len(mol_smiles)} library SMILES")

    target_labels = {}  # uniprot -> binary array (n_mols,)

    for _, row in target_mapping.iterrows():
        target = row['target']
        uniprot = row['uniprot']

        # Try loading ChEMBL active LMDB
        import lmdb
        active_path = os.path.join(data_dir, "targets", target, "ChEMBL", "active.lmdb")
        if not os.path.exists(active_path):
            continue

        env = lmdb.open(active_path, subdir=False, readonly=True, lock=False)
        active_smiles = set()
        with env.begin() as txn:
            n = txn.stat()['entries']
            for i in range(n):
                raw = txn.get(str(i).encode())
                if raw is None:
                    continue
                entry = pickle.loads(raw)
                smi = entry.get('smi', '')
                if smi:
                    # Strip CHEMBLID_ prefix if present
                    if '_' in smi and smi.split('_')[0].startswith('CHEMBL'):
                        smi = smi.split('_', 1)[1]
                    try:
                        mol = Chem.MolFromSmiles(smi)
                        if mol is not None:
                            canon = Chem.MolToSmiles(mol)
                            active_smiles.add(canon)
                    except:
                        pass
        env.close()

        # Build binary label vector
        labels = np.zeros(len(mol_smiles), dtype=np.int32)
        n_found = 0
        for smi in active_smiles:
            if smi in smi_to_idx:
                labels[smi_to_idx[smi]] = 1
                n_found += 1

        if n_found >= 3:
            target_labels[uniprot] = labels
            print(f"  {target} ({uniprot}): {n_found} actives in library (of {len(active_smiles)} ChEMBL)")
        else:
            print(f"  {target} ({uniprot}): only {n_found} actives in library, skipping")

    # Load crystal pocket embeddings (ensemble average)
    print("\nLoading crystal pocket embeddings...")
    crystal_embs_list = []
    crystal_names = None
    for fold in range(6):
        crystal_path = os.path.join(data_dir, "crystal_pocket_embeddings", f"fold{fold}.pkl")
        if not os.path.exists(crystal_path):
            print(f"  WARNING: {crystal_path} not found")
            continue
        embs, names = load_embeddings(crystal_path)
        crystal_embs_list.append(embs)
        if crystal_names is None:
            crystal_names = names

    if not crystal_embs_list:
        print("ERROR: No crystal pocket embeddings found!")
        sys.exit(1)

    crystal_embs = np.mean(crystal_embs_list, axis=0)
    crystal_embs = crystal_embs / np.linalg.norm(crystal_embs, axis=1, keepdims=True)
    print(f"  Crystal pockets: {crystal_embs.shape}")

    # Load AF2 pocket embeddings (ensemble average)
    print("\nLoading AF2 pocket embeddings...")
    af2_embs_list = []
    af2_names = None
    for fold in range(6):
        af2_path = os.path.join(data_dir, "pocket_embeddings", f"fold{fold}.pkl")
        embs, names = load_embeddings(af2_path)
        af2_embs_list.append(embs)
        if af2_names is None:
            af2_names = names

    af2_embs = np.mean(af2_embs_list, axis=0)
    af2_embs = af2_embs / np.linalg.norm(af2_embs, axis=1, keepdims=True)
    print(f"  AF2 pockets: {af2_embs.shape}")

    # Load RMSD data
    rmsd_df = pd.read_csv(os.path.join(results_dir, "crystal_vs_af2_rmsd.tsv"), sep="\t")

    # For each target: compute crystal AUC, AF2 AUC, cosine similarity
    print("\n" + "="*80)
    print("CRYSTAL vs AF2 COMPARISON")
    print("="*80)

    results = []

    for _, row in target_mapping.iterrows():
        target = row['target']
        uniprot = row['uniprot']

        if uniprot not in target_labels:
            continue

        labels = target_labels[uniprot]
        n_actives = labels.sum()

        # Crystal AUC
        crystal_auc, n_crystal = compute_auc_for_target(
            target, uniprot, crystal_embs, crystal_names,
            mol_embs, {uniprot: labels}, None)

        # AF2 AUC — need to find AF2 pockets for this UniProt
        # AF2 pocket names look like: "AF-{UNIPROT}-F1-model_v4_0_pocket5/conformation_0"
        af2_pocket_indices = [i for i, name in enumerate(af2_names)
                             if f"AF-{uniprot}-" in name]

        if af2_pocket_indices:
            af2_pocket_vecs = af2_embs[af2_pocket_indices]
            af2_pocket_names_sub = [af2_names[i] for i in af2_pocket_indices]

            # Compute scores
            scores = mol_embs @ af2_pocket_vecs.T
            medians = np.median(scores, axis=0, keepdims=True)
            mad = np.median(np.abs(scores - medians), axis=0, keepdims=True)
            z_scores = 0.6745 * (scores - medians) / (mad + 1e-6)
            max_z = z_scores.max(axis=1)

            try:
                af2_auc = roc_auc_score(labels, max_z)
            except:
                af2_auc = None
            n_af2 = len(af2_pocket_indices)
        else:
            af2_auc = None
            n_af2 = 0

        # Cosine similarity between crystal and AF2 embeddings
        if af2_pocket_indices and n_crystal > 0:
            c_idx = [i for i, n in enumerate(crystal_names) if n.startswith(f"{target}/")]
            c_vecs = crystal_embs[c_idx]
            a_vecs = af2_embs[af2_pocket_indices]

            sim_matrix = c_vecs @ a_vecs.T
            mean_cos_sim = float(sim_matrix.mean())
            max_cos_sim = float(sim_matrix.max())

            # Centroid similarity
            c_cent = c_vecs.mean(axis=0)
            c_cent /= np.linalg.norm(c_cent)
            a_cent = a_vecs.mean(axis=0)
            a_cent /= np.linalg.norm(a_cent)
            centroid_cos_sim = float(c_cent @ a_cent)
        else:
            mean_cos_sim = max_cos_sim = centroid_cos_sim = None

        # Get RMSD
        rmsd_row = rmsd_df[rmsd_df['target'] == target]
        aligned_rmsd = rmsd_row['aligned_rmsd'].values[0] if len(rmsd_row) > 0 else None
        if pd.isna(aligned_rmsd):
            aligned_rmsd = None

        result = {
            'target': target,
            'uniprot': uniprot,
            'n_actives': int(n_actives),
            'n_crystal_pockets': n_crystal,
            'n_af2_pockets': n_af2,
            'crystal_auc': crystal_auc,
            'af2_auc': af2_auc,
            'auc_delta': (crystal_auc - af2_auc) if (crystal_auc and af2_auc) else None,
            'mean_cos_sim': mean_cos_sim,
            'max_cos_sim': max_cos_sim,
            'centroid_cos_sim': centroid_cos_sim,
            'aligned_rmsd': aligned_rmsd,
        }
        results.append(result)

        print(f"\n{target} ({uniprot}):")
        print(f"  Actives: {n_actives}, Crystal pockets: {n_crystal}, AF2 pockets: {n_af2}")
        if crystal_auc is not None:
            print(f"  Crystal AUC: {crystal_auc:.3f}")
        if af2_auc is not None:
            print(f"  AF2 AUC:     {af2_auc:.3f}")
        if crystal_auc and af2_auc:
            delta = crystal_auc - af2_auc
            print(f"  ΔAUC:        {delta:+.3f} ({'crystal better' if delta > 0 else 'AF2 better'})")
        if mean_cos_sim is not None:
            print(f"  Embedding cosine sim: mean={mean_cos_sim:.3f}, max={max_cos_sim:.3f}, centroid={centroid_cos_sim:.3f}")
        if aligned_rmsd is not None:
            print(f"  Aligned RMSD: {aligned_rmsd:.2f} Å")

    # Save results
    df = pd.DataFrame(results)
    out_path = os.path.join(results_dir, "crystal_vs_af2_comparison.tsv")
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\nSaved to {out_path}")

    # Summary statistics
    valid = df.dropna(subset=['crystal_auc', 'af2_auc'])

    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("SUMMARY: Crystal vs AF2 Pocket Performance")
    summary_lines.append("=" * 70)
    summary_lines.append(f"Targets with both crystal and AF2 AUC: {len(valid)}")
    summary_lines.append(f"")
    summary_lines.append(f"Crystal pocket AUC:")
    summary_lines.append(f"  Mean:   {valid['crystal_auc'].mean():.3f}")
    summary_lines.append(f"  Median: {valid['crystal_auc'].median():.3f}")
    summary_lines.append(f"  Range:  {valid['crystal_auc'].min():.3f} - {valid['crystal_auc'].max():.3f}")
    summary_lines.append(f"")
    summary_lines.append(f"AF2 pocket AUC:")
    summary_lines.append(f"  Mean:   {valid['af2_auc'].mean():.3f}")
    summary_lines.append(f"  Median: {valid['af2_auc'].median():.3f}")
    summary_lines.append(f"  Range:  {valid['af2_auc'].min():.3f} - {valid['af2_auc'].max():.3f}")
    summary_lines.append(f"")
    summary_lines.append(f"AUC degradation (crystal - AF2):")
    summary_lines.append(f"  Mean:   {valid['auc_delta'].mean():+.3f}")
    summary_lines.append(f"  Median: {valid['auc_delta'].median():+.3f}")
    n_degraded = (valid['auc_delta'] > 0.05).sum()
    summary_lines.append(f"  Targets with >0.05 degradation: {n_degraded}/{len(valid)}")
    summary_lines.append(f"")

    # Correlation: RMSD vs AUC delta
    valid_rmsd = valid.dropna(subset=['aligned_rmsd'])
    if len(valid_rmsd) >= 5:
        rho, p = spearmanr(valid_rmsd['aligned_rmsd'], valid_rmsd['auc_delta'])
        summary_lines.append(f"RMSD vs AUC degradation:")
        summary_lines.append(f"  Spearman ρ: {rho:.3f}, p={p:.3f}")
        summary_lines.append(f"  → RMSD does {'NOT ' if p > 0.05 else ''}explain performance degradation")
        summary_lines.append(f"")

    # Correlation: cosine similarity vs AUC delta
    valid_cos = valid.dropna(subset=['centroid_cos_sim'])
    if len(valid_cos) >= 5:
        rho, p = spearmanr(valid_cos['centroid_cos_sim'], valid_cos['auc_delta'])
        summary_lines.append(f"Embedding cosine similarity vs AUC delta:")
        summary_lines.append(f"  Spearman ρ: {rho:.3f}, p={p:.3f}")
        summary_lines.append(f"")

    # Correlation: RMSD vs cosine similarity
    valid_both = valid.dropna(subset=['aligned_rmsd', 'centroid_cos_sim'])
    if len(valid_both) >= 5:
        rho, p = spearmanr(valid_both['aligned_rmsd'], valid_both['centroid_cos_sim'])
        summary_lines.append(f"RMSD vs embedding cosine similarity:")
        summary_lines.append(f"  Spearman ρ: {rho:.3f}, p={p:.3f}")
        summary_lines.append(f"  → Structural similarity does {'NOT ' if p > 0.05 else ''}predict embedding similarity")
        summary_lines.append(f"")

    # Mean cosine similarity
    if valid_cos is not None and len(valid_cos) > 0:
        summary_lines.append(f"Embedding similarity (crystal vs AF2):")
        summary_lines.append(f"  Mean centroid cosine sim: {valid_cos['centroid_cos_sim'].mean():.3f}")
        summary_lines.append(f"  Mean pairwise cosine sim: {valid_cos['mean_cos_sim'].mean():.3f}")
        summary_lines.append(f"  → Pocket embeddings are {'uncorrelated' if abs(valid_cos['centroid_cos_sim'].mean()) < 0.2 else 'correlated'}")

    summary = "\n".join(summary_lines)
    print(f"\n{summary}")

    summary_path = os.path.join(results_dir, "crystal_vs_af2_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary + "\n")
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
