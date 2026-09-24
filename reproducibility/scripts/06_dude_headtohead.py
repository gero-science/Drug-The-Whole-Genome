#!/usr/bin/env python3
"""Direct head-to-head: DUD-E pocket → DUD-E mols vs DUD-E pocket → drug library.

For overlapping targets (in both DUD-E and our 28 targets):
1. Encode DUD-E pocket (holo crystal)
2. Evaluate against DUD-E molecules (actives + decoys) → reproduce paper's high EF1%
3. Evaluate same pocket against our drug library → show near-random
4. Also: evaluate against DUD-E molecules using RANDOM pocket → test decoy bias

This is the definitive test of DUD-E benchmark inflation.
"""

import os
import sys
import pickle
import argparse
import numpy as np
import lmdb
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import torch
torch.serialization.add_safe_globals([argparse.Namespace])

import unicore
from unicore import checkpoint_utils
from unimol.tasks.drugclip import DrugCLIP


# DUD-E name → our target name mapping
DUDE_TO_OURS = {
    'adrb2': 'ADRB2',
    'esr2': 'ESR2',
    'mk01': 'MAPK1',
    'mk10': 'MAPK10',
    'xiap': 'XIAP',
}


def cal_metrics(y_true, y_score):
    """Compute AUC and EF at various cutoffs."""
    try:
        auc = roc_auc_score(y_true, y_score)
    except:
        auc = 0.5

    n = len(y_true)
    n_actives = int(y_true.sum())
    sorted_idx = np.argsort(-y_score)
    sorted_labels = y_true[sorted_idx]

    metrics = {'auc': auc, 'n_actives': n_actives, 'n_total': n}
    for pct in [0.005, 0.01, 0.02, 0.05]:
        top_n = max(1, int(n * pct))
        n_act_top = int(sorted_labels[:top_n].sum())
        ef = (n_act_top / top_n) / (n_actives / n) if n_actives > 0 else 0
        re = n_act_top / n_actives if n_actives > 0 else 0
        metrics[f'ef_{pct}'] = ef
        metrics[f're_{pct}'] = re
    return metrics


def encode_dataloader(dataloader, model, encoder_type, device):
    """Encode from a pre-created dataloader."""
    reps = []
    labels = []

    with torch.no_grad():
        for sample in dataloader:
            if device != 'cpu':
                sample = unicore.utils.move_to_cuda(sample)

            if encoder_type == 'mol':
                prefix = 'mol'
                enc_model = model.mol_model
                proj = model.mol_project
            else:
                prefix = 'pocket'
                enc_model = model.pocket_model
                proj = model.pocket_project

            dist = sample["net_input"][f"{prefix}_src_distance"]
            et = sample["net_input"][f"{prefix}_src_edge_type"]
            st = sample["net_input"][f"{prefix}_src_tokens"]

            padding_mask = st.eq(enc_model.padding_idx)
            x = enc_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = enc_model.gbf(dist, et)
            gbf_result = enc_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)

            outputs = enc_model.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
            encoder_rep = outputs[0][:, 0, :]
            emb = proj(encoder_rep)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            reps.append(emb.detach().cpu().numpy())

            if encoder_type == 'mol':
                labels.extend(sample["target"].detach().cpu().numpy())

    reps = np.concatenate(reps, axis=0)
    labels = np.array(labels, dtype=np.int32) if labels else None
    return reps, labels


def score_with_zscore(pocket_reps, mol_reps):
    """Score molecules against pockets using max z-score."""
    res = pocket_reps @ mol_reps.T
    medians = np.median(res, axis=1, keepdims=True)
    mads = np.median(np.abs(res - medians), axis=1, keepdims=True)
    res = 0.6745 * (res - medians) / (mads + 1e-6)
    return res.max(axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dude-dir", default="./data/DUD-E")
    parser.add_argument("--weights-dir", default="./data/model_weights/6_folds")
    parser.add_argument("--data-root", default=None,
                        help="Root for mol_embeddings/, active_library_meta.pkl, targets/. "
                             "Default: PROJECT_ROOT/data")
    parser.add_argument("--dict-dir", default=None,
                        help="Path to dict/ with dict_mol.txt and dict_pkt.txt. "
                             "Default: PROJECT_ROOT/dict")
    args = parser.parse_args()

    data_root = args.data_root or os.path.join(PROJECT_ROOT, "data")
    dict_dir = args.dict_dir or os.path.join(PROJECT_ROOT, "dict")

    # Load model
    first_ckpt = os.path.join(args.weights_dir, "fold_0.pt")
    state = torch.load(first_ckpt, map_location='cpu', weights_only=False)
    model_args = state['args']
    model_args.data = dict_dir
    model_args.arch = "drugclip"
    model_args.finetune_mol_model = None
    model_args.finetune_pocket_model = None

    task = DrugCLIP.setup_task(model_args)
    model = task.build_model(model_args)
    model = model.to(args.device)
    model.eval()

    # Load pre-encoded drug library embeddings
    print("Loading pre-encoded drug library embeddings...")
    drug_lib_embs_list = []
    for fold in range(6):
        p = os.path.join(data_root, "mol_embeddings", f"fold{fold}.pkl")
        with open(p, 'rb') as f:
            drug_lib_embs_list.append(pickle.load(f)[0])
    drug_lib_embs = np.mean(drug_lib_embs_list, axis=0)
    drug_lib_embs = drug_lib_embs / np.linalg.norm(drug_lib_embs, axis=1, keepdims=True)

    # Load drug library metadata
    meta = pickle.load(open(os.path.join(data_root, "active_library_meta.pkl"), "rb"))
    drug_lib_smiles = meta['smiles']

    from rdkit import Chem
    lib_smi_to_idx = {}
    for i, smi in enumerate(drug_lib_smiles):
        try:
            m = Chem.MolFromSmiles(smi)
            if m:
                lib_smi_to_idx[Chem.MolToSmiles(m)] = i
        except:
            lib_smi_to_idx[smi] = i

    results = []

    # Run 6-fold ensemble evaluation
    for dude_target, our_target in DUDE_TO_OURS.items():
        print(f"\n{'='*70}")
        print(f"Target: DUD-E '{dude_target}' = our '{our_target}'")
        print(f"{'='*70}")

        mols_path = os.path.join(args.dude_dir, dude_target, "mols.lmdb")
        pocket_path = os.path.join(args.dude_dir, dude_target, "pocket.lmdb")

        if not os.path.exists(mols_path) or not os.path.exists(pocket_path):
            print(f"  MISSING: {mols_path} or {pocket_path}")
            continue

        # ============================================================
        # EXTRACT ACTIVE SMILES BEFORE creating datasets
        # (UniCore's LMDBDataset caches LMDB connections, preventing
        #  subsequent lmdb.open() calls on the same file)
        # ============================================================
        dude_active_smiles = set()
        env = lmdb.open(mols_path, subdir=False, readonly=True, lock=False)
        with env.begin() as txn:
            n_entries = txn.stat()['entries']
            for i in range(n_entries):
                raw = txn.get(str(i).encode())
                if raw is None:
                    continue
                entry = pickle.loads(raw)
                label = entry.get('target', 0)
                if label == 1:
                    smi = entry.get('smi', '')
                    if '_' in smi and smi.split('_')[0].startswith('CHEMBL'):
                        smi = smi.split('_', 1)[1]
                    try:
                        m = Chem.MolFromSmiles(smi)
                        if m:
                            dude_active_smiles.add(Chem.MolToSmiles(m))
                    except:
                        pass
        env.close()
        del env

        # Also load ChEMBL actives for this target
        chembl_path = os.path.join(data_root, "targets", our_target, "ChEMBL", "active.lmdb")
        chembl_smiles = set()
        if os.path.exists(chembl_path):
            env2 = lmdb.open(chembl_path, subdir=False, readonly=True, lock=False)
            with env2.begin() as txn:
                n = txn.stat()['entries']
                for i in range(n):
                    raw = txn.get(str(i).encode())
                    if raw is None:
                        continue
                    entry = pickle.loads(raw)
                    smi = entry.get('smi', '')
                    if '_' in smi and smi.split('_')[0].startswith('CHEMBL'):
                        smi = smi.split('_', 1)[1]
                    try:
                        m = Chem.MolFromSmiles(smi)
                        if m:
                            chembl_smiles.add(Chem.MolToSmiles(m))
                    except:
                        pass
            env2.close()
            del env2

        print(f"  Pre-extracted: {len(dude_active_smiles)} DUD-E actives, {len(chembl_smiles)} ChEMBL actives")

        # Now load datasets (LMDB connection gets cached here)
        mol_dataset = task.load_mols_dataset(mols_path, "atoms", "coordinates")
        mol_dataloader = torch.utils.data.DataLoader(
            mol_dataset, batch_size=512, collate_fn=mol_dataset.collater)

        pocket_dataset = task.load_pockets_dataset(pocket_path)
        pocket_dataloader = torch.utils.data.DataLoader(
            pocket_dataset, batch_size=8, collate_fn=pocket_dataset.collater)

        # 6-fold ensemble: average raw scores
        dude_scores_list = []
        lib_scores_list = []

        for fold_i in range(6):
            ckpt_path = os.path.join(args.weights_dir, f"fold_{fold_i}.pt")
            state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.load_state_dict(state["model"], strict=False)
            model.eval()

            # Encode DUD-E molecules
            dude_mol_reps, dude_labels = encode_dataloader(
                mol_dataloader, model, 'mol', args.device)

            # Encode DUD-E pocket (holo crystal)
            dude_pocket_reps, _ = encode_dataloader(
                pocket_dataloader, model, 'pocket', args.device)

            # Raw score matrix for DUD-E: (n_pockets, n_dude_mols)
            dude_raw = dude_pocket_reps @ dude_mol_reps.T
            dude_scores_list.append(dude_raw)

            # Raw score matrix for drug library: (n_pockets, n_lib_mols)
            # Use the fold-specific mol embeddings
            with open(os.path.join(data_root, "mol_embeddings", f"fold{fold_i}.pkl"), "rb") as f:
                fold_mol_embs = pickle.load(f)[0]
            fold_mol_embs = fold_mol_embs / np.linalg.norm(fold_mol_embs, axis=1, keepdims=True)
            lib_raw = dude_pocket_reps @ fold_mol_embs.T
            lib_scores_list.append(lib_raw)

            if fold_i == 0:
                print(f"  DUD-E: {int(dude_labels.sum())} actives / {len(dude_labels)} total")
                print(f"  Pocket: {dude_pocket_reps.shape[0]} conformations")

        # Average across folds, then z-score (matching paper's ensemble method)
        dude_scores_avg = np.mean(dude_scores_list, axis=0)
        lib_scores_avg = np.mean(lib_scores_list, axis=0)

        # Z-score and max
        def zscore_max(raw_scores):
            medians = np.median(raw_scores, axis=1, keepdims=True)
            mads = np.median(np.abs(raw_scores - medians), axis=1, keepdims=True)
            z = 0.6745 * (raw_scores - medians) / (mads + 1e-6)
            return z.max(axis=0)

        dude_final = zscore_max(dude_scores_avg)
        lib_final = zscore_max(lib_scores_avg)

        # 1. DUD-E evaluation
        dude_metrics = cal_metrics(dude_labels, dude_final)
        print(f"\n  === DUD-E pocket → DUD-E molecules ===")
        print(f"  AUC:  {dude_metrics['auc']:.3f}")
        print(f"  EF1%: {dude_metrics['ef_0.01']:.1f}")
        print(f"  EF5%: {dude_metrics['ef_0.05']:.1f}")

        # 2. Same DUD-E pocket → drug library
        # (Active SMILES already extracted above, before dataset loading)
        all_actives = dude_active_smiles | chembl_smiles
        lib_labels = np.zeros(len(drug_lib_smiles), dtype=np.int32)
        for smi in all_actives:
            if smi in lib_smi_to_idx:
                lib_labels[lib_smi_to_idx[smi]] = 1

        n_lib_actives = int(lib_labels.sum())
        print(f"\n  === DUD-E pocket → Drug library ===")
        print(f"  Actives in library: {n_lib_actives} (DUD-E: {len(dude_active_smiles)}, ChEMBL: {len(chembl_smiles)})")

        if n_lib_actives >= 3:
            lib_metrics = cal_metrics(lib_labels, lib_final)
            print(f"  AUC:  {lib_metrics['auc']:.3f}")
            print(f"  EF1%: {lib_metrics['ef_0.01']:.1f}")
            print(f"  EF5%: {lib_metrics['ef_0.05']:.1f}")
        else:
            lib_metrics = {'auc': None, 'ef_0.01': None, 'ef_0.05': None}
            print(f"  Too few actives in library")

        # 3. Random pocket → DUD-E (test decoy bias)
        rng = np.random.RandomState(42)
        random_pocket = rng.randn(1, 128).astype(np.float32)
        random_pocket /= np.linalg.norm(random_pocket)
        # Use fold-0 DUD-E mol reps for random pocket test
        random_dude_scores = random_pocket @ dude_mol_reps.T
        random_final = random_dude_scores[0]  # just 1 pocket, no z-score needed
        random_metrics = cal_metrics(dude_labels, random_final)
        print(f"\n  === Random pocket → DUD-E molecules ===")
        print(f"  AUC:  {random_metrics['auc']:.3f}")
        print(f"  EF1%: {random_metrics['ef_0.01']:.1f}")

        results.append({
            'dude_target': dude_target,
            'our_target': our_target,
            'dude_auc': dude_metrics['auc'],
            'dude_ef1': dude_metrics['ef_0.01'],
            'dude_ef5': dude_metrics['ef_0.05'],
            'dude_n_actives': dude_metrics['n_actives'],
            'dude_n_total': dude_metrics['n_total'],
            'lib_auc': lib_metrics.get('auc'),
            'lib_ef1': lib_metrics.get('ef_0.01'),
            'lib_n_actives': n_lib_actives,
            'random_dude_auc': random_metrics['auc'],
            'random_dude_ef1': random_metrics['ef_0.01'],
        })

    # Summary
    import pandas as pd
    df = pd.DataFrame(results)

    print(f"\n{'='*70}")
    print("SUMMARY: DUD-E BENCHMARK INFLATION")
    print(f"{'='*70}")
    print(f"\n{'Target':<10} {'DUD-E AUC':>10} {'DUD-E EF1%':>11} {'Lib AUC':>8} {'Lib EF1%':>9} {'Rnd AUC':>8}")
    for _, r in df.iterrows():
        lib_auc = f"{r['lib_auc']:.3f}" if r['lib_auc'] else "N/A"
        lib_ef = f"{r['lib_ef1']:.1f}" if r['lib_ef1'] else "N/A"
        print(f"{r['our_target']:<10} {r['dude_auc']:>10.3f} {r['dude_ef1']:>11.1f} {lib_auc:>8} {lib_ef:>9} {r['random_dude_auc']:>8.3f}")

    if len(df) > 0:
        print(f"\nMean DUD-E AUC:     {df['dude_auc'].mean():.3f}")
        print(f"Mean DUD-E EF1%:    {df['dude_ef1'].mean():.1f}")
        valid_lib = df.dropna(subset=['lib_auc'])
        if len(valid_lib) > 0:
            print(f"Mean Library AUC:   {valid_lib['lib_auc'].mean():.3f}")
            print(f"Mean Library EF1%:  {valid_lib['lib_ef1'].mean():.1f}")
        print(f"Mean Random AUC:    {df['random_dude_auc'].mean():.3f}")

    out_path = os.path.join(PROJECT_ROOT, "reproducibility", "results", "dude_headtohead.tsv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
