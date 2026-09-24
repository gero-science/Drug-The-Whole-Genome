#!/usr/bin/env python3
"""Encode molecules through DrugCLIP model (one or more folds).

Mirrors encode_pockets_gpu.py but for the molecule encoder branch.
Uses exact same encoding logic as retrieval_multi_folds() lines 1727-1752.

Output: mol_embeddings/fold{0..5}.pkl
Each pkl contains [embeddings_array(N, 128), smiles_list(N)]

Usage:
    python scripts/encode_mols_gpu.py \
        --mol-lmdb data/active_library.lmdb \
        --weights-dir data/model_weights/6_folds \
        --output-dir data/mol_embeddings \
        --batch-size 64 --device cuda --folds 0
"""

import os
import sys
import pickle
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

torch.serialization.add_safe_globals([argparse.Namespace])

import unicore
from unicore import checkpoint_utils
from unimol.tasks.drugclip import DrugCLIP


def main():
    parser = argparse.ArgumentParser(description="Encode molecules through DrugCLIP")
    parser.add_argument("--mol-lmdb", type=str, required=True)
    parser.add_argument("--weights-dir", type=str,
                        default="./data/model_weights/6_folds")
    parser.add_argument("--output-dir", type=str,
                        default="./data/mol_embeddings")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (mols are smaller than pockets, can use larger)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--folds", type=str, default="0",
                        help="Comma-separated fold indices")
    parser.add_argument("--max-seq-len", type=int, default=512)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = [int(x) for x in args.folds.split(',')]
    use_cuda = args.device == 'cuda'

    # Load model args from first checkpoint
    first_ckpt = os.path.join(args.weights_dir, f"fold_{folds[0]}.pt")
    print(f"Loading model args from {first_ckpt}...")
    state = torch.load(first_ckpt, map_location='cpu', weights_only=False)
    model_args = state['args']
    model_args.data = os.path.join(PROJECT_ROOT, "dict")
    model_args.arch = "drugclip"
    model_args.finetune_mol_model = None
    model_args.finetune_pocket_model = None
    model_args.max_seq_len = args.max_seq_len

    # Build task and model
    task = DrugCLIP.setup_task(model_args)
    model = task.build_model(model_args)
    model = model.to(args.device)
    model.eval()

    # Load molecule dataset
    print(f"Loading molecule dataset from {args.mol_lmdb}...")
    mol_dataset = task.load_mols_dataset(args.mol_lmdb, "atoms", "coordinates")
    mol_data = torch.utils.data.DataLoader(
        mol_dataset, batch_size=args.batch_size,
        collate_fn=mol_dataset.collater
    )
    print(f"  {len(mol_dataset)} molecules, {len(mol_data)} batches of {args.batch_size}")

    # Read molecule SMILES from LMDB
    import lmdb
    env = lmdb.open(args.mol_lmdb, subdir=False, readonly=True, lock=False,
                     readahead=False, meminit=False)
    mol_names = []
    with env.begin() as txn:
        n_raw = txn.get(b'__len__')
        n = int(n_raw.decode()) if n_raw else txn.stat()['entries']
        for i in range(n):
            data = pickle.loads(txn.get(str(i).encode('ascii')))
            mol_names.append(data.get('smi', str(i)))
    env.close()
    print(f"  Read {len(mol_names)} molecule names")

    total_start = time.time()

    for fold_i in folds:
        fold_output = output_dir / f"fold{fold_i}.pkl"
        if fold_output.exists():
            print(f"\nFold {fold_i}: {fold_output} already exists, skipping")
            continue

        ckpt_path = os.path.join(args.weights_dir, f"fold_{fold_i}.pt")
        print(f"\n=== Fold {fold_i}: loading {ckpt_path} ===")
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        model.eval()

        # Encode molecules — exact same logic as retrieval_multi_folds() lines 1727-1752
        mol_reps = []
        fold_start = time.time()

        with torch.no_grad():
            for batch_idx, sample in enumerate(tqdm(mol_data, desc=f"Fold {fold_i}")):
                if use_cuda:
                    sample = unicore.utils.move_to_cuda(sample)

                dist = sample["net_input"]["mol_src_distance"]
                et = sample["net_input"]["mol_src_edge_type"]
                st = sample["net_input"]["mol_src_tokens"]

                mol_padding_mask = st.eq(model.mol_model.padding_idx)
                mol_x = model.mol_model.embed_tokens(st)

                n_node = dist.size(-1)
                gbf_feature = model.mol_model.gbf(dist, et)
                gbf_result = model.mol_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)

                mol_outputs = model.mol_model.encoder(
                    mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
                )
                mol_encoder_rep = mol_outputs[0][:, 0, :]  # CLS token
                mol_emb = model.mol_project(mol_encoder_rep)
                mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
                mol_emb = mol_emb.detach().cpu().numpy()
                mol_reps.append(mol_emb)

        mol_reps = np.concatenate(mol_reps, axis=0).astype(np.float32)
        fold_elapsed = time.time() - fold_start
        print(f"  Fold {fold_i}: {mol_reps.shape[0]} molecules in {fold_elapsed:.1f}s "
              f"({mol_reps.shape[0]/fold_elapsed:.1f} mols/sec)")

        with open(fold_output, 'wb') as f:
            pickle.dump([mol_reps, mol_names], f)
        print(f"  Saved {fold_output} ({mol_reps.shape})")

    total_elapsed = time.time() - total_start
    print(f"\nAll done in {total_elapsed:.1f}s ({total_elapsed/3600:.2f} hours)")


if __name__ == "__main__":
    main()
