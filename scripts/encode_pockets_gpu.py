#!/usr/bin/env python3
"""Encode all proteome pocket conformations through 6-fold DrugCLIP model.

Uses the exact same encoding pipeline as retrieval_multi_folds() but only
saves pocket embeddings (no molecule encoding or screening).

Output: data/pocket_embeddings/fold{0..5}.pkl
Each pkl contains [embeddings_array(N, 128), pocket_names_list(N)]

Usage:
    python scripts/encode_pockets_gpu.py \
        --pocket-lmdb data/proteome_pockets.lmdb \
        --batch-size 32 --device cuda
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

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# PyTorch 2.6+ compatibility
torch.serialization.add_safe_globals([argparse.Namespace])

import unicore
from unicore import checkpoint_utils
from unimol.tasks.drugclip import DrugCLIP


def main():
    parser = argparse.ArgumentParser(description="Encode pockets through 6-fold DrugCLIP")
    parser.add_argument("--pocket-lmdb", type=str, required=True,
                        help="Path to pocket.lmdb")
    parser.add_argument("--weights-dir", type=str,
                        default="./data/model_weights/6_folds",
                        help="Directory with fold_0.pt ... fold_5.pt")
    parser.add_argument("--output-dir", type=str,
                        default="./data/pocket_embeddings",
                        help="Output directory for embeddings")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (32 for 16GB VRAM, 16 for 8GB)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda, cpu, mps")
    parser.add_argument("--folds", type=str, default="0,1,2,3,4,5",
                        help="Comma-separated fold indices to encode")
    parser.add_argument("--max-pocket-atoms", type=int, default=511)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = [int(x) for x in args.folds.split(',')]
    use_cuda = args.device == 'cuda'

    # Load first checkpoint to get model args
    first_ckpt = os.path.join(args.weights_dir, f"fold_{folds[0]}.pt")
    print(f"Loading model args from {first_ckpt}...")
    state = torch.load(first_ckpt, map_location='cpu', weights_only=False)
    model_args = state['args']
    model_args.data = os.path.join(PROJECT_ROOT, "dict")
    model_args.max_pocket_atoms = args.max_pocket_atoms
    model_args.arch = "drugclip"  # checkpoint says 'binding_affinity', registry uses 'drugclip'
    # Null out pretrain paths from training — we load fine-tuned weights below
    model_args.finetune_mol_model = None
    model_args.finetune_pocket_model = None

    # Build task and model (once) — setup_task loads the dictionaries
    task = DrugCLIP.setup_task(model_args)
    model = task.build_model(model_args)
    model = model.to(args.device)
    model.eval()

    # Load pocket dataset (once, reuse across folds — same as retrieval_multi_folds)
    print(f"Loading pocket dataset from {args.pocket_lmdb}...")
    pocket_dataset = task.load_pockets_dataset(args.pocket_lmdb)
    pocket_data = torch.utils.data.DataLoader(
        pocket_dataset, batch_size=args.batch_size,
        collate_fn=pocket_dataset.collater
    )
    print(f"  {len(pocket_dataset)} pocket conformations, "
          f"{len(pocket_data)} batches of {args.batch_size}")

    # Read pocket names from LMDB
    import lmdb
    env = lmdb.open(args.pocket_lmdb, subdir=False, readonly=True, lock=False,
                     readahead=False, meminit=False)
    pocket_names = []
    with env.begin() as txn:
        n = txn.stat()['entries']
        for i in range(n):
            data = pickle.loads(txn.get(str(i).encode('ascii')))
            pocket_names.append(data.get('pocket', str(i)))
    env.close()
    print(f"  Read {len(pocket_names)} pocket names")

    total_start = time.time()

    for fold_i in folds:
        fold_output = output_dir / f"fold{fold_i}.pkl"
        if fold_output.exists():
            print(f"\nFold {fold_i}: {fold_output} already exists, skipping")
            continue

        # Load fold weights
        ckpt_path = os.path.join(args.weights_dir, f"fold_{fold_i}.pt")
        print(f"\n=== Fold {fold_i}: loading {ckpt_path} ===")
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        model.eval()

        # Encode pockets — exact same logic as retrieval_multi_folds() lines 1757-1784
        pocket_reps = []
        fold_start = time.time()

        with torch.no_grad():
            for batch_idx, sample in enumerate(tqdm(pocket_data, desc=f"Fold {fold_i}")):
                if use_cuda:
                    sample = unicore.utils.move_to_cuda(sample)

                dist = sample["net_input"]["pocket_src_distance"]
                et = sample["net_input"]["pocket_src_edge_type"]
                st = sample["net_input"]["pocket_src_tokens"]

                pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
                pocket_x = model.pocket_model.embed_tokens(st)

                n_node = dist.size(-1)
                gbf_feature = model.pocket_model.gbf(dist, et)
                gbf_result = model.pocket_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)

                pocket_outputs = model.pocket_model.encoder(
                    pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
                )
                pocket_encoder_rep = pocket_outputs[0][:, 0, :]  # CLS token
                pocket_emb = model.pocket_project(pocket_encoder_rep)
                pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
                pocket_emb = pocket_emb.detach().cpu().numpy()
                pocket_reps.append(pocket_emb)

        pocket_reps = np.concatenate(pocket_reps, axis=0).astype(np.float32)
        fold_elapsed = time.time() - fold_start
        print(f"  Fold {fold_i}: {pocket_reps.shape[0]} pockets in {fold_elapsed:.1f}s "
              f"({pocket_reps.shape[0]/fold_elapsed:.1f} pockets/sec)")

        # Save as [embeddings_array, names_list]
        with open(fold_output, 'wb') as f:
            pickle.dump([pocket_reps, pocket_names], f)
        print(f"  Saved {fold_output} ({pocket_reps.shape})")

    total_elapsed = time.time() - total_start
    print(f"\nAll done in {total_elapsed:.1f}s ({total_elapsed/3600:.2f} hours)")


if __name__ == "__main__":
    main()
