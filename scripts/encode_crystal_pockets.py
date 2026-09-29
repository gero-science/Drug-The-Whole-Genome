#!/usr/bin/env python3
"""Encode crystal (holo) pockets from data/targets/*/PDB/pocket.lmdb.

Uses the same model and encoding as encode_pockets_gpu.py, but for the
28 crystal structure targets rather than AF2 genome-wide pockets.

Output: data/crystal_pocket_embeddings/fold{0..5}.pkl
Each pkl contains [embeddings_array(N, 128), pocket_names(N)]
"""

import os
import sys
import pickle
import argparse
import time
import lmdb
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
    parser = argparse.ArgumentParser(description="Encode crystal pockets through DrugCLIP")
    parser.add_argument("--targets-dir", type=str, default="./data/targets")
    parser.add_argument("--weights-dir", type=str, default="./data/model_weights/6_folds")
    parser.add_argument("--output-dir", type=str, default="./data/crystal_pocket_embeddings")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--folds", type=str, default="0",
                        help="Comma-separated fold indices")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = [int(x) for x in args.folds.split(',')]
    use_cuda = args.device == 'cuda'

    # Collect all crystal pocket LMDBs
    targets_dir = args.targets_dir
    target_lmdbs = []
    for target in sorted(os.listdir(targets_dir)):
        lmdb_path = os.path.join(targets_dir, target, "PDB", "pocket.lmdb")
        if not os.path.exists(lmdb_path):
            lmdb_path = os.path.join(targets_dir, target, "pocket.lmdb")
        if os.path.exists(lmdb_path):
            target_lmdbs.append((target, lmdb_path))

    print(f"Found {len(target_lmdbs)} targets with crystal pocket LMDBs")

    # Load model args from first checkpoint
    first_ckpt = os.path.join(args.weights_dir, f"fold_{folds[0]}.pt")
    print(f"Loading model args from {first_ckpt}...")
    state = torch.load(first_ckpt, map_location='cpu', weights_only=False)
    model_args = state['args']
    model_args.data = os.path.join(PROJECT_ROOT, "dict")
    model_args.arch = "drugclip"
    model_args.finetune_mol_model = None
    model_args.finetune_pocket_model = None

    task = DrugCLIP.setup_task(model_args)
    model = task.build_model(model_args)
    model = model.to(args.device)
    model.eval()

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

        all_pocket_reps = []
        all_pocket_names = []

        for target, lmdb_path in target_lmdbs:
            print(f"\n  Target: {target} ({lmdb_path})")

            pocket_dataset = task.load_pockets_dataset(lmdb_path)
            pocket_data = torch.utils.data.DataLoader(
                pocket_dataset, batch_size=args.batch_size,
                collate_fn=pocket_dataset.collater
            )

            # Read pocket names
            env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False)
            pocket_names = []
            with env.begin() as txn:
                n_raw = txn.get(b'__len__')
                n = int(n_raw.decode()) if n_raw else txn.stat()['entries']
                for i in range(n):
                    data = pickle.loads(txn.get(str(i).encode()))
                    name = data.get('pocket', f"{target}_{i}")
                    pocket_names.append(f"{target}/{name}")
            env.close()

            pocket_reps = []
            with torch.no_grad():
                for batch_idx, sample in enumerate(tqdm(pocket_data, desc=target)):
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
                    pocket_encoder_rep = pocket_outputs[0][:, 0, :]
                    pocket_emb = model.pocket_project(pocket_encoder_rep)
                    pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
                    pocket_emb = pocket_emb.detach().cpu().numpy()
                    pocket_reps.append(pocket_emb)

            pocket_reps = np.concatenate(pocket_reps, axis=0).astype(np.float32)
            print(f"    Encoded {pocket_reps.shape[0]} pockets ({pocket_reps.shape})")

            all_pocket_reps.append(pocket_reps)
            all_pocket_names.extend(pocket_names)

        all_pocket_reps = np.concatenate(all_pocket_reps, axis=0)
        print(f"\n  Total: {all_pocket_reps.shape[0]} crystal pockets across {len(target_lmdbs)} targets")

        with open(fold_output, 'wb') as f:
            pickle.dump([all_pocket_reps, all_pocket_names], f)
        print(f"  Saved {fold_output}")

    total_elapsed = time.time() - total_start
    print(f"\nAll done in {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
