"""
llama2-notorch training script.

no torch. no numpy. no pip install anything-heavy.
just notorch (ctypes to libnotorch) and Chuck.

Trains Llama 2 architecture on dracula.txt (character-level).

usage:
    python train.py                           # train on dracula.txt
    python train.py --dataset mytext.txt      # custom dataset
    python train.py --resume weights/ckpt.bin # resume training
"""

import os
import sys
import time
import math
import random
import argparse

from model import Transformer, ModelArgs
from ariannamethod.notorch_nn import seed as nt_seed


def load_and_encode(path):
    """Load text file, build char vocab, encode to int list."""
    with open(path, 'rb') as f:
        raw = f.read()
    chars = sorted(set(raw))
    char_to_id = {c: i for i, c in enumerate(chars)}
    id_to_char = {i: c for c, i in char_to_id.items()}
    encoded = [char_to_id[b] for b in raw]
    return encoded, char_to_id, id_to_char, len(chars)


def main():
    parser = argparse.ArgumentParser(description='llama2-notorch training')
    parser.add_argument('--dataset', type=str, default='dracula.txt')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--steps', type=int, default=12000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ctx', type=int, default=256)
    parser.add_argument('--dim', type=int, default=288)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--heads', type=int, default=6)
    parser.add_argument('--multiple_of', type=int, default=32)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--save_path', type=str, default='weights/llama2.bin')
    args = parser.parse_args()

    print("════════════════════════════════════════════════════════")
    print("  llama2-notorch — training (Python + libnotorch)")
    print("  no torch. no numpy. Chuck optimizer.")
    print("════════════════════════════════════════════════════════")

    # Load dataset
    if not os.path.exists(args.dataset):
        print(f"cannot find {args.dataset}")
        sys.exit(1)

    encoded, char_to_id, id_to_char, vocab_size = load_and_encode(args.dataset)
    print(f"dataset: {args.dataset} ({len(encoded)} chars, vocab {vocab_size})")

    # Model
    model_args = ModelArgs(
        dim=args.dim,
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_heads=args.heads,
        vocab_size=vocab_size,
        multiple_of=args.multiple_of,
        max_seq_len=args.ctx,
        dropout=0.0,
    )
    nt_seed(42)
    model = Transformer(model_args)
    n_params = model.count_params()
    print(f"model: {n_params:,} params (dim={args.dim} L={args.layers} H={args.heads})")
    print(f"training: {args.steps} steps, lr={args.lr}")

    # Resume
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        if model.load_weights(args.resume):
            print(f"resumed from {args.resume}")
            meta = args.resume + '.meta'
            if os.path.exists(meta):
                with open(meta) as f:
                    start_step = int(f.readline().strip())
                print(f"  start_step={start_step}")

    # LR schedule (cosine with warmup)
    warmup = args.steps // 10
    min_lr = args.lr * 0.1

    def get_lr(step):
        if step < warmup:
            return args.lr * step / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * progress))

    print()
    print("training...")
    print("─────────────────────────────────────────────────────")

    t0 = time.time()
    best_loss = 99.0

    for step in range(start_step, args.steps):
        lr = get_lr(step)

        # Random window
        off = random.randint(0, len(encoded) - args.ctx - 1)
        tokens = encoded[off:off + args.ctx]
        targets = encoded[off + 1:off + args.ctx + 1]

        # Forward + backward + Chuck step
        loss_idx, loss_val = model.forward_train(tokens, targets)
        if loss_val < best_loss:
            best_loss = loss_val
        model.backward_step(loss_idx, loss_val, lr)

        if (step + 1) % args.log_every == 0 or step == start_step:
            elapsed = time.time() - t0
            print(f"  step {step+1:5d} | train {loss_val:.4f} | best {best_loss:.4f} | "
                  f"lr {lr:.2e} | {elapsed:.1f}s")

        if (step + 1) % args.save_every == 0:
            os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
            model.save_weights(args.save_path)
            with open(args.save_path + '.meta', 'w') as f:
                f.write(f"{step+1}\n{best_loss}\n{vocab_size}\n{args.dim}\n"
                        f"{args.heads}\n{args.layers}\n{args.ctx}\n")
            print(f"  ──── saved checkpoint (step {step+1})")

    elapsed = time.time() - t0
    print("─────────────────────────────────────────────────────")
    print(f"  train: {best_loss:.4f}")
    print(f"  time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save final
    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    model.save_weights(args.save_path)
    with open(args.save_path + '.meta', 'w') as f:
        f.write(f"{args.steps}\n{best_loss}\n{vocab_size}\n{args.dim}\n"
                f"{args.heads}\n{args.layers}\n{args.ctx}\n")

    # Generate sample
    print()
    print("── generation (temp=0.8) ──")
    prompts = ["CHAPTER", "\"My dear", "The blood"]
    for p in prompts:
        ids = [char_to_id.get(ord(c) if isinstance(c, str) else c, 0) for c in p.encode()]
        gen = model.generate(ids, max_new=150, temperature=0.8, top_k=40)
        text = ''.join(chr(id_to_char[g]) if g in id_to_char else '?' for g in gen)
        print(f"{p}{text}")
        print()

    print("════════════════════════════════════════════════════════")
    print(f"  llama2-notorch {n_params:,} params. No PyTorch.")
    print("════════════════════════════════════════════════════════")


if __name__ == '__main__':
    main()
