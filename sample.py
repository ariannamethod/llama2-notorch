"""
Sample from a trained llama2-notorch model.

no torch. no numpy. just notorch.

usage:
    python sample.py                                          # use default checkpoint
    python sample.py --checkpoint weights/llama2.bin          # specify checkpoint
    python sample.py --prompt "Once upon a time"              # specify prompt
"""

import os
import sys
import argparse

from model import Transformer, ModelArgs
from ariannamethod.notorch_nn import seed as nt_seed


def main():
    parser = argparse.ArgumentParser(description='llama2-notorch sampling')
    parser.add_argument('--checkpoint', type=str, default='weights/llama2.bin')
    parser.add_argument('--prompt', type=str, default='CHAPTER')
    parser.add_argument('--max_tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--seed', type=int, default=1337)
    args = parser.parse_args()

    meta_path = args.checkpoint + '.meta'
    if not os.path.exists(args.checkpoint) or not os.path.exists(meta_path):
        print(f"cannot find checkpoint: {args.checkpoint}")
        print("train first: python train.py")
        sys.exit(1)

    # Read meta
    with open(meta_path) as f:
        lines = f.readlines()
    vocab_size = int(lines[2].strip())
    dim = int(lines[3].strip())
    n_heads = int(lines[4].strip())
    n_layers = int(lines[5].strip())
    ctx = int(lines[6].strip())

    # Rebuild char vocab from dataset
    dataset_path = 'dracula.txt'
    if not os.path.exists(dataset_path):
        print(f"cannot find dataset: {dataset_path}")
        sys.exit(1)

    with open(dataset_path, 'rb') as f:
        raw = f.read()
    chars = sorted(set(raw))
    char_to_id = {c: i for i, c in enumerate(chars)}
    id_to_char = {i: c for c, i in char_to_id.items()}

    # Model
    model_args = ModelArgs(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_heads,
        vocab_size=vocab_size,
        max_seq_len=ctx,
    )
    nt_seed(args.seed)
    model = Transformer(model_args)
    model.load_weights(args.checkpoint)
    print(f"loaded {model.count_params():,} params from {args.checkpoint}")

    # Encode prompt
    ids = [char_to_id.get(ord(c) if isinstance(c, str) else c, 0) for c in args.prompt.encode()]

    # Generate
    gen = model.generate(ids, max_new=args.max_tokens,
                         temperature=args.temperature, top_k=args.top_k)
    text = ''.join(chr(id_to_char[g]) if g in id_to_char else '?' for g in gen)
    print(f"{args.prompt}{text}")


if __name__ == '__main__':
    main()
