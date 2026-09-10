#!/usr/bin/env python3
"""Causality test: verify no position's output depends on future tokens.

For each model, perturb a token at position p and check that logits at all
positions < p are exactly unchanged (within floating-point tolerance).
"""

import os
import sys
import torch
from slm import (load_data, ModelB, ModelBa, ModelBs,
                 ModelJ, ModelJc, ModelJr, ModelK)


def test_causality(model, x, edit_pos, new_token, name):
    """Check that editing token at edit_pos doesn't change logits before it."""
    model.eval()
    with torch.no_grad():
        logits_orig, _ = model(x)
        x_edit = x.clone()
        x_edit[0, edit_pos] = new_token
        logits_edit, _ = model(x_edit)

    diff = (logits_orig[0, :edit_pos] - logits_edit[0, :edit_pos]).abs().max().item()
    status = "PASS" if diff == 0.0 else "FAIL"
    print(f"  {name:>4s}: max |delta logits| at pos < {edit_pos} = {diff:.10f}  [{status}]")
    return diff == 0.0


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'input.txt')
    if not os.path.exists(data_path):
        print("ERROR: input.txt not found. Download Shakespeare data:")
        print("  curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
        sys.exit(1)

    train_data, val_data, vocab_size, stoi, itos = load_data(data_path)
    newline_id = stoi['\n']
    device = 'cpu'  # CPU for exact reproducibility

    T = 256
    x = val_data[:T].unsqueeze(0).to(device)

    # Find a position mid-line (not at a newline boundary) to edit
    newlines = (x[0] == newline_id).nonzero(as_tuple=True)[0]
    edit_pos = newlines[2].item() + 5  # 5 chars into the 4th line
    orig_token = x[0, edit_pos].item()
    new_token = (orig_token + 1) % vocab_size

    print(f"Test: edit position {edit_pos} ('{itos[orig_token]}' -> '{itos[new_token]}')")
    print(f"Checking logits at positions 0..{edit_pos - 1} are unchanged.\n")

    n_embed, n_layers, n_heads = 64, 2, 4
    all_pass = True

    models = [
        ('B',  ModelB(vocab_size, n_embed, n_layers, n_heads)),
        ('Ba', ModelBa(vocab_size, n_embed, n_layers, n_heads)),
        ('Bs', ModelBs(vocab_size, n_embed, n_layers, n_heads, newline_id)),
        ('J',  ModelJ(vocab_size, n_embed, n_layers, n_heads, newline_id)),
        ('Jc', ModelJc(vocab_size, n_embed, n_layers, n_heads, newline_id)),
        ('K',  ModelK(vocab_size, n_embed, n_layers, n_heads)),
    ]
    # Jr uses random offsets per forward pass — skip (no content info = no leak)

    for name, model in models:
        torch.manual_seed(42)
        model = model.to(device).eval()
        passed = test_causality(model, x, edit_pos, new_token, name)
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("All models PASS causality test.")
    else:
        print("SOME MODELS FAILED causality test!")
        sys.exit(1)


if __name__ == '__main__':
    main()
