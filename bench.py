#!/usr/bin/env python3
"""Benchmark training step overhead: B vs J vs Jc at various model scales."""

import torch
import time
from slm import (ModelB, ModelJ, ModelJc, load_data, get_batch,
                 make_base_freq)


def bench_config(config_name, n_embed, n_layers, n_heads, ctx, bs,
                 vocab_size, newline_id, train_data, device, N=10):
    print(f"\n=== {config_name} (n={n_embed} l={n_layers}) ===")
    times = {}
    params_str = ""
    for name, cls, kw in [
        ("B",  ModelB,  {}),
        ("J",  ModelJ,  {"newline_id": newline_id}),
        ("Jc", ModelJc, {"newline_id": newline_id}),
    ]:
        try:
            torch.manual_seed(42)
            model = cls(vocab_size, n_embed, n_layers, n_heads,
                        **kw, dropout=0.0).to(device)
            n_params = sum(p.numel() for p in model.parameters())
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
            x, y = get_batch(train_data, bs, ctx, device)
            # Warmup
            _, loss = model(x, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
            torch.cuda.synchronize()
            # Timed loop
            t0 = time.time()
            for _ in range(N):
                _, loss = model(x, y)
                loss.backward()
                opt.step()
                opt.zero_grad()
            torch.cuda.synchronize()
            dt = (time.time() - t0) / N * 1000
            times[name] = dt
            if name == "B":
                params_str = f"{n_params / 1e6:.0f}M"
            del model, opt
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  {name}: OOM")
                torch.cuda.empty_cache()
                times[name] = None
            else:
                raise

    if all(v is not None for v in times.values()):
        base = times["B"]
        parts = []
        for n in ["B", "J", "Jc"]:
            overhead = (times[n] - base) / base * 100
            parts.append(f"{n}={times[n]:.0f}ms ({overhead:+.1f}%)")
        print(f"  ctx={ctx} bs={bs} ({params_str}):  {'  '.join(parts)}")


def main():
    device = "cuda"
    vocab_size = 65
    stoi = {c: i for i, c in enumerate(
        sorted(set(open("input.txt").read())))}
    newline_id = stoi["\n"]
    train_data, _, _, _, _ = load_data("input.txt")

    configs = [
        # (name, n_embed, n_layers, n_heads, ctx, batch_size)
        ("0.8M",  128,  4,  4, 1024, 8),
        ("38M",   512, 12,  8, 1024, 4),
        ("85M",   768, 12, 12, 1024, 4),
        ("302M", 1024, 24, 16, 1024, 2),
        ("680M", 1536, 24, 16, 1024, 1),
    ]

    for name, n_embed, n_layers, n_heads, ctx, bs in configs:
        bench_config(name, n_embed, n_layers, n_heads, ctx, bs,
                     vocab_size, newline_id, train_data, device)


if __name__ == "__main__":
    main()
