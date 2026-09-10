# SLM: Structured Length-generalized Models

Character-level language models comparing positional encoding strategies for length generalization.

## Models

| Model | Description |
|-------|-------------|
| **B** | Standard RoPE — continuous positions 0..T-1 |
| **Ba** | ALiBi — additive linear bias, no rotation |
| **Bs** | RoPE with newline reset (no content offset) |
| **J** | LISformer — RoPE resets at newlines + content-based line addressing (Q: prefix address, K: full-line address) |
| **Jc** | Per-layer LISformer — recomputes offsets per layer from residual stream |
| **Jr** | J with random i.i.d. line addresses instead of content-derived (ablation control) |
| **K** | Purely content-derived angles (no positional info) |

## Key result

| Model | ctx=256 | ctx=512 | ctx=1024 | ctx=2048 | ctx=4096 |
|-------|---------|---------|----------|----------|----------|
| B (RoPE)          | 4.71 | 5.78 | 7.56 | 9.62 | 12.09 |
| J (reset+content) | 5.07 | 5.01 | 5.05 | 5.09 | 5.24  |
| Jr (reset+random) | 4.88 | 4.87 | 4.92 | 4.95 | 5.09  |

(Trained at ctx=256, n_embed=128, n_layers=4, 5K iters on Shakespeare.)

J generalizes to 16x training context (5.07 → 5.24) while B degrades sharply (4.71 → 12.09).

## Causality

Cross-line attention uses two-sided content matching: Q carries a prefix address (cumulative mean of its line so far), K carries the full-line address. Same-line pairs use pure reset-RoPE. At line completion, the prefix address equals the full-line address — the address crystallizes. Run `python causality_test.py` to verify all models pass (exact 0.0 delta).

## Usage

Download Shakespeare data:
```bash
curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Train and evaluate:
```bash
python slm.py --models B J --n_embed 128 --n_layers 4 --iters 5000
```

Smoke test:
```bash
python slm.py --models B J --smoke
```

## License

CC BY-NC-SA 4.0 — see [LICENSE](LICENSE).
