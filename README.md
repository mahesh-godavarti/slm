# SLM: Structured Length-generalized Models

Character-level language models comparing positional encoding strategies for length generalization.

## Models

| Model | Description |
|-------|-------------|
| **B** | Standard RoPE — continuous positions 0..T-1 |
| **Ba** | ALiBi — additive linear bias, no rotation |
| **Bs** | RoPE with newline reset (no content offset) |
| **J** | LISformer — RoPE resets at newlines + content-based line angle offset (causal: offset on cross-line K only) |
| **Jc** | Per-layer LISformer — recomputes offsets per layer from residual stream |
| **Jr** | RoPE reset + random i.i.d. line offsets (ablation) |
| **K** | Purely content-derived angles (no positional info) |

## Key result

| Model | ctx=256 | ctx=512 | ctx=1024 | ctx=2048 | ctx=4096 |
|-------|---------|---------|----------|----------|----------|
| B (RoPE)          | 4.71 | 5.78 | 7.56 | 9.62 | 12.09 |
| J (reset+content) | 4.90 | 4.86 | 4.93 | 5.01 | 5.27  |

(Trained at ctx=256, n_embed=128, n_layers=4, 5K iters on Shakespeare.)

J generalizes to 4x training context (4.90 → 5.27) while B degrades sharply (4.71 → 12.09).

## Causality

The content-based offset is applied only to K for cross-line pairs; same-line pairs use pure reset-RoPE. Run `python causality_test.py` to verify all models pass (exact 0.0 delta).

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
