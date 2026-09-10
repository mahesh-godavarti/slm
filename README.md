# SLM: Structured Length-generalized Models

Character-level language models comparing positional encoding strategies for length generalization.

## Models

| Model | Description |
|-------|-------------|
| **B** | Standard RoPE — continuous positions 0..T-1 |
| **J** | LISformer — RoPE resets at newlines + two-score attention with content-based K addressing |
| **Jr** | J with random i.i.d. K line addresses instead of content-derived (ablation control) |

## Key result

| Model | ctx=256 | ctx=512 | ctx=1024 | ctx=2048 | ctx=4096 |
|-------|---------|---------|----------|----------|----------|
| B (RoPE)          | 4.71 | 5.78 | 7.56 | 9.62 | 12.09 |
| J (reset+content) | 4.90 | 4.86 | 4.93 | 5.01 | 5.27  |
| Jr (reset+random) | 4.90 | 4.91 | 4.98 | 5.05 | 5.25  |

(Trained at ctx=256, n_embed=128, n_layers=4, 5K iters on Shakespeare.)

J generalizes to 16x training context (4.90 → 5.27) while B degrades sharply (4.71 → 12.09). Jr matches J closely, showing the length generalization comes from the structural reset + two-score routing, not the content signal.

## Two-score attention

Cross-line attention uses two scores selected by a same-line mask:

- **Same-line pairs** → plain scores: Q and K rotated by reset-RoPE only (within-line positional matching)
- **Cross-line pairs** → addressed scores: K additionally rotated by its line's content address, Q has no address (zero offset)

K's line address is computed from input embeddings: rotate by reset-RoPE, scatter-add per line, mean-pool, LayerNorm → Linear. Q receives no address rotation — cross-line matching is purely K-sided.

This is fully causal: K addresses summarize completed past lines, Q uses only reset-RoPE positions. Run `python causality_test.py` to verify (exact 0.0 delta).

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
