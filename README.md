# SLM: Structured Length-generalized Models

Character-level language models comparing positional encoding strategies for length generalization.

## Models

| Model | Description |
|-------|-------------|
| **B** | Standard RoPE — continuous positions 0..T-1 |
| **Ba** | ALiBi — additive linear bias, no rotation |
| **Bs** | RoPE with newline reset (no content offset) |
| **J** | LISformer — RoPE resets at newlines + content-based line angle offset |
| **Jc** | Per-layer LISformer — recomputes offsets per layer from residual stream |
| **Jr** | RoPE reset + random i.i.d. line offsets (ablation) |
| **K** | Purely content-derived angles (no positional info) |

## Key result

LISformer (J) generalizes to longer contexts while RoPE (B) degrades:

| Model | ctx=256 | ctx=512 | ctx=1024 | ctx=2048 | ctx=4096 |
|-------|---------|---------|----------|----------|----------|
| B     | 3.00    | 2.72    | 2.73     | 3.72     | 12.09    |
| J     | 3.05    | 2.76    | 2.66     | 3.02     | 3.82     |

(Trained at ctx=256, n_embed=128, n_layers=4, 5K iters on Shakespeare)

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
