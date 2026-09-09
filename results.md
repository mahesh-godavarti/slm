# LISformer: Content-Addressed Line Structure for Length Generalization

## Experiment

Train character-level language models on Shakespeare (~1M chars), compare length generalization. All models are trained on 256-char chunks and evaluated on 256–4096-char chunks.

## Models

**B** (baseline): Standard RoPE. Positions increment continuously across the entire context.

**Ba** (ALiBi): Attention with Linear Biases. No Q/K rotation — positions encoded via an additive bias -m·(i-j) on attention scores, with a different slope m per head (geometric sequence). A well-known alternative to RoPE with claimed length generalization properties.

**Bs** (reset only): RoPE positions reset to 0 at each newline. No content-based offset — all lines have identical positional encoding.

**Jr** (reset + random): RoPE reset at newlines + random i.i.d. angle offset per line per forward pass. Tests whether any distinct per-line signal suffices for length generalization.

**J** (LISformer): RoPE resets to 0 at each newline. Each completed line gets a content-based angle offset: character embeddings are rotated by their within-line RoPE angle (the "operator"), scatter-added per line, mean-pooled, then projected through LayerNorm → Linear to produce an angle offset in C/2 dimensions. This offset is added to the reset RoPE for past lines only — the current (last) line has zero offset, since its content is incomplete. Within-line attention sees pure RoPE (the shared offset cancels in relative angles); cross-line attention uses the offset to distinguish past lines by content.

**Jc** (per-layer LISformer): Like J but recomputes the line offset at each transformer layer from the evolving residual stream, using a shared LayerNorm → Linear projection. The offset is context-dependent rather than static.

**K** (pure content angles): No RoPE, no positional information at all. Each layer produces angles entirely from the residual stream via its own LayerNorm → Linear projection. Attention uses relative angles (θ_i − θ_j) just like RoPE, but both θ_i and θ_j come from content, not position.

## Architecture

- Character-level vocabulary (65 tokens)
- n_embed=128, n_layers=4, n_heads=4
- Multi-head RoPE attention with F.scaled_dot_product_attention
- Pre-norm transformer blocks (LayerNorm → Attn, LayerNorm → FFN)
- AdamW (lr=3e-4, weight_decay=0.1), cosine schedule, grad clip 1.0
- 5000 iterations, batch_size=64, dropout=0.1
- B/Ba/K: 810K params; Bs/Jr/J: 810–819K params; Jc: 819K params

## Results

### PPL by context length (post-training evaluation)

| Model | ctx=256 | ctx=512 | ctx=1024 | ctx=2048 | ctx=4096 |
|-------|---------|---------|----------|----------|----------|
| **B** (RoPE) | 4.71 | 5.78 | 7.56 | 9.62 | 12.09 |
| **Ba** (ALiBi) | 4.96 | 4.94 | 4.96 | 4.93 | 4.93 |
| **K** (content angles) | 6.45 | 8.12 | 12.50 | 16.65 | 19.20 |
| **Bs** (reset only) | 5.18 | 5.63 | 7.77 | 10.51 | 12.70 |
| **Jr** (reset+random) | 4.89 | 4.88 | 4.93 | 4.94 | 4.98 |
| **J** (LISformer) | 4.04 | 3.85 | 3.78 | 3.75 | 3.82 |
| **Jc** (per-layer) | 3.58 | 3.38 | 3.34 | 3.34 | 3.44 |

### Key observations

1. **Three regimes of length generalization.** Models fall into three groups: those that degrade (B, Bs, K), those that are flat but mediocre (Ba, Jr), and those that are flat and excellent (J, Jc).

2. **B degrades sharply beyond training length.** PPL rises from 4.71 to 12.09 as context grows from 256 to 4096 — a 2.6× degradation. This is the classic RoPE length generalization failure: the model encounters position values at eval (257–4095) that it never saw during training.

3. **ALiBi achieves flat PPL but at a worse level than LISformer.** Ba holds steady at ~4.95 across all context lengths, confirming ALiBi's known length generalization. But it is 30% worse than J (4.93 vs 3.82 at ctx=4096) and 43% worse than Jc (4.93 vs 3.44). ALiBi's linear distance bias is a blunt instrument — it penalizes distant tokens uniformly without exploiting the natural line structure of text.

4. **Random offsets match ALiBi.** Jr (reset + random offsets) achieves nearly identical performance to Ba (~4.9 across all lengths). Both solve length generalization by providing *some* mechanism to distinguish contexts, but neither uses content information. This suggests ALiBi's benefit is purely length generalization, not better modeling.

5. **Reset alone is the worst of both worlds.** Bs degrades like B (5.18 → 12.70) because all lines look positionally identical — the model cannot distinguish line 3 from line 30.

6. **Pure content angles (K) fail.** K is worse than B at every length and degrades even faster (6.45 → 19.20). Without any positional signal, the first layer cannot distinguish identical characters at different positions. Subsequent layers inherit this weak foundation. The causal mask provides ordering but not distance.

7. **Content-based addressing dominates.** J beats all non-content models at every context length, including the training length (4.04 vs 4.71 for B, 4.96 for Ba at ctx=256). This shows LISformer's advantage is not just length extrapolation — encoding line structure directly is a better prior for line-structured text.

8. **Per-layer contextualization adds further gains.** Jc improves over J by recomputing line offsets from the evolving residual stream at each layer. At ctx=4096, Jc achieves 3.44 vs J's 3.82 — an additional 10% improvement.

## Ablation ladder

The seven models form a clean ablation:

- **B → Bs**: Adding position reset without content offset *hurts* — lines become indistinguishable.
- **Bs → Jr**: Adding random per-line offsets fixes length generalization — any distinguishing signal suffices.
- **Jr ≈ Ba**: Random offsets and ALiBi achieve similar flat-but-mediocre performance (~4.9).
- **Jr → J**: Replacing random with content-based offsets improves quality by 22% (4.98 → 3.82 at ctx=4096) — content addressing lets the model know *what* it's attending to, not just *that* it's different.
- **J → Jc**: Per-layer contextualized offsets add another 10% (3.82 → 3.44) — context-dependent addressing refines with depth.
- **K**: Removing RoPE entirely and relying solely on content-derived angles fails — you cannot bootstrap position from content at the first layer.

## Why it works

Shakespeare text is naturally structured into lines (dialogue, verse, stage directions). The LISformer exploits this structure:

- **Position reset at newlines** keeps RoPE values small and within the training distribution. A typical Shakespeare line is 20–60 characters, so J never encounters positions beyond ~60 regardless of total context length. B at ctx=4096 must handle positions 0–4095, most of which it has never seen.

- **Content-based line angle offset** gives each completed line a unique positional identity. Without it, all lines would have identical positional encoding (just pos_in_line × base_freq), making cross-line attention unable to distinguish line 3 from line 30. The offset is computed by rotating each character embedding by its within-line RoPE angle ("multiply by operator"), scatter-adding per line, mean-pooling, and projecting to angle space. This produces a position-sensitive summary of each line's content that serves as its "address." The current line's offset is zeroed out since its content is incomplete; within-line attention is unaffected since the shared offset cancels in relative angles anyway.

- **Better inductive bias even at training length.** The 14% PPL improvement at ctx=256 (4.04 vs 4.71) shows this isn't just about length extrapolation. Encoding the line structure directly is a better prior for line-structured text, freeing the model from having to discover this structure implicitly through continuous positions.

## Reproduction

```bash
cd /home/ubuntu/slm
source /home/ubuntu/experiments/venv/bin/activate

# Smoke test (~5s)
python slm.py --smoke

# Full run — all models (~30 min total)
python slm.py --models B Ba Bs Jr J Jc K --n_embed 128 --n_layers 4 --iters 5000
```
