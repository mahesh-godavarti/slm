#!/usr/bin/env python3
"""Character-level SLM: RoPE (B) vs Content-based Line Addressing (J)

Train on Shakespeare, eval PPL at various context lengths.
Model B: Standard RoPE — positions increment continuously.
Model J: RoPE resets at newlines + content-based line angle offset via
         rotate embeddings by RoPE -> scatter-add per line -> mean pool ->
         LayerNorm -> Linear -> angle offset.
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Data
# ============================================================================

def load_data(path, train_frac=0.9):
    with open(path, 'r') as f:
        text = f.read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(len(data) * train_frac)
    return data[:n], data[n:], vocab_size, stoi, itos


def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)
    return x, y


# ============================================================================
# RoPE utilities
# ============================================================================

def make_base_freq(n_embed):
    """Standard RoPE frequency schedule: 1/10000^(2i/d) for each dim pair."""
    return 1.0 / (10000 ** (torch.arange(0, n_embed // 2).float() / (n_embed // 2)))


def apply_rotation(x, angles):
    """Apply RoPE rotation. Works for any shape (..., D) with angles (..., D//2)."""
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    x_pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    x0, x1 = x_pairs[..., 0], x_pairs[..., 1]
    r0 = x0 * cos_a - x1 * sin_a
    r1 = x0 * sin_a + x1 * cos_a
    return torch.stack([r0, r1], dim=-1).reshape_as(x)


# ============================================================================
# Transformer Blocks
# ============================================================================

class RotaryAttention(nn.Module):
    def __init__(self, n_embed, n_heads, dropout=0.1):
        super().__init__()
        assert n_embed % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embed // n_heads
        self.q_proj = nn.Linear(n_embed, n_embed)
        self.k_proj = nn.Linear(n_embed, n_embed)
        self.v_proj = nn.Linear(n_embed, n_embed)
        self.out_proj = nn.Linear(n_embed, n_embed)
        self.attn_drop = dropout

    def forward(self, x, angles):
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)
        # Split angles across heads: (B, T, C//2) -> (B, H, T, D//2)
        a = angles.view(B, T, H, D // 2).transpose(1, 2)
        q = apply_rotation(q, a)
        k = apply_rotation(k, a)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_drop if self.training else 0.0)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C))


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = RotaryAttention(n_embed, n_heads, dropout)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ffn = FeedForward(n_embed, dropout)

    def forward(self, x, angles):
        x = x + self.attn(self.ln1(x), angles)
        x = x + self.ffn(self.ln2(x))
        return x


# ============================================================================
# Addressed Attention (causal content addressing for LISformer)
# ============================================================================

class AddressedAttention(nn.Module):
    """Attention with causal content addressing.

    Same-line pairs: Q and K rotated by reset-RoPE only (pure positional).
    Cross-line pairs: Q rotated by prefix address (line so far),
                      K rotated by full-line address.
    Two-sided content matching without leaking future tokens.
    """

    def __init__(self, n_embed, n_heads, dropout=0.1):
        super().__init__()
        assert n_embed % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embed // n_heads
        self.q_proj = nn.Linear(n_embed, n_embed)
        self.k_proj = nn.Linear(n_embed, n_embed)
        self.v_proj = nn.Linear(n_embed, n_embed)
        self.out_proj = nn.Linear(n_embed, n_embed)
        self.attn_drop = dropout

    def forward(self, x, rope, offset, prefix_offset, line_ids):
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)
        # Split angles across heads: (B, T, C//2) -> (B, H, T, D//2)
        r = rope.view(B, T, H, D // 2).transpose(1, 2)
        o = offset.view(B, T, H, D // 2).transpose(1, 2)
        po = prefix_offset.view(B, T, H, D // 2).transpose(1, 2)
        # Plain: Q and K rotated by reset-RoPE only
        q_rot = apply_rotation(q, r)
        k_plain = apply_rotation(k, r)
        # Addressed: Q by prefix address, K by full-line address
        q_addr = apply_rotation(q_rot, po)
        k_addr = apply_rotation(k_plain, o)
        scale = 1.0 / math.sqrt(D)
        plain_scores = (q_rot @ k_plain.transpose(-2, -1)) * scale
        addr_scores = (q_addr @ k_addr.transpose(-2, -1)) * scale
        # Same-line mask: True where query and key are in the same line
        same_line = (line_ids.unsqueeze(2) == line_ids.unsqueeze(1))  # (B, T, T)
        scores = torch.where(same_line.unsqueeze(1), plain_scores, addr_scores)
        # Causal mask
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        if self.training and self.attn_drop > 0:
            attn = F.dropout(attn, p=self.attn_drop)
        out = attn @ v
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C))


class AddressedBlock(nn.Module):
    def __init__(self, n_embed, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = AddressedAttention(n_embed, n_heads, dropout)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ffn = FeedForward(n_embed, dropout)

    def forward(self, x, rope, offset, prefix_offset, line_ids):
        x = x + self.attn(self.ln1(x), rope, offset, prefix_offset, line_ids)
        x = x + self.ffn(self.ln2(x))
        return x


# ============================================================================
# ALiBi Attention
# ============================================================================

class ALiBiAttention(nn.Module):
    def __init__(self, n_embed, n_heads, dropout=0.1):
        super().__init__()
        assert n_embed % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embed // n_heads
        self.q_proj = nn.Linear(n_embed, n_embed)
        self.k_proj = nn.Linear(n_embed, n_embed)
        self.v_proj = nn.Linear(n_embed, n_embed)
        self.out_proj = nn.Linear(n_embed, n_embed)
        self.attn_drop = dropout
        # ALiBi slopes: geometric sequence 2^(-8*i/n) for i=1..n
        slopes = torch.tensor(
            [2 ** (-8.0 * i / n_heads) for i in range(1, n_heads + 1)])
        self.register_buffer('slopes', slopes)

    def forward(self, x):
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)
        # ALiBi bias: -slope * (i - j), with causal mask
        rel = torch.arange(T, device=x.device).unsqueeze(1) - \
              torch.arange(T, device=x.device).unsqueeze(0)  # (T, T), [i,j] = i-j
        bias = -self.slopes[:, None, None] * rel.float().unsqueeze(0)  # (H, T, T)
        bias = bias.masked_fill(rel.unsqueeze(0) < 0, float('-inf'))
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias.unsqueeze(0).expand(B, -1, -1, -1),
            dropout_p=self.attn_drop if self.training else 0.0)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C))


class ALiBiBlock(nn.Module):
    def __init__(self, n_embed, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = ALiBiAttention(n_embed, n_heads, dropout)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ffn = FeedForward(n_embed, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ============================================================================
# Model Ba: ALiBi (Attention with Linear Biases)
# ============================================================================

class ModelBa(nn.Module):
    """Char-level transformer with ALiBi positional encoding.
    No rotation of Q/K — positions encoded via additive linear bias
    on attention scores: -slope * (i - j) per head."""

    def __init__(self, vocab_size, n_embed, n_layers, n_heads, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.ModuleList(
            [ALiBiBlock(n_embed, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_emb(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ============================================================================
# Model B: Standard RoPE
# ============================================================================

class ModelB(nn.Module):
    """Char-level transformer with standard RoPE (continuous positions)."""

    def __init__(self, vocab_size, n_embed, n_layers, n_heads, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embed)
        self.register_buffer('base_freq', make_base_freq(n_embed))
        self.blocks = nn.ModuleList(
            [Block(n_embed, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_emb(idx)
        pos = torch.arange(T, device=idx.device, dtype=torch.float)
        angles = torch.outer(pos, self.base_freq).unsqueeze(0).expand(B, -1, -1)
        for block in self.blocks:
            x = block(x, angles)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ============================================================================
# Model Bs: RoPE with newline reset only (no content offset)
# ============================================================================

class ModelBs(nn.Module):
    """RoPE positions reset to 0 at each newline. No content-based offset."""

    def __init__(self, vocab_size, n_embed, n_layers, n_heads, newline_id,
                 dropout=0.1):
        super().__init__()
        self.newline_id = newline_id
        self.token_emb = nn.Embedding(vocab_size, n_embed)
        self.register_buffer('base_freq', make_base_freq(n_embed))
        self.blocks = nn.ModuleList(
            [Block(n_embed, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def _line_pos(self, idx):
        B, T = idx.shape
        device = idx.device
        is_nl = (idx == self.newline_id)
        starts = torch.zeros(B, T, dtype=torch.bool, device=device)
        starts[:, 0] = True
        starts[:, 1:] = is_nl[:, :-1]
        gpos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        marker = torch.where(starts, gpos, torch.zeros_like(gpos))
        line_start, _ = marker.cummax(dim=1)
        return (gpos - line_start).float()

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_emb(idx)
        pos_in_line = self._line_pos(idx)
        angles = pos_in_line.unsqueeze(-1) * self.base_freq  # (B, T, C//2)
        for block in self.blocks:
            x = block(x, angles)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ============================================================================
# Model J: Newline-reset RoPE + content-based line angle offset
# ============================================================================

class ModelJ(nn.Module):
    """Char-level transformer with line-aware positional encoding.

    - RoPE position resets to 0 at each newline boundary
    - Each completed line gets a content-based angle offset: rotate
      embeddings by RoPE, scatter-add per line, mean-pool, LN → Linear.
    - Q gets prefix address (cumulative mean of line so far), K gets
      full-line address. Two-sided content matching, causal by construction.
    - Same-line pairs use pure reset-RoPE.
    - random_addresses=True: replaces content addresses with i.i.d. random
      per-line draws (control ablation, formerly ModelJr).
    """

    def __init__(self, vocab_size, n_embed, n_layers, n_heads, newline_id,
                 dropout=0.1, random_addresses=False):
        super().__init__()
        self.newline_id = newline_id
        self.n_embed = n_embed
        self.random_addresses = random_addresses
        self.token_emb = nn.Embedding(vocab_size, n_embed)
        self.register_buffer('base_freq', make_base_freq(n_embed))
        if not random_addresses:
            self.line_ln = nn.LayerNorm(n_embed)
            self.line_proj = nn.Linear(n_embed, n_embed // 2)
        self.blocks = nn.ModuleList(
            [AddressedBlock(n_embed, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def _line_info(self, idx):
        """Compute line_ids, pos_in_line, and line_start from newline positions.

        Newline is the LAST character of a line; the character after it
        starts a new line with position 0.
        """
        B, T = idx.shape
        device = idx.device
        is_nl = (idx == self.newline_id)
        starts = torch.zeros(B, T, dtype=torch.bool, device=device)
        starts[:, 0] = True
        starts[:, 1:] = is_nl[:, :-1]
        line_ids = starts.long().cumsum(dim=1) - 1
        gpos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        marker = torch.where(starts, gpos, torch.zeros_like(gpos))
        line_start, _ = marker.cummax(dim=1)
        pos_in_line = (gpos - line_start).float()
        return line_ids, pos_in_line, line_start

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        x = self.token_emb(idx)
        line_ids, pos_in_line, line_start = self._line_info(idx)

        # RoPE with per-line position reset
        rope = pos_in_line.unsqueeze(-1) * self.base_freq  # (B, T, C//2)

        half = self.n_embed // 2
        if self.random_addresses:
            # Random i.i.d. offset per line (control ablation)
            n_lines = line_ids.max().item() + 1
            line_angles = torch.randn(B, n_lines, half, device=device)
            offset = line_angles.gather(
                1, line_ids.unsqueeze(-1).expand(-1, -1, half))
            prefix_offset = offset  # random has no prefix concept
        else:
            # Content-based line angle offset (full-line mean for K)
            rotated = apply_rotation(x, rope)  # (B, T, C)
            n_lines = line_ids.max().item() + 1
            lid_exp = line_ids.unsqueeze(-1).expand(-1, -1, self.n_embed)
            sums = torch.zeros(B, n_lines, self.n_embed, device=device)
            sums.scatter_add_(1, lid_exp, rotated)
            counts = torch.zeros(B, n_lines, 1, device=device)
            counts.scatter_add_(1, line_ids.unsqueeze(-1),
                                torch.ones(B, T, 1, device=device))
            pooled = sums / counts.clamp(min=1)  # (B, n_lines, C)
            line_angles = self.line_proj(self.line_ln(pooled))
            offset = line_angles.gather(
                1, line_ids.unsqueeze(-1).expand(-1, -1, half))

            # Prefix address (segmented cumulative mean for Q)
            cumsum_full = torch.cumsum(rotated, dim=1)  # (B, T, C)
            ls_prev = (line_start - 1).clamp(min=0)
            boundary = cumsum_full.gather(
                1, ls_prev.unsqueeze(-1).expand(-1, -1, self.n_embed))
            boundary = boundary * (line_start > 0).unsqueeze(-1).float()
            seg_cumsum = cumsum_full - boundary  # per-line cumsum
            prefix_mean = seg_cumsum / (pos_in_line + 1).unsqueeze(-1)
            prefix_offset = self.line_proj(self.line_ln(prefix_mean))

        for block in self.blocks:
            x = block(x, rope, offset, prefix_offset, line_ids)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ============================================================================
# Model Jc: Per-layer contextualized line offset (shared projection)
# ============================================================================

class ModelJc(nn.Module):
    """Like ModelJ but recomputes line offset at each layer from the
    contextualized residual stream. Shared line_ln + line_proj across layers.
    Uses AddressedAttention for causal content addressing."""

    def __init__(self, vocab_size, n_embed, n_layers, n_heads, newline_id,
                 dropout=0.1):
        super().__init__()
        self.newline_id = newline_id
        self.n_embed = n_embed
        self.token_emb = nn.Embedding(vocab_size, n_embed)
        self.register_buffer('base_freq', make_base_freq(n_embed))
        self.line_ln = nn.LayerNorm(n_embed)
        self.line_proj = nn.Linear(n_embed, n_embed // 2)
        self.blocks = nn.ModuleList(
            [AddressedBlock(n_embed, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def _line_info(self, idx):
        B, T = idx.shape
        device = idx.device
        is_nl = (idx == self.newline_id)
        starts = torch.zeros(B, T, dtype=torch.bool, device=device)
        starts[:, 0] = True
        starts[:, 1:] = is_nl[:, :-1]
        line_ids = starts.long().cumsum(dim=1) - 1
        gpos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        marker = torch.where(starts, gpos, torch.zeros_like(gpos))
        line_start, _ = marker.cummax(dim=1)
        pos_in_line = (gpos - line_start).float()
        return line_ids, pos_in_line, line_start

    def _compute_offsets(self, x, rope, line_ids, lid_exp, lid_exp_half,
                         counts, line_start, pos_in_line):
        """Compute full-line offset (for K) and prefix offset (for Q)."""
        rotated = apply_rotation(x, rope)
        B = x.shape[0]
        n_lines = counts.shape[1]
        # Full-line mean (for K)
        sums = torch.zeros(B, n_lines, self.n_embed, device=x.device)
        sums.scatter_add_(1, lid_exp, rotated)
        pooled = sums / counts
        line_angles = self.line_proj(self.line_ln(pooled))
        offset = line_angles.gather(1, lid_exp_half)
        # Prefix mean (for Q)
        cumsum_full = torch.cumsum(rotated, dim=1)
        ls_prev = (line_start - 1).clamp(min=0)
        boundary = cumsum_full.gather(
            1, ls_prev.unsqueeze(-1).expand(-1, -1, self.n_embed))
        boundary = boundary * (line_start > 0).unsqueeze(-1).float()
        seg_cumsum = cumsum_full - boundary
        prefix_mean = seg_cumsum / (pos_in_line + 1).unsqueeze(-1)
        prefix_offset = self.line_proj(self.line_ln(prefix_mean))
        return offset, prefix_offset

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        x = self.token_emb(idx)
        line_ids, pos_in_line, line_start = self._line_info(idx)
        rope = pos_in_line.unsqueeze(-1) * self.base_freq

        # Precompute reusable index tensors
        n_lines = line_ids.max().item() + 1
        lid_exp = line_ids.unsqueeze(-1).expand(-1, -1, self.n_embed)
        lid_exp_half = line_ids.unsqueeze(-1).expand(-1, -1, self.n_embed // 2)
        counts = torch.zeros(B, n_lines, 1, device=device)
        counts.scatter_add_(1, line_ids.unsqueeze(-1),
                            torch.ones(B, T, 1, device=device))
        counts = counts.clamp(min=1)

        for block in self.blocks:
            offset, prefix_offset = self._compute_offsets(
                x, rope, line_ids, lid_exp, lid_exp_half, counts,
                line_start, pos_in_line)
            x = block(x, rope, offset, prefix_offset, line_ids)

        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ============================================================================
# Model K: Purely content-derived angles (no RoPE, no positions)
# ============================================================================

class ModelK(nn.Module):
    """Each layer projects the residual stream to angles via LayerNorm -> Linear.
    No positional information at all — angles come entirely from content.
    Attention uses relative angle (theta_i - theta_j) just like RoPE."""

    def __init__(self, vocab_size, n_embed, n_layers, n_heads, dropout=0.1):
        super().__init__()
        self.n_embed = n_embed
        self.token_emb = nn.Embedding(vocab_size, n_embed)
        self.angle_lns = nn.ModuleList(
            [nn.LayerNorm(n_embed) for _ in range(n_layers)])
        self.angle_projs = nn.ModuleList(
            [nn.Linear(n_embed, n_embed // 2) for _ in range(n_layers)])
        self.blocks = nn.ModuleList(
            [Block(n_embed, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_emb(idx)
        for angle_ln, angle_proj, block in zip(
                self.angle_lns, self.angle_projs, self.blocks):
            angles = angle_proj(angle_ln(x))  # (B, T, C//2)
            x = block(x, angles)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ============================================================================
# Training & Evaluation
# ============================================================================

@torch.no_grad()
def eval_ppl(model, data, block_size, batch_size, device, n_batches=50):
    """Evaluate perplexity at a given context length."""
    model.eval()
    total_loss = 0.0
    n = 0
    for _ in range(n_batches):
        x, y = get_batch(data, batch_size, block_size, device)
        _, loss = model(x, y)
        total_loss += loss.item()
        n += 1
    model.train()
    return math.exp(total_loss / n) if n > 0 else float('inf')


@torch.no_grad()
def eval_ppl_final_line(model, data, block_size, batch_size, device,
                        newline_id, n_batches=50):
    """Evaluate PPL only on tokens in the final line of each chunk.
    These tokens are leak-free: their own line's offset is never used
    in cross-line attention, and all past lines are genuinely complete."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for _ in range(n_batches):
        x, y = get_batch(data, batch_size, block_size, device)
        logits, _ = model(x)
        B, T = x.shape
        is_nl = (x == newline_id)
        starts = torch.zeros(B, T, dtype=torch.bool, device=device)
        starts[:, 0] = True
        starts[:, 1:] = is_nl[:, :-1]
        line_ids = starts.long().cumsum(dim=1) - 1
        max_line = line_ids.max(dim=1, keepdim=True).values
        final_mask = (line_ids == max_line)
        loss_all = F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), reduction='none')
        total_loss += (loss_all.view(B, T) * final_mask.float()).sum().item()
        total_tokens += final_mask.sum().item()
    model.train()
    return math.exp(total_loss / max(total_tokens, 1))


def train_model(model, train_data, val_data, args, name):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)
    model.train()
    for i in range(1, args.iters + 1):
        x, y = get_batch(train_data, args.batch_size, args.train_block_size,
                         args.device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if i % args.eval_interval == 0 or i == 1:
            val_ppl = eval_ppl(model, val_data, args.train_block_size,
                               args.eval_batch_size, args.device,
                               args.eval_batches)
            print(f"[{name}] iter {i:5d} | loss {loss.item():.4f} "
                  f"| val_ppl {val_ppl:.2f}")
            sys.stdout.flush()


def eval_lengths(model, val_data, lengths, device, batch_size, n_batches):
    """Evaluate PPL at multiple context lengths."""
    results = {}
    for L in lengths:
        if len(val_data) <= L + 1:
            results[L] = float('inf')
            continue
        # Scale batch for long contexts (manual attention is O(T²) memory)
        bs = max(1, min(batch_size, batch_size * 512 // L))
        ppl = eval_ppl(model, val_data, L, bs, device, n_batches)
        results[L] = ppl
    return results


def eval_lengths_final_line(model, val_data, lengths, device, batch_size,
                            n_batches, newline_id):
    """Evaluate final-line-only PPL at multiple context lengths."""
    results = {}
    for L in lengths:
        if len(val_data) <= L + 1:
            results[L] = float('inf')
            continue
        bs = max(1, min(batch_size, batch_size * 512 // L))
        ppl = eval_ppl_final_line(model, val_data, L, bs, device,
                                  newline_id, n_batches)
        results[L] = ppl
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Char-level SLM: RoPE (B) vs Line Addressing (J)")
    p.add_argument('--data', default='input.txt')
    p.add_argument('--models', nargs='+', default=['B', 'J'])
    p.add_argument('--n_embed', type=int, default=128)
    p.add_argument('--n_layers', type=int, default=4)
    p.add_argument('--n_heads', type=int, default=4)
    p.add_argument('--train_block_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--eval_batch_size', type=int, default=32)
    p.add_argument('--eval_batches', type=int, default=50)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--iters', type=int, default=5000)
    p.add_argument('--eval_interval', type=int, default=500)
    p.add_argument('--eval_lengths', nargs='+', type=int,
                   default=[256, 512, 1024, 2048, 4096])
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    if args.smoke:
        args.iters = 100
        args.eval_interval = 50
        args.eval_batches = 5
        args.eval_lengths = [256, 512]
        args.n_embed = 64
        args.n_layers = 2

    torch.manual_seed(args.seed)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, args.data)
    train_data, val_data, vocab_size, stoi, itos = load_data(data_path)
    newline_id = stoi['\n']

    print(f"Vocab: {vocab_size} | Train: {len(train_data):,} | "
          f"Val: {len(val_data):,}")
    print(f"Device: {args.device} | n_embed={args.n_embed} "
          f"n_layers={args.n_layers} n_heads={args.n_heads}")
    print(f"Train ctx={args.train_block_size} | "
          f"Eval lengths: {args.eval_lengths}")
    print()

    all_results = {}
    for name in args.models:
        print(f"{'=' * 60}")
        print(f"Model {name}")
        print(f"{'=' * 60}")
        torch.manual_seed(args.seed)
        if name == 'B':
            model = ModelB(vocab_size, args.n_embed, args.n_layers,
                           args.n_heads, args.dropout)
        elif name == 'Ba':
            model = ModelBa(vocab_size, args.n_embed, args.n_layers,
                            args.n_heads, args.dropout)
        elif name == 'Bs':
            model = ModelBs(vocab_size, args.n_embed, args.n_layers,
                            args.n_heads, newline_id, args.dropout)
        elif name == 'J':
            model = ModelJ(vocab_size, args.n_embed, args.n_layers,
                           args.n_heads, newline_id, args.dropout)
        elif name == 'Jr':
            model = ModelJ(vocab_size, args.n_embed, args.n_layers,
                           args.n_heads, newline_id, args.dropout,
                           random_addresses=True)
        elif name == 'Jc':
            model = ModelJc(vocab_size, args.n_embed, args.n_layers,
                            args.n_heads, newline_id, args.dropout)
        elif name == 'K':
            model = ModelK(vocab_size, args.n_embed, args.n_layers,
                           args.n_heads, args.dropout)
        else:
            print(f"Unknown model: {name}")
            continue

        n_params = sum(pp.numel() for pp in model.parameters())
        print(f"Params: {n_params:,}")
        model = model.to(args.device)

        t0 = time.time()
        train_model(model, train_data, val_data, args, name)
        dt = time.time() - t0
        print(f"Time: {dt:.1f}s\n")

        print("Eval PPL by context length:")
        res = eval_lengths(model, val_data, args.eval_lengths, args.device,
                           args.eval_batch_size, args.eval_batches)
        for L in sorted(res):
            print(f"  ctx={L:5d}  PPL={res[L]:.2f}")
        print()

        print("Eval PPL (final-line only) by context length:")
        res_fl = eval_lengths_final_line(model, val_data, args.eval_lengths,
                                         args.device, args.eval_batch_size,
                                         args.eval_batches, newline_id)
        for L in sorted(res_fl):
            print(f"  ctx={L:5d}  PPL={res_fl[L]:.2f}")
        print()
        all_results[name] = res

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("SUMMARY: PPL by context length")
        print(f"{'=' * 60}")
        lengths = sorted(args.eval_lengths)
        hdr = f"{'Model':>6}"
        for L in lengths:
            hdr += f"  ctx={L:>5d}"
        print(hdr)
        print("-" * len(hdr))
        for name in args.models:
            if name not in all_results:
                continue
            row = f"{name:>6}"
            for L in lengths:
                ppl = all_results[name].get(L, float('inf'))
                row += f"  {ppl:>10.2f}"
            print(row)


if __name__ == '__main__':
    main()
