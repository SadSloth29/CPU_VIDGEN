import torch
import torch.nn.functional as F


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
):
    """
    CPU-friendly attention implementation (no flash attention).

    q: [B, Lq, Nq, C]
    k: [B, Lk, Nk, C]
    v: [B, Lk, Nk, Cv]
    """

    B, Lq, Nq, C = q.shape
    _, Lk, Nk, _ = k.shape

    assert Nq == Nk, "For simplicity, require same number of heads"

    if q_scale is not None:
        q = q * q_scale

    if softmax_scale is None:
        softmax_scale = 1.0 / (C ** 0.5)

    # Move heads forward: [B, N, L, C]
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)

    # Compute attention scores
    # [B, N, Lq, Lk]
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale

    # --- Causal mask ---
    if causal:
        causal_mask = torch.triu(
            torch.ones(Lq, Lk, device=attn_scores.device),
            diagonal=1
        ).bool()
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

    # --- Windowed / Local attention ---
    if window_size != (-1, -1):
        left, right = window_size
        idx_q = torch.arange(Lq, device=attn_scores.device).unsqueeze(1)
        idx_k = torch.arange(Lk, device=attn_scores.device).unsqueeze(0)

        lower = idx_q - left
        upper = idx_q + right

        local_mask = (idx_k < lower) | (idx_k > upper)
        attn_scores = attn_scores.masked_fill(local_mask, float("-inf"))

    # Softmax
    attn_probs = F.softmax(attn_scores, dim=-1)

    if dropout_p > 0.0:
        attn_probs = F.dropout(attn_probs, p=dropout_p)

    # Attention output
    out = torch.matmul(attn_probs, v)

    # Back to [B, L, N, C]
    out = out.transpose(1, 2).contiguous()

    return out.to(q.dtype)