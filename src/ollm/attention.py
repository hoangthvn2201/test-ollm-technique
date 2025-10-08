import torch
import math
from typing import Optional

def online_chunked_grouped_attention_rope_no_mask(
    q,                    # (B, Hq, Lq, D)  -- RoPE already applied to q
    k,                    # (B, Hkv, Lk, D) -- RoPE already applied to k
    v,                    # (B, Hkv, Lk, D)
    position_ids: Optional[torch.Tensor] = None,  # (B, L) or (L,) -- not used for masking here
    q_block_size: int = 32768, #64,
    k_block_size: int = 1024, #512,
    eps: float = 1e-12,
):
    """
    Online chunked grouped multi-head attention WITHOUT masking (inference-only).
    - q: (B, Hq, Lq, D) (RoPE already applied)
    - k: (B, Hkv, Lk, D) (RoPE already applied)
    - v: (B, Hkv, Lk, D)
    - position_ids: accepted for API compatibility (came from RoPE) but NOT used here.
    Returns:
      out: (B, Hq, Lq, D)
    Notes:
      * This does streaming softmax (log-sum-exp merge) across k-blocks.
      * Internal accumulators are float32 even if inputs are fp16 for numeric stability.
      * No causal or other masking is applied — it's pure full attention (softmax over all keys).
    """
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    B, Hq, Lq, D = q.shape
    _, Hkv, Lk, Dk = k.shape
    assert D == Dk and v.shape == (B, Hkv, Lk, D)

    device = q.device
    dtype = q.dtype
    scale = 1.0 / math.sqrt(D)

    # head mapping Hq -> Hkv (bucket each consecutive group to a kv head)
    group_size = (Hq + Hkv - 1) // Hkv 
    head_mapping = torch.arange(Hq, device=device) // group_size  # (Hq,)
    groups = []
    for hkv in range(Hkv):
        q_head_idxs = (head_mapping == hkv).nonzero(as_tuple=False).squeeze(1)
        groups.append(q_head_idxs if q_head_idxs.numel() > 0 else None)

    # position_ids are not used here, but keep signature for compatibility
    # (they should match the RoPE you applied to q/k upstream).
    if position_ids is not None:
        # basic sanity move to device (we don't use it further)
        if position_ids.dim() == 1:
            _ = position_ids.unsqueeze(0).expand(B, -1).to(device)
        else:
            _ = position_ids.to(device)

    out = torch.zeros((B, Hq, Lq, D), device=device, dtype=dtype)

    # iterate over each KV head group (vectorized over batch and group's q-heads)
    for hkv_idx, q_head_idxs in enumerate(groups):
        if q_head_idxs is None:
            continue

        # k_h, v_h: (B, Lk, D) float32
        k_h = k[:, hkv_idx].to(torch.float32)
        v_h = v[:, hkv_idx].to(torch.float32)

        # q_sub: (B, Hq_g, Lq, D) float32
        q_sub = q[:, q_head_idxs].to(torch.float32)
        Hq_g = q_sub.shape[1]

        # process q in q-blocks so we keep accumulators small
        for q_start in range(0, Lq, q_block_size):
            q_end = min(Lq, q_start + q_block_size)
            Bq = q_end - q_start

            q_block = q_sub[:, :, q_start:q_end, :]  # (B, Hq_g, Bq, D)

            # accumulators (float32)
            # m: running max, s: running sum of exp in that frame, wv: running weighted V
            m = torch.full((B, Hq_g, Bq), float("-inf"), device=device, dtype=torch.float32)
            s = torch.zeros((B, Hq_g, Bq), device=device, dtype=torch.float32)
            wv = torch.zeros((B, Hq_g, Bq, D), device=device, dtype=torch.float32)

            # iterate over k-blocks
            for k_start in range(0, Lk, k_block_size):
                k_end = min(Lk, k_start + k_block_size)
                Bk = k_end - k_start

                k_block = k_h[:, k_start:k_end, :]   # (B, Bk, D)
                v_block = v_h[:, k_start:k_end, :]   # (B, Bk, D)

                # scores: (B, Hq_g, Bq, Bk)
                scores = torch.einsum("b h q d, b k d -> b h q k", q_block, k_block) * scale

                # compute local max per row over k
                local_max = torch.amax(scores, dim=-1)  # (B, Hq_g, Bq)
                exp_scores = torch.exp(scores - local_max.unsqueeze(-1))  # (B, Hq_g, Bq, Bk)
                sum_exp = exp_scores.sum(dim=-1)  # (B, Hq_g, Bq)
                weighted_v_chunk = torch.einsum("b h q k, b k d -> b h q d", exp_scores, v_block)  # (B, Hq_g, Bq, D)

                # merge with global accumulators using log-sum-exp merging
                first_chunk_mask = torch.isinf(m)  # True where uninitialized
                # initialize where first_chunk_mask is True
                if first_chunk_mask.any():
                    init_idx = first_chunk_mask
                    m[init_idx] = local_max[init_idx]
                    s[init_idx] = sum_exp[init_idx]
                    wv[init_idx] = weighted_v_chunk[init_idx]

                # merge where already initialized
                merge_idx = ~first_chunk_mask
                if merge_idx.any():
                    m_old = m[merge_idx]
                    s_old = s[merge_idx]
                    wv_old = wv[merge_idx]
                    lm_new = local_max[merge_idx]
                    se_new = sum_exp[merge_idx]
                    wv_new = weighted_v_chunk[merge_idx]

                    new_m = torch.maximum(m_old, lm_new)
                    alpha = torch.exp(m_old - new_m)
                    beta = torch.exp(lm_new - new_m)

                    s[merge_idx] = s_old * alpha + se_new * beta
                    wv[merge_idx] = wv_old * alpha.unsqueeze(-1) + wv_new * beta.unsqueeze(-1)
                    m[merge_idx] = new_m

                # release temporaries
                del scores, local_max, exp_scores, sum_exp, weighted_v_chunk, k_block, v_block

            # finalize output for this q_block
            denom = s.unsqueeze(-1) + eps  # (B, Hq_g, Bq, 1)
            out_block = (wv / denom).to(dtype)  # cast back to original dtype

            # write into output tensor
            out[:, q_head_idxs, q_start:q_end, :] = out_block

            # free accumulators
            del m, s, wv, out_block, denom, q_block

    return out


def online_chunked_grouped_attention_rope(
    q,                    # (B, Hq, Lq, D)
    k,                    # (B, Hkv, Lk, D)
    v,                    # (B, Hkv, Lk, D)
    position_ids: Optional[torch.Tensor] = None,  # (B, L) or (L,)
    causal: bool = True,
    q_block_size: int = 64,
    k_block_size: int = 512,
    eps: float = 1e-12,
):
    """
    Online chunked grouped multi-head attention (RoPE already applied to q,k).
    - q: queries (B, Hq, Lq, D)
    - k,v: keys/values with Hkv heads (B, Hkv, Lk, D)
    - position_ids: used only for causal masking; must match RoPE positions used earlier
    Returns:
      out: (B, Hq, Lq, D)
    """
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    B, Hq, Lq, D = q.shape
    _, Hkv, Lk, Dk = k.shape
    assert D == Dk and v.shape == (B, Hkv, Lk, D)
    print("attention1.", q.shape, k.shape, v.shape)
    device = q.device
    dtype = q.dtype
    scale = 1.0 / math.sqrt(D)

    # Build head mapping Hq -> Hkv (round-robin / bucket)
    group_size = (Hq + Hkv - 1) // Hkv
    head_mapping = torch.arange(Hq, device=device) // group_size  # (Hq,)
    # For vectorization, collect which q-head indices map to each kv-head
    groups = []
    for hkv in range(Hkv):
        q_head_idxs = (head_mapping == hkv).nonzero(as_tuple=False).squeeze(1)
        if q_head_idxs.numel() == 0:
            groups.append(None)
        else:
            groups.append(q_head_idxs)

    # position ids -> (B, L)
    if position_ids is None:
        pos = torch.arange(Lq, device=device).unsqueeze(0).expand(B, Lq)
    else:
        if position_ids.dim() == 1:
            pos = position_ids.unsqueeze(0).expand(B, -1).to(device)
        else:
            pos = position_ids.to(device)

    out = torch.zeros((B, Hq, Lq, D), device=device, dtype=dtype)

    # We'll iterate over kv-head groups; inside we iterate over q-blocks and k-blocks
    for hkv_idx, q_head_idxs in enumerate(groups):
        if q_head_idxs is None:
            continue

        # slice views for this kv head
        # k_h, v_h: (B, Lk, D)
        k_h = k[:, hkv_idx].to(torch.float32)
        v_h = v[:, hkv_idx].to(torch.float32)

        # q_sub: (B, Hq_g, Lq, D)
        q_sub = q[:, q_head_idxs].to(torch.float32)  # may copy but simpler; small Hq_g usually

        Hq_g = q_sub.shape[1]

        # Process q in blocks (so we don't allocate giant accumulators for full Lq)
        for q_start in range(0, Lq, q_block_size):
            q_end = min(Lq, q_start + q_block_size)
            Bq = q_end - q_start

            q_block = q_sub[:, :, q_start:q_end, :]  # (B, Hq_g, Bq, D)
            q_pos_block = pos[:, q_start:q_end]      # (B, Bq)

            # accumulators per (B, Hq_g, Bq)
            # use float32 accumulators for numeric stability
            m = torch.full((B, Hq_g, Bq), float("-inf"), device=device, dtype=torch.float32)
            s = torch.zeros((B, Hq_g, Bq), device=device, dtype=torch.float32)
            wv = torch.zeros((B, Hq_g, Bq, D), device=device, dtype=torch.float32)

            # iterate over k blocks
            for k_start in range(0, Lk, k_block_size):
                k_end = min(Lk, k_start + k_block_size)
                Bk = k_end - k_start

                k_block = k_h[:, k_start:k_end, :]     # (B, Bk, D) float32
                v_block = v_h[:, k_start:k_end, :]     # (B, Bk, D) float32
                k_pos_block = pos[:, k_start:k_end]    # (B, Bk)

                # scores: (B, Hq_g, Bq, Bk) via einsum
                # q_block: (B, Hq_g, Bq, D); k_block: (B, Bk, D)
                scores = torch.einsum("b h q d, b k d -> b h q k", q_block, k_block) * scale

                # mask for causal: True where masked (k_pos > q_pos)
                if causal:
                    # shape (B, 1, Bq, Bk) broadcastable
                    mask = (k_pos_block.unsqueeze(1).unsqueeze(2) > q_pos_block.unsqueeze(1).unsqueeze(3))
                else:
                    mask = torch.zeros((B, 1, Bq, Bk), dtype=torch.bool, device=device)

                # masked_scores: set masked positions to -inf
                masked_scores = scores.masked_fill(mask, float("-inf"))  # (B, Hq_g, Bq, Bk)

                # local_max per row over k (keep shape (B,Hq_g,Bq))
                local_max = torch.amax(masked_scores, dim=-1)  # (B, Hq_g, Bq)

                # For rows where all positions were masked in this (q_block,k_block) slice,
                # local_max will be -inf; we must handle merging carefully.
                # Compute exp(scores - local_max) but avoid NaN when local_max == -inf by marking invalid rows.
                valid_rows = ~torch.isinf(local_max)  # (B, Hq_g, Bq) boolean

                if valid_rows.any():
                    # For valid rows compute exp and weighted_v
                    # exp_scores shape (B, Hq_g, Bq, Bk)
                    exp_scores = torch.exp(masked_scores - local_max.unsqueeze(-1))
                    sum_exp = exp_scores.sum(dim=-1)  # (B, Hq_g, Bq)
                    # weighted_v_chunk: (B, Hq_g, Bq, D) = einsum over k
                    weighted_v_chunk = torch.einsum("b h q k, b k d -> b h q d", exp_scores, v_block)

                    # Merge streaming softmax results:
                    # If m is -inf (first chunk for that row), initialize. Else merge by log-sum-exp rules.
                    first_chunk_mask = torch.isinf(m)  # (B, Hq_g, Bq) boolean

                    # Where first_chunk_mask is True, initialize m,s,wv
                    init_idx = first_chunk_mask & valid_rows
                    if init_idx.any():
                        # set by indexing
                        m[init_idx] = local_max[init_idx]
                        s[init_idx] = sum_exp[init_idx]
                        wv[init_idx] = weighted_v_chunk[init_idx]

                    # For rows that were already initialized and are valid, merge
                    merge_idx = (~first_chunk_mask) & valid_rows
                    if merge_idx.any():
                        # extract slices
                        m_old = m[merge_idx]                      # (N,)
                        s_old = s[merge_idx]                      # (N,)
                        wv_old = wv[merge_idx]                    # (N, D)
                        lm_new = local_max[merge_idx]             # (N,)
                        se_new = sum_exp[merge_idx]               # (N,)
                        wv_new = weighted_v_chunk[merge_idx]      # (N, D)

                        new_m = torch.maximum(m_old, lm_new)
                        alpha = torch.exp(m_old - new_m)
                        beta = torch.exp(lm_new - new_m)

                        s[merge_idx] = s_old * alpha + se_new * beta
                        # broadcasting alpha,beta to match wv shapes
                        wv[merge_idx] = wv_old * alpha.unsqueeze(-1) + wv_new * beta.unsqueeze(-1)
                        m[merge_idx] = new_m

                    # free temporaries (helpful when memory is tight)
                    del exp_scores, sum_exp, weighted_v_chunk

                # else: if no valid rows in this slice, nothing to merge

                # free per-block tensors we no longer need
                del scores, masked_scores, local_max, mask, k_block, v_block, k_pos_block

            # after all k-blocks: finalize output for this q_block
            # denom (B, Hq_g, Bq, 1)
            denom = s.unsqueeze(-1) + eps
            out_block = (wv / denom).to(dtype)  # cast back to original dtype

            # write into out: out[:, q_head_idxs, q_start:q_end, :]
            # out_block shape (B, Hq_g, Bq, D)
            out[:, q_head_idxs, q_start:q_end, :] = out_block

            # free accumulators for this q_block
            del m, s, wv, out_block, denom, q_block

    return out



#=== Try torch.compile later reduce overhead ===
"""
# Option A: balanced / safe
compiled_attn = torch.compile(
    online_chunked_grouped_attention_rope,
    backend="inductor",      # default; uses triton + generated kernels on GPU
    mode="reduce-overhead"   # good for inference: lower overhead, uses CUDA graphs
)

# Option B: maximum steady-state speed (long compile time; autotunes GEMMs)
compiled_attn_max = torch.compile(
    online_chunked_grouped_attention_rope,
    backend="inductor",
    mode="max-autotune"      # autotunes GEMM backends; long first-run compile
)

# Call it exactly like the original call
out = compiled_attn(q, k, v, position_ids=pos, causal=True, q_block_size=64, k_block_size=512)
"""

if __name__=="__main__":
    # -------------------------
    # Test with your sample
    # -------------------------
    B, Hq, Hkv, L, D = 2, 16, 4, 256, 64
    q = torch.randn(B, Hq, L, D, device="cuda")
    k = torch.randn(B, Hkv, L, D, device="cuda")
    v = torch.randn(B, Hkv, L, D, device="cuda")

    out = online_chunked_grouped_attention_rope(q, k, v, q_block_size=64, k_block_size=128)
    print(out.shape)  # should be (2,16,256,64)
