import  torch

def apply_2d_rotary_pos_emb(q, k, H, W):
    """
    q, k: [B, N, C] (N = H*W)
    H, W: spatial dimensions of the feature map

    Returns rotated q, k with 2D rotary positional embeddings
    """
    B, N, C = q.shape
    assert N == H * W, "Mismatch between number of tokens and H*W"

    # Compute 2D sine/cosine frequencies
    dim = C // 2
    freqs = torch.arange(dim, device=q.device, dtype=torch.float32)
    freqs = 1.0 / (10000 ** (freqs / dim))  # [dim]

    # Get x/y coordinates for each token
    ys, xs = torch.meshgrid(torch.arange(H, device=q.device), torch.arange(W, device=q.device), indexing='ij')
    xs = xs.flatten().float()  # [N]
    ys = ys.flatten().float()  # [N]

    # Compute rotation angles
    theta_x = xs[:, None] * freqs[None, :]  # [N, dim]
    theta_y = ys[:, None] * freqs[None, :]  # [N, dim]

    # Concatenate for full C dims
    sin = torch.cat([theta_x.sin(), theta_y.sin()], dim=1)  # [N, C]
    cos = torch.cat([theta_x.cos(), theta_y.cos()], dim=1)

    # Apply rotation: (q_cos, q_sin)
    q_rot = q * cos[None, :, :] + rotate_half(q) * sin[None, :, :]
    k_rot = k * cos[None, :, :] + rotate_half(k) * sin[None, :, :]
    return q_rot, k_rot

def rotate_half(x):
    """Helper: rotate last dim by half"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)
