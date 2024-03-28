import functools
import pdb

from torch import nn, Tensor
import torch.nn.functional as F
from transformers import PreTrainedModel

from .config import DemoModelConfig

""" debug """


def debug_func(func):
    """Decorator to debug a function when error is encountered."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            print(
                "Error encountered! Starting debug session from the beginning of the function."
            )
            pdb.runcall(func, *args, **kwargs)

    return wrapper


import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

try:
    import flash_attn

    flash_attn_enabled = True
except ImportError:
    flash_attn_enabled = False

cuda_available = torch.cuda.is_available()
if not flash_attn_enabled or not cuda_available:
    print(
        "WARNING: flash_attn not found or CUDA not available, will produce incorrect results."
    )


def find_nearest_128(x: int) -> int:
    return 128 * math.ceil(x / 128)


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    device: torch.device | None = None,
    base: int = 10000,
    condense_ratio: int = 1,
) -> tuple[Tensor, Tensor]:
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

    return torch.cos(idx_theta), torch.sin(idx_theta)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    head_dim = x.size(-1)
    x1 = x[..., : head_dim // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_dim // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        norm_x = torch.mean(x * x, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return (self.weight * x_normed).to(dtype=dtype)


class Attention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        rope_n_elem: int,
        sliding_window: int,
    ):
        super().__init__()
        self.rope_n_elem = rope_n_elem
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.sliding_window = sliding_window

        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = RMSNorm(hidden_dim)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = self.norm.forward(x)

        q, k, v = self.qkv_proj.forward(x).chunk(3, dim=-1)  # (b, s, d)
        q, k, v = [
            rearrange(t, "b s (nh hd) -> b nh s hd", hd=self.head_dim)
            for t in (q, k, v)
        ]
        q_roped, k_roped = [
            apply_rope(t[..., : self.rope_n_elem], cos, sin) for t in (q, k)
        ]
        q, k = [
            torch.cat((roped, t[..., self.rope_n_elem :]), dim=-1)
            for roped, t in zip((q_roped, k_roped), (q, k))
        ]
        if flash_attn_enabled and cuda_available:
            q, k, v = [rearrange(t, "b nh s hd -> b s nh hd") for t in (q, k, v)]
            x = flash_attn.flash_attn_func(
                q, k, v, causal=True, window_size=(self.sliding_window, 0)
            )

        # x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = rearrange(x, "bs s nh hd -> bs s (nh hd)")
        return self.o_proj.forward(x)


class Mlp(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.fc3 = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.norm = RMSNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm.forward(x)
        x = F.silu(self.fc1.forward(x)) * self.fc2.forward(x)
        return self.fc3.forward(x)


# class Patch(nn.Module):
#     """Resamples and rewidths the signal with a linear proj."""

#     def __init__(
#         self,
#         factor: int,
#         in_dim: int,
#         out_dim: int,
#         resample: str = "down",  # down, up
#         bias: bool = True,
#     ):
#         super().__init__()
#         self.factor = factor
#         self.resample = resample
#         self.in_dim = in_dim
#         self.out_dim = out_dim

#         assert out_dim % factor == 0, "Output dimension must be divisible by factor."

#         if factor == 1:
#             self.proj = nn.Identity()
#         else:
#             linear = nn.Linear(
#                 in_dim,
#                 (out_dim // factor if resample == "down" else out_dim * factor),
#                 bias=bias,
#             )
#             rearrange_op = (
#                 Rearrange(f"b (s f) d -> b s (f d)", f=factor)
#                 if resample == "down"
#                 else Rearrange(f"b s (f d) -> b (s f) d", f=factor)
#             )
#             self.proj = nn.Sequential(linear, rearrange_op)

#     def __repr__(self):
#         return f"Patch(factor={self.factor}, resample={self.resample}, in_dim={self.in_dim}, out_dim={self.out_dim})"

#     def forward(self, x: Tensor) -> Tensor:
#         return self.proj.forward(x)


class Patch(nn.Module):
    """Resamples using averaging or duplication and rewidths the signal with a linear projection."""

    def __init__(
        self,
        factor: int,
        in_dim: int,
        out_dim: int,
        resample: str,  # "down" or "up"
        bias: bool = True,
    ):
        super().__init__()
        self.factor = factor
        self.resample = resample
        self.in_dim = in_dim
        self.out_dim = out_dim

        assert out_dim % factor == 0, "Output dimension must be divisible by factor."

        self.proj = (
            nn.Linear(
                in_dim,
                out_dim,
                bias=bias,
            )
            if factor != 1
            else nn.Identity()
        )

    def __repr__(self):
        return f"PatchWithResampling(factor={self.factor}, resample={self.resample}, in_dim={self.in_dim}, out_dim={self.out_dim})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.resample == "down" and self.factor > 1:  # TODO
            x = torch.roll(x, shifts=self.factor - 1, dims=1)
            x[:, : self.factor - 1] = 0
            # import pdb; pdb.set_trace()

        if self.resample == "down" and self.factor > 1:
            # Averaging for downsampling
            x = F.avg_pool1d(x.permute(0, 2, 1), self.factor).permute(0, 2, 1)
        elif self.resample == "up" and self.factor > 1:
            # Duplication for upsampling
            x = torch.repeat_interleave(x, self.factor, dim=1)

        # Apply linear projection
        return self.proj.forward(x)


class Block(nn.Module):
    """Transformer attention and mlp block"""

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        head_dim: int,
        rope_n_elem: int,
        sliding_window: int = -1,
    ):
        super().__init__()

        num_heads = hidden_dim // head_dim

        self.attn = Attention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            rope_n_elem=rope_n_elem,
            sliding_window=sliding_window,
        )
        self.mlp = Mlp(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
        )

        self.use_checkpointing = False

    def checkpoint(self, function, *args, **kwargs):
        if self.use_checkpointing and self.training:
            kwargs.setdefault("use_reentrant", True)
            return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)
        else:
            return function(*args, **kwargs)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.checkpoint(self.attn, x, cos, sin)
        return x + self.checkpoint(self.mlp, x)


class Hourglass(nn.Module):
    """Hourglass transformer backbone"""

    def __init__(
        self,
        factors: list[int] = [2, 4, 8, 1],  # [2, 2, 1],
        hidden_dims: list[int] = [64, 128, 256, 512],  # [64, 128, 256],
        blocks: list[int] = [4, 2, 2, 2],  # [4, 4, 4],
        sliding_windows: list[int] = [-1, -1, -1, -1],
        head_dim: int = 32,
        mlp_factor: float = 8 / 3,
        block_size: int = 768,
        norm_cond_dim: int = 512,
        rope_base: int = 1000000,
        rope_rotary_percentage: float = 1.0,
        rope_condense_ratio: int = 4,
        init_method: str = "gaussian",
        patch_bias: bool = True,
    ):
        super().__init__()

        # Ensure hidden dimensions are divisible by head dimension
        assert all(
            d % head_dim == 0 for d in hidden_dims
        ), "Hidden dimensions must be divisible by head dimension."
        assert factors[-1] == 1, "Last factor must be 1."
        assert factors[0] != 1 or len(factors) > 1, "First factor must not be 1."

        # Configuration for rotary position embeddings
        self.rope_n_elems = int(rope_rotary_percentage * head_dim)
        self.rope_condense_ratio = rope_condense_ratio
        self.rope_base = rope_base
        self.block_size = block_size
        self.max_seq_length = block_size  # Maximum sequence length for attention

        # Initialize blocks and patches for encoder, bottleneck, and decoder
        self.blocks = nn.ModuleList()
        self.patches = nn.ModuleList()

        # Prepare configurations for symmetric encoder-decoder architecture
        full_factors = factors + factors[::-1][1:]
        full_hidden_dims = hidden_dims + hidden_dims[::-1]
        full_sliding_windows = sliding_windows + sliding_windows[::-1][1:]
        full_blocks = blocks + blocks[::-1][1:]

        # Find the index where decoder starts
        self.decoder_start_idx = len(full_factors) - full_factors[::-1].index(1, 1)

        # Create blocks and patches for each stage
        for stage in range(len(full_factors)):
            in_dim, out_dim = full_hidden_dims[stage], full_hidden_dims[stage + 1]
            factor = full_factors[stage]
            sliding_window = full_sliding_windows[stage]
            intermediate_dim = find_nearest_128(int(in_dim * mlp_factor))

            # Initialize Patch
            resample = "down" if stage < self.decoder_start_idx else "up"
            self.patches.append(Patch(factor, in_dim, out_dim, resample, patch_bias))

            # Adjust in_dim for decoder blocks where signal flows in reverse
            if stage >= self.decoder_start_idx:
                in_dim, out_dim = out_dim, in_dim

            for _ in range(full_blocks[stage]):
                # Initialize Blocks per stage
                self.blocks.append(
                    Block(
                        in_dim,
                        intermediate_dim,
                        head_dim,
                        self.rope_n_elems,
                        sliding_window,
                    )
                )
        # Final normalization layer
        self.norm = RMSNorm(full_hidden_dims[-1])

        # misc attributes
        self.hidden_dim = hidden_dims[0]
        self.norm_cond_dim = norm_cond_dim
        self.cumsum_blocks = torch.cat(
            (torch.tensor([0]), torch.cumsum(torch.tensor(full_blocks), 0))
        )
        self.cumprod_factors = torch.cat(  # downsample factor per stage
            (
                f := torch.tensor(factors)
                .cumprod(0)
                .roll(1, 0)
                .index_fill(0, torch.tensor(0), torch.tensor(1)),
                f.flip(0)[1:],
            )
        )

        # Initialize weights
        self.initialize_weights(init_method)

    def initialize_weights(self, init_method: str = "gaussian"):
        def basic_init(module: nn.Module, depth: int = None):
            if isinstance(module, nn.Linear):
                if init_method == "xavier":
                    gain = 1
                    if depth is not None:
                        gain = gain / math.sqrt(2 * depth)
                    nn.init.xavier_uniform_(module.weight, gain=gain)

                elif init_method == "gaussian":
                    input_dim = module.weight.size(0)
                    std = 1 / math.sqrt(input_dim)
                    if depth is not None:
                        std = std / math.sqrt(2 * depth)
                    nn.init.trunc_normal_(module.weight, std=std, a=-3 * std, b=3 * std)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(basic_init)

        # zero init residual path in attn and mlp, o_proj and fc3
        for i, block in enumerate(self.blocks):
            nn.init.zeros_(block.attn.o_proj.weight)
            nn.init.zeros_(block.mlp.fc3.weight)

        # zero init all patches after middle idxs # to keep identity func within each hierarchy
        for i, patch in enumerate(self.patches):
            if i >= self.decoder_start_idx:
                nn.init.zeros_(patch.proj.weight)
                if patch.proj.bias is not None:
                    nn.init.zeros_(patch.proj.bias)

    def rope_cache(self, device: torch.device | None = None) -> tuple[Tensor, Tensor]:
        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.rope_n_elems,
            device=device,
            condense_ratio=self.rope_condense_ratio,
            base=self.rope_base,
        )

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int):
        """
        When doing inference, the sequences used might be shorter than the model's context length.
        This allows setting a smaller number to avoid allocating unused memory
        """
        if value > self.block_size:
            raise ValueError(
                f"Cannot attend to {value}, block size is only {self.block_size}"
            )
        self._max_seq_length = value
        if not hasattr(self, "cos"):
            # first call
            cos, sin = self.rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        # overrides
        elif self.cos.device.type == "meta":
            self.cos, self.sin = self.rope_cache()
        elif value != self.cos.size(0):
            self.cos, self.sin = self.rope_cache(device=self.cos.device)

    def forward(self, x: Tensor, norm_cond: Tensor) -> Tensor:
        if self.training:
            assert x.size(1) % max(self.cumprod_factors) == 0, (
                f"Sequence length {x.size(1)} must be divisible by "
                f"max cumprod factor {max(self.cumprod_factors)} during training."
            )

        if self.max_seq_length < x.size(1):
            raise ValueError(
                f"Cannot forward sequence of length {x.size(1)}, max seq length is only {self.max_seq_length}."
            )
        cos_full = self.cos[: x.size(1)]
        sin_full = self.sin[: x.size(1)]

        active_stages = x.size(1) % self.cumprod_factors == 0

        # TODO
        # active_stages = [False] * len(self.cumprod_factors)
        # active_stages[1:-1] = torch.tensor([False] * (len(active_stages) - 2))
        # print(active_stages)

        residuals = []
        for stage, patch in enumerate(self.patches):
            blocks: list[Block] = self.blocks[
                self.cumsum_blocks[stage] : self.cumsum_blocks[stage + 1]
            ]
            patch: Patch

            if not active_stages[stage]:
                continue

            if stage >= self.decoder_start_idx:  # decoders
                x = patch.forward(x) if active_stages[stage - 1] else 0
                # TODO
                # x = 0
                x = x + residuals.pop()

            cos = cos_full[: x.size(1)]
            sin = sin_full[: x.size(1)]
            for b in blocks:
                x = b.forward(x, norm_cond, cos, sin)

            if (
                stage < self.decoder_start_idx and patch.factor != 1
            ):  # encoders, no residuals or patching for bottleneck
                residuals.append(x)
                x = patch.forward(x) if active_stages[stage + 1] else x

        return self.norm.forward(x, norm_cond)
