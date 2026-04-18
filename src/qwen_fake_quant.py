"""
W8A8 fake-quantization for Qwen2.

"Fake" means we quantize values to INT8 representation and back but run
compute in FP16/BF16 — so the numerics match a true INT8 deployment but
there's no actual speedup. This is what we use to measure the ACCURACY
effect of SmoothQuant. For actual SPEEDUP we use llm-compressor + vLLM
(see notebooks/03_llmcompressor_export.ipynb), which produces a
checkpoint vLLM can run with real INT8 kernels.

Reference:
    Upstream: https://github.com/mit-han-lab/smoothquant/blob/main/smoothquant/fake_quant.py
    We follow the same per-token activation quantization + per-channel
    weight quantization scheme described in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def quantize_weight_per_channel_absmax(w: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """
    Symmetric per-output-channel absmax quantization of a Linear's weight.

    Shape convention: nn.Linear stores weight as [out_features, in_features].
    We compute one scale per output channel (row) because the downstream
    INT8 GEMM accumulates along the input (column) dimension.
    """
    # scale per row: [out_features, 1]
    q_max = 2 ** (n_bits - 1) - 1     # e.g. 127 for INT8
    scales = w.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5) / q_max
    w_int = torch.round(w / scales).clamp(-q_max - 1, q_max)
    # Dequantize back to original dtype (simulated INT8)
    return (w_int * scales).to(w.dtype)


@torch.no_grad()
def quantize_activation_per_token_absmax(x: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """
    Symmetric per-token absmax quantization of the Linear input.

    Shape convention: x is [..., in_features]. We compute one scale per
    token (i.e. over the last dim), keeping other dims. Per-token (rather
    than per-tensor) activation quant is critical for accuracy because
    activation distributions change rapidly between tokens.
    """
    q_max = 2 ** (n_bits - 1) - 1
    scales = x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5) / q_max
    x_int = torch.round(x / scales).clamp(-q_max - 1, q_max)
    return (x_int * scales).to(x.dtype)


class W8A8Linear(nn.Module):
    """
    Drop-in replacement for nn.Linear that simulates W8A8 quantization.

    Forward pass:
        1. Quantize input per-token to INT8 (simulated — kept in FP16)
        2. Use pre-quantized weight (stored at construction time)
        3. Compute matmul in high precision (FP16/BF16)
        4. Add bias unchanged (if any)

    Preserving the original Linear's dtype/device on construction.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_quant: str = "per_channel",
        act_quant: str = "per_token",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_quant_mode = weight_quant
        self.act_quant_mode = act_quant

        # We store the already-quantized weight — it's computed once at
        # module-replacement time and never updated (PTQ, no training).
        self.register_buffer(
            "weight", torch.empty(out_features, in_features, dtype=torch.float16)
        )
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_quant_mode == "per_token":
            x_q = quantize_activation_per_token_absmax(x, n_bits=8)
        elif self.act_quant_mode == "none":
            x_q = x
        else:
            raise ValueError(f"Unknown act_quant_mode: {self.act_quant_mode}")
        return F.linear(x_q, self.weight, self.bias)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        weight_quant: str = "per_channel",
        act_quant: str = "per_token",
    ) -> "W8A8Linear":
        """Construct a W8A8Linear by quantizing an existing nn.Linear's weight."""
        new = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            weight_quant=weight_quant,
            act_quant=act_quant,
        )
        if weight_quant == "per_channel":
            w_q = quantize_weight_per_channel_absmax(linear.weight.data, n_bits=8)
        elif weight_quant == "none":
            w_q = linear.weight.data.clone()
        else:
            raise ValueError(f"Unknown weight_quant_mode: {weight_quant}")
        new.weight = w_q.to(linear.weight.dtype).to(linear.weight.device)
        if linear.bias is not None:
            new.bias = linear.bias.data.to(linear.weight.dtype).to(linear.weight.device)
        return new

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bias={self.bias is not None}, W8A8 fake-quant")


@torch.no_grad()
def quantize_qwen2_w8a8(model: nn.Module) -> nn.Module:
    """
    Replace every nn.Linear inside Qwen2 decoder layers with W8A8Linear.

    We skip:
        - The embedding (not a Linear)
        - The final lm_head (Linear, but standard practice to leave it
          in FP16 since it's outside the decoder stack and small relative
          to total compute)

    Mutates `model` in place and also returns it for chaining.
    """
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    replaced = 0
    for layer in model.modules():
        if not isinstance(layer, Qwen2DecoderLayer):
            continue
        # Attention projections
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            orig = getattr(layer.self_attn, attr)
            setattr(layer.self_attn, attr, W8A8Linear.from_linear(orig))
            replaced += 1
        # MLP projections
        for attr in ("gate_proj", "up_proj", "down_proj"):
            orig = getattr(layer.mlp, attr)
            setattr(layer.mlp, attr, W8A8Linear.from_linear(orig))
            replaced += 1

    print(f"Replaced {replaced} Linears with W8A8Linear (fake-quant)")
    return model


__all__ = [
    "quantize_weight_per_channel_absmax",
    "quantize_activation_per_token_absmax",
    "W8A8Linear",
    "quantize_qwen2_w8a8",
]
