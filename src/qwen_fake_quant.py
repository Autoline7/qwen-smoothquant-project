"""
W8A8 fake-quantization for Qwen2.

"Fake" means we quantize values to INT8 representation and back but run
compute in FP16/BF16 — so the numerics match a true INT8 deployment but
there's no actual speedup. This is what we use to measure the ACCURACY
effect of SmoothQuant. For actual SPEEDUP we use llm-compressor + vLLM
(see notebooks/03_llmcompressor_export.ipynb), which produces a
checkpoint vLLM can run with real INT8 kernels.

This file is a close port of the upstream MIT SmoothQuant fake_quant to
Qwen2. It supports the same configurations as the reference repo:

    weight_quant        : "per_channel"  or  "per_tensor"
    act_quant           : "per_token"    or  "per_tensor"
    quantize_bmm_input  : True  -> q/k/v outputs are also quantized, so
                                   the attention BMM runs with INT8
                                   activations on both sides (O3-style)

Both activation modes are DYNAMIC (scale recomputed at runtime from the
current input tensor), matching the upstream reference. The paper's O3
setting (per-tensor STATIC with calibrated scales) is not implemented
here — use the llm-compressor pipeline (notebook 03) for that.

Upstream reference:
    https://github.com/mit-han-lab/smoothquant/blob/main/smoothquant/fake_quant.py
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Core quant primitives — direct ports from the upstream fake_quant.py.
# All four are in-place on the input tensor.
# ----------------------------------------------------------------------------

@torch.no_grad()
def quantize_weight_per_channel_absmax(w: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """Symmetric per-output-channel absmax weight quant (in-place)."""
    # w: [out_features, in_features] — one scale per row
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """Symmetric per-tensor absmax weight quant (in-place)."""
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(t: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """Symmetric per-token absmax activation quant (dynamic)."""
    # t: [..., in_features] — one scale per token (row)
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """Symmetric per-tensor absmax activation quant (dynamic: runtime max)."""
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


# ----------------------------------------------------------------------------
# W8A8Linear — drop-in nn.Linear replacement that simulates W8A8 inference.
# ----------------------------------------------------------------------------

class W8A8Linear(nn.Module):
    """
    Simulated W8A8 linear layer.

    Weight is pre-quantized once at construction time (PTQ, no training).
    Activation is quantized at every forward call, dynamically, using
    whichever scheme was selected by `act_quant`. If `quantize_output`
    is True, the output of the linear is also quantized with the same
    scheme — this is how we simulate an INT8 BMM input for attention.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        act_quant: str = "per_token",
        quantize_output: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randn(out_features, in_features, dtype=torch.float16,
                        requires_grad=False),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros((1, out_features), dtype=torch.float16,
                            requires_grad=False),
            )
        else:
            self.register_buffer("bias", None)

        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=8)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=8)
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda y: y

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_x = self.act_quant(x)
        y = F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(
        module: nn.Linear,
        weight_quant: str = "per_channel",
        act_quant: str = "per_token",
        quantize_output: bool = False,
    ) -> "W8A8Linear":
        """Build a W8A8Linear by quantizing an existing nn.Linear's weight."""
        assert isinstance(module, nn.Linear)
        new_module = W8A8Linear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
        )
        if weight_quant == "per_channel":
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=8)
        elif weight_quant == "per_tensor":
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=8)
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self) -> str:
        return (f"W8A8Linear({self.in_features}, {self.out_features}, "
                f"bias={self.bias is not None}, "
                f"weight_quant={getattr(self, 'weight_quant_name', '?')}, "
                f"act_quant={self.act_quant_name}, "
                f"output_quant={self.output_quant_name})")


# ----------------------------------------------------------------------------
# Model-level replacement for Qwen2 — mirrors upstream's quantize_llama_like.
# ----------------------------------------------------------------------------

@torch.no_grad()
def quantize_qwen2_w8a8(
    model: nn.Module,
    weight_quant: str = "per_channel",
    act_quant: str = "per_token",
    quantize_bmm_input: bool = False,
) -> nn.Module:
    """
    Replace every nn.Linear inside Qwen2 decoder layers with W8A8Linear.

    Mirrors `quantize_llama_like` in the upstream repo, since Qwen2 has
    the same attention + MLP structure (q/k/v/o projections, gate/up/down
    MLP). Skips the embedding and lm_head, matching upstream convention.

    Args:
        model: a Qwen2 HF model (e.g. Qwen2.5-Coder-7B-Instruct)
        weight_quant: "per_channel" (default) or "per_tensor"
        act_quant: "per_token" (default) or "per_tensor"
        quantize_bmm_input: if True, also quantize the output of
            q/k/v_proj so the attention BMM gets INT8 activations on
            both sides (matches upstream's O3-style setting)

    Returns:
        the model, mutated in place
    """
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Attention,
        Qwen2MLP,
    )

    replaced = 0
    for m in model.modules():
        if isinstance(m, Qwen2MLP):
            m.gate_proj = W8A8Linear.from_float(
                m.gate_proj, weight_quant=weight_quant, act_quant=act_quant)
            m.up_proj = W8A8Linear.from_float(
                m.up_proj, weight_quant=weight_quant, act_quant=act_quant)
            m.down_proj = W8A8Linear.from_float(
                m.down_proj, weight_quant=weight_quant, act_quant=act_quant)
            replaced += 3
        elif isinstance(m, Qwen2Attention):
            # quantize_bmm_input=True => q/k/v outputs also quantized so
            # that the attention BMM sees INT8 activations on both sides.
            m.q_proj = W8A8Linear.from_float(
                m.q_proj, weight_quant=weight_quant, act_quant=act_quant,
                quantize_output=quantize_bmm_input)
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, weight_quant=weight_quant, act_quant=act_quant,
                quantize_output=quantize_bmm_input)
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, weight_quant=weight_quant, act_quant=act_quant,
                quantize_output=quantize_bmm_input)
            m.o_proj = W8A8Linear.from_float(
                m.o_proj, weight_quant=weight_quant, act_quant=act_quant)
            replaced += 4

    print(f"Replaced {replaced} Linears with W8A8Linear "
          f"(weight_quant={weight_quant}, act_quant={act_quant}, "
          f"quantize_bmm_input={quantize_bmm_input})")
    return model


__all__ = [
    "quantize_weight_per_channel_absmax",
    "quantize_weight_per_tensor_absmax",
    "quantize_activation_per_token_absmax",
    "quantize_activation_per_tensor_absmax",
    "W8A8Linear",
    "quantize_qwen2_w8a8",
]