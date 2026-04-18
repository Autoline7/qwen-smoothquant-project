"""
SmoothQuant smoothing adapted for the Qwen2 architecture.

The MIT-HAN-Lab `smoothquant` repo officially supports OPT, Llama, Mistral,
Mixtral, Falcon, and BLOOM. Qwen2 is structurally very similar to Llama
(RMSNorm + SwiGLU + GQA) but has two key differences that require adaptation:

    1. Qwen2 attention has bias on q_proj, k_proj, v_proj (Llama does not).
       This does NOT affect the smoothing math — bias is added post-matmul
       and is unchanged by the smoothing transformation:
           y = (W * s)(x / s) + b = Wx + b
       But it means we cannot just `isinstance(ln, LlamaRMSNorm)` check the
       way the upstream repo does.

    2. The upstream `smooth_ln_fcs` asserts `isinstance(ln, (LlamaRMSNorm,
       MistralRMSNorm, MixtralRMSNorm))` — we need to add Qwen2RMSNorm.

Reference:
    Xiao et al. SmoothQuant. ICML 2023. https://arxiv.org/abs/2211.10438
    Upstream: https://github.com/mit-han-lab/smoothquant/blob/main/smoothquant/smooth.py
"""

import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2RMSNorm,
)


@torch.no_grad()
def smooth_ln_fcs_qwen2(
    ln: Qwen2RMSNorm,
    fcs: list,
    act_scales: torch.Tensor,
    alpha: float = 0.5,
) -> None:
    """
    Apply SmoothQuant's mathematically equivalent scaling transformation:

        ln.weight -> ln.weight / s
        fc.weight -> fc.weight * s    for each fc in fcs

    where s is the per-channel smoothing factor:

        s_j = (|act|_j)^alpha / (|weight|_j)^(1 - alpha)

    This migrates quantization difficulty from activations to weights.
    Alpha controls the tradeoff: larger alpha -> more difficulty moved to
    weights (easier activations, harder weights). Paper recommends 0.5 for
    Llama-like models, 0.85 for OPT. Alpha must be in [0, 1].

    Args:
        ln: a Qwen2RMSNorm whose output feeds into all Linears in `fcs`
        fcs: list of nn.Linear modules sharing the same input (e.g., q/k/v,
             or gate/up)
        act_scales: 1-D tensor of per-channel absolute-max activation
                    values at the Linear input, shape [in_features]
        alpha: smoothing strength, in [0, 1]

    Mutates `ln` and `fcs` in place.
    """
    if not isinstance(ln, Qwen2RMSNorm):
        raise TypeError(
            f"Expected Qwen2RMSNorm, got {type(ln).__name__}. "
            "If you're adapting to a different Qwen variant (e.g. Qwen2-VL), "
            "import the appropriate RMSNorm class."
        )

    for fc in fcs:
        if not isinstance(fc, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(fc).__name__}")
        if ln.weight.numel() != fc.in_features:
            raise ValueError(
                f"Size mismatch: ln has {ln.weight.numel()} features, "
                f"fc has {fc.in_features} input features. They must match "
                "because the smoothing scale is applied elementwise to both."
            )
        if fc.in_features != act_scales.numel():
            raise ValueError(
                f"act_scales has {act_scales.numel()} values but fc expects "
                f"{fc.in_features}. Did you use scales from a different layer?"
            )

    device = fcs[0].weight.device
    dtype = fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)

    # Per-channel max absolute weight value across all fcs in this group.
    # For QKV (which share an input), we take the max over all three to ensure
    # the smoothing factor balances weight magnitudes across the group.
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    # The SmoothQuant scale: s_j = a_j^alpha / w_j^(1-alpha)
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device=device, dtype=dtype)
    )

    # Apply the equivalent transformation.
    # RMSNorm absorbs 1/s by scaling its per-channel weight.
    # Each following Linear absorbs s by scaling each input column by s[j].
    ln.weight.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_qwen2(
    model: nn.Module,
    act_scales: dict,
    alpha: float = 0.5,
) -> None:
    """
    Walk a Qwen2-based model and apply SmoothQuant at every place where an
    RMSNorm feeds into a Linear layer.

    For each Qwen2DecoderLayer we smooth at two points:

        1. `input_layernorm` -> {q_proj, k_proj, v_proj}
           (all three share the same input via GQA; we group them so the
           smoothing factor is consistent across Q/K/V)

        2. `post_attention_layernorm` -> {gate_proj, up_proj}
           (both gate and up receive the same RMSNorm output; we group them.
           down_proj has a separate input and is not smoothed here — it
           receives the output of SiLU(gate) * up which is not immediately
           downstream of an RMSNorm. Following the paper, we only smooth at
           RMSNorm-Linear interfaces.)

    Args:
        model: a Qwen2ForCausalLM (or any module tree containing
               Qwen2DecoderLayers)
        act_scales: dict mapping module path (str) -> 1-D tensor of
                    per-channel activation absmax. Produced by calibrate.py.
                    Keys are expected to match `model.named_modules()` paths,
                    e.g. `model.layers.0.self_attn.q_proj`.
        alpha: SmoothQuant alpha in [0, 1]. Paper recommends 0.5 for
               Llama-family, 0.85 for OPT. Qwen2 is Llama-family
               structurally but we sweep alpha experimentally.

    Mutates `model` in place.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    smoothed_count = 0
    skipped = []

    for name, module in model.named_modules():
        if not isinstance(module, Qwen2DecoderLayer):
            continue

        # 1. Attention block: input_layernorm -> {q,k,v}_proj
        attn_ln = module.input_layernorm
        q_proj = module.self_attn.q_proj
        k_proj = module.self_attn.k_proj
        v_proj = module.self_attn.v_proj

        q_name = f"{name}.self_attn.q_proj"
        if q_name in act_scales:
            smooth_ln_fcs_qwen2(
                attn_ln,
                [q_proj, k_proj, v_proj],
                act_scales[q_name],
                alpha=alpha,
            )
            smoothed_count += 1
        else:
            skipped.append(q_name)

        # 2. MLP block: post_attention_layernorm -> {gate,up}_proj
        mlp_ln = module.post_attention_layernorm
        gate_proj = module.mlp.gate_proj
        up_proj = module.mlp.up_proj

        gate_name = f"{name}.mlp.gate_proj"
        if gate_name in act_scales:
            smooth_ln_fcs_qwen2(
                mlp_ln,
                [gate_proj, up_proj],
                act_scales[gate_name],
                alpha=alpha,
            )
            smoothed_count += 1
        else:
            skipped.append(gate_name)

    print(f"SmoothQuant: smoothed {smoothed_count} RMSNorm-Linear groups (alpha={alpha})")
    if skipped:
        print(f"  Warning: missing act_scales for {len(skipped)} modules "
              f"(first few: {skipped[:3]})")


__all__ = ["smooth_ln_fcs_qwen2", "smooth_qwen2"]
