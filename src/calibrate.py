"""
Activation scale calibration for SmoothQuant.

For each Linear layer's input, we need a per-channel measure of how
outlier-heavy the activations are. SmoothQuant uses the per-channel
absolute maximum (infinity norm) over a calibration dataset. These scales
are then used to compute the smoothing factors in `qwen_smooth.smooth_qwen2`.

The calibration process is architecture-agnostic: we register forward
pre-hooks on every nn.Linear and track the running max of |input|.

Reference:
    Xiao et al. SmoothQuant. ICML 2023. https://arxiv.org/abs/2211.10438
    Upstream: https://github.com/mit-han-lab/smoothquant/blob/main/smoothquant/calibration.py
"""

import functools
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm


@torch.no_grad()
def get_act_scales(
    model: nn.Module,
    tokenizer,
    dataset_name: str = "mit-han-lab/pile-val-backup",
    num_samples: int = 512,
    seq_len: int = 512,
    device: str = "cuda",
) -> dict:
    """
    Collect per-channel absolute-max activations at every Linear input.

    Args:
        model: a HuggingFace causal LM, already on `device` in eval mode.
               We don't move it here to avoid surprising the caller about
               memory.
        tokenizer: matching tokenizer
        dataset_name: HF dataset with a 'text' field. Default is the same
                      Pile validation subset the SmoothQuant paper used.
                      For code models you can also try
                      'bigcode/the-stack-smol' which is more in-distribution.
        num_samples: calibration samples (paper uses 512)
        seq_len: truncate each sample to this length (paper uses 512)
        device: where to run forward passes. Must match model's device.

    Returns:
        dict: {module_name: torch.Tensor of shape [in_features]} where
              module_name matches `model.named_modules()` paths. All values
              are on CPU for easy saving.
    """
    model.eval()
    act_scales: dict[str, torch.Tensor] = {}

    def stat_tensor(name: str, tensor: torch.Tensor) -> None:
        """Update the running max of |tensor| along the channel dimension."""
        hidden_dim = tensor.shape[-1]
        # Flatten everything except the last (channel) dim, take absmax
        flat = tensor.detach().abs().view(-1, hidden_dim)
        comming_max = flat.max(dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(module, inputs, name):
        # pre-hook: inputs is a tuple, inputs[0] is the tensor we care about
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        stat_tensor(name, x)

    # Register a forward pre-hook on every Linear
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook = module.register_forward_pre_hook(
                functools.partial(stat_input_hook, name=name)
            )
            hooks.append(hook)

    try:
        # Load calibration data
        dataset = load_dataset(dataset_name, split="validation", streaming=False)
        dataset = dataset.shuffle(seed=42).select(range(num_samples))

        for sample in tqdm(dataset, desc="Calibrating", total=num_samples):
            text = sample["text"]
            if not text or not text.strip():
                continue
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=seq_len,
                truncation=True,
            ).to(device)
            # We only care about the forward pass populating the hooks;
            # logits go to /dev/null.
            _ = model(**inputs)
    finally:
        for h in hooks:
            h.remove()

    return act_scales


def save_act_scales(scales: dict, path: str) -> None:
    """Save activation scales dict to disk. All tensors on CPU."""
    cpu_scales = {k: v.cpu() for k, v in scales.items()}
    torch.save(cpu_scales, path)
    print(f"Saved {len(cpu_scales)} act_scales to {path}")


def load_act_scales(path: str) -> dict:
    """Load activation scales from disk."""
    scales = torch.load(path, map_location="cpu", weights_only=True)
    print(f"Loaded {len(scales)} act_scales from {path}")
    return scales


__all__ = ["get_act_scales", "save_act_scales", "load_act_scales"]
