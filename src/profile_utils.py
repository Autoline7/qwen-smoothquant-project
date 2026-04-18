"""
Profiling utilities for the project rubric: per-stage timing, FLOPs
counting, and memory measurement. Keeping them in one module so notebook
06 stays readable.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class StageTimings:
    """Container for per-stage wall-clock timings for one prompt."""
    tokenization_ms: float = 0.0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    detokenization_ms: float = 0.0
    num_prompt_tokens: int = 0
    num_generated_tokens: int = 0

    @property
    def total_ms(self) -> float:
        return (self.tokenization_ms + self.prefill_ms
                + self.decode_ms + self.detokenization_ms)

    @property
    def decode_tokens_per_sec(self) -> float:
        if self.decode_ms == 0 or self.num_generated_tokens == 0:
            return 0.0
        return self.num_generated_tokens / (self.decode_ms / 1000.0)


@contextmanager
def cuda_timer():
    """
    Context manager that yields a list with a single float — wall-clock
    ms for the block. Synchronizes CUDA before and after so the number
    reflects actual GPU work, not async launch overhead.
    """
    result = [0.0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield result
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result[0] = (time.perf_counter() - t0) * 1000.0


@torch.no_grad()
def profile_single_prompt(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    system_prompt: str = "You are a coding assistant. Return only Python code.",
) -> StageTimings:
    """
    Generate one completion and measure per-stage wall time.

    Breakdown:
        - tokenization: prompt string -> input_ids
        - prefill: forward pass over the whole prompt (populates KV cache)
        - decode: autoregressive generation of new tokens
        - detokenization: output_ids -> string

    Note: HuggingFace's model.generate() fuses prefill and decode. To
    separate them we do one manual forward pass for prefill, then call
    generate() for the decode loop with the cached state.
    """
    t = StageTimings()
    device = model.device

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Stage 1: tokenization (CPU)
    with cuda_timer() as ms:
        inputs = tokenizer([text], return_tensors="pt").to(device)
    t.tokenization_ms = ms[0]
    t.num_prompt_tokens = inputs.input_ids.shape[1]

    # Stage 2 + 3: we'll use the full generate() and then separately
    # time a manual prefill to tease them apart. This is the most
    # accurate way that also matches real generation performance.

    # First, just time a prefill-only forward
    with cuda_timer() as ms:
        _ = model(**inputs, use_cache=True)
    t.prefill_ms = ms[0]

    # Now time the full generate() and subtract prefill for decode
    with cuda_timer() as ms:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_generate_ms = ms[0]
    t.num_generated_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    # decode time = total_generate_time - (one prefill is done inside)
    # This approximation assumes prefill cost is roughly constant.
    t.decode_ms = max(0.0, full_generate_ms - t.prefill_ms)

    # Stage 4: detokenization (CPU)
    generated = outputs[0][inputs.input_ids.shape[1]:]
    with cuda_timer() as ms:
        _ = tokenizer.decode(generated, skip_special_tokens=True)
    t.detokenization_ms = ms[0]

    return t


def count_parameters(model) -> dict:
    """Return total and non-embedding parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    # Embedding is typically tied with lm_head in Qwen; count the unique set
    embed = 0
    for name, p in model.named_parameters():
        if "embed_tokens" in name:
            embed += p.numel()
    return {
        "total": total,
        "non_embedding": total - embed,
        "embedding": embed,
    }


def measure_peak_vram(func, *args, **kwargs):
    """Run `func(*args, **kwargs)` and return (result, peak_vram_gb)."""
    if not torch.cuda.is_available():
        return func(*args, **kwargs), 0.0
    torch.cuda.reset_peak_memory_stats()
    result = func(*args, **kwargs)
    peak_bytes = torch.cuda.max_memory_allocated()
    return result, peak_bytes / (1024 ** 3)


def estimate_flops_per_decode_token(num_params: int) -> float:
    """
    Rough FLOPs estimate for one autoregressive decode step.

    The standard approximation for transformer decode (ignoring KV cache
    reads) is 2 * num_params FLOPs per token: one multiply-add per
    weight per output token. For exact numbers including attention
    compute, use calflops on an actual forward pass.
    """
    return 2.0 * num_params


__all__ = [
    "StageTimings",
    "cuda_timer",
    "profile_single_prompt",
    "count_parameters",
    "measure_peak_vram",
    "estimate_flops_per_decode_token",
]
