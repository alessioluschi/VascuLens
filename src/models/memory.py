"""VRAM management utilities for the Ulcer Classification Pipeline.

All training and explainability code must use gpu_memory_guard to ensure
clean VRAM state between pipeline phases.
"""

import gc
import logging
from contextlib import contextmanager
from typing import Generator

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_gpu_memory_status() -> dict:
    """Return current VRAM statistics in megabytes.

    Returns:
        Dictionary with keys: total, allocated, reserved, free (all in MB).
    """
    if not torch.cuda.is_available():
        return {"total": 0.0, "allocated": 0.0, "reserved": 0.0, "free": 0.0}

    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    allocated_bytes = torch.cuda.memory_allocated(0)
    reserved_bytes = torch.cuda.memory_reserved(0)
    mb = 1024 ** 2

    return {
        "total": total_bytes / mb,
        "allocated": allocated_bytes / mb,
        "reserved": reserved_bytes / mb,
        "free": free_bytes / mb,
    }


def log_gpu_memory(tag: str) -> None:
    """Log current VRAM usage with a descriptive tag.

    Args:
        tag: Short label for the log message (e.g., "BEFORE Training BiomedCLIP").
    """
    s = get_gpu_memory_status()
    logger.info(
        f"[VRAM | {tag}] "
        f"Total: {s['total']:.0f} MB | "
        f"Allocated: {s['allocated']:.0f} MB | "
        f"Reserved: {s['reserved']:.0f} MB | "
        f"Free: {s['free']:.0f} MB"
    )


def release_model(model: nn.Module) -> None:
    """Move model to CPU, delete the reference, empty CUDA cache, and run GC.

    Args:
        model: The PyTorch module to release from VRAM.
    """
    try:
        model.cpu()
    except Exception:
        pass
    del model
    safe_cuda_empty_cache()
    logger.debug("Model released from VRAM.")


def safe_cuda_empty_cache() -> None:
    """Call torch.cuda.empty_cache() and gc.collect() with a debug log."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.debug("CUDA cache emptied and garbage collected.")


@contextmanager
def gpu_memory_guard(tag: str) -> Generator:
    """Context manager that logs VRAM before/after the block and cleans up on exit.

    Usage::

        with gpu_memory_guard("Training BiomedCLIP"):
            train(model, ...)
        # VRAM is cleaned up automatically on exit

    Args:
        tag: Descriptive label for the protected block.

    Yields:
        None
    """
    log_gpu_memory(f"BEFORE {tag}")
    try:
        yield
    finally:
        safe_cuda_empty_cache()
        log_gpu_memory(f"AFTER  {tag}")


def estimate_batch_size(
    model: nn.Module,
    input_shape: tuple,
    target_vram_usage: float = 0.8,
) -> int:
    """Estimate the maximum batch size that fits within the target VRAM fraction.

    Runs a single forward+backward pass with batch_size=2 and linearly
    extrapolates to the available VRAM budget.

    Args:
        model: The PyTorch model (must already be on the target device).
        input_shape: Shape of a single input sample, e.g. (3, 224, 224).
        target_vram_usage: Fraction of free VRAM to target (default 0.8).

    Returns:
        Estimated safe batch size (minimum 1).
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available — returning default batch size 8.")
        return 8

    device = next(model.parameters()).device

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    baseline_mem = torch.cuda.memory_allocated(device)

    probe_batch = 2
    dummy = torch.randn(probe_batch, *input_shape, device=device)

    try:
        with torch.cuda.amp.autocast():
            out = model(dummy)
            out = out[0] if isinstance(out, (tuple, list)) else out
            loss = out.sum()
        loss.backward()

        peak_mem = torch.cuda.max_memory_allocated(device)
        mem_per_sample = max(1, (peak_mem - baseline_mem) / probe_batch)

        free_mem = torch.cuda.mem_get_info(0)[0]
        available_mem = free_mem * target_vram_usage
        estimated = max(1, int(available_mem / mem_per_sample))

        logger.info(
            f"Estimated max batch size: {estimated} "
            f"(mem/sample: {mem_per_sample / (1024**2):.1f} MB)"
        )
        return estimated

    except Exception as exc:
        logger.warning(f"Batch size estimation failed ({exc}). Returning default 8.")
        return 8

    finally:
        del dummy
        gc.collect()
        torch.cuda.empty_cache()
