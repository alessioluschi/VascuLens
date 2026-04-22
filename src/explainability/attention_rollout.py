"""Attention Rollout for ViT architectures (BiomedCLIP and UNI).

Root issue with naive output hooks: open_clip calls
    self.attn(q, k, v, need_weights=False)
so output[1] is always None — no weights captured.

Fix: hook on the attention module inputs and manually compute Q@K
attention weights, bypassing the need_weights flag entirely. Works
for both nn.MultiheadAttention (BiomedCLIP) and timm custom Attention (UNI).

Rollout note: discard_ratio is applied ONCE to the final CLS attention
vector, not per-layer.  Per-layer discarding compounds across deep ViTs
(UNI ViT-L = 24 layers) and zeros out the rollout signal, producing
all-blue maps.  Post-rollout discard keeps top (1-discard_ratio) patches.

Architecture detection order:
  1. model.transformer.resblocks   → open_clip ViT (BiomedCLIP)
  2. model.blocks                  → timm ViT (UNI)
  3. Recursive nn.MultiheadAttention search → fallback
"""

import logging
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image

from src.models.memory import gpu_memory_guard, release_model

logger = logging.getLogger(__name__)


# ===========================================================================
# Hook factories (module-level so they avoid closure/recursion issues)
# ===========================================================================


def _make_mha_weight_hook(mha: nn.MultiheadAttention, store: list):
    """Hook for nn.MultiheadAttention — computes per-head weights from Q,K inputs.

    Extracts Q and K projections from the hook's input tensors and
    computes scaled dot-product attention without calling the module again.
    Handles both batch_first=True and batch_first=False layouts.

    Args:
        mha: The nn.MultiheadAttention module.
        store: List to append attention tensors to.

    Returns:
        Hook function compatible with register_forward_hook.
    """
    def hook_fn(module: nn.Module, inputs: tuple, output) -> None:
        try:
            q_in, k_in = inputs[0], inputs[1]
            E = module.embed_dim
            H = module.num_heads
            head_dim = E // H

            W = module.in_proj_weight          # (3E, E)
            b = module.in_proj_bias            # (3E,) or None
            W_q, W_k = W[:E], W[E : 2 * E]
            b_q = b[:E] if b is not None else None
            b_k = b[E : 2 * E] if b is not None else None

            with torch.no_grad():
                Q = F.linear(q_in, W_q, b_q)
                K = F.linear(k_in, W_k, b_k)

                batch_first = getattr(module, "batch_first", False)
                if not batch_first:
                    # (seq, B, E) → (B, seq, E)
                    Q = Q.transpose(0, 1)
                    K = K.transpose(0, 1)

                B, seq_q, _ = Q.shape
                seq_k = K.shape[1]

                # (B, seq, E) → (B, H, seq, head_dim)
                Q = Q.reshape(B, seq_q, H, head_dim).transpose(1, 2)
                K = K.reshape(B, seq_k, H, head_dim).transpose(1, 2)

                scale = head_dim ** -0.5
                attn = (Q @ K.transpose(-2, -1)) * scale  # (B, H, Sq, Sk)
                attn = attn.softmax(dim=-1)
                store.append(attn.detach().cpu())

        except Exception as exc:
            logger.debug(f"MHA weight hook error: {exc}")

    return hook_fn


def _make_timm_weight_hook(attn_module: nn.Module, store: list):
    """Hook for timm custom Attention module — computes Q@K attention from input x.

    Calls attn_module.qkv(x) to split Q, K, V and computes scaled
    dot-product attention weights.

    Args:
        attn_module: The timm Attention module (has .qkv, .num_heads, .scale).
        store: List to append attention tensors to.

    Returns:
        Hook function compatible with register_forward_hook.
    """
    def hook_fn(module: nn.Module, inputs: tuple, output) -> None:
        try:
            x = inputs[0]                       # (B, N, C)
            B, N, C = x.shape
            H = attn_module.num_heads
            head_dim = C // H
            scale = getattr(attn_module, "scale", head_dim ** -0.5)

            with torch.no_grad():
                qkv = attn_module.qkv(x)        # (B, N, 3*C)
                qkv = qkv.reshape(B, N, 3, H, head_dim).permute(2, 0, 3, 1, 4)
                q, k, _ = qkv.unbind(0)         # each (B, H, N, head_dim)
                attn = (q @ k.transpose(-2, -1)) * scale
                attn = attn.softmax(dim=-1)     # (B, H, N, N)
                store.append(attn.detach().cpu())

        except Exception as exc:
            logger.debug(f"timm weight hook error: {exc}")

    return hook_fn


# ===========================================================================
# AttentionRollout
# ===========================================================================


class AttentionRollout:
    """Compute Attention Rollout for a ViT model.

    Registers per-input hooks on all transformer attention layers to capture
    per-head attention matrices, then performs the rollout computation.

    Args:
        model: ViT model (from open_clip or timm). Must be in eval mode.
        head_fusion: Strategy for fusing multi-head attention:
            ``"mean"``, ``"max"``, or ``"min"``.
        discard_ratio: Fraction of lowest attention values to zero out
            before rollout multiplication (e.g., 0.9 discards the lowest 90%).
    """

    def __init__(
        self,
        model: nn.Module,
        head_fusion: str = "mean",
        discard_ratio: float = 0.9,
    ) -> None:
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self._attention_store: List[torch.Tensor] = []
        self._hooks: List = []
        self._register_hooks()

    # ------------------------------------------------------------------ #
    # Hook registration                                                    #
    # ------------------------------------------------------------------ #

    def _register_hooks(self) -> None:
        """Register input-based attention hooks on all transformer blocks.

        Tries six detection strategies in order, covering all known
        open_clip and timm ViT layouts:

        1. ``model.transformer.resblocks``  — open_clip VisionTransformer
        2. ``model.blocks``                 — timm ViT (direct)
        3. ``model.trunk.blocks``           — open_clip TimmModel wrapper
        4. ``model.model.blocks``           — some HF hub wrappers
        5. Recursive block-list search      — any nested ModuleList of blocks
        6. Deep scan for attn modules       — last-resort named_modules walk
        """
        registered = 0

        # Collect candidate block lists from all known paths
        block_lists: list = []

        for path in [
            ["transformer", "resblocks"],   # open_clip VisionTransformer
            ["blocks"],                      # timm ViT
            ["trunk", "blocks"],             # open_clip TimmModel
            ["model", "blocks"],             # HF hub wrappers
            ["visual", "transformer", "resblocks"],  # full CLIP model passed by mistake
            ["visual", "trunk", "blocks"],
        ]:
            obj = self.model
            try:
                for attr in path:
                    obj = getattr(obj, attr)
                blocks = list(obj)
                if blocks:
                    block_lists.append(blocks)
                    logger.debug(
                        f"Found {len(blocks)} blocks via path: "
                        f"model.{'.'.join(path)}"
                    )
            except (AttributeError, TypeError):
                pass

        # Also search recursively: find any nn.ModuleList/Sequential whose
        # children all have an 'attn' attribute (transformer block pattern)
        if not block_lists:
            for _name, module in self.model.named_modules():
                if isinstance(module, (nn.ModuleList, nn.Sequential)):
                    children = list(module.children())
                    if len(children) >= 2 and all(
                        hasattr(c, "attn") for c in children
                    ):
                        block_lists.append(children)
                        logger.debug(
                            f"Found {len(children)} blocks via recursive "
                            f"ModuleList scan at '{_name}'."
                        )
                        break

        # Register hooks on the first valid block list found
        for blocks in block_lists:
            for block in blocks:
                h = self._hook_block(block)
                if h is not None:
                    self._hooks.append(h)
                    registered += 1
            if registered:
                break

        # Last resort: walk all named modules and hook any attention-like leaf
        if registered == 0:
            for _name, module in self.model.named_modules():
                if isinstance(module, nn.MultiheadAttention):
                    h = module.register_forward_hook(
                        _make_mha_weight_hook(module, self._attention_store)
                    )
                    self._hooks.append(h)
                    registered += 1
                elif (
                    hasattr(module, "qkv")
                    and hasattr(module, "num_heads")
                    and isinstance(getattr(module, "qkv", None), nn.Linear)
                ):
                    h = module.register_forward_hook(
                        _make_timm_weight_hook(module, self._attention_store)
                    )
                    self._hooks.append(h)
                    registered += 1
            if registered:
                logger.debug(f"Deep attn scan: hooked {registered} modules.")

        if registered == 0:
            # Print top-level children to help diagnose
            children_info = {
                n: type(c).__name__
                for n, c in self.model.named_children()
            }
            raise RuntimeError(
                f"Could not locate transformer attention blocks.\n"
                f"Top-level children of model: {children_info}\n"
                f"Tip: run  python -c \"import open_clip; m,_,_="
                f"open_clip.create_model_and_transforms('hf-hub:microsoft/"
                f"BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'); "
                f"[print(n) for n,_ in m.visual.named_modules()]\" "
                f"to inspect the real structure."
            )

    def _hook_block(self, block: nn.Module):
        """Register the appropriate hook on a single transformer block.

        Dispatches to MHA hook (open_clip) or timm hook based on the
        type of block.attn.

        Args:
            block: A transformer block module.

        Returns:
            The registered hook handle, or None if dispatch failed.
        """
        if not hasattr(block, "attn"):
            return None

        attn = block.attn
        if isinstance(attn, nn.MultiheadAttention):
            return attn.register_forward_hook(
                _make_mha_weight_hook(attn, self._attention_store)
            )
        elif hasattr(attn, "qkv") and hasattr(attn, "num_heads"):
            # timm-style custom Attention module
            return attn.register_forward_hook(
                _make_timm_weight_hook(attn, self._attention_store)
            )
        elif hasattr(attn, "num_heads"):
            # Unknown attention type but has num_heads — try timm hook anyway
            logger.warning(
                f"Unknown attn type {type(attn).__name__} with num_heads — "
                f"attempting timm hook."
            )
            return attn.register_forward_hook(
                _make_timm_weight_hook(attn, self._attention_store)
            )
        else:
            logger.warning(
                f"Unrecognised attn type {type(attn).__name__} "
                f"(no qkv, no num_heads, not MHA) — skipping block."
            )
            return None

    # ------------------------------------------------------------------ #
    # Inference + rollout                                                  #
    # ------------------------------------------------------------------ #

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        """Run a forward pass and compute the attention rollout map.

        Args:
            x: Input image tensor of shape (1, 3, H, W).

        Returns:
            Attention map normalised to [0, 1], shape (side, side) or (N_patches,).
        """
        self._attention_store.clear()
        with torch.no_grad():
            _ = self.model(x)

        if not self._attention_store:
            raise RuntimeError(
                "No attention weights captured. Forward hooks did not fire. "
                "Check that the model is in eval mode and the input is on the correct device."
            )

        return self._rollout()

    def _rollout(self) -> np.ndarray:
        """Accumulate attention rollout from all layers.

        Discard is applied ONCE to the final CLS attention vector (not
        per-layer).  Applying discard per-layer compounds across deep ViTs
        (e.g. UNI ViT-L/24 layers) and drives the rollout toward the
        identity, producing an all-zero (all-blue) map.  Post-rollout
        discard keeps the top ``(1 - discard_ratio)`` fraction of patch
        tokens highlighted without degrading signal through multiplication.

        Returns:
            2-D attention map (side × side) normalised to [0, 1], or
            1-D array if patch count is not a perfect square.
        """
        n_tokens = self._attention_store[0].shape[-1]
        rollout = torch.eye(n_tokens)  # (N, N) identity as starting point

        for attn in self._attention_store:
            # attn: (B, H, N, N) — squeeze batch dim (we always use B=1)
            if attn.ndim == 4:
                attn = attn[0]   # (H, N, N)

            # Fuse heads
            if self.head_fusion == "mean":
                attn_fused = attn.mean(dim=0)
            elif self.head_fusion == "max":
                attn_fused = attn.max(dim=0).values
            elif self.head_fusion == "min":
                attn_fused = attn.min(dim=0).values
            else:
                attn_fused = attn.mean(dim=0)

            # Add identity (residual connection) and row-normalise
            attn_fused = attn_fused + torch.eye(n_tokens)
            row_sums = attn_fused.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            attn_fused = attn_fused / row_sums

            rollout = rollout @ attn_fused

        # CLS token row → attentions from CLS to patch tokens (skip CLS itself)
        cls_attn = rollout[0, 1:]   # (N_patches,)

        # Apply discard once on the final CLS vector: zero out the lowest
        # discard_ratio fraction to sharpen the visualisation.
        if self.discard_ratio > 0.0:
            threshold = torch.quantile(cls_attn, self.discard_ratio)
            cls_attn = cls_attn.clone()
            cls_attn[cls_attn < threshold] = 0.0

        # Reshape to 2-D spatial grid
        n_patches = cls_attn.shape[0]
        side = int(math.sqrt(n_patches))
        if side * side == n_patches:
            attn_map = cls_attn.reshape(side, side).numpy()
        else:
            logger.warning(
                f"n_patches={n_patches} not a perfect square — "
                "returning flat attention vector."
            )
            attn_map = cls_attn.numpy()

        # Normalise to [0, 1]
        lo, hi = attn_map.min(), attn_map.max()
        if hi > lo:
            attn_map = (attn_map - lo) / (hi - lo)

        return attn_map


# ===========================================================================
# Public entry point
# ===========================================================================


def generate_attention_rollout_heatmaps(
    cfg: DictConfig,
    backbone_name: str,
    checkpoint_path: str,
    image_paths: List[str],
) -> Dict[str, np.ndarray]:
    """Load a ViT backbone, generate attention rollout maps, then release.

    For UNI (ViT-L), images are processed one at a time to avoid OOM from
    large attention matrices (24 layers × 16 heads × 197×197 patches).

    Args:
        cfg: Full OmegaConf configuration.
        backbone_name: ``"biomedclip"`` or ``"uni"``.
        checkpoint_path: Path to the saved backbone checkpoint (.pt).
        image_paths: List of image paths to generate attention maps for.

    Returns:
        Dictionary mapping image path string to 2-D attention map in [0, 1].
    """
    device = cfg.general.device
    arcfg = cfg.explainability.attention_rollout
    heatmaps: Dict[str, np.ndarray] = {}

    with gpu_memory_guard(f"Attention Rollout — {backbone_name}"):
        # --- Load backbone ---
        if backbone_name == "biomedclip":
            from src.models.backbone_biomedclip import BiomedCLIPBackbone
            model_wrapper = BiomedCLIPBackbone(cfg)
            vit_model = model_wrapper.vision_encoder
        elif backbone_name == "uni":
            from src.models.backbone_uni import UNIBackbone
            model_wrapper = UNIBackbone(cfg)
            vit_model = model_wrapper.backbone
        else:
            raise ValueError(
                f"Attention rollout not supported for backbone '{backbone_name}'. "
                f"Use 'biomedclip' or 'uni'."
            )

        # Load checkpoint weights
        ckpt = torch.load(checkpoint_path, map_location=device)
        model_wrapper.load_state_dict(ckpt["model_state_dict"])
        model_wrapper.to(device)
        vit_model = (
            model_wrapper.vision_encoder
            if backbone_name == "biomedclip"
            else model_wrapper.backbone
        )
        vit_model.eval()

        # Build rollout engine (registers hooks on vit_model)
        rollout = AttentionRollout(
            model=vit_model,
            head_fusion=arcfg.head_fusion,
            discard_ratio=arcfg.discard_ratio,
        )

        from src.data.augmentation import get_val_transforms
        transform = get_val_transforms(cfg, backbone_name)

        for img_path in image_paths:
            try:
                pil_img = Image.open(img_path).convert("RGB")
                input_tensor = transform(pil_img).unsqueeze(0).to(device)

                with torch.cuda.amp.autocast(enabled=device == "cuda"):
                    attn_map = rollout(input_tensor)

                # Resize map to image_size
                size = cfg.data.image_size
                if attn_map.ndim == 2:
                    attn_pil = Image.fromarray(
                        (attn_map * 255).astype(np.uint8)
                    ).resize((size, size), Image.BILINEAR)
                    heatmaps[img_path] = np.array(attn_pil, dtype=np.float32) / 255.0
                else:
                    heatmaps[img_path] = attn_map

                logger.debug(
                    f"Attention rollout OK: {Path(img_path).name} "
                    f"map_shape={attn_map.shape}"
                )

            except Exception as exc:
                logger.warning(f"Attention rollout failed for {img_path}: {exc}")

        rollout.remove_hooks()
        release_model(model_wrapper)

    logger.info(
        f"Generated {len(heatmaps)} attention rollout maps for '{backbone_name}'."
    )
    return heatmaps
