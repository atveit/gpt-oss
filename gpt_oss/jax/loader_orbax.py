"""Orbax checkpoint loader for gpt-oss-20b.

This module provides fast weight loading from pre-converted Orbax checkpoints.
Much faster than loading SafeTensors (5s vs 90s).

Supports MXFP4 quantized weight unpacking for models like GPT-OSS-20B.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import orbax.checkpoint as ocp
import jax.numpy as jnp

# Import MXFP4 unpacking utilities
from .mx_formats import unpack_quantized_param_tree


def translate_orbax_to_model_structure(orbax_params: Dict[str, Any]) -> Dict[str, Any]:
    """Translate Orbax checkpoint structure to JAX model structure.

    Orbax checkpoint uses:
        - embed_tokens/embedding
        - layers/0/...
        - lm_head/kernel
        - norm/scale

    JAX model expects:
        - embedding/embedding
        - block_0/...
        - unembedding/kernel
        - norm/scale

    Args:
        orbax_params: Parameters from Orbax checkpoint

    Returns:
        Translated parameters matching JAX model structure
    """
    translated = {}

    for key, value in orbax_params.items():
        if key == 'embed_tokens':
            # Map embed_tokens → embedding
            translated['embedding'] = value
        elif key == 'layers':
            # Map layers/N → block_N
            for layer_idx, layer_params in value.items():
                block_key = f'block_{layer_idx}'
                translated[block_key] = layer_params
        elif key == 'lm_head':
            # Map lm_head → unembedding
            translated['unembedding'] = value
        elif key == 'norm':
            # Keep norm as-is
            translated['norm'] = value
        else:
            # Keep other keys as-is
            translated[key] = value

    return translated


class OrbaxWeightLoader:
    """Load weights from Orbax checkpoint format.

    Orbax checkpoints are pre-converted from SafeTensors and load much faster
    (5-6s vs 90s for SafeTensors + MXFP4 decompression).

    Usage:
        loader = OrbaxWeightLoader('/path/to/orbax/checkpoint')
        params = loader.load_params()
    """

    def __init__(self, checkpoint_path: str):
        """Initialize loader.

        Args:
            checkpoint_path: Path to Orbax checkpoint directory (should contain '0' subdirectory)
        """
        self.checkpoint_path = Path(checkpoint_path).resolve()  # Use absolute path for Orbax
        assert self.checkpoint_path.exists(), \
            f"Checkpoint path not found: {self.checkpoint_path}"

        # Load quantization metadata if present
        self.quantization_metadata = None
        quant_path = self.checkpoint_path / "_quantization_metadata.json"
        if quant_path.exists():
            with open(quant_path, 'r') as f:
                self.quantization_metadata = json.load(f)
                print(f"  Found quantization metadata: {len(self.quantization_metadata)} quantized parameters")

    def load_params(
        self,
        show_progress: bool = True,
        unpack_quantized: bool = False,
        validate_unpacking: bool = True
    ) -> Dict[str, Any]:
        """Load parameters from Orbax checkpoint.

        Args:
            show_progress: Show loading progress
            unpack_quantized: Automatically unpack MXFP4 quantized weights to float16
            validate_unpacking: Validate unpacking invariants (recommended)

        Returns:
            Parameter tree compatible with JAX/Flax models
        """
        # Check for state subdirectory (common Orbax structure)
        checkpoint_dir = self.checkpoint_path / "0"
        state_path = checkpoint_dir / "state"
        if state_path.exists() and (state_path / "_METADATA").exists():
            checkpoint_dir = state_path

        if show_progress:
            print(f"  Loading from: {checkpoint_dir}")

        # Load checkpoint
        checkpointer = ocp.PyTreeCheckpointer()
        params = checkpointer.restore(str(checkpoint_dir))

        # Translate Orbax structure to JAX model structure
        if show_progress:
            print(f"  Translating parameter structure...")
        params = translate_orbax_to_model_structure(params)

        if show_progress:
            print(f"  ✓ Loaded {len(params)} top-level parameter groups")

            # Show size estimate
            def count_params(tree):
                if isinstance(tree, dict):
                    return sum(count_params(v) for v in tree.values())
                elif isinstance(tree, jnp.ndarray):
                    return tree.size
                return 0

            total_params = count_params(params)
            print(f"  ✓ Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")

        # Unpack quantized weights if requested
        if unpack_quantized and self.quantization_metadata:
            if show_progress:
                print(f"\n  Unpacking {len(self.quantization_metadata)} quantized parameters...")

            params, timing_info = unpack_quantized_param_tree(
                params,
                self.quantization_metadata,
                validate=validate_unpacking,
                show_progress=show_progress,
                parallel=True,
                backend='auto'
            )

            if show_progress:
                print(f"  ✓ Unpacked in {timing_info['total_time']:.2f}s (backend: {timing_info['backend']})")

        return params


def load_config_from_orbax(checkpoint_path: str) -> Dict[str, Any]:
    """Load model config inferred from Orbax checkpoint.

    Note: Orbax checkpoints don't include config.json,
    so we return hardcoded config for gpt-oss-20b.

    Args:
        checkpoint_path: Path to Orbax checkpoint

    Returns:
        Dictionary with model configuration
    """
    # For now, return the known gpt-oss-20b config
    # In future, could infer from checkpoint structure
    return {
        "num_hidden_layers": 24,
        "hidden_size": 2880,
        "head_dim": 64,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "sliding_window": 128,
        "intermediate_size": 2880,
        "num_experts": 32,
        "experts_per_token": 4,
        "vocab_size": 201088,
        "swiglu_limit": 7.0,
        "rope_theta": 150000.0,
        "rope_scaling_factor": 32.0,
        "rope_ntk_alpha": 1.0,
        "rope_ntk_beta": 32.0,
        "initial_context_length": 4096,
    }


if __name__ == "__main__":
    """Test Orbax loader."""
    import time

    checkpoint_path = "../atsentia-orbax-to-jaxflaxmock/orbaxmodels/gpt-oss-20b"

    print("="*80)
    print("Testing Orbax Weight Loader")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}\n")

    print("Loading parameters...")
    t0 = time.time()
    loader = OrbaxWeightLoader(checkpoint_path)
    params = loader.load_params(show_progress=True)
    load_time = time.time() - t0

    print(f"\n✓ Loading completed in {load_time:.2f}s")
    print(f"  (Compare to SafeTensors: ~90s)")
    print("="*80)
