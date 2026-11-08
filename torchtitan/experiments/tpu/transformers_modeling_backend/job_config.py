"""Job config for the HF backend on the TPU lane."""

import dataclasses

from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig

# The folded segment mask has 128 - 64 = 64 spare lanes: one is the padding
# tail's segment, the rest are documents.
ESMC_FOLDED_MAX_DOCUMENTS_PER_ROW = 63


@dataclasses.dataclass(kw_only=True, slots=True)
class TPUTransformersBackendConfig(TPUTrainerConfig):
  """A TPU job config that also carries the HuggingFace model id.

  The HF backend reads the id off the *job* config -- ``HFTransformerModel.
  Config.update_from_config`` does ``getattr(config, "hf_model", "")`` -- so it
  has to be a top-level field rather than one of the per-model sub-configs.
  ``TPUTrainerConfig`` is a slotted dataclass, so it cannot simply be set at
  runtime; subclassing is what adds the slot.

  Subclassing ``TPUTrainerConfig`` (rather than the backend's
  ``SFTTrainer.Config``) is what lets the TPU entry point accept this config
  unchanged: ``gmain`` reaches for ``config.tpu_config``, and ``train.py``
  gates its TPU behavior on ``isinstance(config, TPUTrainerConfig)``.
  """

  hf_model: str = ""
  # A JSON object applied to the loaded HF config after the flavor's
  # overrides, e.g. ``{"attn_pdrop": 0, "resid_pdrop": 0}`` to train a
  # checkpoint without its dropout. Read by the shared
  # backend's ``update_from_config`` through the same ``getattr(config, ...)``
  # path as ``hf_model``.
  hf_config_overrides: str = ""
  # ESM-C only: run QK-LayerNorm + head split + RoPE + heads-first as one
  # fused Pallas kernel each way (kernels/qk_norm_rope.py) instead of XLA's
  # norm / relayout-copy / RoPE passes. Read by the model config's
  # ``update_from_config`` and carried as ``tpu_fused_qk_norm_rope``.
  esmc_fused_qk_norm_rope: bool = False
  # ESM-C only: store each block's input_layernorm / post_attention_layernorm
  # output once (an XLA optimization barrier) instead of letting XLA re-run
  # the LayerNorm as the prologue of every matmul tile that reads it.
  esmc_materialize_layernorms: bool = False
  # ESM-C only, needs ``esmc_fused_qk_norm_rope``: fold the packed-document
  # mask into the 64 padding lanes of q/k (written by the fused prologue), so
  # splash attention runs without segment ids. Encodes at most 64 segments per
  # row (documents plus the padding tail), so the packer must cap documents per
  # row: requires ``dataloader.pack_rows_to_context`` and
  # ``dataloader.max_documents_per_row <= ESMC_FOLDED_MAX_DOCUMENTS_PER_ROW``.
  esmc_fold_segment_mask: bool = False
  # ESM-C only: SwiGLU that saves sigmoid(gate) so the backward does not
  # recompute it in the fused down-projection dX epilogue (+1 activation of
  # (tokens, intermediate_size) per layer).
  esmc_swiglu_saved_sigmoid: bool = False
  # ESM-C only: optimization barrier on each block's input LayerNorm output
  # (forward only), so the QKV matmuls read it instead of recomputing the
  # LayerNorm per tile; its gradient stays fused.
  esmc_barrier_input_layernorm: bool = False
  # ESM-C only (with esmc_fold_segment_mask): q/k in the split-halves lane
  # order, so the fused prologue's rotate_half is one 64-lane roll per lane
  # group. A per-step row gather of q_proj/k_proj/q_norm/k_norm weights.
  esmc_qk_split_halves: bool = False

  def __post_init__(self):
    super(TPUTransformersBackendConfig, self).__post_init__()
    if self.esmc_qk_split_halves and not self.esmc_fold_segment_mask:
      raise ValueError(
          "esmc_qk_split_halves requires esmc_fold_segment_mask (each head"
          " keeps its own lanes only in the widened q/k)"
      )
    if self.esmc_fold_segment_mask:
      if not self.esmc_fused_qk_norm_rope:
        raise ValueError(
            "esmc_fold_segment_mask requires esmc_fused_qk_norm_rope (the "
            "fused prologue writes the folded q/k lanes)"
        )
      cap = getattr(self.dataloader, "max_documents_per_row", None)
      if not getattr(self.dataloader, "pack_rows_to_context", False) or (
          cap is None or cap > ESMC_FOLDED_MAX_DOCUMENTS_PER_ROW
      ):
        raise ValueError(
            "esmc_fold_segment_mask encodes at most "
            f"{ESMC_FOLDED_MAX_DOCUMENTS_PER_ROW + 1} segments per 2048-token "
            "row (documents plus the padding tail); set "
            "--dataloader.pack_rows_to_context and "
            "--dataloader.max_documents_per_row <= "
            f"{ESMC_FOLDED_MAX_DOCUMENTS_PER_ROW} (got {cap})"
        )
