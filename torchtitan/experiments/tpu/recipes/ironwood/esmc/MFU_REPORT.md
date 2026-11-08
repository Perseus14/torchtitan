# ESMC-600M on TPU7x: MFU campaign report

Same model (`biohub/ESMC-600M`), config, data and hyperparameters as
`hf_esmc_mlm_uniref50_graph`; no architecture changes. One tpu7x-8 host
(8 TensorCores), torch_tpu 0.2.0, libtpu 0.0.51, 16 x 2048 tokens per rank,
one microbatch per optimizer step. The per-change table and launch commands
are in [README.md](README.md); the per-component profile analysis is in
[HEADROOM_ANALYSIS.md](HEADROOM_ANALYSIS.md).

| Config | tok/s/TC | MFU |
|---|---:|---:|
| `hf_esmc_mlm_uniref50_graph` (baseline) | 70.2k | 27.2% |
| `hf_esmc600m_mlm_uniref50_graph_opt` | **86.1k** | **33.4%** |
| `hf_esmc600m_mlm_uniref50_graph_opt_pstate7` | **97.4k** | **37.7%** |

MFU divides by 1,153.5 TFLOP/s/TC, the P7 (2.2 GHz) peak. Without the p-state
flag the chip runs at the nominal P3 clock (1.9 GHz, ~996 TFLOP/s/TC), so
33.4% nominal is ~38.7% of the peak at the clock actually in use. The p-state 7
row's gain is clock, not kernel efficiency.

## Numerics

Similar, not bit-identical: fixed-max softmax, the compact logsumexp, the
reduce-window flag and the forward kv compute sub-block each change fusion or
fp32 accumulation order. 600-step UniRef50 loss, seed 42, untouched baseline
vs the optimized stack:

| step | baseline loss | optimized loss | baseline grad_norm | optimized grad_norm |
|---:|---:|---:|---:|---:|
| 25 | 2.77997 | 2.78116 | 2.944 | 3.027 |
| 100 | 2.62538 | 2.62682 | 1.518 | 1.344 |
| 200 | 2.61443 | 2.61429 | 0.638 | 0.743 |
| 300 | 2.56797 | 2.56726 | 0.569 | 0.592 |
| 400 | 2.57633 | 2.57652 | 0.443 | 0.365 |
| 500 | 2.56487 | 2.56509 | 0.308 | 0.296 |
| 600 | **2.54881** | **2.54839** | 0.285 | 0.303 |

The table is the 30.3% stack. The final 33.4% stack ends at **2.5483** at
step 600 (steps 100 / 300 / 600: 2.6339 / 2.5698 / 2.5483), the 32.8% stack
at 2.5473.

## What moved MFU, and why

- **Fixed softmax max** (`sa_max_logit_const=0`): QK-LayerNorm bounds the
  logits, so the forward drops the running row max and its rescale.
- **Compact logsumexp:** the forward wrote lse as a 128-lane-replicated
  `[H, S, 128]` f32 buffer (~300 MB per layer) that the backward relaid out;
  it is now `[H, 1, S]`. The dkv kernel also gets the LP LLO scheduler.
- **`optimizer_in_graph`:** forward, backward, clip and AdamW in one compiled
  program; no per-step dispatch gaps.
- **Folded segment mask:** head_dim 64 already occupies 128-lane tiles. The
  QK prologue writes `onehot(document)` into q's spare 64 lanes and `0` /
  `-2^14` (same / other document) into k's, so `q . k` is unchanged within a
  protein and `exp` underflows to exactly 0 across proteins. Splash runs with
  no segment ids. Even heads keep their values in lanes `[0, 64)`, odd heads in
  `[64, 128)`, so no head is lane-shifted.
- **`xla_tpu_reduce_window_reduction_dim_max=0`:** XLA rewrote each
  LayerNorm-backward row reduce + broadcast into a `1x1x2303` reduce-window
  over the 1152 hidden lanes (148 instances, ~21% of HBM roofline). The flag
  keeps reduce + broadcast (+3.7%).
- **SwiGLU saves `sigmoid(gate)`:** the backward is multiplies and adds in the
  down-projection dX epilogue instead of a recomputed sigmoid (+1.7%).
- **`qk_norm_rope` kernel:** LLO feedback showed a critical path of cross-lane
  (XLU) reductions and rotates. Lane groups are now added elementwise and
  reduced across lanes once per row tile; the smaller live set lets `block_s`
  grow to 1024 (+2.6% together with the no-roll layout).
- **Split-halves q/k lanes:** with each 128-lane group holding
  `[h0[:32] | h1[:32] | h0[32:] | h1[32:]]`, `rotate_half` is one 64-lane
  roll (was two plus a select). The q/k projection and norm weights are
  row-gathered per step. Microbench: fwd 0.443 -> 0.404 ms, bwd 0.573 ->
  0.484 ms per layer (+0.45% end to end).
- **Input-LayerNorm barrier:** forward-only barrier on LN1 so the QKV matmuls
  stop recomputing it per tile (+0.2%, +2.5 GiB).
- **Splash block re-sweep:** 2048 x 2048 blocks and dkv compute 1024 remain
  best; with no segment ids the forward kv compute sub-block moves 512 -> 2048
  (+0.4%).

## Tried and dropped (measured)

| Experiment | Result |
|---|---|
| LayerNorm materialization barrier | helped (+1.2%) until the reduce-window flag; then -0.4% and +4.9 GiB |
| Pallas LayerNorm fwd/bwd | -0.4%, +5 GiB (faster in isolation; loses XLA residual-add fusions) |
| Fused gate+up matmul (concatenated weights) | -1.0% |
| `o_proj` as an einsum on the heads-first attention output | -2.4%, +1.2 GiB |
| SwiGLU saving `silu(gate)` instead of `gate` | -2.1% |
| SwiGLU fusion-split optimization barriers (fwd / bwd / both) | -1.2% / -1.4% / -2.5% |
| `qk_norm_rope` rotate-half on the MXU (0/1 permutation matmul) | ~neutral in microbench; VMEM OOM at block_s 1024 |
| Barriers on the LayerNorm gradients (unfuse LN bwd from the dX matmul) | -0.7% to -1.3% |
| Barrier on LN2's forward output (alone / with LN1) | -0.6% / +0.1% |
| P cast to bf16 before the forward P.V matmul | no change: Mosaic already does it |
| Splash block re-sweep (8 other variants) | -0.3% to -2.6% |
| Collective / scheduler flags (SparseCore offload, LHS, async all-gather, ...) | within +-1% |
| `dot_strength_reduction=false` | -10% |
| Smaller splash blocks with document skipping | block overhead exceeds the savings |
| bf16 exponentials / bf16 P.V | slower / already done by Mosaic |

## Remaining headroom

About 45% of the step is attention. LLO shows both splash kernels with a
critical path that is >99% MXU (bf16, with masked K=64 contractions in dkv):
they are bound by the MXU shape head_dim 64 implies, not by vector work. About
42% is matmuls at 65-88% of peak against a ~90% ceiling (d = 1152 = 4.5 x 256),
with LayerNorm / SwiGLU epilogues fused in (splitting them was measured slower).
See HEADROOM_ANALYSIS.md for the per-component table. Beyond ~38% MFU at p-state 7
needs FP8 matmuls or an architecture change (e.g. head_dim 128).

## Caveats

- **63 documents per row** (`esmc_fold_segment_mask`): the folded mask encodes
  at most 64 segments per 2048-token row; the config caps the packer at
  `dataloader.max_documents_per_row = 63`. This recipe's UniRef50 subset (the
  first 4M FASTA records, which are the *longest* proteins, median ~650-1,960
  residues) holds at most 3 documents per row, so the cap never binds. On full
  UniRef50 or short-peptide data, re-measure before relying on it (turning it
  off costs ~1.7%).
- **`optimizer_in_graph`** supports one microbatch per optimizer step only (it
  raises otherwise) and skips the host-side non-finite loss check.
- **`sa_max_logit_const=0`** relies on QK-LayerNorm keeping the scaled logits
  within fp32 `exp2` range (roughly [-87, 88]).
- **`xla_tpu_reduce_window_reduction_dim_max=0`** is a workaround for an XLA
  TPU fusion choice; an upstream XLA bug is still to be filed.
