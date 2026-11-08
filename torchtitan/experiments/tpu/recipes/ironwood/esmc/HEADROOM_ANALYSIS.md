# ESMC-600M on TPU7x: headroom analysis

Per-component TensorCore time against a roofline, re-measured after each round of changes. Round 1 analyzes the 30.3% configuration, round 2 the 32.0% one, round 3 the 32.9% one. Final: 85.3k tok/s/TC, 33.1% MFU (97.0k / 37.6% with p-state 7). See [MFU_REPORT.md](MFU_REPORT.md).

## Round 1 (30.3% configuration)

**Workload:** `hf_esmc600m_mlm_uniref50_graph_opt` on one tpu7x-8 host (8 TensorCores), 16 x 2048 tokens/TC, one microbatch per step.
**Measured:** 78.3k tok/s/TC, 30.3% MFU, 419 ms per step.
**Profile:** fresh XProf of the final configuration, 19 steps; times are TensorCore milliseconds per step.

> [!NOTE]
> **Clock and MFU denominator.** torchtitan's MFU divides by 1,153.5 TFLOP/s/TC, the **P7 (2.2 GHz)** peak. Without the p-state flag the chip runs at the nominal **P3 (1.9 GHz, 996 TFLOP/s/TC)**, which matches the measured 1.14x from p-state 7. Against the clock actually in use, the 30.3% configuration is at **35.1% MFU**. The roofline below uses the P3 rates: 996 TFLOP/s BF16 and 3.7 TB/s HBM per TC.

## Component-level headroom table

FLOPs and bytes are per TC per step: T = 32,768 tokens, L = 36, d = 1152, F = 3072, H = 18, head_dim = 64.

| Category | Component | Roofline (ms) | Measured (ms) | % of step | % of roofline | Bound | Priority | Opportunity |
|:--|:--|--:|--:|--:|--:|:--|:-:|:--|
| Attention | splash dkv (bwd), 27.8 TF executed | 111.8 *(25% MXU cap)* / 27.9 *(full MXU)* | 118.8 | 29.2% | **94%** of cap | MXU at hd = 64, plus VALU | **Do not touch** | At the head_dim-64 cap; only an architecture change helps |
| Attention | splash fwd, 11.1 TF | 44.7 *(cap)* / 11.2 | 51.7 | 12.7% | 86% of cap | MXU at hd = 64, plus VALU | P2 | ~7 ms left |
| Matmul | bwd dX, 37.6 TF | 37.7 | 56.0 | 13.8% | 67% | compute | **P0** | 18.3 ms |
| Matmul | bwd dW, 37.6 TF (f32 out) | 37.7 | 50.9 | 12.5% | 74% | compute | **P1** | 13.2 ms |
| Matmul | fwd FFN gate+up, 16.7 TF | 16.8 | 21.8 | 5.4% | 77% | compute | P1 | 5.0 ms |
| Matmul | fwd O + FFN down, 11.5 TF | 11.5 | 14.0 | 3.4% | 83% | compute | P2 | 2.5 ms |
| Matmul | fwd QKV, 9.4 TF | 9.4 | 12.2 | 3.0% | 77% | compute | P2 | 2.8 ms |
| Norm | LayerNorm fwd + bwd (2/layer) | 7.3 *(~27 GB)* | 34.3 | 8.4% | **21%** | HBM | **P0** | **27 ms**: bwd output fusion is ~55%, f32 dx ~15%, dgamma reductions ~11% |
| Attention prologue | qk_norm_rope bwd | 7.3 *(~27 GB)* | 21.3 | 5.2% | 34% | HBM | **P0** | 14 ms |
| Attention prologue | qk_norm_rope fwd | 5.9 *(~22 GB)* | 11.3 | 2.8% | 52% | HBM | P1 | 5.4 ms |
| Glue (roofline 0) | copies / relayout, loop fusions, broadcast, dynamic-update-slice, TC-visible collectives | 0 | 12.4 | 3.0% | 0% | overhead | **P1** | 12.4 ms |
| Optimizer | AdamW in graph (72M fp32 params/TC) | 0.55 | ~1.0 | 0.3% | ~55% | HBM | -- | negligible |
| Host (roofline 0) | step-boundary gap (419 ms wall vs 407 ms device) | 0 | ~12 | 2.9% | 0% | host dispatch | P1 | 12 ms |
| **Total** | | **~ 291** *(hd-64 cap)* | **~ 419** | 100% | **69%** | | | **~ 128 ms recoverable** |

**Speed of light for this architecture:** about 291 ms per step, i.e. about 112k tok/s/TC, **about 43.7% nominal MFU** (about 51% at the actual P3 clock). Even with perfect execution, head_dim 64 caps the step about 1.4x above where it is now. Attention alone accounts for 156 ms of the 291 ms at the 25% MXU cap.

**Calibration notes:**
- The **25% MXU cap** assumes every attention matmul pays for a 64-wide dimension on the 256x256 MXU. LLO feedback (dkv kernel: VALU busy in 73% of bundles, vld 72%, vst 68%) shows VALU and load/store work co-limit as well, so the cap is a reasonable floor, not a guaranteed one.
- The **matmul ceiling** is realistically about 90%, because d = 1152 = 4.5 x 256 leaves a half tile.
- **LayerNorm bytes** assume bf16 activations, plus the extra write and read that the materialization barrier adds. The recoverable share is still at least 20 ms if the residual stream turns out to be fp32.

## Round 1 ROI-ranked plan

ROI = recoverable ms x % of step. The gain estimates assume about 60% of the gap is recovered.

| Rank | Hypothesis | Evidence | Est. gain | Effort | Numerics |
|:-:|:--|:--|--:|:-:|:--|
| 1 | **LayerNorm backward**: fix the slow bwd output fusion. Inspect its HLO layout (f32 `[16,2048,1152]` T(8,128); reductions over the 1152 lane axis). Either fuse dgamma/dbeta into one pass, or use a Pallas LN fwd/bwd kernel keyed to the 128-lane layout. | 34.3 ms at 21% of HBM roofline | **+4-5%** | M | similar |
| 2 | **qk_norm_rope bwd/fwd**: sweep `block_s` (512 -> 1024/2048, currently hard-coded) and the VMEM limit; drop redundant f32 upcasts of `gq`/`gk`. | 32.6 ms at 34-52% of roofline | **+2-3%** | S | identical (block_s) / similar |
| 3 | **Matmul bwd dX/dW**: XProf per-op tile/fusion check for elementwise prologues (SwiGLU bwd, LN bwd) fused into the dX convolutions; try `bf16` dW accumulation outputs with an fp32 reduce-scatter. | 107 ms at 67-74% | **+2-4%** | M | similar |
| 4 | **Host step gap**: dispatch step N+1's traced graph before step N finishes (the `make_fx` unwrap/wrap of subclasses costs ~5-10 ms of Python per step). | ~12 ms of wall, device idle | **+2%** | M | identical |
| 5 | **Glue**: remove the remaining `as_strided` relayout copies around attention output (`ContiguousTranspose12`) and stray loop fusions. | 12.4 ms, roofline 0 | **+1-2%** | S-M | identical |
| -- | **Do not touch:** splash dkv (94% of the hd-64 cap); collectives (already overlapped on SparseCore, 4 neutral trials); scheduler flags (2 neutral). | | | | |

**Cumulative prediction** if ranks 1-5 land: about 360 ms per step -> **about 91k tok/s/TC, about 35% nominal MFU** without p-state, or about 40% with p-state 7.


---

# Round 2: re-profile of the 82.5k configuration (2026-11-08)

**Config:** `opt` plus `xla_tpu_reduce_window_reduction_dim_max=0` and `esmc_swiglu_saved_sigmoid`.
**Measured:** 82.5k tok/s/TC, 32.0% MFU (36.9% at the P3 clock), 395 ms per step on the device.
**Profile:** fresh XProf of that configuration, 18 steps.

## What happened to the round 1 plan

| Rank | Item | Outcome |
|:-:|:--|:--|
| 1 | LayerNorm bwd | **Landed, +3.7%.** XLA had rewritten each 1152-lane reduce + broadcast into a 2303-wide reduce-window; the flag above turns that off. LN time fell from 34.3 to 12.6 ms, and about 7 ms of it moved into loop fusions (see the table). |
| 2 | qk_norm_rope `block_s` / VMEM | **Exhausted.** 512 is best, 256 is 20% slower, and 1024 and 2048 run out of VMEM. |
| 3a | SwiGLU saves `sigmoid(gate)` | **Landed, +1.7%.** |
| 3b | One combined gate+up matmul (concatenated weights) | **Reverted, -1.0%.** Concatenating and splitting the weights costs more than the larger matmul saves. |
| 4 | Host step gap | **Already gone.** `optimizer_in_graph` leaves 0.4 ms of device idle per step. |
| 5 | `o_proj` as an einsum on the heads-first attention output | **Reverted, -2.4%, +1.2 GiB.** XLA moves the relayout inside the dot. |

## Updated per-component table (TensorCore ms per step)

| Component | Round 1 | Round 2 | Bound | Notes |
|:--|--:|--:|:--|:--|
| splash dkv | 118.8 | 120.4 | MXU at hd-64 plus VALU | Do not touch |
| splash fwd | 51.7 | 54.5 | MXU at hd-64 plus VALU | |
| matmul bwd dX | 56.0 | 54.7 | compute / HBM | The SwiGLU-bwd + down_proj dX fusion is **HBM-bound at 54% of peak**. It reads gate, up and s and writes d_gate and d_up, about 1 GB per layer. |
| matmul bwd dW | 50.9 | 46.9 | compute, 79-89% | Near the ~90% ceiling set by d = 1152 = 4.5 x 256 |
| matmul fwd (QKV, O, FFN) | 48.0 | 50.1 | compute, 69-82% | The gate+up multi-output fusion runs at 69% |
| LayerNorm (labelled) | 34.3 | 12.6 | HBM | |
| loop fusions (residual add + LN bwd output, LN fwd) | 4.4 | 11.4 | HBM, **62-77% of peak** | `bf16[16,2048,1152]` written in T(8,128), which is not bf16's native T(16,128) tile |
| attention `di = rowsum(o * dO)` fusion | (in loop) | ~3.4 | HBM, **~40% of peak** | head_dim 64 is padded to 128 lanes, so it reads twice the bytes it needs |
| qk_norm_rope fwd + bwd | 32.6 | 33.5 | HBM | Exhausted |
| copies, collectives, optimizer, other | ~9 | ~9 | | |

## Round 2 hypotheses (ROI order)

| Rank | Hypothesis | Est. gain | Effort | Numerics |
|:-:|:--|--:|:-:|:--|
| R2-1 | **Remove the LayerNorm materialization barrier** now that the reduce-window rewrite is gone: the barrier forces a T(8,128) bf16 write and an extra HBM round trip. Single flag (`--no-esmc_materialize_layernorms`). | 0 to +2% | XS | similar |
| R2-2 | Re-check `esmc_fold_segment_mask` against the new stack (keep it only if it still wins; it carries the 63-documents-per-row caveat). | +-1% | XS | similar |
| R2-3 | **Fewer bytes in the SwiGLU bwd**: save `silu(gate)` instead of `gate`, so that `d_up = dh * silu` and `d_gate = dh * up * (s + silu * (1 - s))`. Bytes are unchanged, but the down_proj epilogue does less VALU work. Alternatively, store `s` in bf16 inside a single packed `[.., 2F]` tensor. | +0.5-1% | S | similar |
| R2-4 | **`di` fusion**: emit `di` from the splash kernel's backward entry on lane-dense (S/2, 128) views of `o` and `dO`, or compute it inside the dkv prologue once per q block. | +0.5% | M | identical / similar |
| R2-5 | Matmul tiling or fusion knobs for the FFN gate+up fwd multi-output (69%) | +0.5-1% | M | identical |

The remaining gap is now dominated by attention at the head_dim-64 cap (175 ms of 395) and by matmuls near the 1152-dimension ceiling. After R2-1 to R2-5, expect diminishing returns of about 1-3% in total.


## Round 2 results

| Rank | Trial | tok/s/TC | delta vs previous best | Decision |
|:-:|:--|--:|--:|:--|
| R2-1 | Drop the LayerNorm materialization barrier | 82.9k | +0.4%, -4.9 GiB | **Adopted** |
| R2-2 | Turn off the folded segment mask | 81.1k | -1.7% | Fold stays |
| R2-3 | SwiGLU saves `silu(gate)` instead of `gate` | 81.2k | -2.1% | Reverted |
| R2-4 | `di` fusion | -- | -- | **Dropped:** the reduce already runs near the padded-bytes roofline (~2.9 ms of the 3.4 ms); fixing it needs a kernel layout change for at most ~0.4% |
| R2-6 *(new)* | `qk_norm_rope`: sum lane groups with elementwise adds, then do **one** cross-lane reduce (LLO showed a critical path that is 99.7% XLU: `vrot.lane` 60%, `vadd.xlane` 40%) | 84.1k | +1.4% | **Adopted** |
| R2-6b | Same, plus `block_s` 1024 and 64 MiB VMEM (1024 now fits) | **84.6k** | +2.1% | **Adopted** |
| -- | Rotate-half done on the MXU with a 0/1 permutation matmul | microbench 0.754 vs 0.787 ms; out of VMEM at 1024 | | Dropped |
| -- | `qk_norm_rope` bwd two-pass recompute, or an f32 scratch buffer | microbench 1.12 / 0.89 vs 0.78 ms | | Dropped (VMEM is not the limit) |


---

# Round 3 plan: can p-state 7 reach 40% MFU?

**Gap:** 37.3% -> 40.0% under p-state 7 is **+7.3% throughput** (96.2k -> about 103.3k tok/s/TC). Without p-state the same gain would take 32.8% to about 35.2%.

**Where the 387 ms step goes now** (82.5k profile, adjusted for the round 2 `qk_norm_rope` win):

| Component | ms | Headroom |
|:--|--:|:--|
| Attention (splash fwd + dkv) | ~175 | Close to the head_dim-64 cap. Only block re-tuning is left: the kernel now runs **without segment ids** (folded mask), so the round 1 block choice may no longer be optimal. |
| Matmuls | ~150 | 70-89%, against a ~90% ceiling (d = 1152 = 4.5 x 256). The FFN gate+up fwd (69%) and the SwiGLU-bwd + down dX fusion (HBM-bound at 54%) are the outliers. |
| Residual + LayerNorm loop fusions | ~24 | HBM-bound at 62-77%. bf16 is written in T(8,128), which is not bf16's native tile. |
| `qk_norm_rope` fwd + bwd | ~28 | The bwd still spends its critical path on lane rotates; the fwd hasn't been LLO-audited. Roughly 2x the HBM roofline. |
| Other (copies, `di`, optimizer, collectives) | ~10 | Small |

## Candidate hypotheses (ROI order)

| Rank | Hypothesis | Est. gain | Effort | Numerics |
|:-:|:--|--:|:-:|:--|
| R3-1 | **Splash block re-sweep** for fwd (`block_kv_compute` 256/512/1024, `block_q` 1024/2048) and dkv (`block_kv_dkv_compute` 512/1024/2048) on the folded-mask stack | 0 to +2% | XS (config) | identical |
| R3-2 | **LLO audit of the `qk_norm_rope` fwd**, then apply the same lane-group fixes as the bwd | +1-1.5% | S | similar (one bf16 ulp) |
| R3-3 | **`qk_norm_rope` bwd: remove the remaining rotates.** Either one roll with a pre-signed partner table, or swap whole vregs by pairing the first halves of 2 heads in one vreg and the second halves in the next | +0.5-1% | M | identical |
| R3-4 | **Unfuse the SwiGLU bwd from the down_proj dX matmul** (optimization barrier on `dh`): the matmul becomes compute-bound and the elementwise pass runs at HBM speed | -1 to +1% | XS | identical |
| R3-5 | **Split the FFN gate+up fwd multi-output fusion** (barrier between the matmuls and the SwiGLU), currently 69% | 0 to +0.5% | XS | identical |
| R3-6 | **Pallas residual-add + LayerNorm fwd/bwd kernel** in native bf16 tiles, to take the 24 ms of loop fusions from 62-77% to about 90% of HBM | +2-3% | M-L | similar |
| R3-7 | **p-state probe:** check whether libtpu accepts a higher throughput state than 7 on tpu7x. A clock gain only, reported separately. | 0 / clock | XS | identical |
| -- | *Out of scope (numerics / architecture):* FP8 matmuls (+10-15% plausible, but numerics change materially); an architecture change (head_dim 128). | | | |

**Projection:** if every row lands at its estimate, the total is about +5 to +9%, i.e. **p-state 7 at about 39-40.5%** and no p-state at about 34.5-35.7%. 40% is reachable but not likely: most of the remaining time is attention at the head_dim-64 cap and matmuls near the 1152 ceiling.

**Order run:** R3-1, R3-4, R3-5 and R3-7 first (config or barrier only, about 1 hour of TPU time). Then R3-2 and R3-3 (kernel work driven by LLO feedback). Then R3-6 (a new kernel, the largest piece of work). Each change was kept only if it beat the current best.


## Round 3 results

| Rank | Trial | tok/s/TC | delta vs best | Decision |
|:-:|:--|--:|--:|:--|
| R3-1 | Splash block re-sweep | 85.3k | +0.4% | See below: only the fwd kv compute sub-block moved |
| R3-7 | Higher p-state | -- | -- | Skipped: 7 is the maximum |
| R3-4 | Barrier on `dh` (unfuse the SwiGLU bwd from down_proj dX) | 83.5k | -1.4% | Reverted |
| R3-5 | Barrier between gate/up and the SwiGLU fwd | 83.6k | -1.2% | Reverted |
| R3-4+5 | Both | 82.5k | -2.5% | Reverted |
| R3-6 | Pallas LayerNorm fwd/bwd (2/layer) | 84.3k | -0.4%, +5 GiB | Removed: microbench 0.084/0.130 ms vs XLA 0.115/0.323 ms, but it loses XLA's residual-add fusions in the graph |
| R3-2/3 | `qk_norm_rope`: odd heads keep their upper lanes (no head-shift roll); bwd recomputes `xhat`. LLO: rotates were 86% of the critical path in both kernels. | **85.0k** | +0.5% | **Adopted** (bit-identical loss) |

**Final:** 85.0k tok/s/TC, 32.9% MFU; with p-state 7, 96.6k / 37.4%. The remaining time is attention at the head_dim-64 cap (~45%) and matmuls near the d = 1152 ceiling (~39%). Getting past ~38% at p-state 7 needs FP8 or an architecture change.

### R3-1 block re-sweep

| Variant (vs 2048/2048 blocks, fwd kv compute 512, dkv compute 1024) | tok/s/TC | delta |
|:--|--:|--:|
| Reference | 85.0k | -- |
| **fwd kv compute 2048** (repeat: 85.3k) | **85.3k** | **+0.4%** -- adopted |
| fwd kv compute 1024 / 256 | 85.2k / 85.0k | +0.2% / 0 |
| fwd `block_q` 1024 / `block_kv` 1024 | 84.1k / 84.0k | -1.0% / -1.2% |
| dkv compute 512 / 2048 (2048 now fits) | 84.7k / 84.6k | -0.3% / -0.4% |
| dkv `block_q_dkv` 1024 / `block_kv_dkv` 1024 | 83.5k / 82.8k | -1.7% / -2.6% |

**Final:** 85.3k tok/s/TC, 33.1% MFU; with p-state 7, 97.0k / 37.6%.

## Round 4: speed-of-light check of the final configuration

Profile of the 85.3k configuration (381.7 ms per step on the device);
limits use P3 rates (996 TFLOP/s, 3.7 TB/s per TC).

| Component | Measured (ms) | Limit (ms) | % | Bound (LLO / profile) |
|:--|--:|--:|--:|:--|
| Splash dkv | 120.4 | 44.7 at full MXU | -- | MXU: critical path >99% bf16 MXU, a third masked K=64 contractions |
| Splash fwd | 54.0 | 16.8 at full MXU | -- | MXU: critical path >99% MXU (P already fed as bf16) |
| Matmul dW | 49.0 | 37.6 (41.8 at the ~90% tile ceiling) | 77% | compute |
| Matmul dX (+ fused LN-bwd and SwiGLU-bwd epilogues) | 57.4 | 37.6 | 65% | one LN-bwd fusion is HBM-bound (17 ms), but splitting it is slower |
| Matmul fwd (+ fused LN prologue, SwiGLU epilogue) | 53.4 | 37.6 | 70% | QKV at 62% (LN1 recompute) |
| `qk_norm_rope` fwd / bwd | 9.9 / 14.1 | 6.6 / 8.1 (HBM) | 67% / 57% | lane rotates (86% of the critical path) |
| Residual adds, `di`, LN stats, relayout copies | ~20 | ~8 (HBM) | ~40% | HBM, partly non-native layouts |
| Optimizer, exposed collectives, loss | ~3.5 | ~0.6 | -- | negligible |

| Hypothesis | Result | Decision |
|:--|--:|:--|
| A: barrier on LayerNorm gradients (both / LN1 / LN2) | -1.3% / -0.7% / -1.1% | Dropped |
| B: forward barrier on LN1 (repeated) | +0.2% (+2.5 GiB) | **Adopted** (`esmc_barrier_input_layernorm`) |
| B': forward barrier on LN2 / both | -0.6% / +0.1% | Dropped |
| C: row sum via a ones lane in V | -- | Not built: the forward is MXU-bound, this adds MXU work |
| bf16 P for the forward P.V | identical output and time | Dropped (Mosaic already rounds P) |
| D: dkv vector-op audit | -- | Not pursued: dkv is MXU-bound, not VALU-bound |
| F: split-halves q/k lanes (one roll for `rotate_half`) | +0.45% | **Adopted** (`esmc_qk_split_halves`) |

**Final:** 86.1k tok/s/TC, 33.4% MFU; 97.4k / 37.7% with p-state 7. 600-step
loss 2.5483 vs 2.5488 baseline.
