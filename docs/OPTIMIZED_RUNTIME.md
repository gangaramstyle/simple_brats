# Optimized A40 runtime

The optimized runtime changes execution, not the registered SSL experiment. It must preserve the
pinned manifest/split/case-grid inputs, subject-balanced absolute-step schedule, materialized patch
identities, hard symmetric matching objective, target-modality exclusion, modality-specific source
tokens, blind teacher inputs, model architecture, optimizer hyperparameters, and checkpoint/evaluation
contracts.

## Runtime changes

- prepare the exact upcoming scheduled cases with eight background workers and a 16-case ordered
  prefetch queue, overlap its cold start with compilation, retain a readiness barrier before the
  first optimizer step, then replenish exactly four keys per low-watermark refill to avoid
  continuous CPU contention;
- use the rotating cache only as storage: cache residency never chooses a subject, visit, or bag;
- keep verified canonical volumes in a bounded GPU cache and batch patch extraction by modality;
- vectorize the 512-candidate geometric conflict table while preserving the reference planner's
  candidate order and selected patch identities;
- run the forward path under CUDA bf16 autocast and `torch.compile`;
- use fused CUDA AdamW with the same parameter groups and hyperparameters;
- move full fixed-probe/SVD diagnostics and W&B writes to a registered cadence while retaining
  per-step loss, accuracy, non-finite-gradient failure, checkpoint, and wall-time guarantees;
- have the prelaunch A40 gate record synchronized steady-state timestamps, excluded compile and
  calibration time, cache hits/misses, prefetch stalls, successful/failed/running refill counts,
  compile counters, and peak CUDA allocator memory.

## Correctness gates

Before a scientific launch, the optimized path must demonstrate:

1. exact plan and identity-table equality with the reference path over cold, warm, and resumed steps;
2. CPU-reference versus batched-GPU patch agreement under a locked numeric tolerance;
3. no target pixels, target-modality overlap, labels, scan statistics, or coordinates entering the
   blind teacher;
4. fresh-process checkpoint resume equivalence under the compiled bf16/fused runtime;
5. identical absolute-step subject/case/bag assignments regardless of worker timing or cache size;
6. no train/validation/test subject-boundary change;
7. at least 2 optimizer steps/second over completed steps 65 through 160 on one A40.

The 160-step A40 gate crosses the startup-prefetch boundary: with eight bags per case, the first
case supplied by a post-startup refill is consumed at completed step 137. Its exact accounting is
one synchronous calibration stall followed by 19 ready case consumptions, 33 total submissions,
and 13 pending cases at the end. At least the first five pending keys must form a successful ready
prefix; failures, substitutions, and discarded keys are forbidden. The gate reports median, p95,
p99, maximum, and the ten slowest synchronized step intervals so GPFS tails remain inspectable, but
it applies no individual-interval cutoff. Only aggregate throughput over steps 65 through 160 is a
performance pass/fail criterion.

Because bf16, compilation, and fused optimizer kernels define a different numerical trajectory from
the existing float32 checkpoint, the optimized scientific run starts from step zero.
