# AFD MoE profile reproduction

This branch supports two explicit simulation modes:

- Generic AIC: use the configured SGLang operation database for both AGG and AFD.
- Measured MoE: replace only an exact matching MoE-stage point in both arms and retain AIC attention, router, memory, MTP progress, and uncovered layer work.

Measured lookup never interpolates. The full key is model, system, stage,
topology, logical batch per source rank, MTP `nextn`, microbatch count, MoE
layer count, precision, and backend. A B200 point therefore cannot calibrate a
GB200 simulation, and a MegaMoE point cannot calibrate a DeepEP+DeepGEMM arm.

## Where the measured values live

`afd_moe_time_ms` is a single-run SDK/CLI override; it is not the calibration
database. The reproducible path is:

1. A kernel measurement producer writes one raw JSON result per exact workload.
2. Its qualification step validates those results and exports
   `afd_moe_stage_profile.json`.
3. This AIC branch loads that profile with `--afd-moe-profile` and records the
   exact matched entry in every retained AGG or AFD row.

The current colocated MegaMoE path exports
`moe_precision=w4a8_mxfp4_mxfp8` (E2M1 plus UE8M0 block-32 weights and E4M3
activations). It must not be used to calibrate an `nvfp4` candidate.

The measured profile can be loaded to audit its schema and provenance. Select
the simulation hardware with `--system`; its node width comes from the AIC
system specification. A GB200 sweep rejects a B200 profile by design, so a
GB200 measured-only run requires a profile whose entries use `system=gb200`.
The topology must match as well. For example, the current single-node B200
`4A4F` and `2A6F` split measurements are evidence only for a `b200_sxm` AFD
sweep because that AIC system has 8 GPUs per node and therefore starts at an
`8A8F` service unit. Its colocated `ep8` entries remain exact AGG matches.

For AGG, the measured source-rank batch is
`agg_local_batch / attention_tp`, because `agg_local_batch` is per attention-DP
replica and the MoE EP stage consumes the tokens distributed across its source
GPU ranks. A measured point is not used when that division is non-integral.
For AFD, `batch_per_a_gpu` is already the source-rank batch.

## 1. Environment

```bash
git checkout pr1323-afd-moe-eval
uv sync --extra dev
git lfs pull
```

`git lfs pull` is required for HYBRID/default performance database queries.

## 2. Validate a measured profile

The measurement pipeline exports `aic.afd-moe-stage-profile.v2` JSON. Version 2
requires `moe_backend` in every exact key. Version 1 remains readable and is
interpreted as legacy MegaMoE-only data. Loading the file is a strict
validation step:

```bash
uv run python -c \
  'from aiconfigurator.sdk.afd_moe_profile import AFDMoEStageProfile; AFDMoEStageProfile.load("/path/to/afd_moe_stage_profile.json")'
```

AGG entries must have stable timing and a passing matched-output check. Measured
speedup fields are evidence, not an admission filter: a valid slower backend is
kept so backend comparisons are not biased. AFD entries must be stable and
carry the paired validation evidence emitted by the measurement pipeline.

## 3. Run the fixed-pool sweep

Generic AIC SGLang/FlashInfer/TensorRT-LLM MoE control:

```bash
uv run python tools/afd_multimodel_mtp_experiment.py \
  --output /path/to/generic_sweep.json \
  --system gb200 \
  --models qwen3_235b minimax_m25 minimax_m3 deepseek_v4_flash deepseek_v4_pro \
  --workloads 8k 16k \
  --total-gpus 16 24 36 48 72 \
  --profile-scope all \
  --moe-backends trtllm
```

Use exact MegaMoE points and keep an explicitly labeled generic fallback when
a key is absent:

```bash
uv run python tools/afd_multimodel_mtp_experiment.py \
  --output /path/to/prefer_measured_sweep.json \
  --system gb200 \
  --models qwen3_235b minimax_m25 minimax_m3 deepseek_v4_flash deepseek_v4_pro \
  --workloads 8k 16k \
  --total-gpus 16 24 36 48 72 \
  --profile-scope all \
  --moe-backends megamoe \
  --afd-moe-profile /path/to/afd_moe_stage_profile.json
```

For an apples-to-apples measured-MoE comparison, drop every candidate without
an exact point:

```bash
uv run python tools/afd_multimodel_mtp_experiment.py \
  --output /path/to/measured_only_sweep.json \
  --system gb200 \
  --models qwen3_235b minimax_m25 minimax_m3 deepseek_v4_flash deepseek_v4_pro \
  --workloads 8k 16k \
  --total-gpus 16 24 36 48 72 \
  --profile-scope all \
  --moe-backends megamoe \
  --afd-moe-profile /path/to/afd_moe_stage_profile.json \
  --require-measured-moe
```

Run the same measured-only command with
`--moe-backends deepep_deepgemm` for DeepEP+DeepGEMM. The committed B200
profile currently contains colocated AGG points for that backend but no split
AFD points, so it cannot yet produce a paired DeepEP AFD comparison. The
measured-only policy drops those missing arms instead of substituting another
backend.

The values passed to `--total-gpus` are fixed comparison-pool sizes. For AFD,
the sweep independently evaluates every node-aligned service-unit size from
two nodes through the largest selected pool, then the report packs identical
units into each fixed pool and charges any idle remainder. On `gb200`, whose
system specification has 4 GPUs per node, an exact `4A4F` measurement can
therefore participate in a 16/24/36/48/72-GPU comparison. On `b200_sxm`, whose
node width is 8, the smallest AFD topology is `8A8F`. AGG is optimized directly
at each fixed pool size, including all valid node-aligned worker sizes.

Each row records `moe_measurement.used`, the exact lookup key, measured
latency, generic residual, source commit/tree hash, and backend contract. The
residual is limited to MTP auxiliary layer-equivalents and decoder layers not
covered by the measured MoE boundary.

## 4. Render the self-contained HTML report

```bash
uv run python tools/render_afd_multimodel_mtp_report.py \
  --sweep /path/to/measured_only_sweep.json \
  --moe-reference-profile /path/to/afd_moe_stage_profile.json \
  --moe-reference-url https://example.invalid/measured-profile-provenance \
  --output-dir /path/to/measured_report \
  --speed-floor 30
```

Open `/path/to/measured_report/index.html`. Every chart is inline SVG, so the
report directory has no external image dependency. Model pages list the exact
measured MoE-stage key, latency, residual AIC work, validation speedup bound,
and source commit used by each selected point.

`--moe-reference-profile` is also valid when rendering a generic GB200 sweep
from the qualified B200 profile. In that case the report places the B200
latency and speedup ranges in a separate evidence table and explicitly records
that zero B200 points were injected into the GB200 system simulation.

## 5. Optional Dynamo Mocker replay

Mocker does not predict kernel time. It replays selected AIC service points to
check worker count, concurrency, scheduling, and the throughput/latency
accounting used by the report.

```bash
uv run python tools/afd_multimodel_mtp_mocker_replay.py \
  --sweep /path/to/measured_only_sweep.json \
  --dynamo /path/to/dynamo \
  --output-dir /path/to/mocker_replay \
  --models qwen3_235b minimax_m25 minimax_m3 deepseek_v4_flash deepseek_v4_pro \
  --workloads 8k 16k \
  --total-gpus 16 24 36 48 72
```

The default one-wave run intentionally includes startup, drain, and the
stochastic final-request tail. Its output throughput is therefore not expected
to equal AIC's saturated steady-state rate, especially with MTP. To validate
steady-state convergence for one selected point, use a longer output and many
waves:

```bash
uv run python tools/afd_multimodel_mtp_mocker_replay.py \
  --sweep /path/to/measured_only_sweep.json \
  --dynamo /path/to/dynamo \
  --output-dir /path/to/mocker_steady_state \
  --models qwen3_235b --workloads 8k --total-gpus 16 \
  --output-tokens 128 --waves 64
```

Compare `mean_tpot_ms` with `expected_steady_state_tpot_ms` first. The JSON
field `finite_wave_efficiency_vs_aic_steady_state` separately shows how much
fill/drain remains in the finite replay.

Pass one or more generated summaries back to the renderer to include the
accounting audit in the index without embedding NPZ files or per-case logs:

```bash
uv run python tools/render_afd_multimodel_mtp_report.py \
  --sweep /path/to/sweep.json \
  --mocker-summary /path/to/mocker_replay/mocker_summary.json \
                   /path/to/mocker_steady_state/mocker_summary.json \
  --output-dir /path/to/report --speed-floor 30
```

## 6. Tests

```bash
uv run ruff check src/aiconfigurator/sdk/afd_moe_profile.py \
  src/aiconfigurator/sdk/inference_session.py \
  tools/afd_multimodel_mtp_experiment.py
uv run pytest -m unit tests/unit/sdk/test_afd_moe_profile.py \
  tests/unit/cli/test_afd_phase_completion.py
```
