# AFD MoE profile reproduction

This branch supports two explicit simulation modes:

- Generic AIC: use the configured SGLang operation database for both AGG and AFD.
- Measured MoE: replace only an exact matching MoE-stage point in both arms and retain AIC attention, router, memory, MTP progress, and uncovered layer work.

Measured lookup never interpolates. The full key is model, system, stage,
topology, logical batch per source rank, MTP `nextn`, microbatch count, MoE
layer count, and precision. A B200 point therefore cannot calibrate a GB200
simulation.

## Where the measured values live

`afd_moe_time_ms` is a single-run SDK/CLI override; it is not the calibration
database. The reproducible path is:

1. FastAFD writes one raw JSON result per exact workload through
   `scripts/experiments/afd/run_megamoe_{colocated,m2n}_model_benchmark.sh`.
2. `scripts/experiments/afd/summarize_megamoe_model_results.py` validates those
   JSON files and exports `afd_moe_stage_profile.json`.
3. This AIC branch loads that profile with `--afd-moe-profile` and records the
   exact matched entry in every retained AGG or AFD row.

The current colocated MegaMoE path exports
`moe_precision=w4a8_mxfp4_mxfp8` (E2M1 plus UE8M0 block-32 weights and E4M3
activations). It must not be used to calibrate an `nvfp4` candidate.

The collection commands and qualification gates are in FastAFD
`scripts/experiments/afd/README_megamoe_multimodel.md` on branch
`megamoe-multimodel-b200`.

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

The measurement pipeline exports `aic.afd-moe-stage-profile.v1` JSON. Loading
the file is a strict validation step:

```bash
uv run python -c \
  'from aiconfigurator.sdk.afd_moe_profile import AFDMoEStageProfile; AFDMoEStageProfile.load("/path/to/afd_moe_stage_profile.json")'
```

AGG entries must have stable timing, a passing matched-output check, and
same-point speedup greater than one. AFD entries must be stable and carry the
paired validation evidence emitted by the measurement pipeline.

## 3. Run the fixed-pool sweep

Generic AIC only:

```bash
uv run python tools/afd_multimodel_mtp_experiment.py \
  --output /path/to/generic_sweep.json \
  --models qwen3_235b minimax_m25 minimax_m3 deepseek_v4_flash deepseek_v4_pro \
  --workloads 8k 16k \
  --total-gpus 16 24 36 48 72 \
  --profile-scope primary
```

Use exact measured points and keep an explicitly labeled generic fallback when
a key is absent:

```bash
uv run python tools/afd_multimodel_mtp_experiment.py \
  --output /path/to/prefer_measured_sweep.json \
  --models qwen3_235b minimax_m25 minimax_m3 deepseek_v4_flash deepseek_v4_pro \
  --workloads 8k 16k \
  --total-gpus 16 24 36 48 72 \
  --profile-scope primary \
  --afd-moe-profile /path/to/afd_moe_stage_profile.json
```

For an apples-to-apples measured-MoE comparison, drop every candidate without
an exact point:

```bash
uv run python tools/afd_multimodel_mtp_experiment.py \
  --output /path/to/measured_only_sweep.json \
  --models qwen3_235b minimax_m25 minimax_m3 deepseek_v4_flash deepseek_v4_pro \
  --workloads 8k 16k \
  --total-gpus 16 24 36 48 72 \
  --profile-scope primary \
  --afd-moe-profile /path/to/afd_moe_stage_profile.json \
  --require-measured-moe
```

Each row records `moe_measurement.used`, the exact lookup key, measured
latency, generic residual, source commit/tree hash, and backend contract. The
residual is limited to MTP auxiliary layer-equivalents and decoder layers not
covered by the measured MoE boundary.

## 4. Optional Dynamo Mocker replay

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

## 5. Tests

```bash
uv run ruff check src/aiconfigurator/sdk/afd_moe_profile.py \
  src/aiconfigurator/sdk/inference_session.py \
  tools/afd_multimodel_mtp_experiment.py
uv run pytest -m unit tests/unit/sdk/test_afd_moe_profile.py \
  tests/unit/cli/test_afd_phase_completion.py
```
