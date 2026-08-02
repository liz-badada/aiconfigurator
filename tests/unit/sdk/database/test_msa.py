# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax Sparse Attention (MSA) op: SOL and the cross-op (XOP) transfer from DSA.

MSA has no own silicon data, so its empirical path is a cross-op transfer gated by the
XOP transfer kind. These tests cover the SOL path and the XOP gate (the headline new op
otherwise had no coverage)."""

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.errors import EmpiricalNotImplementedError

pytestmark = pytest.mark.unit


def _ctx_msa():
    from aiconfigurator.sdk.operations.msa import ContextMSAModule

    # M3-like per-GPU shape: 8 q / 1 kv heads, head_dim 128, v 128, top-16 blocks * 128.
    return ContextMSAModule(
        "msa",
        1.0,
        num_heads=8,
        num_kv_heads=1,
        hidden_size=4096,
        head_dim=128,
        v_head_dim=128,
        index_n_heads=4,
        index_head_dim=128,
        index_topk=2048,
        block_size=128,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.fp8,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
    )


def _gen_msa():
    from aiconfigurator.sdk.operations.msa import GenerationMSAModule

    return GenerationMSAModule(
        "msa",
        1.0,
        num_heads=8,
        num_kv_heads=1,
        hidden_size=4096,
        head_dim=128,
        v_head_dim=128,
        index_n_heads=4,
        index_head_dim=128,
        index_topk=2048,
        block_size=128,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.fp8,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
    )


def test_msa_sol_scales_with_workload(comprehensive_perf_db):
    """SOL mode computes the three-group MSA SOL (gemm + fp8 indexer + sparse attn). Assert it
    RESPONDS to the workload rather than returning a constant: more new tokens (s) add work, and
    a longer cached prefix adds indexer/attention work (full_s > index_topk)."""
    comprehensive_perf_db.set_default_database_mode(common.DatabaseMode.SOL)
    try:
        op = _ctx_msa()
        small = float(op.query(comprehensive_perf_db, batch_size=8, s=512, prefix=0))
        large = float(op.query(comprehensive_perf_db, batch_size=8, s=2048, prefix=0))
        with_prefix = float(op.query(comprehensive_perf_db, batch_size=8, s=2048, prefix=2048))
        assert 0 < small < large  # scales with new-token count
        assert with_prefix > large  # cached prefix adds indexer work beyond index_topk
    finally:
        comprehensive_perf_db.set_default_database_mode(common.DatabaseMode.SILICON)


def test_msa_xop_gating(comprehensive_perf_db, monkeypatch):
    """The DSA-to-MSA utilization transfer is gated and tagged as XOP."""
    from aiconfigurator.sdk.operations import util_empirical

    util_queries = []

    def dsa_util(_database, **kwargs):
        util_queries.append(kwargs)
        return 0.5

    monkeypatch.setattr("aiconfigurator.sdk.operations.msa._dsa_context_util", dsa_util)
    comprehensive_perf_db.set_default_database_mode(common.DatabaseMode.HYBRID)
    kw = dict(batch_size=8, s=2048, prefix=0)
    try:
        comprehensive_perf_db.set_transfer_policy(["xshape", "xquant"])  # no XOP
        with pytest.raises(EmpiricalNotImplementedError) as exc:
            _ctx_msa().query(comprehensive_perf_db, **kw)
        assert "xop" in str(exc.value).lower()  # gated at the policy, not a data miss
        assert util_queries == []

        comprehensive_perf_db.set_transfer_policy(None)  # XOP allowed
        with util_empirical.capture_provenance() as tags:
            assert float(_ctx_msa().query(comprehensive_perf_db, **kw)) > 0
        assert len(util_queries) == 1
        assert util_empirical.worst_provenance(tags) == "xop"
    finally:
        comprehensive_perf_db.set_transfer_policy(None)
        comprehensive_perf_db.set_default_database_mode(common.DatabaseMode.SILICON)


def test_msa_generation_verification_reuses_shared_prefix(comprehensive_perf_db):
    from aiconfigurator.sdk.operations.msa import _msa_attention_sol

    comprehensive_perf_db.set_default_database_mode(common.DatabaseMode.SOL)
    try:
        op = _gen_msa()
        one_query = float(op.query(comprehensive_perf_db, batch_size=8, s=8192, query_len=1))
        four_queries = float(op.query(comprehensive_perf_db, batch_size=8, s=8192, query_len=4))
        assert 1 < four_queries / one_query < 4.01

        sol_kwargs = {
            "database": comprehensive_perf_db,
            "is_context": False,
            "s": 8192,
            "prefix": 0,
            "num_heads": op._num_heads,
            "num_kv_heads": op._num_kv_heads,
            "hidden_size": op._hidden_size,
            "head_dim": op._head_dim,
            "v_head_dim": op._v_head_dim,
            "index_n_heads": op._index_n_heads,
            "index_head_dim": op._index_head_dim,
            "index_topk": op._index_topk,
            "block_size": op._block_size,
            "kvcache_quant_mode": op._kvcache_quant_mode,
            "fmha_quant_mode": op._fmha_quant_mode,
            "gemm_quant_mode": op._gemm_quant_mode,
        }
        fused_mem = _msa_attention_sol(**sol_kwargs, b=8, query_len=4)[2]
        independent_mem = _msa_attention_sol(**sol_kwargs, b=32, query_len=1)[2]
        assert fused_mem < independent_mem
    finally:
        comprehensive_perf_db.set_default_database_mode(common.DatabaseMode.SILICON)
