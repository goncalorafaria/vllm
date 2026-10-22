# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compact vLLM logprobs with parallel normalization and sampled-token ranks."""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors


@triton.jit
def _partials(
    X,
    Sample,
    Max,
    Sum,
    Rank,
    V: tl.constexpr,
    STRIDE: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    j = tl.arange(0, BLOCK)
    ids = tile * BLOCK + j
    x = tl.load(X + row * STRIDE + ids, ids < V, other=-float("inf")).to(tl.float32)
    selected = tl.load(Sample + row)
    selected_x = tl.load(X + row * STRIDE + selected).to(tl.float32)
    m = tl.max(x, 0)
    z = tl.sum(tl.exp(x - tl.maximum(m, -3.402823466e38)), 0)
    rank = tl.sum(((x >= selected_x) & (ids < V)).to(tl.int32), 0)
    offset = row * TILES + tile
    tl.store(Max + offset, m)
    tl.store(Sum + offset, z)
    tl.store(Rank + offset, rank)


@triton.jit
def _finish_dense(
    X,
    Sample,
    Max,
    Sum,
    Rank,
    TopIds,
    OutIds,
    OutLP,
    OutRank,
    STRIDE: tl.constexpr,
    TILES: tl.constexpr,
    K: tl.constexpr,
    REDUCE: tl.constexpr,
    HEAD: tl.constexpr,
):
    row = tl.program_id(0)
    t = tl.arange(0, REDUCE)
    m = tl.load(Max + row * TILES + t, t < TILES, other=-float("inf"))
    z = tl.load(Sum + row * TILES + t, t < TILES, other=0.0)
    r = tl.load(Rank + row * TILES + t, t < TILES, other=0)
    maximum = tl.max(m, 0)
    normalizer = tl.log(tl.sum(z * tl.exp(m - maximum), 0))
    sampled = tl.load(Sample + row)
    selected = tl.load(X + row * STRIDE + sampled).to(tl.float32)
    tl.store(OutIds + row * (K + 1), sampled)
    tl.store(OutLP + row * (K + 1), selected - maximum - normalizer)
    tl.store(OutRank + row, tl.sum(r, 0))
    if K > 0:
        j = tl.arange(0, HEAD)
        token = tl.load(TopIds + row * K + j, j < K, other=0)
        value = tl.load(X + row * STRIDE + token, j < K, other=-float("inf")).to(
            tl.float32
        )
        tl.store(OutIds + row * (K + 1) + 1 + j, token, j < K)
        tl.store(OutLP + row * (K + 1) + 1 + j, value - maximum - normalizer, j < K)


def compute_compact_scores(logits, num_logprobs, sampled_token_ids, block=4096):
    batch, vocab = logits.shape
    k = num_logprobs
    tiles = triton.cdiv(vocab, block)
    if k and vocab >= 4096 and logits.is_contiguous():
        from flashinfer import top_k

        # Extraction only: these IDs never restrict the sampling distribution.
        top_ids = top_k(logits, k, sorted=True, deterministic=True)[1]
    else:
        top_ids = torch.topk(logits, k, dim=-1).indices if k else sampled_token_ids
    maxima = torch.empty((batch, tiles), device=logits.device, dtype=torch.float32)
    sums = torch.empty_like(maxima)
    ranks = torch.empty((batch, tiles), device=logits.device, dtype=torch.int32)
    _partials[(batch, tiles)](
        logits,
        sampled_token_ids,
        maxima,
        sums,
        ranks,
        vocab,
        logits.stride(0),
        tiles,
        block,
    )
    out_ids = torch.empty((batch, k + 1), device=logits.device, dtype=torch.int64)
    out_lp = torch.empty((batch, k + 1), device=logits.device, dtype=torch.float32)
    out_rank = torch.empty((batch,), device=logits.device, dtype=torch.int64)
    _finish_dense[(batch,)](
        logits,
        sampled_token_ids,
        maxima,
        sums,
        ranks,
        top_ids,
        out_ids,
        out_lp,
        out_rank,
        logits.stride(0),
        tiles,
        k,
        triton.next_power_of_2(tiles),
        triton.next_power_of_2(max(k, 1)),
    )
    return LogprobsTensors(out_ids, out_lp, out_rank)
