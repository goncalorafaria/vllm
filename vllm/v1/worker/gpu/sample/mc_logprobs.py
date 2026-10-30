# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent categorical probes, with replacement, from processed logits."""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.sample.compact_logprobs import _partials


@triton.jit
def _draw(
    X,
    Sample,
    Max,
    Sum,
    Rank,
    Seeds,
    Mapping,
    Positions,
    OutIds,
    OutLP,
    OutRank,
    V: tl.constexpr,
    STRIDE: tl.constexpr,
    TILES: tl.constexpr,
    REDUCE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, draw = tl.program_id(0), tl.program_id(1)
    t = tl.arange(0, REDUCE)
    m = tl.load(Max + row * TILES + t, t < TILES, other=-float("inf"))
    z = tl.load(Sum + row * TILES + t, t < TILES, other=0.0)
    maximum = tl.max(m, 0)
    mass = z * tl.exp(m - maximum)
    cdf = tl.cumsum(mass, 0)
    total = tl.sum(mass, 0)
    req = tl.load(Mapping + row)
    seed = tl.load(Seeds + req).to(tl.uint64) ^ 0x6A09E667F3BCC909
    pos = tl.load(Positions + row).to(tl.uint32)
    # A separate Philox domain leaves the response token's RNG untouched.
    u = tl.rand(seed, pos * 32 + draw)
    cdf_total = tl.sum(tl.where(t == TILES - 1, cdf, 0.0), 0)
    target = tl.minimum(u * cdf_total, cdf_total * (1.0 - 1.1920928955078125e-7))
    tile = tl.min(tl.where((cdf > target) & (mass > 0), t, TILES), 0)
    before = tl.sum(tl.where(t == tile - 1, cdf, 0.0), 0)
    tile_m = tl.sum(tl.where(t == tile, m, 0.0), 0)
    # Avoid 0 * -inf when reducing masked empty tiles.
    local_scale = tl.exp(tile_m - maximum)
    j = tl.arange(0, BLOCK)
    ids = tile * BLOCK + j
    x = tl.load(X + row * STRIDE + ids, ids < V, other=-float("inf")).to(tl.float32)
    weights = tl.exp(x - tile_m)
    local_cdf = tl.cumsum(weights, 0)
    local_total = tl.sum(tl.where(j == BLOCK - 1, local_cdf, 0.0), 0)
    local_target = tl.minimum(
        tl.maximum((target - before) / local_scale, 0.0),
        local_total * (1.0 - 1.1920928955078125e-7),
    )
    token = tl.min(tl.where((local_cdf > local_target) & (weights > 0), ids, V), 0)
    value = tl.load(X + row * STRIDE + token).to(tl.float32)
    logz = maximum + tl.log(total)
    tl.store(OutIds + row * 33 + draw + 1, token)
    tl.store(OutLP + row * 33 + draw + 1, value - logz)
    if draw == 0:
        sampled = tl.load(Sample + row)
        selected = tl.load(X + row * STRIDE + sampled).to(tl.float32)
        r = tl.load(Rank + row * TILES + t, t < TILES, other=0)
        tl.store(OutIds + row * 33, sampled)
        tl.store(OutLP + row * 33, selected - logz)
        tl.store(OutRank + row, tl.sum(r, 0))


def compute_mc_scores(logits, sampled_token_ids, seeds, mapping, positions, block=4096):
    """Return response token plus 32 IID probes; duplicate IDs are meaningful.

    Inputs must have at least one finite logit per row and no NaNs/+inf.
    Sampling is categorical up to FP32/PRNG precision. Memory is O(B * V/block),
    not O(B * V * 32); normalization is exact over the processed support.
    """
    batch, vocab = logits.shape
    if logits.stride(1) != 1 or sampled_token_ids.numel() != batch:
        raise ValueError(
            "MC32 requires contiguous vocabulary rows and one token per row"
        )
    tiles = triton.cdiv(vocab, block)
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
    ids = torch.empty((batch, 33), device=logits.device, dtype=torch.int64)
    lp = torch.empty((batch, 33), device=logits.device, dtype=torch.float32)
    rank = torch.empty(batch, device=logits.device, dtype=torch.int64)
    _draw[(batch, 32)](
        logits,
        sampled_token_ids,
        maxima,
        sums,
        ranks,
        seeds,
        mapping,
        positions,
        ids,
        lp,
        rank,
        vocab,
        logits.stride(0),
        tiles,
        triton.next_power_of_2(tiles),
        block,
    )
    return LogprobsTensors(ids, lp, rank)
