# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NumPy-backed MC32 storage; never deduplicate independent draws."""

import numpy as np

from vllm.logprobs import FlatLogprobs


class MCLogprobs(FlatLogprobs):
    def __init__(self):
        self._size = 0
        self._ids = np.empty((64, 33), dtype=np.int32)
        self._lp = np.empty((64, 33), dtype=np.float32)
        self._ranks = np.empty(64, dtype=np.int64)

    def append_arrays(self, ids, logprobs, rank):
        ids, logprobs = np.asarray(ids), np.asarray(logprobs)
        if ids.shape != (33,) or logprobs.shape != (33,):
            raise ValueError("MC32 requires response token plus exactly 32 probes")
        if self._size == len(self._ids):
            capacity = max(64, 2 * self._size)
            for name, shape, dtype in (
                ("_ids", (capacity, 33), np.int32),
                ("_lp", (capacity, 33), np.float32),
                ("_ranks", (capacity,), np.int64),
            ):
                old = getattr(self, name)
                new = np.empty(shape, dtype=dtype)
                new[: self._size] = old[: self._size]
                setattr(self, name, new)
        self._ids[self._size] = ids
        self._lp[self._size] = logprobs
        self._ranks[self._size] = rank
        self._size += 1

    @property
    def start_indices(self):
        return np.arange(self._size, dtype=np.int64) * 33

    @property
    def end_indices(self):
        return self.start_indices + 33

    @property
    def token_ids(self):
        return self._ids[: self._size].reshape(-1)

    @property
    def logprobs(self):
        return self._lp[: self._size].reshape(-1)

    @property
    def ranks(self):
        ranks = np.zeros((self._size, 33), dtype=np.int64)
        ranks[:, 0] = self._ranks[: self._size]
        return ranks.reshape(-1)

    @property
    def decoded_tokens(self):
        return np.full(self._size * 33, None, dtype=object)

    def __getitem__(self, index):
        if not isinstance(index, slice):
            raise TypeError("MC probes cannot be converted to token-keyed dictionaries")
        result = MCLogprobs()
        result._ids = self._ids[: self._size][index].copy()
        result._lp = self._lp[: self._size][index].copy()
        result._ranks = self._ranks[: self._size][index].copy()
        result._size = len(result._ids)
        return result
