"""Counting records inside many boxes at once (plan Algorithm 4, step 1).

Records are stored as packed bit masks: for every feature j and every cell index a, one mask of the
records with c_j >= a and one of the records with c_j <= a. The records inside a box are the AND of
at most two masks per constrained feature, and n_B and k_B are bit counts. Boxes are processed in
batches. This gives the same n_B and k_B as a row-by-row membership test.
"""
from __future__ import annotations

import numpy as np

_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount_table(words: np.ndarray) -> np.ndarray:
    """Bit counts per row of a uint64 array, by a 256-entry table (any NumPy version)."""
    b = words.view(np.uint8).reshape(words.shape[0], -1)
    return _TABLE[b].sum(axis=1, dtype=np.int64)


def _popcount(words: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(words).sum(axis=1, dtype=np.int64)
    return _popcount_table(words)


def _pack(mask: np.ndarray, n_words: int) -> np.ndarray:
    bits = np.packbits(mask.astype(np.uint8), bitorder="little")
    out = np.zeros(n_words * 8, dtype=np.uint8)
    out[: len(bits)] = bits
    return out.view(np.uint64)


def box_counts(keys, box_class, C, y=None, batch: int = 256):
    """n_B (records inside) and k_B (records whose outcome differs from the box class) for each key.

    keys: list of ((feature, a, b), ...) with a <= c_j <= b; box_class: class per box;
    C: cell indices of the records; y: outcomes of the records, or None to count only n_B.
    """
    C = np.asarray(C)
    n_rec, d = C.shape
    n_words = max(1, (n_rec + 63) // 64)
    kmax = C.max(axis=0) if n_rec else np.zeros(d, dtype=np.int64)
    used = sorted({(j, a, b) for key in keys for (j, a, b) in key})
    rows, index = [np.full(n_words, np.uint64(0xFFFFFFFFFFFFFFFF))], {}
    if n_rec % 64:
        rows[0][-1] = np.uint64((1 << (n_rec % 64)) - 1)
    full = rows[0]
    for j, a, b in used:
        if a > 0 and ("ge", j, a) not in index:
            index[("ge", j, a)] = len(rows); rows.append(_pack(C[:, j] >= a, n_words) & full)
        if b < kmax[j] and ("le", j, b) not in index:
            index[("le", j, b)] = len(rows); rows.append(_pack(C[:, j] <= b, n_words) & full)
    T = np.stack(rows)
    side_lists = []
    for key in keys:
        s = []
        for j, a, b in key:
            if a > 0:
                s.append(index[("ge", j, a)])
            if b < kmax[j]:
                s.append(index[("le", j, b)])
        side_lists.append(s)
    width = max([1] + [len(s) for s in side_lists])
    idx = np.zeros((len(keys), width), dtype=np.int64)       # row 0 is all records, the neutral element
    for i, s in enumerate(side_lists):
        idx[i, : len(s)] = s
    ypos = None if y is None else _pack(np.asarray(y).astype(bool), n_words)
    n_out = np.zeros(len(keys), dtype=np.int64)
    pos_out = np.zeros(len(keys), dtype=np.int64)
    for s in range(0, len(keys), batch):
        M = np.bitwise_and.reduce(T[idx[s:s + batch]], axis=1)
        n_out[s:s + batch] = _popcount(M)
        if ypos is not None:
            pos_out[s:s + batch] = _popcount(M & ypos)
    if y is None:
        return n_out, None
    box_class = np.asarray(box_class)
    k_out = np.where(box_class == 1, n_out - pos_out, pos_out)
    return n_out, k_out
