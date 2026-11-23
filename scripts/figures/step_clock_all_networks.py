"""Import this module first in a step-clock script; it changes the loaders imported after it.

1. All 512 networks of each optimizer enter (none is left out for not grokking).
2. The field "age" of the push loaders holds the training step of the push, not the steps from the onset.
Reads and writes nothing itself. Usage: `import step_clock_all_networks` before the other project imports.
"""
import contextlib
import io

import sgld_512_grokked

sgld_512_grokked.not_grokked = lambda: []          # read by every script imported after this line

with contextlib.redirect_stdout(io.StringIO()):
    import adamw_fullbatch_512_kicks as K

K.sk.not_grokked = lambda: []
_adamw, _sgld = K.load_adamw, K.sk.load


def load_adamw(h="0.1", every=True):
    D, delays, chunks, dropped = _adamw(h, every=True)
    for L in D:
        D[L]["age"] = D[L]["kstep"].astype(float)
    return D, delays, chunks, dropped


def load_sgld(suffix="", filtered=False):
    data, taus, fields = _sgld(suffix, filtered)
    for L in data:
        data[L]["age"] = data[L]["kstep"].astype(float)
    return data, taus, fields


K.load_adamw, K.sk.load = load_adamw, load_sgld
