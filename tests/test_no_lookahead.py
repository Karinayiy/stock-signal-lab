"""Leak tests: prove that no feature uses information from the future.

Idea: build the feature table twice, once from all the data and once from data
cut off at some date D. If every feature on or before D is identical in both
tables, then no feature could have looked past its own date. Run:

    .venv/bin/python tests/test_no_lookahead.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import FEATURES, build_panel  # noqa: E402
from fetch import load  # noqa: E402

CUTOFFS = ["2018-06-29", "2021-03-15", "2024-10-31"]


def truncated_inputs(close, volume, macro, earnings, cutoff):
    c = pd.Timestamp(cutoff)
    # earnings reports dated after the cutoff did not exist yet
    earn = {t: e[e.index.normalize() <= c] for t, e in earnings.items()}
    return close[close.index <= c], volume[volume.index <= c], macro[macro.index <= c], earn


def test_features_ignore_the_future():
    close, volume, macro, earnings, _ = load()
    full = build_panel(close, volume, macro, earnings).set_index(["date", "ticker"])
    for cutoff in CUTOFFS:
        part = build_panel(*truncated_inputs(close, volume, macro, earnings, cutoff)).set_index(["date", "ticker"])
        common = part.index.intersection(full.index)
        a = full.loc[common, FEATURES].astype(float)
        b = part.loc[common, FEATURES].astype(float)
        both_nan = a.isna() & b.isna()
        same = np.isclose(a.fillna(0), b.fillna(0), rtol=1e-9, atol=1e-12) | both_nan
        bad = (~same).sum()
        assert bad.sum() == 0, f"cutoff {cutoff}: features changed when future data was removed:\n{bad[bad > 0]}"
        print(f"ok  cutoff {cutoff}: {len(common):,} rows x {len(FEATURES)} features identical")


def test_targets_do_look_ahead():
    """Sanity check of the test itself: the *targets* must change when the future
    is removed (they are future returns). If they didn't, the test above would
    prove nothing."""
    close, volume, macro, earnings, _ = load()
    full = build_panel(close, volume, macro, earnings).set_index(["date", "ticker"])
    part = build_panel(*truncated_inputs(close, volume, macro, earnings, CUTOFFS[0])).set_index(["date", "ticker"])
    last_day = part.index.get_level_values("date").max()
    rows = part.xs(last_day, level="date")
    assert rows["fwd_ret_h"].isna().all(), "targets at the cutoff should be unknown"
    assert full.xs(last_day, level="date")["fwd_ret_h"].notna().all(), "full data should know them"
    print("ok  targets correctly depend on future data")


if __name__ == "__main__":
    test_features_ignore_the_future()
    test_targets_do_look_ahead()
    print("all leak tests passed")
