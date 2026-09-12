"""Snapshot construction: the channel-channel commenter-overlap network.

Each snapshot is one non-overlapping `WINDOW_DAYS` window. Two channels are
joined by an edge when at least one person commented on both inside that
window; the edge weight is the symmetric overlap ratio

    w(a, b) = 0.5 * (|A n B| / |A| + |A n B| / |B|)

which is bounded in [0, 1] and does not simply track channel size, unlike the
raw shared-commenter count (available as `weight_mode='raw'`).
"""

from collections import defaultdict
from datetime import datetime, timedelta

import networkx as nx
import numpy as np

from . import config


def build_snapshot(daily_authors, channel_ids, end_date, window_days,
                   min_weight, weight_mode=None):
    """Build one weighted snapshot graph ending on (and including) `end_date`.

    Channels with no edges in this window are excluded from the snapshot: they
    had no meaningful commenter overlap with anyone, so they are not part of the
    network for this period.
    """
    weight_mode = config.NETWORK_WEIGHT_MODE if weight_mode is None else weight_mode
    start_date = end_date - timedelta(days=window_days - 1)

    # Aggregate author sets per channel across the window
    win_authors = defaultdict(set)
    d = start_date
    while d <= end_date:
        if d in daily_authors:
            for cid, auths in daily_authors[d].items():
                win_authors[cid].update(auths)
        d += timedelta(days=1)

    G = nx.Graph()
    for i in range(len(channel_ids)):
        for j in range(i + 1, len(channel_ids)):
            a, b = channel_ids[i], channel_ids[j]
            Aset, Bset = win_authors.get(a, set()), win_authors.get(b, set())
            if not Aset or not Bset:
                continue
            inter = len(Aset & Bset)
            if inter == 0:
                continue
            if weight_mode == 'raw':
                w = float(inter)
            else:
                w = 0.5 * (inter / len(Aset) + inter / len(Bset))
            if w >= min_weight:
                G.add_edge(a, b, weight=w)

    return G


def build_snapshots(daily_authors, channel_ids, start_date=None, end_date=None,
                    window_days=None, step_days=None, min_weight=None,
                    weight_mode=None, verbose=True):
    """Step a window across the study period, one snapshot graph per position.

    Returns `(snapshot_dates, snapshots)` where `snapshot_dates[t]` is the first
    day of snapshot `t`. A trailing partial window is dropped, so every snapshot
    covers the same number of days.

    Unset arguments fall back to `config` at call time, not at import time, so a
    notebook can override a study parameter and re-run without reimporting.
    """
    start_date = config.START_DATE if start_date is None else start_date
    end_date = config.END_DATE if end_date is None else end_date
    window_days = config.WINDOW_DAYS if window_days is None else window_days
    step_days = config.STEP_DAYS if step_days is None else step_days
    min_weight = config.MIN_EDGE_WEIGHT if min_weight is None else min_weight
    weight_mode = config.NETWORK_WEIGHT_MODE if weight_mode is None else weight_mode

    start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()

    snapshot_dates, snapshots = [], []
    window_start = start_dt
    while window_start + timedelta(days=window_days - 1) <= end_dt:
        window_end = window_start + timedelta(days=window_days - 1)
        snapshots.append(build_snapshot(daily_authors, channel_ids, window_end,
                                        window_days, min_weight, weight_mode))
        snapshot_dates.append(window_start)
        window_start += timedelta(days=step_days)

    if verbose and snapshots:
        print(f'Built {len(snapshots)} snapshots from {snapshot_dates[0]} to {snapshot_dates[-1]}')
        print('Median edges per snapshot:',
              int(np.median([G.number_of_edges() for G in snapshots])))
        print('Median nodes per snapshot:',
              int(np.median([G.number_of_nodes() for G in snapshots])))

    return snapshot_dates, snapshots
