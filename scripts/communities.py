"""Stage 1 of the tracking algorithm: static community detection per snapshot.

Louvain is run independently on each snapshot graph. Cross-snapshot identity is
established afterwards, in `tracking.py` -- Louvain itself has no memory between
snapshots and its community labels are not comparable across them.
"""

import numpy as np
from networkx.algorithms.community import louvain_communities

from . import config


def detect_communities(G, resolution=None, seed=None, min_size=None):
    """Louvain community detection on one weighted snapshot graph.

    Isolated nodes (no edges) are excluded -- they are not part of the active
    network for this snapshot. Communities smaller than `min_size` are dropped
    to avoid tracking noise from weakly connected pairs.

    Unset arguments fall back to `config` at call time, not at import time, so a
    notebook can override a study parameter and re-run without reimporting.
    """
    resolution = config.LOUVAIN_RESOLUTION if resolution is None else resolution
    seed = config.LOUVAIN_SEED if seed is None else seed
    min_size = config.MIN_COMMUNITY_SIZE if min_size is None else min_size

    if G.number_of_edges() == 0:
        return []
    parts = louvain_communities(G, weight='weight', resolution=resolution, seed=seed)
    return [frozenset(p) for p in parts if len(p) >= min_size]


def detect_all(snapshots, resolution=None, seed=None, min_size=None, verbose=True):
    """Run `detect_communities` over every snapshot."""
    snapshot_communities = [detect_communities(G, resolution, seed, min_size)
                            for G in snapshots]
    if verbose:
        counts = [len(c) for c in snapshot_communities]
        print('Community counts per snapshot:', counts)
        print('Median # communities:', int(np.median(counts)) if counts else 0)
    return snapshot_communities
