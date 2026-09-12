"""Critical (core) channels within each dynamic community (section 16g).

Two independently-ranked notions of "core", both scoped to the community's own
induced subgraph rather than the whole observed network:

* **Hub** -- weighted degree strength (and eigenvector centrality): channels
  whose audience overlaps heavily with the rest of the faction.
* **Broker** -- weighted betweenness: channels that many within-faction shortest
  paths route through. A broker can have modest degree strength and still be
  structurally load-bearing, connecting otherwise loosely-tied subgroups.

`plot_core_nodes` draws both rankings side by side, one row per faction: left
column the hub core, right column the brokers. They are deliberately not the
same ranking -- a broker can have modest degree strength and still be the
channel holding two sub-groups of the faction together, which is exactly the
structural role a degree ranking hides.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from . import config
from .cohesion import select_top_dcs, top_dc_label
from .style import DC_PALETTE, save_figure
from .tracking import dc_summary, dominant_wing

TITLE_TRUNCATE = 20


def community_core_scores(H):
    """Weighted degree strength and eigenvector centrality within subgraph `H`.

    Degree strength is normalised by `n - 1` so it is comparable across
    communities of different sizes. Eigenvector centrality is only defined on a
    connected graph, so it is NaN for a disconnected or edgeless community.
    """
    n = H.number_of_nodes()
    if n < 2:
        return {}, {}
    strength = dict(H.degree(weight='weight'))
    degree_strength = {node: s / (n - 1) for node, s in strength.items()}
    if H.number_of_edges() > 0 and nx.is_connected(H):
        try:
            eigen = nx.eigenvector_centrality(H, weight='weight', max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            eigen = {node: np.nan for node in H.nodes()}
    else:
        eigen = {node: np.nan for node in H.nodes()}
    return degree_strength, eigen


def community_betweenness(H):
    """Weighted betweenness centrality within subgraph `H`.

    `weight` here is a similarity (overlap ratio -- higher means closer), the
    opposite of the distance networkx's shortest-path routines expect, so
    betweenness is computed over `1 / weight`: stronger ties become shorter
    distances. `normalized=True` keeps scores comparable across community sizes.
    Small or edgeless subgraphs correctly yield all-zero betweenness -- there is
    no intermediary role to play either way.
    """
    n = H.number_of_nodes()
    if n < 2:
        return {}
    H_dist = H.copy()
    for _, _, d in H_dist.edges(data=True):
        d['distance'] = 1.0 / d['weight']
    return nx.betweenness_centrality(H_dist, weight='distance', normalized=True)


def core_nodes_table(dcs, snapshots, snapshot_dates, channel_info_map, verbose=True):
    """Score every (DC, snapshot, channel) and roll up to (DC, channel).

    Returns `(core_df, dc_core_nodes_df)`. The rolled-up frame is sorted so the
    top rows of each DC's block are its hub core; brokers are ranked separately
    by `mean_betweenness`.
    """
    rows = []
    for dc in dcs.values():
        wing = dominant_wing(dc.latest_front(), channel_info_map)
        for t, front in dc.fronts.items():
            H = snapshots[t].subgraph(front)
            degree_strength, eigen = community_core_scores(H)
            betweenness = community_betweenness(H)
            for cid in front:
                rows.append({'t': t, 'date': snapshot_dates[t], 'dc_id': dc.id,
                             'wing': wing, 'channel_id': cid,
                             'title': channel_info_map.get(cid, {}).get('title', cid),
                             'degree_strength': degree_strength.get(cid, np.nan),
                             'eigenvector': eigen.get(cid, np.nan),
                             'betweenness': betweenness.get(cid, np.nan)})

    core_df = pd.DataFrame(rows)

    dc_core_nodes_df = (core_df.groupby(['dc_id', 'channel_id'])
        .agg(title=('title', 'first'), wing=('wing', 'first'),
             n_snapshots_present=('t', 'size'),
             mean_degree_strength=('degree_strength', 'mean'),
             mean_eigenvector=('eigenvector', 'mean'),
             mean_betweenness=('betweenness', 'mean'))
        .reset_index()
        .sort_values(['dc_id', 'mean_degree_strength', 'mean_eigenvector'],
                     ascending=[True, False, False]))

    if verbose:
        print(f'Scored {len(dc_core_nodes_df)} (DC, channel) core-member rows '
              f'across {len(dcs)} DCs')

    return core_df, dc_core_nodes_df


def plot_core_nodes(dc_core_nodes_df, dc_cohesion_summary_df, dcs, channel_info_map,
                    top_k=None, n_shown=None, name='core_nodes', save=True):
    """Hub/broker bar grid, one row per selected dynamic community."""
    top_k = config.TOP_K_DCS if top_k is None else top_k
    n_shown = config.N_CORE_SHOWN if n_shown is None else n_shown

    top = select_top_dcs(dc_cohesion_summary_df, top_k)

    fig, axes = plt.subplots(len(top), 2, figsize=(13, 2.0 * len(top)), squeeze=False)
    for slot, (_, row) in enumerate(top.iterrows()):
        dc_id = row['dc_id']
        color = DC_PALETTE[slot % len(DC_PALETTE)]
        ax_deg, ax_btw = axes[slot]
        dc_rows = dc_core_nodes_df[dc_core_nodes_df['dc_id'] == dc_id]

        # `dc_core_nodes_df` arrives already sorted by mean degree strength
        # within each DC block, so head() is the hub ranking.
        deg_sub = dc_rows.head(n_shown)
        _barh(ax_deg, deg_sub, 'mean_degree_strength', color,
              'Mean weighted degree (within-DC)',
              dc_summary(dc_id, dcs, channel_info_map))

        btw_sub = (dc_rows.dropna(subset=['mean_betweenness'])
                   .sort_values('mean_betweenness', ascending=False).head(n_shown))
        _barh(ax_btw, btw_sub, 'mean_betweenness', color,
              'Mean betweenness (within-DC)', 'Top brokers / bridges')

    fig.suptitle(f'Core channels, {len(top)} {top_dc_label()} dynamic communities\n'
                 '(left: hub / degree-strength core; right: broker / betweenness bridges)',
                 y=1.02, fontsize=13)
    fig.tight_layout()
    if save:
        save_figure(fig, name)
    return fig


def _barh(ax, sub, column, color, xlabel, title):
    labels = [t[:TITLE_TRUNCATE] for t in sub['title']]
    ax.barh(range(len(sub)), sub[column], color=color, alpha=0.8)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_title(title, fontsize=9, loc='left')
