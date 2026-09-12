"""Per-community cohesion: internal density and conductance (section 16f).

Two complementary quantities, both computed per (dynamic community, snapshot):

* **Internal density** -- how tightly the faction's own members overlap with
  each other. Weighted density is the mean overlap strength across all possible
  member pairs; unweighted density is the fraction of possible pairs that have
  any overlap edge at all.
* **Conductance** -- how separated the faction is from the rest of the observed
  network that week: cut weight to the complement over
  `min(volume(S), volume(complement))`. Low conductance means a well-isolated
  faction.

Density is measured inside the community; conductance is measured against the
*full* snapshot graph, not a wing-restricted subgraph.

The four `plot_*` functions below turn `cohesion_tables()`'s output into the
paper's cohesion figures:

1. `plot_wing_cohesion` -- do the two wings differ in how tightly their
   factions hold together, and does that change over the study period?
2. `plot_top_dc_cohesion` -- how does each individual faction move on its own?
3. `plot_cohesion_scatter` -- across every observation, does denser also mean
   better separated?
4. `plot_cohesion_scatter_by_dc` -- is that relationship the same within each
   faction as it is across them?
"""

from math import comb

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from networkx.algorithms.cuts import conductance as nx_conductance

from . import config
from .style import DC_PALETTE, WING_COLOR, save_figure
from .tracking import dominant_wing


def community_density(G, members):
    """(weighted, unweighted) internal density of `members` within `G`.

    Both are NaN for communities smaller than 2 -- there are no possible pairs.
    """
    n = len(members)
    if n < 2:
        return np.nan, np.nan
    H = G.subgraph(members)
    max_pairs = comb(n, 2)
    weighted = sum(d['weight'] for _, _, d in H.edges(data=True)) / max_pairs
    unweighted = H.number_of_edges() / max_pairs
    return weighted, unweighted


def community_conductance(G, members):
    """Conductance of `members` against the rest of the full snapshot graph.

    NaN when `members` covers every node in `G`: the complement is empty, so
    conductance is undefined rather than zero.
    """
    S = set(members)
    if len(S) >= G.number_of_nodes():
        return np.nan
    return nx_conductance(G, S, weight='weight')


def cohesion_tables(dcs, snapshots, snapshot_dates, channel_info_map, verbose=True):
    """Compute cohesion for every (DC, snapshot) the tracking produced.

    Returns three frames:

    * `cohesion_df` -- one row per (DC, snapshot) observation.
    * `dc_cohesion_summary_df` -- per-DC lifetime profile. Also the canonical
      DC ordering used to pick the "longest-lived" set for the per-DC panels and
      for the core-node figure, so both figures colour the same DC identically.
    * `wing_cohesion_df` -- per-snapshot, per-wing rollup (mean across the DCs
      active in that wing that week).
    """
    rows = []
    for dc in dcs.values():
        wing = dominant_wing(dc.latest_front(), channel_info_map)
        for t, front in dc.fronts.items():
            G = snapshots[t]
            dens_w, dens_u = community_density(G, front)
            rows.append({'t': t, 'date': snapshot_dates[t], 'dc_id': dc.id,
                         'wing': wing, 'size': len(front),
                         'density_weighted': dens_w, 'density_unweighted': dens_u,
                         'conductance': community_conductance(G, front)})

    cohesion_df = pd.DataFrame(rows)

    dc_cohesion_summary_df = (cohesion_df.groupby('dc_id')
        .agg(wing=('wing', 'first'), n_snapshots=('t', 'size'),
             mean_density_weighted=('density_weighted', 'mean'),
             std_density_w=('density_weighted', 'std'),
             mean_conductance=('conductance', 'mean'),
             std_conductance=('conductance', 'std'),
             mean_size=('size', 'mean'))
        .reset_index().sort_values('dc_id', ascending=True))

    wing_cohesion_df = (cohesion_df.groupby(['t', 'date', 'wing'])
        .agg(mean_density_weighted=('density_weighted', 'mean'),
             mean_density_unweighted=('density_unweighted', 'mean'),
             mean_conductance=('conductance', 'mean'),
             n_dcs=('dc_id', 'nunique'))
        .reset_index())

    if verbose:
        print(f'Computed cohesion for {len(cohesion_df)} (DC, snapshot) observations '
              f'across {len(dcs)} DCs')

    return cohesion_df, dc_cohesion_summary_df, wing_cohesion_df


def _date_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')


def plot_wing_cohesion(wing_cohesion_df, name='cohesion_by_wing', save=True):
    """Mean internal density and conductance per wing over time."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for wing in ('left', 'right'):
        sub = wing_cohesion_df[wing_cohesion_df['wing'] == wing]
        axes[0].plot(sub['date'], sub['mean_density_weighted'], color=WING_COLOR[wing],
                     linewidth=1.8, marker='.', label=f'{wing.capitalize()} DCs')
        axes[1].plot(sub['date'], sub['mean_conductance'], color=WING_COLOR[wing],
                     linewidth=1.8, marker='.', label=f'{wing.capitalize()} DCs')

    axes[0].set_ylabel('Mean internal density\n(weighted)')
    axes[0].set_title('Per-wing dynamic-community cohesion over time', fontsize=13)
    axes[0].legend(loc='upper left', fontsize=9)
    axes[1].set_ylabel('Mean conductance')
    axes[1].set_xlabel('Date')
    _date_axis(axes[1])

    fig.tight_layout()
    if save:
        save_figure(fig, name)
    return fig


def select_top_dcs(dc_cohesion_summary_df, top_k=None, selection=None):
    """The subset of dynamic communities shown in the per-DC panels.

    `selection='dc_id'` (the default, and what the paper's figures use) takes
    the first `top_k` dynamic communities by id -- ids are assigned in birth
    order, so this is the earliest-established set. `selection='n_snapshots'`
    takes the longest-lived ones instead.

    Either way the result is re-sorted by id, so a DC lands in the same palette
    slot here and in the core-node figure and keeps one colour across figures.
    """
    top_k = config.TOP_K_DCS if top_k is None else top_k
    selection = config.TOP_DC_SELECTION if selection is None else selection

    if selection == 'n_snapshots':
        chosen = dc_cohesion_summary_df.sort_values('n_snapshots', ascending=False).head(top_k)
    else:
        chosen = dc_cohesion_summary_df.sort_values('dc_id').head(top_k)
    return chosen.sort_values('dc_id')


def top_dc_label(selection=None):
    selection = config.TOP_DC_SELECTION if selection is None else selection
    return 'longest-lived' if selection == 'n_snapshots' else 'earliest-established'


def plot_top_dc_cohesion(cohesion_df, dc_cohesion_summary_df,
                         top_k=None, name='cohesion_top_dcs', save=True):
    """Density and conductance trend for each of the selected DCs."""
    top_k = config.TOP_K_DCS if top_k is None else top_k
    top = select_top_dcs(dc_cohesion_summary_df, top_k)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for slot, (_, row) in enumerate(top.iterrows()):
        sub = cohesion_df[cohesion_df['dc_id'] == row['dc_id']].sort_values('t')
        color = DC_PALETTE[slot % len(DC_PALETTE)]
        label = f"dc{row['dc_id']} ({row['wing']}, n={row['n_snapshots']})"
        axes[0].plot(sub['date'], sub['density_weighted'], color=color,
                     linewidth=1.6, marker='.', alpha=0.9, label=label)
        axes[1].plot(sub['date'], sub['conductance'], color=color,
                     linewidth=1.6, marker='.', alpha=0.9, label=label)

    axes[0].set_ylabel('Internal density\n(weighted)')
    axes[0].set_title(f'Cohesion trend, {len(top)} {top_dc_label()} dynamic communities',
                      fontsize=13)
    axes[0].legend(loc='upper left', fontsize=7, ncol=2)
    axes[1].set_ylabel('Conductance')
    axes[1].set_xlabel('Date')
    _date_axis(axes[1])

    fig.tight_layout()
    if save:
        save_figure(fig, name)
    return fig


def plot_cohesion_scatter(cohesion_df, name='cohesion_scatter', save=True):
    """Density vs. conductance across every (DC, snapshot) observation."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for wing in ('left', 'right', 'mixed', 'unknown'):
        sub = (cohesion_df[cohesion_df['wing'] == wing]
               .dropna(subset=['density_weighted', 'conductance']))
        if sub.empty:
            continue
        ax.scatter(sub['density_weighted'], sub['conductance'], s=sub['size'] * 4,
                   color=WING_COLOR[wing], alpha=0.5, edgecolors='none',
                   label=wing.capitalize())

    ax.set_xlabel('Internal density (weighted)')
    ax.set_ylabel('Conductance')
    ax.set_title('Cohesion vs. separation, all (DC, snapshot) observations\n'
                 '(point size ~ community size)', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)

    fig.tight_layout()
    if save:
        save_figure(fig, name)
    return fig


def plot_cohesion_scatter_by_dc(cohesion_df, dc_cohesion_summary_df,
                                top_k=None, name='cohesion_scatter_by_dc', save=True):
    """The same trade-off space, coloured per DC rather than per wing.

    Reuses the palette slots from `plot_top_dc_cohesion`, so a DC keeps its
    colour across both figures.
    """
    top_k = config.TOP_K_DCS if top_k is None else top_k
    top = select_top_dcs(dc_cohesion_summary_df, top_k)
    sub_top = (cohesion_df[cohesion_df['dc_id'].isin(top['dc_id'])]
               .dropna(subset=['density_weighted', 'conductance']))

    fig, ax = plt.subplots(figsize=(8, 6.5))
    for slot, (_, row) in enumerate(top.iterrows()):
        sub = sub_top[sub_top['dc_id'] == row['dc_id']]
        ax.scatter(sub['density_weighted'], sub['conductance'], s=sub['size'] * 4,
                   color=DC_PALETTE[slot % len(DC_PALETTE)], alpha=0.7, edgecolors='none',
                   label=f"dc{row['dc_id']} ({row['wing']}, n={row['n_snapshots']})")

    ax.set_xlabel('Internal density (weighted)')
    ax.set_ylabel('Conductance')
    ax.set_title(f'Cohesion vs. separation, {len(top)} {top_dc_label()} dynamic communities\n'
                 '(point size ~ community size)', fontsize=12)
    ax.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    if save:
        save_figure(fig, name)
    return fig


def plot_cohesion(cohesion_df, dc_cohesion_summary_df, wing_cohesion_df,
                  top_k=None, save=True):
    """Draw all four cohesion panels. Returns them in order."""
    return [
        plot_wing_cohesion(wing_cohesion_df, save=save),
        plot_top_dc_cohesion(cohesion_df, dc_cohesion_summary_df, top_k, save=save),
        plot_cohesion_scatter(cohesion_df, save=save),
        plot_cohesion_scatter_by_dc(cohesion_df, dc_cohesion_summary_df, top_k, save=save),
    ]
