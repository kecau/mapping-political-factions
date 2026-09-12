"""User polarity distributions (sections 16l/16m, reported as 16n).

A polarity score places one commenter on a 0-1 axis between two channel sets:
0 means all their in-window comments went to the first set, 1 means all went to
the second. Two variants are computed:

* **Bipartite** -- the fixed left-vs-right wing split across the whole study.
  This is the baseline panel: how polarised is the audience between the two
  political camps at all?
* **Same-wing DC pair** -- two factions *inside* one wing, scored over the
  window in which both were active. This is the paper's actual question: does
  the same audience-level separation appear between factions of the same
  political side?

Both aggregate per-author counts in SQL. Pulling raw comment rows for a wide,
well-populated channel set means millions of rows through an unbounded
`fetchall()`; aggregating with `GROUP BY author_name` bounds the result set by
the number of *distinct commenting authors* instead, which is orders of
magnitude smaller.

`plot_polarity_distributions` draws the bipartite baseline alongside one panel
per same-wing pair: a bimodal baseline with a unimodal same-wing panel would
mean intra-wing factions share an audience; bimodality in both means the
separation reproduces itself inside a single political camp.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

from . import config
from .style import WING_COLOR, save_figure
from .tracking import (dc_active_window, dc_all_members, dc_snapshot_presence_count,
                       dominant_wing)


def _distinct_comment_subquery(placeholders_combined):
    """Author-level counts over comments de-duplicated on (author, channel, time).

    The DISTINCT guards against the same comment being counted twice when a row
    was re-collected; it mirrors the de-duplication the cache layer applies.
    """
    return f"""
        SELECT DISTINCT author_name, channel_id, published_at
        FROM youtube_comments
        WHERE channel_id IN ({placeholders_combined})
          AND published_at BETWEEN %s AND %s
          AND author_name IS NOT NULL AND author_name != ''
    """


def _score_author_counts(rows, min_comments):
    """Turn per-author (c_a, c_b) counts into scores in [0, 1].

    MySQL returns `SUM()` of an integer expression as DECIMAL, which PyMySQL
    hands back as `decimal.Decimal`; cast to float here so downstream consumers
    (seaborn's KDE, numpy) get plain numbers.
    """
    scores = []
    for row in rows:
        c_a, c_b = float(row['c_a'] or 0), float(row['c_b'] or 0)
        total = c_a + c_b
        if total >= min_comments:
            scores.append(c_b / total)
    return scores


def compute_bipartite_polarity(connection, channel_info_map, start_date=None,
                               end_date=None, min_comments=None, verbose=True):
    """Per-user polarity across the fixed left/right split. 0 = left, 1 = right."""
    start_date = config.START_DATE if start_date is None else start_date
    end_date = config.END_DATE if end_date is None else end_date
    min_comments = (config.MIN_COMMENTS_PER_USER_POLARITY if min_comments is None
                    else min_comments)

    left_ids = [cid for cid, info in channel_info_map.items() if info.get('wing') == 'left']
    right_ids = [cid for cid, info in channel_info_map.items() if info.get('wing') == 'right']
    combined = left_ids + right_ids

    ph_left = ', '.join(['%s'] * len(left_ids))
    ph_right = ', '.join(['%s'] * len(right_ids))
    ph_combined = ', '.join(['%s'] * len(combined))

    connection.ping(reconnect=True)
    with connection.cursor() as cur:
        cur.execute(
            f"""SELECT author_name,
                       SUM(channel_id IN ({ph_left}))  AS c_a,
                       SUM(channel_id IN ({ph_right})) AS c_b
                FROM ({_distinct_comment_subquery(ph_combined)}) AS distinct_comments
                GROUP BY author_name""",
            tuple(left_ids) + tuple(right_ids) + tuple(combined)
            + (f'{start_date} 00:00:00', f'{end_date} 23:59:59')
        )
        rows = cur.fetchall()

    scores = _score_author_counts(rows, min_comments)
    if verbose:
        print(f'{len(scores):,} users scored (>= {min_comments} left+right comments).')
    return scores


def compute_pair_polarity(connection, dc_a, dc_b, snapshot_dates, window_days=None,
                          min_comments=None, min_users=None):
    """Per-user polarity between two dynamic communities over their shared window.

    Scope is the temporal intersection of the two DCs' real-calendar spans, and
    membership is each DC's *lifetime* channel set, not its latest front.

    A channel that was ever a member of both DCs' histories ("contested") is
    resolved by majority vote: it is assigned wholly to whichever DC it appeared
    in more often across that DC's own tracked snapshots. A true tie leaves the
    channel shared -- counted toward both factions -- rather than discarded,
    since a genuinely 50/50-contested channel has no clean home and a user
    concentrated there really should read as more centrist between the two.

    Score convention: 0 = comments exclusively on `dc_a`'s channels, 1 =
    exclusively on `dc_b`'s.

    Always returns a result dict rather than raising on "no data", with a
    `status` in {'ok', 'no_overlap', 'degenerate_membership', 'no_comments',
    'too_few_users'} so the caller can report *why* a pair was skipped.
    """
    window_days = config.WINDOW_DAYS if window_days is None else window_days
    min_comments = (config.MIN_COMMENTS_PER_USER_POLARITY if min_comments is None
                    else min_comments)
    min_users = config.MIN_USERS_PER_PAIR if min_users is None else min_users

    start_a, end_a = dc_active_window(dc_a, snapshot_dates, window_days)
    start_b, end_b = dc_active_window(dc_b, snapshot_dates, window_days)
    overlap_start = max(start_a, start_b)
    overlap_end = min(end_a, end_b)

    result = {
        'dc_a': dc_a.id, 'dc_b': dc_b.id,
        'overlap_start': overlap_start, 'overlap_end': overlap_end,
        'status': None, 'scores': [], 'n_users': 0, 'n_comments': 0,
        'n_contested_channels': 0, 'n_reassigned_a': 0, 'n_reassigned_b': 0,
        'n_kept_shared': 0,
    }

    if overlap_start >= overlap_end:
        result['status'] = 'no_overlap'
        return result

    members_a_all = dc_all_members(dc_a)
    members_b_all = dc_all_members(dc_b)
    contested = members_a_all & members_b_all

    members_a = set(members_a_all - contested)
    members_b = set(members_b_all - contested)
    for cid in contested:
        n_a = dc_snapshot_presence_count(dc_a, cid)
        n_b = dc_snapshot_presence_count(dc_b, cid)
        if n_a > n_b:
            members_a.add(cid)
            result['n_reassigned_a'] += 1
        elif n_b > n_a:
            members_b.add(cid)
            result['n_reassigned_b'] += 1
        else:
            members_a.add(cid)
            members_b.add(cid)
            result['n_kept_shared'] += 1
    result['n_contested_channels'] = len(contested)

    if not members_a or not members_b:
        result['status'] = 'degenerate_membership'
        return result

    members_a_list = list(members_a)
    members_b_list = list(members_b)
    combined = list(members_a | members_b)
    ph_a = ', '.join(['%s'] * len(members_a_list))
    ph_b = ', '.join(['%s'] * len(members_b_list))
    ph_combined = ', '.join(['%s'] * len(combined))

    # Called once per pair in a loop -- reconnect transparently if the
    # connection went idle between pairs.
    connection.ping(reconnect=True)
    with connection.cursor() as cur:
        cur.execute(
            f"""SELECT author_name,
                       SUM(channel_id IN ({ph_a})) AS c_a,
                       SUM(channel_id IN ({ph_b})) AS c_b
                FROM ({_distinct_comment_subquery(ph_combined)}) AS distinct_comments
                GROUP BY author_name""",
            tuple(members_a_list) + tuple(members_b_list) + tuple(combined)
            + (f'{overlap_start.isoformat()} 00:00:00', f'{overlap_end.isoformat()} 23:59:59')
        )
        rows = cur.fetchall()

    if not rows:
        result['status'] = 'no_comments'
        return result

    result['n_comments'] = int(sum(float(r['c_a'] or 0) + float(r['c_b'] or 0) for r in rows))
    result['scores'] = _score_author_counts(rows, min_comments)
    result['n_users'] = len(result['scores'])
    result['status'] = 'ok' if result['n_users'] >= min_users else 'too_few_users'
    return result


def resolve_dc_pairs(dcs, dc_cohesion_summary_df, channel_info_map,
                     configured_pairs=None, verbose=True):
    """Pick the same-wing DC pairs to plot.

    Uses `config.POLARITY_DC_PAIRS` when those ids exist in this run. They only
    identify the paper's factions under the paper's exact parameters, so if a
    replication changes any of them the ids will not line up -- in that case
    fall back to the longest-lived same-wing pair per wing and say so.
    """
    configured_pairs = config.POLARITY_DC_PAIRS if configured_pairs is None else configured_pairs
    if all(a in dcs and b in dcs for a, b in configured_pairs):
        return list(configured_pairs)

    missing = [p for p in configured_pairs if p[0] not in dcs or p[1] not in dcs]
    if verbose:
        print(f'Configured polarity pairs {missing} are not present in this run '
              f'(the ids depend on the tracking parameters). '
              f'Falling back to the longest-lived same-wing pairs.')

    ranked = dc_cohesion_summary_df.sort_values('n_snapshots', ascending=False)
    fallback = []
    for wing in ('left', 'right'):
        ids = [int(r.dc_id) for r in ranked.itertuples()
               if dominant_wing(dcs[int(r.dc_id)].latest_front(), channel_info_map) == wing]
        if len(ids) >= 2:
            fallback.append((ids[0], ids[1]))
    return fallback


def polarity_for_pairs(connection, dcs, pairs, snapshot_dates, channel_info_map,
                       window_days=None, verbose=True):
    """Run `compute_pair_polarity` over a list of (dc_id, dc_id) pairs.

    Returns `(plottable, skipped)`; each result carries a `wing` key naming the
    shared wing (or 'mixed' if the two DCs disagree).
    """
    results = []
    for id_a, id_b in pairs:
        res = compute_pair_polarity(connection, dcs[id_a], dcs[id_b],
                                    snapshot_dates, window_days)
        wing_a = dominant_wing(dcs[id_a].latest_front(), channel_info_map)
        wing_b = dominant_wing(dcs[id_b].latest_front(), channel_info_map)
        res['wing'] = wing_a if wing_a == wing_b else 'mixed'
        results.append(res)

    plottable = [r for r in results if r['status'] == 'ok']
    skipped = [r for r in results if r['status'] != 'ok']
    if skipped and verbose:
        print(f'{len(skipped)} pair(s) skipped (status != ok):')
        for r in skipped:
            print(f"  DC {r['dc_a']} vs DC {r['dc_b']}: {r['status']}")
    return plottable, skipped


def plot_polarity_distributions(bipartite_scores, pair_results, name=None, save=True):
    """KDE panels: the bipartite baseline plus one panel per same-wing DC pair."""
    n_panels = 1 + len(pair_results)
    n_cols = min(3, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows),
                             squeeze=False)
    flat_axes = axes.flatten()
    panel_labels = [chr(ord('a') + i) for i in range(n_panels)]

    # Panel (a): the fixed left/right baseline.
    _kde_panel(
        flat_axes[0], bipartite_scores, '#555555',
        ['0 = Left-exclusive, 1 = Right-exclusive', f'n = {len(bipartite_scores):,} users'],
        panel_labels[0],
    )

    # Panels (b), (c), ...: same-wing faction pairs.
    for i, result in enumerate(pair_results, start=1):
        _kde_panel(
            flat_axes[i], result['scores'], WING_COLOR.get(result['wing'], '#7A5B9A'),
            [f"0 = DC{result['dc_a']}-exclusive, 1 = DC{result['dc_b']}-exclusive",
             f"n = {result['n_users']:,} users"],
            panel_labels[i],
        )

    for j in range(n_panels, len(flat_axes)):
        flat_axes[j].axis('off')

    fig.tight_layout()
    if save:
        save_figure(fig, name or f'polarity_distributions_{config.START_DATE}_{config.END_DATE}')
    return fig


def _kde_panel(ax, scores, color, legend_lines, panel_label):
    """One density panel. Titles are omitted -- captions are added externally."""
    sns.kdeplot(scores, ax=ax, color=color, linewidth=2, fill=True,
                alpha=0.25, clip=(0, 1), bw_adjust=0.8)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Polarity score', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)

    # Text-only legend entries via invisible handles: this is a caption block
    # pinned inside the panel, not a series key.
    ax.legend(handles=[Line2D([], [], color='none', label=line) for line in legend_lines],
              loc='upper right', fontsize=8, frameon=True,
              handlelength=0, handletextpad=0)
    sns.despine(ax=ax)
    ax.text(0.5, -0.18, f'({panel_label})', transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='center')
