"""Stage 2: Greene et al. (2010) cross-snapshot community matching.

    Greene, D., Doyle, D., & Cunningham, P. (2010). Tracking the Evolution of
    Communities in Dynamic Social Networks. ASONAM '10.

A community `C` at snapshot `t` is matched to a dynamic community `DC` when
either the Jaccard similarity `J(C, DC) >= THETA` or the containment
`|C n DC| / |C| >= PHI` (or the mirror containment on the DC's side). The
matching structure between consecutive snapshots yields the event log: birth,
death, growth, contraction, merge, split, and plain continuation.

This module also holds the small dynamic-community helpers that other modules
need -- wing labelling, lifetime membership, active window -- so they are
defined once rather than per figure, plus the two figures built directly on
the tracking output: `plot_timeline` (every DC's lifeline over the study
period) and `plot_event_graph` (the same output, compressed for print).
"""

from collections import defaultdict
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch

from . import config
from .style import (GRACE_COLOR, WING_COLOR, WING_ORDER, add_political_events,
                    format_month_axis, save_figure)


def jaccard(A, B):
    if not A and not B:
        return 0.0
    inter = len(A & B)
    if inter == 0:
        return 0.0
    return inter / len(A | B)


class DynamicCommunity:
    """A single community tracked across snapshots.

    `fronts[t]` is the member frozenset at snapshot `t`. A DC is alive if it has
    been matched within the last `GRACE` snapshots. A merge does not mark its
    parent DCs dead outright -- they simply stop being updated, so they stay
    grace-period eligible until either a later snapshot rematches them directly
    or the grace period expires.
    """

    __slots__ = ('id', 'birth_t', 'death_t', 'fronts', 'parents', 'children')

    def __init__(self, dc_id, t, members, parents=()):
        self.id = dc_id
        self.birth_t = t
        self.death_t = None
        self.fronts = {t: frozenset(members)}
        self.parents = tuple(parents)
        self.children = ()

    def latest_t(self):
        return max(self.fronts)

    def latest_front(self):
        return self.fronts[self.latest_t()]

    def is_alive(self, t, grace):
        return self.death_t is None and (t - self.latest_t()) <= grace


def track(snapshot_communities, theta=None, beta=None, grace=None, phi=None,
          verbose=True):
    """Match communities across snapshots and record the event log.

    Returns `(dcs, events)`: a `{dc_id: DynamicCommunity}` map and a list of
    event dicts, each carrying at least `t`, `type` and `dc_id`.

    Unset arguments fall back to `config` at call time, not at import time, so a
    notebook can override a study parameter and re-run without reimporting.
    """
    theta = config.THETA if theta is None else theta
    beta = config.BETA if beta is None else beta
    grace = config.GRACE if grace is None else grace
    phi = config.PHI if phi is None else phi

    dcs = {}
    events = []
    next_id = 0

    # ---- Initialize with snapshot 0 ----
    for c in snapshot_communities[0]:
        dcs[next_id] = DynamicCommunity(next_id, 0, c)
        events.append({'t': 0, 'type': 'birth', 'dc_id': next_id, 'size': len(c)})
        next_id += 1

    # ---- Iterate over subsequent snapshots ----
    for t in range(1, len(snapshot_communities)):
        current = snapshot_communities[t]
        active = [dc for dc in dcs.values() if dc.is_alive(t - 1, grace)]

        # Match if Jaccard >= theta OR containment |C n DC| / |C| >= phi
        raw_matches = []
        for i, c in enumerate(current):
            for dc in active:
                front = dc.latest_front()
                inter = len(c & front)
                j = inter / len(c | front) if (c | front) else 0.0
                containment_c = inter / len(c) if c else 0.0
                containment_dc = inter / len(front) if front else 0.0
                if j >= theta or containment_c >= phi or containment_dc >= phi:
                    raw_matches.append((i, dc.id, j))

        # Priority fix: group raw candidate matches per current community.
        # If a current community's candidates mix a *fresh* DC (last seen at
        # t-1) with a *grace-period* DC (last seen earlier but still within
        # `grace`), that is not a genuine merge -- it is the same community
        # front explained two competing ways. Collapse the mix to a single
        # match: whichever candidate has the higher Jaccard score. A DC that
        # loses this comparison for community i is not discarded outright --
        # if some *other* current community also matches it, it still counts
        # toward that DC's own match set and can register as a split.
        raw_by_current = defaultdict(list)
        for i, dc_id, j in raw_matches:
            raw_by_current[i].append((dc_id, j))

        matches = []
        for i, partners in raw_by_current.items():
            fresh = [(dc_id, j) for dc_id, j in partners if dcs[dc_id].latest_t() == t - 1]
            grace_period = [(dc_id, j) for dc_id, j in partners if dcs[dc_id].latest_t() != t - 1]
            if fresh and grace_period:
                best_dc_id, best_j = max(partners, key=lambda p: p[1])
                matches.append((i, best_dc_id, best_j))
            else:
                matches.extend((i, dc_id, j) for dc_id, j in partners)

        # Pattern analysis: matches per current community, per DC
        m_per_current = defaultdict(list)   # curr_idx -> [(dc_id, j)]
        m_per_dc = defaultdict(list)        # dc_id    -> [(curr_idx, j)]
        for i, dc_id, j in matches:
            m_per_current[i].append((dc_id, j))
            m_per_dc[dc_id].append((i, j))

        # ---------- 1. Process each current community ----------
        handled_splits = set()
        handled_merges = set()

        for i, c in enumerate(current):
            partners = m_per_current[i]

            if not partners:
                # ---- BIRTH ----
                dcs[next_id] = DynamicCommunity(next_id, t, c)
                events.append({'t': t, 'type': 'birth', 'dc_id': next_id, 'size': len(c)})
                next_id += 1
                continue

            if len(partners) > 1:
                # ---- MERGE: current i is the merged result of multiple DCs.
                # Parents are NOT marked dead here -- they simply stop being
                # updated, so they stay grace-period-eligible in case this
                # merge turns out to be temporary (see priority fix above).
                if i in handled_merges:
                    continue
                handled_merges.add(i)
                parent_ids = [pid for pid, _ in partners]
                dcs[next_id] = DynamicCommunity(next_id, t, c, parents=parent_ids)
                for pid in parent_ids:
                    dcs[pid].children = tuple(list(dcs[pid].children) + [next_id])
                events.append({'t': t, 'type': 'merge', 'dc_id': next_id,
                               'parents': parent_ids, 'size': len(c)})
                next_id += 1
                continue

            dc_id, jval = partners[0]
            if len(m_per_dc[dc_id]) > 1:
                # ---- SPLIT: DC dc_id splits into multiple current communities ----
                if dc_id in handled_splits:
                    continue
                handled_splits.add(dc_id)
                child_ids = []
                for ci, cj in m_per_dc[dc_id]:
                    dcs[next_id] = DynamicCommunity(next_id, t, current[ci], parents=(dc_id,))
                    child_ids.append(next_id)
                    next_id += 1
                dcs[dc_id].children = tuple(child_ids)
                events.append({'t': t, 'type': 'split', 'dc_id': dc_id, 'children': child_ids})
                continue

            # ---- CONTINUATION / GROWTH / CONTRACTION ----
            dc = dcs[dc_id]
            prev = dc.latest_front()
            new_sz = len(c)
            prev_sz = len(prev)
            delta = (new_sz - prev_sz) / prev_sz if prev_sz else 0.0
            if delta > beta:
                ev_type = 'growth'
            elif delta < -beta:
                ev_type = 'contraction'
            else:
                ev_type = 'continuation'
            events.append({'t': t, 'type': ev_type, 'dc_id': dc_id,
                           'size': new_sz, 'prev_size': prev_sz,
                           'jaccard': round(jval, 3)})
            dc.fronts[t] = frozenset(c)

        # ---------- 2. Deaths ----------
        for dc in active:
            if dc.death_t is not None:
                continue
            if dc.id in m_per_dc:
                continue
            if (t - dc.latest_t()) >= grace:
                dc.death_t = t
                events.append({'t': t, 'type': 'death', 'dc_id': dc.id})

    if verbose:
        print(f'Tracked {len(dcs)} dynamic communities')
        print(f'Recorded {len(events)} events')

    return dcs, events


# ── Dynamic-community helpers shared by other modules ───────────────────────

def dominant_wing(members, channel_info_map):
    """Label a community by the political leaning of its member channels.

    A community counts as left- or right-dominant only at a 1.5:1 majority;
    anything closer is 'mixed'. The threshold keeps a single crossover channel
    from flipping a faction's label.
    """
    left = sum(1 for cid in members if channel_info_map.get(cid, {}).get('wing') == 'left')
    right = sum(1 for cid in members if channel_info_map.get(cid, {}).get('wing') == 'right')
    if left == 0 and right == 0:
        return 'unknown'
    if left > right * 1.5:
        return 'left'
    if right > left * 1.5:
        return 'right'
    return 'mixed'


def dc_summary(dc_id, dcs, channel_info_map):
    """One-line human-readable label for a DC: wing, size, and a few titles."""
    dc = dcs[dc_id]
    front = dc.latest_front()
    titles = [channel_info_map.get(cid, {}).get('title', cid)[:20] for cid in front]
    return (f"dc{dc_id} ({dominant_wing(front, channel_info_map)}, {len(front)}ch: "
            f"{', '.join(titles[:3])}{'...' if len(titles) > 3 else ''})")


def dc_active_window(dc, snapshot_dates, window_days=None):
    """Real-calendar [start, end) span of a DC's activity.

    Start is the snapshot the DC was born in; end is the snapshot it died in,
    or -- for a DC still alive at the end of the study -- one window past the
    final snapshot. Same convention as `plot_timeline`.
    """
    window_days = config.WINDOW_DAYS if window_days is None else window_days
    start = snapshot_dates[dc.birth_t]
    end = (snapshot_dates[dc.death_t] if dc.death_t is not None
           else snapshot_dates[-1] + timedelta(days=window_days))
    return start, end


def dc_all_members(dc):
    """Every channel that was ever in this DC's front, across its whole lifetime.

    Not just `latest_front()`, which is a point-in-time label: a comment-level
    analysis over an extended date range should credit engagement with any
    channel that was ever part of the faction, since membership drifts.
    """
    return frozenset().union(*dc.fronts.values())


def dc_snapshot_presence_count(dc, channel_id):
    """How many of this DC's own tracked snapshots included `channel_id`.

    Used to break ties when a channel belonged to two different same-wing DCs
    at some point in each one's history.
    """
    return sum(1 for front in dc.fronts.values() if channel_id in front)


def event_dataframe(events, snapshot_dates):
    """The event log as a DataFrame with real dates attached."""
    ev_df = pd.DataFrame(events)
    ev_df['date'] = ev_df['t'].map(lambda t: snapshot_dates[t])
    return ev_df


def event_log(events, snapshot_dates, dcs, channel_info_map):
    """Full event log with human-readable DC labels and merge/split lineage."""
    log_rows = []
    for ev in events:
        row = {'t': ev['t'], 'date': snapshot_dates[ev['t']], 'type': ev['type'],
               'dc_id': ev['dc_id']}
        row['dc_summary'] = (dc_summary(ev['dc_id'], dcs, channel_info_map)
                             if ev['dc_id'] in dcs else '')
        row['size'] = ev.get('size', '')
        row['prev_size'] = ev.get('prev_size', '')
        row['jaccard'] = ev.get('jaccard', '')
        row['parents'] = ','.join(str(p) for p in ev.get('parents', []))
        row['children'] = ','.join(str(c) for c in ev.get('children', []))
        log_rows.append(row)
    return pd.DataFrame(log_rows)


# ── Figure: dynamic-community timeline ───────────────────────────────────────
#
# One horizontal lane per dynamic community. Bar thickness encodes how many
# channels the community held that week; colour encodes its dominant wing. A
# confirmed front is drawn in the wing colour for exactly one window; any
# remaining stretch before the next confirmed front is drawn grey, so time a
# community survives only on its grace allowance never reads as a genuine
# rematch. Arcs connect split parents to children and merge parents to their
# merged result.

TIMELINE_MARKER = {'birth': '^', 'death': 'v', 'split': '>', 'merge': '<',
                   'growth': '+', 'contraction': 'x'}

TIMELINE_Y_SPACING = 0.5
TIMELINE_LW_MIN, TIMELINE_LW_MAX = 2, 16


def plot_timeline(dcs, events, snapshot_dates, channel_info_map,
                  window_days=None, name=None, save=True):
    """Draw the dynamic-community timeline. Returns the Figure."""
    window_days = config.WINDOW_DAYS if window_days is None else window_days

    sorted_dcs = sorted(dcs.values(), key=lambda d: (d.birth_t, d.id))
    dc_y = {dc.id: i * TIMELINE_Y_SPACING for i, dc in enumerate(sorted_dcs)}

    all_sizes = [len(f) for dc in sorted_dcs for f in dc.fronts.values()]
    max_size = max(all_sizes) if all_sizes else 1

    fig, ax = plt.subplots(figsize=(14, max(6, len(sorted_dcs) * TIMELINE_Y_SPACING * 0.7 + 2)))

    # ── 1. DC bars, width proportional to community size ────────────────────
    for dc in sorted_dcs:
        y = dc_y[dc.id]
        color = WING_COLOR[dominant_wing(dc.latest_front(), channel_info_map)]
        ts = sorted(dc.fronts)

        for k, t in enumerate(ts):
            size = len(dc.fronts[t])
            lw = TIMELINE_LW_MIN + (TIMELINE_LW_MAX - TIMELINE_LW_MIN) * size / max_size
            x_start = snapshot_dates[t]

            if k < len(ts) - 1:
                x_end = snapshot_dates[ts[k + 1]]
            elif dc.death_t is not None:
                x_end = snapshot_dates[dc.death_t]
            else:
                # Still alive: extend to the end of the study window so an
                # ongoing, unresolved grace period is visible as a trailing
                # grey stretch rather than stopping at the last front.
                x_end = snapshot_dates[-1] + timedelta(days=window_days)

            seg_end = min(x_start + timedelta(days=window_days), x_end)
            ax.hlines(y, x_start, seg_end, color=color, linewidth=lw, alpha=0.75)
            if x_end > seg_end:
                ax.hlines(y, seg_end, x_end, color=GRACE_COLOR, linewidth=lw, alpha=0.75)

    # ── 2. Lineage arcs for split / merge ───────────────────────────────────
    for ev in events:
        x_ev = snapshot_dates[ev['t']]

        if ev['type'] == 'split':
            y_parent = dc_y[ev['dc_id']]
            for child_id in ev.get('children', []):
                y_child = dc_y.get(child_id)
                if y_child is None:
                    continue
                _timeline_arc(ax, x_ev, y_parent, y_child)

        elif ev['type'] == 'merge':
            y_child = dc_y.get(ev['dc_id'])
            if y_child is None:
                continue
            for parent_id in ev.get('parents', []):
                y_parent = dc_y.get(parent_id)
                if y_parent is None:
                    continue
                _timeline_arc(ax, x_ev, y_parent, y_child)

    # ── 3. Event glyphs ─────────────────────────────────────────────────────
    for ev in events:
        if ev['type'] not in TIMELINE_MARKER:
            continue
        y = dc_y.get(ev['dc_id'])
        if y is None:
            continue
        ax.scatter([snapshot_dates[ev['t']]], [y], marker=TIMELINE_MARKER[ev['type']],
                   color='black', s=60, zorder=5)

    # ── 4. Axes ─────────────────────────────────────────────────────────────
    ax.set_yticks([dc_y[dc.id] for dc in sorted_dcs])
    ax.set_yticklabels([f'DC {dc.id}' for dc in sorted_dcs], fontsize=10)
    format_month_axis(ax)
    add_political_events(ax)

    # ── 5. Legend ───────────────────────────────────────────────────────────
    size_steps = sorted({1, max(1, max_size // 2), max_size})
    handles = (
        [Patch(facecolor=WING_COLOR['left'], label='left-dominant DC'),
         Patch(facecolor=WING_COLOR['right'], label='right-dominant DC'),
         Patch(facecolor=GRACE_COLOR, label='grace period')]
        + [Line2D([0], [0], marker=m, color='k', linestyle='None', label=lbl)
           for lbl, m in TIMELINE_MARKER.items()]
        + [Line2D([0], [0], color='#444444', linewidth=1.8, label='split / merge link'),
           Line2D([0], [0], color='#000000', linestyle='--', linewidth=1, alpha=0.5,
                  label='political event')]
        + [Line2D([0], [0], color='gray',
                  linewidth=TIMELINE_LW_MIN + (TIMELINE_LW_MAX - TIMELINE_LW_MIN) * s / max_size,
                  label=f'{s} channel{"s" if s != 1 else ""}')
           for s in size_steps]
    )
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1),
              fontsize=10, frameon=True)

    fig.tight_layout()
    if save:
        save_figure(fig, name or _timeline_output_name())
    return fig


def _timeline_arc(ax, x, y_from, y_to):
    """Curved connector between two lanes at the same date."""
    rad = 0.35 if y_to > y_from else -0.35
    ax.annotate('', xy=(x, y_to), xytext=(x, y_from),
                arrowprops=dict(arrowstyle='-', color='#444444', lw=1.8, alpha=0.70,
                                connectionstyle=f'arc3,rad={rad}'))


def _timeline_output_name():
    return (f'timeline_{config.START_DATE}_{config.END_DATE}'
            f'_w{config.WINDOW_DAYS}_s{config.STEP_DAYS}'
            f'_t{config.THETA}_k{config.GRACE}')


# ── Figure: static event graph ───────────────────────────────────────────────
#
# A compact, categorical view of the same tracking output as the timeline: one
# node per (dynamic community, snapshot) front, edges for continuation, merge
# and split. Two compressions keep it legible in a single column:
#
# * Snapshots where nothing happened are dropped. A run of more than two
#   consecutive event-free snapshots collapses to a break mark, and the edge
#   that spans it is drawn in a distinct 'cut' style so a shortened stretch is
#   never mistaken for an adjacent one.
# * The x-axis is categorical (uniform column pitch), not a real date axis, so
#   quiet months do not consume horizontal space.
#
# The node layout and the continuation-type lookup were originally defined
# inside the source notebook's interactive plotly section; they are
# reimplemented here in pure matplotlib, so this package has no plotly
# dependency.

EVENTGRAPH_COL_PITCH, EVENTGRAPH_GAP_PITCH = 1.0, 1.8
EVENTGRAPH_NODE_MIN_D, EVENTGRAPH_NODE_SCALE = 60, 50

EVENTGRAPH_STYLE = {
    'continuation': dict(ls='solid',       color='#888888', lw=0.9),
    'growth':       dict(ls='solid',       color='#888888', lw=0.9),
    'contraction':  dict(ls='solid',       color='#888888', lw=0.9),
    'resurgence':   dict(ls=(0, (3, 2)),   color='#888888', lw=0.9),
    'merge':        dict(ls='solid',       color='#222222', lw=1.0),
    'split':        dict(ls='solid',       color='#222222', lw=1.0),
    'cut':          dict(ls=(0, (1, 1.5)), color='#AAAAAA', lw=0.9),
}


def _eventgraph_interesting_snapshots(dcs, events):
    """Snapshots that must be kept because something observable happens there."""
    keep = {ev['t'] for ev in events
            if ev['type'] in ('birth', 'death', 'merge', 'split')}
    keep |= {dc.birth_t for dc in dcs.values()}
    keep |= {dc.death_t for dc in dcs.values() if dc.death_t is not None}
    keep |= {dc.latest_t() for dc in dcs.values() if dc.death_t is None}

    # Resurgences: a DC reappearing after a gap.
    for dc in dcs.values():
        ts = sorted(dc.fronts)
        for k in range(len(ts) - 1):
            if ts[k + 1] - ts[k] > 1:
                keep.add(ts[k + 1])

    # Merge/split arrows anchor on the parent's front as of just before the
    # event, so that anchor snapshot must never be dropped even if nothing else
    # makes it interesting on its own.
    for ev in events:
        if ev['type'] == 'merge':
            for pid in ev['parents']:
                keep.add(max(t for t in dcs[pid].fronts if t < ev['t']))
        elif ev['type'] == 'split':
            keep.add(max(t for t in dcs[ev['dc_id']].fronts if t < ev['t']))

    return keep


def _eventgraph_kept_columns(interesting_ts, n_snapshots):
    """Drop runs of more than two consecutive event-free snapshots."""
    boring_runs, run = [], []
    for t in range(n_snapshots):
        if t in interesting_ts:
            if run:
                boring_runs.append(run)
                run = []
        else:
            run.append(t)
    if run:
        boring_runs.append(run)

    dropped = {t for r in boring_runs if len(r) > 2 for t in r}
    dropped -= {0, n_snapshots - 1}   # always keep the study-window bookends
    return [t for t in range(n_snapshots) if t not in dropped]


def _eventgraph_display_positions(kept_ts):
    """Uniform x per kept column, with a wider gap wherever columns were cut."""
    display_x, break_xs = {}, []
    x = 0.0
    for i, t in enumerate(kept_ts):
        if i > 0:
            if kept_ts[i] - kept_ts[i - 1] == 1:
                x += EVENTGRAPH_COL_PITCH
            else:
                x += EVENTGRAPH_GAP_PITCH
                break_xs.append(x - EVENTGRAPH_GAP_PITCH / 2)
        display_x[t] = x
    return display_x, break_xs


def plot_event_graph(dcs, events, snapshot_dates, channel_info_map,
                     name=None, save=True, verbose=True):
    """Draw the compressed static event graph. Returns the Figure."""
    n_snapshots = len(snapshot_dates)
    interesting_ts = _eventgraph_interesting_snapshots(dcs, events)
    kept_ts = _eventgraph_kept_columns(interesting_ts, n_snapshots)
    display_x, break_xs = _eventgraph_display_positions(kept_ts)

    # Continuation-type lookup, so a continuation edge can be drawn in the style
    # of the specific event (growth/contraction) that produced it.
    cont_type = {(ev['dc_id'], ev['t']): ev['type'] for ev in events
                 if ev['type'] in ('continuation', 'growth', 'contraction')}

    # ── Node layout: one node per (dc, kept snapshot), rows sorted by wing ──
    nodes_by_t = defaultdict(list)
    for dc in dcs.values():
        for t, front in dc.fronts.items():
            if t not in display_x:
                continue
            nodes_by_t[t].append({'dc_id': dc.id, 't': t, 'members': front,
                                  'size': len(front),
                                  'wing': dominant_wing(front, channel_info_map)})

    node_of = {}
    for t, nodes in nodes_by_t.items():
        nodes.sort(key=lambda n: (WING_ORDER[n['wing']], n['dc_id']))
        for row, n in enumerate(nodes):
            n['x'] = display_x[t]
            n['y'] = -row * 1.0
            node_of[(n['dc_id'], n['t'])] = n

    max_rows = max((len(v) for v in nodes_by_t.values()), default=1)
    fig_width = min(10.0, max(6.0, 0.35 * len(kept_ts) + 2.0))
    fig, ax = plt.subplots(figsize=(fig_width, max(3.0, max_rows * 0.4 + 1.2)))

    # ── Within-DC edges, re-derived over the kept-only front sequence ───────
    # A pair adjacent in the DC's real front sequence keeps its true
    # classification; a pair that skips over dropped fronts is forced to the
    # 'cut' style regardless of type (by construction, anything spanning a
    # dropped run can only ever have been a continuation).
    for dc in dcs.values():
        real_ts = sorted(dc.fronts)
        kept_dc_ts = [t for t in real_ts if t in display_x]
        real_idx = {t: i for i, t in enumerate(real_ts)}
        for k in range(len(kept_dc_ts) - 1):
            t0, t1 = kept_dc_ts[k], kept_dc_ts[k + 1]
            src, dst = node_of[(dc.id, t0)], node_of[(dc.id, t1)]
            if (real_idx[t1] - real_idx[t0]) > 1:
                _eventgraph_edge(ax, src, dst, EVENTGRAPH_STYLE['cut'])
                continue
            ev_type = ('resurgence' if (t1 - t0) > 1
                       else cont_type.get((dc.id, t1), 'continuation'))
            _eventgraph_edge(ax, src, dst, EVENTGRAPH_STYLE[ev_type])

    # ── Merge / split edges, anchored on the parent's pre-event front ───────
    for ev in events:
        if ev['type'] == 'merge':
            dst = node_of.get((ev['dc_id'], ev['t']))
            if dst is None:
                continue
            for pid in ev['parents']:
                parent_t = max(t for t in dcs[pid].fronts if t < ev['t'])
                src = node_of.get((pid, parent_t))
                if src is not None:
                    _eventgraph_edge(ax, src, dst, EVENTGRAPH_STYLE['merge'], arrow=True)
        elif ev['type'] == 'split':
            parent_t = max(t for t in dcs[ev['dc_id']].fronts if t < ev['t'])
            src = node_of.get((ev['dc_id'], parent_t))
            if src is None:
                continue
            for cid in ev['children']:
                dst = node_of.get((cid, ev['t']))
                if dst is not None:
                    _eventgraph_edge(ax, src, dst, EVENTGRAPH_STYLE['split'], arrow=True)

    # ── Nodes, drawn after edges so they visually terminate each line ───────
    for nodes in nodes_by_t.values():
        for n in nodes:
            is_first = n['t'] == dcs[n['dc_id']].birth_t
            ax.scatter([n['x']], [n['y']],
                       s=EVENTGRAPH_NODE_MIN_D + EVENTGRAPH_NODE_SCALE * np.sqrt(n['size']),
                       color=WING_COLOR[n['wing']], edgecolors='white', linewidths=0.6,
                       alpha=1.0 if is_first else 0.6, zorder=3)
            ax.text(n['x'], n['y'], f"C{n['dc_id']}", fontsize=8, color='white',
                    ha='center', va='center', zorder=4)

    # ── Break marks over each compressed gap ────────────────────────────────
    y_bottom, y_top = -max_rows * 1.0 - 0.5, 1.5
    for bx in break_xs:
        ax.axvline(bx, color='#CCCCCC', linewidth=8, alpha=0.5, zorder=0)
        ax.plot([bx], [(y_bottom + y_top) / 2], marker='x', markersize=5,
                color='#888888', zorder=2)

    # ── Axes: categorical, one label per kept column ────────────────────────
    ax.set_yticks([])
    ax.set_ylim(y_bottom, y_top)
    ax.set_xlim(-0.5, (display_x[kept_ts[-1]] if kept_ts else 0) + 0.5)
    ax.set_xticks([display_x[t] for t in kept_ts])
    ax.set_xticklabels([snapshot_dates[t].strftime('%d %b %Y') for t in kept_ts],
                       rotation=45, ha='right', fontsize=9)

    handles = (
        [Patch(facecolor=WING_COLOR[w], label=w.capitalize()) for w in ('left', 'right')]
        + [Line2D([0], [0], color='#888888', lw=0.9, label='Continuation'),
           Line2D([0], [0], color='#888888', lw=0.9, linestyle=(0, (3, 2)), label='Resurgence'),
           Line2D([0], [0], color='#222222', lw=1.1, marker='>', markersize=4,
                  label='Merge / Split'),
           Line2D([0], [0], color='#AAAAAA', lw=0.9, linestyle=(0, (1, 1.5)),
                  label='Cut (collapsed)')]
    )
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1),
              fontsize=9, frameon=True)

    fig.tight_layout()
    if verbose:
        print(f'  {len(kept_ts)}/{n_snapshots} snapshots kept, '
              f'{len(break_xs)} compressed gaps')
    if save:
        save_figure(fig, name or _eventgraph_output_name())
    return fig


def _eventgraph_edge(ax, src, dst, style, arrow=False):
    ax.add_patch(FancyArrowPatch(
        (src['x'], src['y']), (dst['x'], dst['y']),
        arrowstyle='-|>' if arrow else '-', mutation_scale=8,
        linestyle=style['ls'], color=style['color'], linewidth=style['lw'],
        shrinkA=6, shrinkB=6, zorder=1,
    ))


def _eventgraph_output_name():
    return (f'eventgraph_{config.START_DATE}_{config.END_DATE}'
            f'_w{config.WINDOW_DAYS}_s{config.STEP_DAYS}'
            f'_t{config.THETA}_k{config.GRACE}')
