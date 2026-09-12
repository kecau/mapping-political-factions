"""Channel metadata, the study's data summary, and daily comment volume.

These are the direct-from-MySQL loaders. Everything heavier (per-day commenter
sets, video titles/tags) goes through the Parquet cache in `cache.py` instead.

`plot_daily_volume` turns `daily_comment_volume()`'s output into the paper's
context figure: how much commenting activity each side of the corpus carried
day by day. It is what the community-level figures should be read against -- a
spike in activity around an election is not the same finding as a change in
community structure.
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from . import config
from .style import (EVENT_COLOR, WING_COLOR, add_political_events,
                    format_month_axis, save_figure)


def load_channel_info(connection, channel_ids):
    """Return `{channel_id: {'title': ..., 'wing': ...}}` for the study set.

    Channels with no coded `wing` or no title are dropped: they cannot be
    placed on the left/right axis the whole analysis is built on. The returned
    dict is the authoritative channel set for every downstream step -- callers
    should take their channel id list from `list(channel_info_map)`, not from
    `channels.CHANNEL_IDS`, so a channel missing from the database never leaks
    into a query.
    """
    placeholders = ', '.join(['%s'] * len(channel_ids))
    with connection.cursor() as cur:
        cur.execute(
            f"""SELECT channel_id, title, wing FROM youtube_channels
                WHERE channel_id IN ({placeholders})
                  AND wing IS NOT NULL AND title IS NOT NULL""",
            tuple(channel_ids)
        )
        return {r['channel_id']: {'title': r['title'], 'wing': r['wing'].lower()}
                for r in cur.fetchall()}


def _counts_for(connection, channel_ids, start_date, end_date):
    """(video count, comment count) for a set of channels inside the window."""
    if not channel_ids:
        return 0, 0
    placeholders = ', '.join(['%s'] * len(channel_ids))
    params = tuple(channel_ids) + (f'{start_date} 00:00:00', f'{end_date} 23:59:59')
    with connection.cursor() as cur:
        cur.execute(
            f"""SELECT COUNT(*) AS n FROM youtube_videos
                WHERE channel_id IN ({placeholders})
                  AND published_at BETWEEN %s AND %s""",
            params
        )
        n_videos = cur.fetchone()['n']

        cur.execute(
            f"""SELECT COUNT(*) AS n FROM youtube_comments
                WHERE channel_id IN ({placeholders})
                  AND published_at BETWEEN %s AND %s""",
            params
        )
        n_comments = cur.fetchone()['n']
    return n_videos, n_comments


def data_summary(connection, channel_info_map, start_date=None, end_date=None):
    """Corpus size overall and split by wing, as a DataFrame.

    Rows: `all`, `left`, `right`. Columns: `channels`, `videos`, `comments`.
    """
    start_date = config.START_DATE if start_date is None else start_date
    end_date = config.END_DATE if end_date is None else end_date

    channel_ids = list(channel_info_map)
    left_ids = [cid for cid in channel_ids if channel_info_map[cid]['wing'] == 'left']
    right_ids = [cid for cid in channel_ids if channel_info_map[cid]['wing'] == 'right']

    rows = []
    for label, ids in (('all', channel_ids), ('left', left_ids), ('right', right_ids)):
        n_videos, n_comments = _counts_for(connection, ids, start_date, end_date)
        rows.append({'scope': label, 'channels': len(ids),
                     'videos': n_videos, 'comments': n_comments})

    return pd.DataFrame(rows).set_index('scope')


def format_summary(summary_df, start_date=None, end_date=None):
    """Render `data_summary()` as the plain-text block the paper quotes."""
    start_date = config.START_DATE if start_date is None else start_date
    end_date = config.END_DATE if end_date is None else end_date

    lines = [f'Data summary ({start_date} -> {end_date}):']
    # 'Left ' is padded so the two wing headings line up under each other.
    for scope, label in (('all', None), ('left', 'Left '), ('right', 'Right')):
        row = summary_df.loc[scope]
        if label is None:
            lines.append(f'  Channels : {row["channels"]}')
            indent = '  '
        else:
            lines.append('')
            lines.append(f'  {label} channels : {row["channels"]}')
            indent = '    '
        lines.append(f'{indent}Videos   : {row["videos"]:,}')
        lines.append(f'{indent}Comments : {row["comments"]:,}')
    return '\n'.join(lines)


def daily_comment_volume(connection, channel_info_map, start_date=None, end_date=None):
    """Daily comment counts per wing, reindexed over every calendar day.

    Aggregated in SQL (`GROUP BY channel_id, DATE(published_at)`) rather than by
    pulling raw comment rows and summing in pandas -- the window spans tens of
    millions of rows.
    """
    start_date = config.START_DATE if start_date is None else start_date
    end_date = config.END_DATE if end_date is None else end_date

    channel_ids = list(channel_info_map)
    placeholders = ', '.join(['%s'] * len(channel_ids))
    with connection.cursor() as cur:
        cur.execute(
            f"""SELECT channel_id, DATE(published_at) AS d, COUNT(*) AS n
                FROM youtube_comments
                WHERE channel_id IN ({placeholders})
                  AND published_at BETWEEN %s AND %s
                GROUP BY channel_id, d""",
            tuple(channel_ids) + (f'{start_date} 00:00:00', f'{end_date} 23:59:59')
        )
        volume_rows = cur.fetchall()

    volume_df = pd.DataFrame(volume_rows)
    volume_df['wing'] = volume_df['channel_id'].map(
        lambda cid: channel_info_map.get(cid, {}).get('wing'))

    full_days = pd.date_range(start_date, end_date, freq='D').date
    return (volume_df[volume_df['wing'].isin(['left', 'right'])]
            .groupby(['d', 'wing'])['n'].sum()
            .unstack(fill_value=0)
            .reindex(columns=['left', 'right'], fill_value=0)
            .reindex(full_days, fill_value=0))


def plot_daily_volume(daily_wing_counts, name='daily_comment_volume', save=True):
    """Plot daily comment counts for left- and right-leaning channels."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily_wing_counts.index, daily_wing_counts['left'],
            color=WING_COLOR['left'], linewidth=1.3, label='left channels')
    ax.plot(daily_wing_counts.index, daily_wing_counts['right'],
            color=WING_COLOR['right'], linewidth=1.3, label='right channels')

    ax.set_ylabel('# comments')
    format_month_axis(ax)
    add_political_events(ax)

    wing_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=wing_handles + [
        Line2D([0], [0], color=EVENT_COLOR, linestyle='--', linewidth=1, alpha=0.5,
               label='political event'),
    ], loc='upper left', fontsize=10)

    fig.tight_layout()
    if save:
        save_figure(fig, name)
    return fig
