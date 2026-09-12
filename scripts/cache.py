"""Monthly Parquet cache for per-day commenter sets and video metadata.

`youtube_comments` holds tens of millions of rows, so pulling the raw
(channel, day, author) triples from MySQL on every run is slow and
memory-heavy. This module streams one calendar month at a time through an
unbuffered server-side cursor -- so a whole month is never materialised as
Python objects at once -- writes it to a small Parquet file under
`config.CACHE_DIR`, and reuses that file on later runs.

Past months are treated as immutable once cached; the current month is always
refetched.

Trimmed from `utils/analysis_cache.py`: only the author-set and video caches are
kept, since no figure in this package needs raw comment text.
"""

import os
from collections import defaultdict
from datetime import date, datetime

import pandas as pd
import pymysql.cursors

from . import config

# Resolved per call, not at import, so overriding `config.CACHE_DIR` in a
# notebook takes effect without reimporting the module.
COMMENT_SUBDIR = 'comments_by_month'
VIDEO_SUBDIR = 'videos_by_month'

_MONTH_QUERY = """
    SELECT DISTINCT channel_id, DATE(published_at) AS d, author_name
    FROM youtube_comments
    WHERE published_at >= %s AND published_at < %s
      AND author_name IS NOT NULL AND author_name != ''
"""

_VIDEO_MONTH_QUERY = """
    SELECT channel_id, video_id, title, tags, published_at
    FROM youtube_videos
    WHERE published_at >= %s AND published_at < %s
"""


def _parse_date(value):
    if isinstance(value, date):
        return value
    return datetime.strptime(value, '%Y-%m-%d').date()


def _month_bounds(year, month):
    start = date(year, month, 1)
    end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return start, end


def _months_in_range(start_date, end_date):
    months = []
    y, m = start_date.year, start_date.month
    while (y, m) <= (end_date.year, end_date.month):
        months.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return months


def _is_current_month(year, month):
    today = date.today()
    return (year, month) == (today.year, today.month)


def _cache_path(year, month):
    return os.path.join(config.CACHE_DIR, COMMENT_SUBDIR, f'{year:04d}-{month:02d}.parquet')


def _video_cache_path(year, month):
    return os.path.join(config.CACHE_DIR, VIDEO_SUBDIR, f'{year:04d}-{month:02d}.parquet')


def _stream_month(connection, query, params, columns, chunk_size):
    cursor = connection.cursor(pymysql.cursors.SSDictCursor)
    try:
        cursor.execute(query, params)
        chunks = []
        while True:
            batch = cursor.fetchmany(chunk_size)
            if not batch:
                break
            chunks.append(pd.DataFrame(batch, columns=columns))
    finally:
        cursor.close()
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=columns)


def refresh_month_cache(year, month, connection, force=False, chunk_size=200_000):
    """(Re)build the (channel, date, author) cache for one calendar month."""
    path = _cache_path(year, month)
    if os.path.exists(path) and not force and not _is_current_month(year, month):
        return path

    start, end = _month_bounds(year, month)
    print(f'  Refreshing comment cache for {year:04d}-{month:02d}...')
    df = _stream_month(connection, _MONTH_QUERY, (start, end),
                       ['channel_id', 'd', 'author_name'], chunk_size)
    df = df.rename(columns={'d': 'date'})
    df['date'] = pd.to_datetime(df['date']).dt.date

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f'  Cached {len(df):,} (channel, date, author) rows for {year:04d}-{month:02d}')
    return path


def refresh_video_month_cache(year, month, connection, force=False, chunk_size=200_000):
    """(Re)build the video title/tag cache for one calendar month."""
    path = _video_cache_path(year, month)
    if os.path.exists(path) and not force and not _is_current_month(year, month):
        return path

    start, end = _month_bounds(year, month)
    print(f'  Refreshing video cache for {year:04d}-{month:02d}...')
    df = _stream_month(connection, _VIDEO_MONTH_QUERY, (start, end),
                       ['channel_id', 'video_id', 'title', 'tags', 'published_at'], chunk_size)
    df['published_at'] = pd.to_datetime(df['published_at']).dt.date

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f'  Cached {len(df):,} video rows for {year:04d}-{month:02d}')
    return path


def get_daily_authors(channel_ids, start_date, end_date, connection):
    """Return `daily_authors[date][channel_id] -> set(author_name)`.

    Ensures every month the window touches is cached, then reads back only the
    relevant Parquet files and filters them to the requested channels/dates.
    """
    start, end = _parse_date(start_date), _parse_date(end_date)
    channel_id_set = set(channel_ids)
    months = _months_in_range(start, end)

    for year, month in months:
        refresh_month_cache(year, month, connection)

    daily_authors = defaultdict(lambda: defaultdict(set))
    for year, month in months:
        path = _cache_path(year, month)
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        df = df[df['channel_id'].isin(channel_id_set)
                & (df['date'] >= start) & (df['date'] <= end)]
        for row in df.itertuples(index=False):
            daily_authors[row.date][row.channel_id].add(row.author_name)

    return daily_authors


def get_videos(channel_ids, start_date, end_date, connection):
    """Return a DataFrame of (channel_id, video_id, title, tags, published_at)."""
    start, end = _parse_date(start_date), _parse_date(end_date)
    channel_id_set = set(channel_ids)
    months = _months_in_range(start, end)

    for year, month in months:
        refresh_video_month_cache(year, month, connection)

    frames = []
    for year, month in months:
        path = _video_cache_path(year, month)
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        df = df[df['channel_id'].isin(channel_id_set)
                & (df['published_at'] >= start) & (df['published_at'] <= end)]
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=['channel_id', 'video_id', 'title', 'tags', 'published_at'])
    return pd.concat(frames, ignore_index=True).sort_values('published_at').reset_index(drop=True)
