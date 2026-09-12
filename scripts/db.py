"""Database connection, schema initialisation, and CSV bulk loading.

The analysis reads from three MySQL tables. `SCHEMA_DDL` below reproduces them
exactly as the collection pipeline writes them, so a replication run can stand
up an empty schema, load a released data bundle into it, and then execute the
same SQL the study did.

Credentials come from a `.env` file (see `.env.example`), never from hardcoded
paths.
"""

import csv
import os
import sys

import pymysql
import pymysql.cursors
from dotenv import dotenv_values

from . import config

# ── Schema ──────────────────────────────────────────────────────────────────
# This mirrors the schema the collection pipeline actually wrote, so a bundle
# loaded into a fresh database behaves identically to the study's own tables:
# a surrogate `id` primary key with the natural key as a UNIQUE constraint (which
# is what makes the loader's `INSERT IGNORE` de-duplicate on `comment_id` /
# `video_id` / `channel_id`).
#
# The indexes are load-bearing, not cosmetic. Every analysis query filters
# `channel_id IN (...) AND published_at BETWEEN ... AND ...`, and the polarity
# queries additionally aggregate `GROUP BY author_name` over tens of millions of
# comment rows -- which is why the comment index is the three-column
# (channel_id, published_at, author_name) rather than two separate ones: it
# covers those queries outright. Without it they fall back to full table scans
# and take hours instead of seconds.

SCHEMA_DDL = {
    'youtube_channels': """
        CREATE TABLE IF NOT EXISTS youtube_channels (
            id               INT          NOT NULL AUTO_INCREMENT,
            channel_url      VARCHAR(255) NOT NULL,
            channel_id       VARCHAR(255) NOT NULL,
            title            VARCHAR(255) DEFAULT NULL,
            wing             VARCHAR(20)  DEFAULT NULL,
            published_at     DATE         DEFAULT NULL,
            subscriber_count BIGINT       DEFAULT NULL,
            video_count      INT          DEFAULT NULL,
            view_count       BIGINT       DEFAULT NULL,
            created_at       TIMESTAMP    NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (id),
            UNIQUE KEY channel_id (channel_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    'youtube_videos': """
        CREATE TABLE IF NOT EXISTS youtube_videos (
            id                 INT          NOT NULL AUTO_INCREMENT,
            channel_id         VARCHAR(255) DEFAULT NULL,
            video_id           VARCHAR(255) DEFAULT NULL,
            title              VARCHAR(500) DEFAULT NULL,
            tags               TEXT,
            published_at       DATE         DEFAULT NULL,
            channel_title      VARCHAR(255) DEFAULT NULL,
            view_count         BIGINT       DEFAULT NULL,
            like_count         BIGINT       DEFAULT NULL,
            category           VARCHAR(255) DEFAULT NULL,
            comments_collected INT          DEFAULT 0,
            PRIMARY KEY (id),
            UNIQUE KEY uq_video_id (video_id),
            KEY idx_videos_channel_published (channel_id, published_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    'youtube_comments': """
        CREATE TABLE IF NOT EXISTS youtube_comments (
            id           INT          NOT NULL AUTO_INCREMENT,
            comment_id   VARCHAR(255) NOT NULL,
            channel_id   VARCHAR(255) NOT NULL,
            video_id     VARCHAR(255) NOT NULL,
            author_name  VARCHAR(255) DEFAULT NULL,
            comment_text TEXT,
            published_at DATETIME     DEFAULT NULL,
            updated_at   DATETIME     DEFAULT NULL,
            like_count   INT          DEFAULT 0,
            PRIMARY KEY (id),
            UNIQUE KEY comment_id (comment_id),
            KEY idx_comments_channel_date_author (channel_id, published_at, author_name)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
}

# Column order used by `load_csv_dir()`. A CSV may omit trailing columns; any
# column it does not carry is written as NULL.
TABLE_COLUMNS = {
    'youtube_channels': ['channel_id', 'title', 'wing', 'published_at',
                         'subscriber_count', 'video_count', 'view_count', 'channel_url'],
    'youtube_videos':   ['video_id', 'channel_id', 'title', 'published_at', 'channel_title',
                         'tags', 'view_count', 'like_count', 'category', 'comments_collected'],
    'youtube_comments': ['comment_id', 'channel_id', 'video_id', 'author_name',
                         'comment_text', 'published_at', 'updated_at', 'like_count'],
}

CSV_FILES = {
    'youtube_channels': 'channels.csv',
    'youtube_videos':   'videos.csv',
    'youtube_comments': 'comments.csv',
}


def connect(env_path=None):
    """Open a pymysql connection using credentials from `.env`.

    Returns a connection with a DictCursor, matching what every loader in this
    package expects. The caller owns the connection and is responsible for
    closing it.
    """
    env_path = env_path or config.ENV_PATH
    env = dotenv_values(env_path)
    # Environment variables win over the file, so CI can inject credentials
    # without writing a .env to disk.
    settings = {key: os.environ.get(key, env.get(key))
                for key in ('DB_HOST', 'DB_USER', 'DB_PASSWORD', 'DB_NAME', 'DB_PORT')}

    missing = [k for k in ('DB_HOST', 'DB_USER', 'DB_PASSWORD', 'DB_NAME') if not settings[k]]
    if missing:
        raise RuntimeError(
            f'Missing database settings {missing}. Copy .env.example to {env_path} '
            f'and fill it in, or set the variables in the environment.'
        )

    return pymysql.connect(
        host=settings['DB_HOST'],
        port=int(settings['DB_PORT'] or 3306),
        user=settings['DB_USER'],
        password=settings['DB_PASSWORD'],
        db=settings['DB_NAME'],
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor,
    )


def create_schema(connection, verbose=True):
    """Create the three analysis tables if they do not already exist."""
    with connection.cursor() as cur:
        for table, ddl in SCHEMA_DDL.items():
            cur.execute(ddl)
            if verbose:
                print(f'  ensured table {table}')
    connection.commit()


def _null_if_blank(value):
    return None if value is None or value == '' else value


def load_csv(connection, table, csv_path, chunk_size=5000, verbose=True):
    """Bulk-load one CSV into one table with `INSERT IGNORE`.

    `INSERT IGNORE` rather than a plain insert so a partially-loaded table can
    be topped up by re-running the loader: rows whose primary key is already
    present are skipped instead of aborting the whole load.
    """
    columns = TABLE_COLUMNS[table]
    inserted = 0

    with open(csv_path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        present = [c for c in columns if c in reader.fieldnames]
        if not present:
            raise ValueError(f'{csv_path} has none of the expected columns for {table}: {columns}')

        placeholders = ', '.join(['%s'] * len(present))
        sql = f"INSERT IGNORE INTO {table} ({', '.join(present)}) VALUES ({placeholders})"

        batch = []
        with connection.cursor() as cur:
            for row in reader:
                batch.append(tuple(_null_if_blank(row.get(c)) for c in present))
                if len(batch) >= chunk_size:
                    cur.executemany(sql, batch)
                    inserted += len(batch)
                    batch = []
            if batch:
                cur.executemany(sql, batch)
                inserted += len(batch)
    connection.commit()

    if verbose:
        print(f'  loaded {inserted:,} rows from {os.path.basename(csv_path)} into {table}')
    return inserted


def load_csv_dir(connection, data_dir, verbose=True):
    """Load `channels.csv`, `videos.csv` and `comments.csv` from `data_dir`.

    Files that are absent are skipped with a note rather than treated as an
    error, so a bundle that ships only part of the data still loads.
    """
    totals = {}
    for table, filename in CSV_FILES.items():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            print(f'  {filename} not found in {data_dir} -- skipping {table}', file=sys.stderr)
            continue
        totals[table] = load_csv(connection, table, path, verbose=verbose)
    return totals
