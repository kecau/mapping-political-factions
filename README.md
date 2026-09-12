# Dynamic communities in the Korean political YouTube commenting network

Replication code for the accompanying paper. Everything here reproduces the
paper's quantitative results and figures from the study database: the two-stage
dynamic community tracking, the cohesion and centrality measures, the audience
polarity distributions, and the topic model.

```
mapping_political_factions/
├── replication.ipynb        # run this -- narrative + one call per result
└── scripts/                 # every module: computation and its figure(s) together
    ├── config.py            # every study parameter
    ├── channels.py          # the 96-channel study set
    ├── db.py                # connection, schema DDL, CSV loader
    ├── cache.py             # monthly Parquet cache for commenter sets and videos
    ├── style.py             # fonts, palettes, date axes, event markers, saving
    ├── data.py              # channel metadata, data summary + plot_daily_volume
    ├── network.py           # snapshot construction
    ├── communities.py       # stage 1: Louvain per snapshot
    ├── tracking.py          # stage 2: Greene et al. matching -> dynamic communities
    │                        #   + plot_timeline, plot_event_graph
    ├── cohesion.py          # internal density/conductance + its 4 figure panels
    ├── core_nodes.py        # hub/broker centrality + plot_core_nodes
    ├── polarity.py          # per-user polarity scores + plot_polarity_distributions
    ├── topics.py            # Korean tokenisation, global LDA + its 2 figures
    ├── init_db.py           # create the schema, load a data bundle
    └── run_all_figures.py   # regenerate everything headless
```

There is no separate `figures/` tree: a figure's plotting code sits directly in
the module that computes its inputs (`cohesion.py` has both `cohesion_tables()`
and `plot_wing_cohesion()`, and so on), so a result and the code that draws it
are never more than a scroll apart.

## Setup

**1. Dependencies** (Python 3.11 or newer):

```bash
pip install -r requirements.txt
```

`kiwipiepy` downloads its Korean language model on first use, so the first topic
run needs network access.

**2. Korean font.** Channel titles and topic keywords are Korean. Matplotlib
ships no CJK font, so install one if your system has none — AppleGothic (macOS)
and Malgun Gothic (Windows) are present by default; on Linux install
`fonts-nanum` or Noto Sans CJK KR. `style.apply_style()` picks the first
available and warns if it finds none (labels would otherwise render as boxes).

**3. Database credentials:**

```bash
cp .env.example .env   # then fill in DB_HOST / DB_USER / DB_PASSWORD / DB_NAME
```

**4. Schema and data:**

```bash
python -m scripts.init_db --create              # create the three tables
python -m scripts.init_db --create --load data/ # ...and load a data bundle
python -m scripts.init_db --check               # report row counts
```

`--load` expects `channels.csv`, `videos.csv` and `comments.csv` in the given
directory, with headers matching the column names below. Rows are inserted with
`INSERT IGNORE`, so an interrupted load resumes by re-running the same command.

## Running

Everything under `scripts/` uses relative imports, so run it with `-m` from
the project root (`python -m scripts.run_all_figures`), never as
`python scripts/run_all_figures.py` directly.

Either open the notebook:

```bash
jupyter lab replication.ipynb
```

or regenerate everything headless:

```bash
python -m scripts.run_all_figures                  # all of it
python -m scripts.run_all_figures --skip-topics    # without the LDA stage
```

Figures land in `figures/` (PNG at 200 dpi plus PDF), result tables in
`tables/`. Both paths are configurable via `MPF_FIGURES` / `MPF_TABLES`.

The first run streams the whole comment history into a Parquet cache under
`cache/` and takes tens of minutes; later runs read the cache and take a few
minutes. The polarity queries and the LDA fit are the slowest remaining steps.

## Data schema

Three tables, defined in `scripts/db.py`. Each carries a surrogate `id` primary key
with the natural key as a UNIQUE constraint — which is what lets the loader's
`INSERT IGNORE` de-duplicate on re-runs.

**`youtube_channels`** — one row per channel.

| Column | Type | Description |
|---|---|---|
| `channel_id` | VARCHAR(255), UNIQUE | YouTube channel id |
| `title` | VARCHAR(255) | Channel display name |
| `wing` | VARCHAR(20) | Manually coded political leaning: `left`, `middle` or `right` |
| `published_at` | DATE | Channel creation date |
| `subscriber_count`, `video_count`, `view_count` | | Channel statistics at collection time |
| `channel_url` | VARCHAR(255) | Canonical channel URL |

**`youtube_videos`** — one row per video.

| Column | Type | Description |
|---|---|---|
| `video_id` | VARCHAR(255), UNIQUE | YouTube video id |
| `channel_id` | VARCHAR(255) | Publishing channel |
| `title` | VARCHAR(500) | Video title — half the topic model's input |
| `tags` | TEXT | Comma-separated tags — the other half |
| `published_at` | DATE | Publication date |
| `channel_title`, `view_count`, `like_count`, `category`, `comments_collected` | | Collection metadata |

**`youtube_comments`** — one row per comment.

| Column | Type | Description |
|---|---|---|
| `comment_id` | VARCHAR(255), UNIQUE | YouTube comment id |
| `channel_id`, `video_id` | VARCHAR(255) | Where the comment was posted |
| `author_name` | VARCHAR(255) | Commenter identifier — the unit of the overlap network |
| `comment_text` | TEXT | Comment body (not used by any figure in this package) |
| `published_at`, `updated_at` | DATETIME | Timestamps |
| `like_count` | INT | Likes at collection time |

`author_name` is only ever used to group and count: the analysis needs a stable
identifier per commenter, not a real handle, so a released bundle can substitute
a per-dataset hash without changing any result. `comment_text` is never read by
this package and can be omitted from a bundle entirely.

The indexes are load-bearing rather than cosmetic. Every analysis query filters
`channel_id IN (...) AND published_at BETWEEN ...`, and the polarity queries
additionally `GROUP BY author_name` over tens of millions of rows — which is why
the comment index is the three-column `(channel_id, published_at, author_name)`
rather than two separate ones: it covers those queries outright. Without it they
fall back to full table scans.

## Parameters

All of these live in `scripts/config.py`. The values below are the ones the paper
reports. They are read when a function is *called*, so overriding one in the
notebook takes effect on the next cell without reimporting.

| Parameter | Value | Meaning |
|---|---|---|
| `START_DATE` – `END_DATE` | 2024-01-01 – 2026-08-20 | Study window |
| `WINDOW_DAYS` | 14 | Snapshot width |
| `STEP_DAYS` | 14 | Step between snapshots (= width, so non-overlapping) |
| `THETA` (θ) | 0.50 | Jaccard threshold for matching a community to a dynamic community |
| `PHI` (φ) | 0.50 | Containment threshold: \|C ∩ DC\| / \|C\| ≥ φ also matches |
| `BETA` (β) | 0.10 | Relative size change separating growth/contraction from continuation |
| `GRACE` (k) | 20 | Snapshots a community may go unmatched before it is declared dead |
| `LOUVAIN_RESOLUTION` | 1.0 | Louvain resolution |
| `LOUVAIN_SEED` | 42 | Louvain seed — fixed, so the partition is reproducible |
| `NETWORK_WEIGHT_MODE` | `normalized` | Symmetric overlap ratio, bounded [0, 1] |
| `MIN_EDGE_WEIGHT` | 0.01 | Minimum overlap ratio to keep an edge |
| `MIN_COMMUNITY_SIZE` | 2 | Communities smaller than this are dropped |
| `TOP_K_DCS` | 8 | Communities shown in the per-community panels |
| `N_TOPICS_GLOBAL` | 14 | Topics in the whole-period LDA |
| `MIN_COMMENTS_PER_USER_POLARITY` | 3 | Comments a user needs to be scored |
| `MIN_USERS_PER_PAIR` | 20 | Scorable users a community pair needs to be plotted |
| `POLARITY_DC_PAIRS` | (6, 7), (1, 5) | Same-wing pairs in the polarity figure |

Two of these deserve a note.

`POLARITY_DC_PAIRS` identifies specific factions, so the ids are only meaningful
under the parameters above. If a replication changes any of them, the ids will
not line up; `polarity.resolve_dc_pairs()` detects that, says so, and falls back
to the longest-lived same-wing pair per wing.

`TOP_DC_SELECTION` controls which communities the per-community panels show.
The default, `'dc_id'`, takes the first `TOP_K_DCS` by id — ids are assigned in
birth order, so this is the earliest-established set, and it is what the paper's
figures use. Set it to `'n_snapshots'` for the longest-lived set instead.

## Outputs

| Result | Produced by | Output |
|---|---|---|
| Data summary | `data.data_summary` | `tables/data_summary.csv` |
| Community detection and tracking | `communities.detect_all`, `tracking.track` | `tables/event_log.csv` |
| Dynamic-community timeline | `tracking.plot_timeline` | `figures/timeline_*.png` |
| Static event graph | `tracking.plot_event_graph` | `figures/eventgraph_*.png` |
| Daily comment volume | `data.plot_daily_volume` | `figures/daily_comment_volume.png` |
| Cohesion: density and conductance | `cohesion.cohesion_tables`, `cohesion.plot_cohesion` | `figures/cohesion_*.png`, `tables/dc_cohesion_summary.csv` |
| Core and broker channels | `core_nodes.core_nodes_table`, `core_nodes.plot_core_nodes` | `figures/core_nodes.png`, `tables/dc_core_nodes.csv` |
| Polarity distributions | `polarity.*`, `polarity.plot_polarity_distributions` | `figures/polarity_distributions_*.png` |
| Global LDA topic model | `topics.fit_global_lda` | printed topic table |
| DC × topic heatmap | `topics.plot_dc_topic_heatmap` | `figures/dc_topic_heatmap_*.png` |
| Topic prevalence over time | `topics.plot_topic_prevalence` | `figures/topic_prevalence_*.png` |

## Method

Two stages, following Greene, Doyle & Cunningham (2010), *Tracking the Evolution
of Communities in Dynamic Social Networks* (ASONAM '10):

1. **Static detection.** Each 14-day window becomes a weighted channel-channel
   graph, where an edge means the two channels shared commenters that window and
   its weight is the symmetric overlap ratio. Louvain runs independently on each.

2. **Cross-snapshot matching.** A community at snapshot *t* is matched to a
   dynamic community when Jaccard similarity ≥ θ or either containment ≥ φ. The
   matching structure yields the event log: birth, death, growth, contraction,
   merge, split, continuation. An unmatched community is not declared dead
   immediately — it stays eligible for `GRACE` snapshots, so a faction that goes
   quiet for a few weeks and returns is tracked as one community rather than two.

Two departures from the paper's source are worth recording. The bipartite
polarity computation aggregates per author in SQL instead of pulling raw comment
rows into pandas — same scores, but bounded by the number of distinct authors
rather than the number of comments. And the political-event markers are clipped
to each figure's data range, so a replication over a shorter window does not
stretch its x-axis out to the last event in the list.

## Citation

> [Citation to be added upon publication]

## License

> [License to be specified upon release]
