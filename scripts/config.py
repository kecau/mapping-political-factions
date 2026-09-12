"""Study parameters.

Every number the analysis depends on lives here, so a replication run is fully
described by this one module. The values below are the ones the paper reports;
changing any of them changes the tracked communities and therefore every
downstream figure.
"""

import os

# ── Paths ───────────────────────────────────────────────────────────────────
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PACKAGE_DIR)

ENV_PATH   = os.environ.get('MPF_ENV',     os.path.join(PROJECT_DIR, '.env'))
CACHE_DIR  = os.environ.get('MPF_CACHE',   os.path.join(PROJECT_DIR, 'cache'))
FIGURE_DIR = os.environ.get('MPF_FIGURES', os.path.join(PROJECT_DIR, 'figures'))
TABLE_DIR  = os.environ.get('MPF_TABLES',  os.path.join(PROJECT_DIR, 'tables'))

# ── Study window ────────────────────────────────────────────────────────────
START_DATE = '2024-01-01'
END_DATE   = '2026-08-20'

# ── Snapshot generation ─────────────────────────────────────────────────────
WINDOW_DAYS = 14   # window width in days
STEP_DAYS   = 14   # step between consecutive snapshots (= WINDOW_DAYS -> non-overlapping)

# ── Greene et al. (2010) tracking parameters ────────────────────────────────
THETA = 0.50   # Jaccard threshold for matching a community to a dynamic community
PHI   = 0.50   # containment threshold: |C n DC| / |C| >= PHI also triggers a match
BETA  = 0.10   # relative size-change threshold separating growth/contraction from continuation
GRACE = 20     # snapshots a dynamic community may go unmatched before it is declared dead

# ── Louvain parameters ──────────────────────────────────────────────────────
LOUVAIN_RESOLUTION = 1.0   # < 1 favours larger communities, > 1 favours smaller ones
LOUVAIN_SEED       = 42

# ── Graph construction / filtering ──────────────────────────────────────────
# 'normalized' -> symmetric overlap ratio, bounded [0, 1]
# 'raw'        -> shared-commenter count, unbounded
NETWORK_WEIGHT_MODE = 'normalized'
MIN_EDGE_WEIGHT = {
    'normalized': 0.01,   # minimum symmetric overlap ratio to keep an edge
    'raw':        1,      # minimum shared-commenter count to keep an edge
}[NETWORK_WEIGHT_MODE]
MIN_COMMUNITY_SIZE = 2    # communities smaller than this are dropped after detection

# ── Reporting ───────────────────────────────────────────────────────────────
TOP_K_DCS    = 8   # how many dynamic communities appear in the per-DC panels
N_CORE_SHOWN = 5   # how many core/broker channels are listed per dynamic community

# Which dynamic communities the per-DC panels show.
#   'dc_id'       -> the first TOP_K_DCS by id. Ids are assigned in birth order,
#                    so this is the earliest-established set. This is what the
#                    paper's figures use.
#   'n_snapshots' -> the longest-lived ones instead.
TOP_DC_SELECTION = 'dc_id'

# ── Polarity (section 16n) ──────────────────────────────────────────────────
MIN_COMMENTS_PER_USER_POLARITY = 3    # a user needs this many in-scope comments to be scored
MIN_USERS_PER_PAIR             = 20   # a DC pair needs this many scorable users to be plotted

# Same-wing dynamic-community pairs shown alongside the left-vs-right baseline.
# These ids are only meaningful for a run with the parameters above; if they are
# absent from the tracked set, `polarity.resolve_dc_pairs()` warns and falls back
# to the longest-lived same-wing pairs.
POLARITY_DC_PAIRS = [
    (6, 7),   # left  vs. left
    (1, 5),   # right vs. right
]

# ── Topic model (section 18h) ───────────────────────────────────────────────
N_TOPICS_GLOBAL = 14   # fixed topic count for the whole-period LDA -- a shared axis
                       # every dynamic community's lifetime corpus is projected onto
LDA_RANDOM_STATE = 42
LDA_MIN_DF = 5         # a term must appear in at least this many videos
LDA_MAX_DF = 0.9       # ...and in at most this share of them
TOPIC_TOP_WORDS = 8    # keywords listed per topic

# ── Figure output ───────────────────────────────────────────────────────────
DPI = 200
SAVE_PDF = True
